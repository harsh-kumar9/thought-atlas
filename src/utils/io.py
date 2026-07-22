"""Validated parquet discovery helpers.

Broad globs such as ``traces_*.parquet`` also match per-worker shards.  Loading both
the canonical file and its shards silently duplicates observations, while loading an
incomplete shard set silently drops observations.  All trace consumers should use
``resolve_trace_paths`` instead of calling ``glob`` directly.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path


_SHARD_RE = re.compile(r"^(?P<base>.+)\.shard(?P<idx>\d+)of(?P<total>\d+)\.parquet$")


def resolve_trace_paths(pattern: str) -> list[Path]:
    """Resolve a trace glob without duplicate or incomplete shard inputs.

    If a canonical parquet exists, matching shards for that canonical file are
    excluded.  If only shards exist, their declared total and exact index coverage
    are validated before they are returned.
    """
    matched = [Path(p) for p in sorted(glob.glob(pattern))]
    canonical: list[Path] = []
    groups: dict[str, list[tuple[Path, int, int]]] = {}
    for path in matched:
        m = _SHARD_RE.match(str(path))
        if m:
            groups.setdefault(m.group("base"), []).append(
                (path, int(m.group("idx")), int(m.group("total")))
            )
        else:
            canonical.append(path)

    canonical_bases = {str(p.with_suffix("")) for p in canonical}
    resolved = list(canonical)
    for base, entries in groups.items():
        if base in canonical_bases:
            continue
        totals = {total for _, _, total in entries}
        if len(totals) != 1:
            raise ValueError(f"inconsistent shard totals for {base}: {sorted(totals)}")
        total = totals.pop()
        indices = [idx for _, idx, _ in entries]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate shard indices for {base}: {indices}")
        expected = set(range(total))
        found = set(indices)
        if found != expected:
            missing = sorted(expected - found)
            extra = sorted(found - expected)
            raise ValueError(
                f"incomplete shard set for {base}: missing={missing}, extra={extra}, total={total}"
            )
        resolved.extend(path for path, _, _ in sorted(entries, key=lambda x: x[1]))
    return sorted(resolved)
