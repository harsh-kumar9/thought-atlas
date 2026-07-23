"""Validated, atomic merging for behavior and quality judge shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import polars as pl


KINDS = ("trackA_counts", "trackB_full", "trackB_isolated", "quality",
         "answer_extractions")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_judge_kind(kind: str, tag: str, out_dir: Path, num_shards: int) -> Path:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind}")
    rx = re.compile(rf"^{re.escape(kind)}__{re.escape(tag)}\.shard(\d+)of(\d+)\.parquet$")
    entries = []
    for path in sorted(out_dir.glob(f"{kind}__{tag}.shard*of*.parquet")):
        m = rx.match(path.name)
        if not m:
            raise ValueError(f"malformed judge shard: {path.name}")
        entries.append((int(m.group(1)), int(m.group(2)), path))
    if not entries:
        canonical = out_dir / f"{kind}__{tag}.parquet"
        if num_shards == 1 and canonical.exists():
            entries = [(0, 1, canonical)]
        else:
            raise FileNotFoundError(f"no {kind} shards for {tag}")
    if {n for _, n, _ in entries} != {num_shards}:
        raise ValueError(f"{kind}: declared shard totals do not equal {num_shards}")
    indices = [i for i, _, _ in entries]
    if len(indices) != len(set(indices)) or set(indices) != set(range(num_shards)):
        raise ValueError(f"{kind}: expected indices 0..{num_shards - 1}, found {indices}")
    frames = [pl.read_parquet(p) for _, _, p in sorted(entries)]
    if any(f.height == 0 for f in frames):
        raise ValueError(f"{kind}: an empty shard was produced")
    df = pl.concat(frames, how="diagonal_relaxed")
    key = ["trace_id", "seg_idx"] if kind.startswith("trackB") else ["trace_id"]
    if any(c not in df.columns for c in key):
        raise ValueError(f"{kind}: missing merge key {key}")
    if df.select(pl.struct(key).is_duplicated().any()).item():
        raise ValueError(f"{kind}: overlapping shard rows for {key}")
    for column in ("score_version", "judge_model"):
        if column not in df.columns or len(df[column].drop_nulls().unique()) != 1:
            raise ValueError(f"{kind}: expected one non-null {column} across all shards")
    artifact_fingerprint = None
    if kind == "answer_extractions":
        column = "extraction_fingerprint"
        if column not in df.columns or len(df[column].drop_nulls().unique()) != 1:
            raise ValueError(
                f"{kind}: expected one non-null {column} across all shards")
        artifact_fingerprint = df[column].drop_nulls().unique().item()
    out = out_dir / f"{kind}__{tag}.parquet"
    if kind == "answer_extractions" and out.exists() and all(
            out != path for _, _, path in entries):
        existing = pl.read_parquet(out)
        existing_fingerprints = (
            existing["extraction_fingerprint"].drop_nulls().unique().to_list()
            if "extraction_fingerprint" in existing.columns else [])
        if existing_fingerprints != [artifact_fingerprint]:
            raise ValueError(
                f"{out} has a different extraction fingerprint; use a fresh output directory")
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.sort(key).write_parquet(tmp); tmp.replace(out)
    manifest = {"schema_version": 2, "kind": kind, "judge_tag": tag,
                "rows": df.height, "sha256": _sha(out), "key": key,
                "source_shards": [{"index": i, "file": p.name, "sha256": _sha(p)}
                                  for i, _, p in sorted(entries)]}
    if artifact_fingerprint is not None:
        manifest["artifact_fingerprint"] = artifact_fingerprint
    mp = out.with_suffix(".manifest.json"); mt = mp.with_suffix(mp.suffix + ".tmp")
    mt.write_text(json.dumps(manifest, indent=2) + "\n"); mt.replace(mp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="judge model id with '/' replaced by '_'")
    ap.add_argument("--out-dir", default="data/judge/prod")
    ap.add_argument("--num-shards", required=True, type=int)
    ap.add_argument("--kinds", nargs="+", choices=KINDS, required=True)
    a = ap.parse_args()
    for kind in a.kinds:
        out = merge_judge_kind(kind, a.tag, Path(a.out_dir), a.num_shards)
        print(f"[judge-merge] validated -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
