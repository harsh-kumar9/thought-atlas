"""Validate and atomically merge trace shards into a canonical parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import polars as pl


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_trace_shards(model: str, out_dir: Path, num_shards: int | None = None,
                       *, delete_shards: bool = False) -> Path:
    pattern = re.compile(
        rf"^traces_{re.escape(model)}\.shard(?P<idx>\d+)of(?P<total>\d+)\.parquet$"
    )
    entries = []
    for path in sorted(out_dir.glob(f"traces_{model}.shard*of*.parquet")):
        m = pattern.match(path.name)
        if not m:
            raise ValueError(f"malformed shard filename: {path.name}")
        entries.append((int(m.group("idx")), int(m.group("total")), path))
    if not entries:
        canonical = out_dir / f"traces_{model}.parquet"
        if canonical.exists() and num_shards in (None, 1):
            entries = [(0, 1, canonical)]
        else:
            raise FileNotFoundError(f"no shards for {model} in {out_dir}")

    totals = {total for _, total, _ in entries}
    if len(totals) != 1:
        raise ValueError(f"inconsistent shard totals: {sorted(totals)}")
    declared = totals.pop()
    expected_n = num_shards if num_shards is not None else declared
    if declared != expected_n:
        raise ValueError(f"shards declare {declared}, expected {expected_n}")
    indices = [idx for idx, _, _ in entries]
    if len(indices) != len(set(indices)) or set(indices) != set(range(expected_n)):
        raise ValueError(f"shard coverage must be exactly 0..{expected_n - 1}; found {indices}")

    frames = []
    for idx, _, path in sorted(entries):
        frame = pl.read_parquet(path)
        if frame.height == 0:
            raise ValueError(f"empty shard {idx}: {path}")
        frames.append(frame)
    df = pl.concat(frames, how="diagonal_relaxed")
    key = ["gen_model", "instance_id", "seed"]
    missing = [c for c in key + ["trace_id", "generation_fingerprint"] if c not in df.columns]
    if missing:
        raise ValueError(f"v2 trace columns missing: {missing}")
    if df.select(pl.struct(key).is_duplicated().any()).item():
        raise ValueError("overlapping shards: duplicate (gen_model, instance_id, seed)")
    if df["trace_id"].n_unique() != df.height:
        raise ValueError("duplicate trace_id across shards")
    fingerprints = df["generation_fingerprint"].drop_nulls().unique().to_list()
    if len(fingerprints) != 1:
        raise ValueError(f"expected one generation fingerprint, found {fingerprints}")

    out = out_dir / f"traces_{model}.parquet"
    if out.exists() and all(out != p for _, _, p in entries):
        existing = pl.read_parquet(out)
        if "generation_fingerprint" not in existing.columns:
            raise ValueError(f"refusing to overwrite legacy canonical file {out}; use a fresh directory")
        existing_fp = existing["generation_fingerprint"].drop_nulls().unique().to_list()
        if existing_fp != fingerprints:
            raise ValueError(f"canonical {out} has a different generation fingerprint; use a fresh directory")
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.sort(key).write_parquet(tmp)
    tmp.replace(out)
    counts = (df.group_by(["gen_model", "task_type"]).len().sort(["gen_model", "task_type"])
              .to_dicts())
    manifest = {
        "schema_version": 2,
        "canonical": out.name,
        "sha256": _sha256(out),
        "rows": df.height,
        "generation_fingerprint": fingerprints[0],
        "natural_key": key,
        "counts": counts,
        "source_shards": [{"index": i, "file": p.name, "sha256": _sha256(p)}
                          for i, _, p in sorted(entries)],
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_tmp.replace(manifest_path)
    if delete_shards:
        for _, _, path in entries:
            if path != out:
                path.unlink()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="data/traces")
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--delete-shards", action="store_true",
                    help="delete source shards only after validation and atomic merge")
    args = ap.parse_args()
    out = merge_trace_shards(args.model, Path(args.out_dir), args.num_shards,
                             delete_shards=args.delete_shards)
    print(f"[merge] validated -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
