"""src/generate/merge_shards.py — concat per-shard trace parquets into traces_<model>.parquet.

Each data-parallel replica writes traces_<model>.shardNNofKK.parquet (no write collisions).
This merges them into the canonical traces_<model>.parquet. Idempotent; safe to re-run after a
resume. Downstream (judge/analysis) also accepts the shard glob, so a missed merge isn't fatal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="data/traces")
    ap.add_argument("--keep-shards", action="store_true", help="don't delete shard files after merge")
    args = ap.parse_args()

    d = Path(args.out_dir)
    shards = sorted(d.glob(f"traces_{args.model}.shard*of*.parquet"))
    if not shards:
        print(f"[merge] no shards for {args.model} in {d}; nothing to do")
        return 0
    df = pl.concat([pl.read_parquet(p) for p in shards], how="diagonal_relaxed")
    # de-dup on the natural key in case a resumed shard overlapped
    if {"gen_model", "instance_id", "seed"}.issubset(df.columns):
        df = df.unique(subset=["gen_model", "instance_id", "seed"], keep="last", maintain_order=True)
    out = d / f"traces_{args.model}.parquet"
    tmp = out.with_suffix(".parquet.tmp"); df.write_parquet(tmp); tmp.replace(out)
    print(f"[merge] {len(shards)} shards -> {out} ({df.height} traces)")
    if not args.keep_shards:
        for p in shards:
            p.unlink()
        print(f"[merge] removed {len(shards)} shard files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
