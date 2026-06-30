"""src/generate/repair_byte_bpe.py — undo byte-level BPE mojibake in stored trace text.

Some vLLM/tokenizer combinations returned `o.outputs[0].text` as the pre-byte-decode token
concatenation (spaces shown as 'Ġ', newlines as 'Ċ', etc.) instead of decoded UTF-8. The mapping
is the GPT-2/Llama bytes_to_unicode table and is fully reversible — so existing traces can be
repaired in place without regenerating. Idempotent: text with no stand-in chars passes through.

Usage:
    python -m src.generate.repair_byte_bpe --glob "data/traces/traces_*.parquet"
    python -m src.generate.repair_byte_bpe --glob "data/traces/_calib*.parquet" --dry-run
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import polars as pl

TEXT_COLS = ["full_text", "think_text", "answer_text", "reasoning_text_for_analysis"]


def _bytes_to_unicode() -> dict[str, int]:
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}   # stand-in char -> original byte


_U2B = _bytes_to_unicode()


def repair(s: str | None) -> str | None:
    if s is None:
        return None
    # fast path: nothing to do
    if "Ġ" not in s and "Ċ" not in s:
        return s
    out = bytearray()
    for ch in s:
        b = _U2B.get(ch)
        if b is not None:
            out.append(b)
        else:
            out.extend(ch.encode("utf-8"))
    return out.decode("utf-8", errors="replace")


def is_corrupt(df: pl.DataFrame) -> bool:
    if "full_text" not in df.columns or df.height == 0:
        return False
    sample = "".join(str(x) for x in df["full_text"].head(20).to_list() if x)
    return ("Ġ" in sample) or ("Ċ" in sample)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"no files match {args.glob}")
        return 1
    for f in files:
        df = pl.read_parquet(f)
        if not is_corrupt(df):
            print(f"[ok]   {f} — already clean, skipping")
            continue
        cols = [c for c in TEXT_COLS if c in df.columns]
        before = df[cols[0]][0]
        df = df.with_columns([
            pl.col(c).map_elements(repair, return_dtype=pl.Utf8).alias(c) for c in cols
        ])
        after = df[cols[0]][0]
        print(f"[fix]  {f} — repaired {len(cols)} cols")
        print(f"        before: {repr(str(before)[:70])}")
        print(f"        after : {repr(str(after)[:70])}")
        if not args.dry_run:
            tmp = Path(f).with_suffix(".parquet.tmp")
            df.write_parquet(tmp); tmp.replace(f)
    if args.dry_run:
        print("\n(dry run — no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
