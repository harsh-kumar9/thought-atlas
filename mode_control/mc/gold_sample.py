"""Human-audit sample for the behaviour labels (D-65a).

Draws segments stratified by behaviour x judge label x source from existing Track B outputs of the primary judge, attaches
the segment text and its preceding context by re-running the repo segmenter on the text the judge saw, and writes a blind
coding sheet plus the answer key and the sampling weights.

Sources (name: where the Track B rows and the traces come from):
  atlas_math, atlas_idea  canonical atlas Track B (full context) joined to data/v2/traces/traces_qwen35_9b.parquet,
                          task_type math / idea, unsteered
  a2_math                 p1_budget, condition ol_math_A2, seed 1 (judge/trackB_s1; staged text in judge_in_s1)
  mathvec_idea            sigx_idea, condition ol_math_A2_samevec_g3p0 (judge/job*/trackB_full*)
Any other source can be given as name=run:condition[:seed].

Outputs under data/gold/: gold_items.parquet (everything, including the full preceding trace), coding_sheet.csv (blind:
item_id, context, segment, empty label columns, notes), answer_key.csv (item_id, source, condition, trace_id, seg_idx, the
eight judge labels, cell and its sampling probability).

Usage: python -m mode_control.mc.gold_sample --n-per-cell 8 --context 5 --seed 20261014
"""
from __future__ import annotations

import argparse
import glob
import json
import random

import polars as pl

from .common import BEH, DATA, JUDGE_MODEL, TRACES_BASELINE, TRACKB_BASELINE
from src.judge.run_judge import _analysis_segment_text
from src.segment.segmenter import segment_text

GOLD = DATA / "gold"
DEFAULT_SOURCES = {
    "atlas_math": "atlas:math",
    "atlas_idea": "atlas:idea",
    "a2_math": "p1_budget:ol_math_A2:1",
    "mathvec_idea": "sigx_idea:ol_math_A2_samevec_g3p0",
}


def _seg_text(seg) -> str:
    if isinstance(seg, dict):
        for k in ("text", "sentence", "segment", "content"):
            if k in seg and seg[k] is not None:
                return str(seg[k])
        return json.dumps(seg)
    return str(seg)


def _trackb_files(run: str) -> list[str]:
    return sorted(glob.glob(str(DATA / "runs" / run / "judge" / "**" / "trackB_full__*.parquet"), recursive=True))


def _primary_judge(tb: pl.DataFrame) -> pl.DataFrame:
    """Keep the primary judge's rows only (a second judge writes files with the same name pattern)."""
    if "judge_model" in tb.columns:
        tb = tb.filter(pl.col("judge_model") == JUDGE_MODEL)
    if "section_type" in tb.columns:
        tb = tb.filter(pl.col("section_type") == "think")
    if "context_mode" in tb.columns:
        tb = tb.filter(pl.col("context_mode") == "full")
    return tb


def load_source(spec: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (track B rows with a condition column, traces with the text the judge saw)."""
    parts = spec.split(":")
    if parts[0] == "atlas":
        task = parts[1]
        tr = pl.read_parquet(TRACES_BASELINE).filter(pl.col("task_type") == task).with_columns(pl.lit("unsteered").alias("condition"))
        tb = _primary_judge(pl.read_parquet(TRACKB_BASELINE).filter(pl.col("trace_id").is_in(tr["trace_id"].implode())))
        return tb.with_columns(pl.lit("unsteered").alias("condition")), tr
    run, cond = parts[0], parts[1]
    seed = int(parts[2]) if len(parts) > 2 else None
    rd = DATA / "runs" / run
    staged = rd / "judge_in_s1" / f"traces_{cond}.parquet"
    tr = pl.read_parquet(staged) if staged.exists() else pl.read_parquet(rd / f"traces_{cond}.parquet")
    if seed is not None and "seed" in tr.columns:
        tr = tr.filter(pl.col("seed") == seed)
    files = _trackb_files(run)
    if not files:
        raise SystemExit(f"no Track B files under {rd / 'judge'}")
    tb = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    tb = _primary_judge(tb).unique(["trace_id", "seg_idx"], keep="last")
    tb = tb.filter(pl.col("trace_id").is_in(tr["trace_id"].implode()))
    return tb.with_columns(pl.lit(cond).alias("condition")), tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=8)
    ap.add_argument("--context", type=int, default=5, help="preceding segments shown with the target on the sheet")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--sources", default=None, help="comma list name=spec; default = the four D-65a sources")
    a = ap.parse_args()
    sources = DEFAULT_SOURCES if a.sources is None else dict(s.split("=", 1) for s in a.sources.split(","))
    rng = random.Random(a.seed)
    GOLD.mkdir(parents=True, exist_ok=True)

    picked, used = [], set()
    traces_by_source: dict[str, pl.DataFrame] = {}
    cell_sizes = []
    for name, spec in sources.items():
        tb, tr = load_source(spec)
        traces_by_source[name] = tr
        print(f"[gold] {name}: {tb.height} labelled segments from {tb['trace_id'].n_unique()} traces")
        for beh in BEH:
            for lab in (1, 0):
                cand = tb.filter(pl.col(beh) == lab).select(["trace_id", "seg_idx", "condition"] + BEH).to_dicts()
                n_avail = len(cand)
                cand = [c for c in cand if (name, c["trace_id"], c["seg_idx"]) not in used]
                rng.shuffle(cand)
                take = cand[: a.n_per_cell]
                cell_sizes.append({"source": name, "cell_behaviour": beh, "cell_label": lab, "n_available": n_avail, "n_sampled": len(take),
                                   "sampling_prob": (len(take) / n_avail) if n_avail else 0.0})
                if len(take) < a.n_per_cell:
                    print(f"[gold]   {name} {beh}={lab}: only {len(take)} available")
                for c in take:
                    used.add((name, c["trace_id"], c["seg_idx"]))
                    picked.append({"source": name, "cell_behaviour": beh, "cell_label": lab, **c})

    items, skipped = [], 0
    seg_cache: dict[tuple[str, str], list] = {}
    for p in picked:
        key = (p["source"], p["trace_id"])
        if key not in seg_cache:
            row = traces_by_source[p["source"]].filter(pl.col("trace_id") == p["trace_id"]).to_dicts()[0]
            seg_cache[key] = segment_text(_analysis_segment_text(row))
        segs = seg_cache[key]
        i = int(p["seg_idx"])
        if i >= len(segs):
            skipped += 1
            continue
        prev = [_seg_text(s) for s in segs[:i]]
        items.append({**p, "n_segments_resegmented": len(segs), "context": "\n".join(prev[-a.context:]) if a.context else "",
                      "context_full": "\n".join(prev), "segment": _seg_text(segs[i])})
    print(f"[gold] {len(items)} items, {skipped} skipped (segment index beyond the re-segmented trace)")

    rng.shuffle(items)
    for n, it in enumerate(items):
        it["item_id"] = f"g{n:03d}"
    df = pl.DataFrame(items)
    sizes = pl.DataFrame(cell_sizes)
    df = df.join(sizes, on=["source", "cell_behaviour", "cell_label"], how="left")
    df.write_parquet(GOLD / "gold_items.parquet")
    sizes.write_csv(GOLD / "cell_sizes.csv")
    sheet = df.select(["item_id", "context", "segment"]).with_columns([pl.lit("").alias(b) for b in BEH] + [pl.lit("").alias("notes")])
    sheet.write_csv(GOLD / "coding_sheet.csv")
    df.select(["item_id", "source", "condition", "trace_id", "seg_idx", "cell_behaviour", "cell_label", "sampling_prob"] + BEH).write_csv(GOLD / "answer_key.csv")
    print(sizes.sort(["source", "cell_behaviour", "cell_label"]))
    print(f"[gold] wrote {GOLD / 'gold_items.parquet'}, coding_sheet.csv, answer_key.csv, cell_sizes.csv")


if __name__ == "__main__":
    main()
