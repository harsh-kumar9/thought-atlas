"""D-64 stage 1: readout corpus and held-out split for Gemma-4-12B from the atlas (no new generation).

For each domain in math, gpqa, planning, idea: take the atlas Gemma-4-12B traces (data/v2/traces/traces_gemma4_12b.parquet)
that have full-context Track B labels from the primary judge, split their instance_ids with a seeded shuffle into a readout
set (300) and a held-out set (the rest, about 200), and write
  data/gemma4_12b/runs/atlas_<dom>/traces_baseline.parquet            readout traces (condition "baseline")
  data/gemma4_12b/runs/atlas_<dom>/judge/trackB_atlas/trackB_full__<judge tag>.parquet   their Track B rows
  data/gemma4_12b/heldout_split.json                                   readout / held-out instance ids per domain
  data/gemma4_12b/tasks_heldout_<dom>.parquet                          held-out task rows (from data/v2/tasks)
The runs/atlas_<dom> layout is what signature_ctrl extract, archetypes and success_dir read.
Run with MC_MODEL=gemma4_12b.
"""
from __future__ import annotations

import argparse
import json
import random

import polars as pl

from .common import DATA, GEN_MODEL_KEY, JUDGE_MODEL, REPO, TASKS_DIR, TRACKB_BASELINE

DOMS = ["math", "gpqa", "planning", "idea"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-readout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261014)
    a = ap.parse_args()
    assert GEN_MODEL_KEY == "gemma4_12b", "run with MC_MODEL=gemma4_12b"
    tag = JUDGE_MODEL.replace("/", "_")
    tr = pl.read_parquet(REPO / "data/v2/traces/traces_gemma4_12b.parquet")
    tb = pl.read_parquet(TRACKB_BASELINE)
    if "judge_model" in tb.columns:
        tb = tb.filter(pl.col("judge_model") == JUDGE_MODEL)
    if "context_mode" in tb.columns:
        tb = tb.filter(pl.col("context_mode") == "full")
    if "section_type" in tb.columns:
        tb = tb.filter(pl.col("section_type") == "think")
    labelled = set(tb["trace_id"].unique().to_list())
    split = {}
    rng = random.Random(a.seed)
    for dom in DOMS:
        d = tr.filter((pl.col("task_type") == dom) & pl.col("trace_id").is_in(list(labelled)) & pl.col("reasoning_text_for_analysis").is_not_null())
        ids = sorted(d["instance_id"].unique().to_list())
        rng.shuffle(ids)
        readout = sorted(ids[: a.n_readout])
        tasks = pl.read_parquet(TASKS_DIR / f"{dom}.parquet")
        held = sorted(set(tasks["instance_id"].to_list()) - set(readout))      # every task not in the readout set, labelled or not
        split[dom] = {"readout": readout, "heldout": held}
        rd = DATA / "runs" / f"atlas_{dom}"; (rd / "judge" / "trackB_atlas").mkdir(parents=True, exist_ok=True)
        rt = d.filter(pl.col("instance_id").is_in(readout)).with_columns(pl.lit("baseline").alias("condition"), pl.lit(1).alias("seed"))
        rt.write_parquet(rd / "traces_baseline.parquet")
        tb.filter(pl.col("trace_id").is_in(rt["trace_id"].implode())).write_parquet(rd / "judge" / "trackB_atlas" / f"trackB_full__{tag}.parquet")
        th = tasks.filter(pl.col("instance_id").is_in(held))
        th.write_parquet(DATA / f"tasks_heldout_{dom}.parquet")
        print(f"[gstage] {dom}: {d.height} labelled traces, readout {rt.height} traces / {len(readout)} tasks, held-out {len(held)} tasks ({th.height} task rows); "
              f"mean reasoning chars {int(rt['reasoning_text_for_analysis'].str.len_chars().mean())}")
    (DATA / "heldout_split.json").write_text(json.dumps({"seed": a.seed, "n_readout": a.n_readout, "domains": split}, indent=1))
    print("->", DATA / "heldout_split.json")


if __name__ == "__main__":
    main()
