"""Re-grade the atlas (data/v2) with the free-text answer extraction (D-47 applied to the main pipeline, D-48).

The repo extractor's JSON grammar makes gemma answer "no_answer" for almost every math/MCQ trace (D-46), so the
canonical data/v2/perf/success_grades.parquet fell back to the deterministic parser, which misreads bold answers.
This module stages the atlas math/gpqa/planning/security traces for mc.extract_free and writes a parallel grade file;
it never overwrites the canonical one.

  stage    data/v2/traces/traces_<model>.parquet -> data/v2/extract_free_atlas/in/traces_<model>.parquet
  grade    after `mode.sbatch freexa` shards: extract_free grading -> data/v2/perf/success_grades_freex.parquet
           and a model x domain comparison with the canonical grades (data/v2/extract_free_atlas/compare.csv)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "data/v2"
BASE = V2 / "extract_free_atlas"
DOMS = ["math", "gpqa", "planning", "security"]
COLS = ["trace_id", "gen_model", "instance_id", "task_type", "seed", "reference_answer", "prompt", "full_text", "think_text",
        "answer_text", "n_new_tokens", "completed", "finish_reason", "parse_status", "hard_loop"]


def stage():
    out = BASE / "in"; out.mkdir(parents=True, exist_ok=True)
    tot = 0
    for p in sorted((V2 / "traces").glob("traces_*.parquet")):
        if ".shard" in p.name:
            continue
        t = pl.read_parquet(p, columns=COLS).filter(pl.col("task_type").is_in(DOMS))
        t.write_parquet(out / p.name); tot += t.height
        print(f"[stage] {p.name}: {t.height} traces ({t.group_by('task_type').len().sort('task_type').rows()})")
    print(f"[stage] {tot} traces -> {out}")


def grade():
    os.environ["XF_IN"] = str(BASE / "in"); os.environ["XF_OUT"] = str(BASE / "out")
    import importlib
    from . import extract_free
    importlib.reload(extract_free)
    extract_free.cmd_grade()
    g = pl.read_parquet(BASE / "out" / "grades.parquet")
    tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "gen_model", "instance_id"]) for p in sorted((BASE / "in").glob("traces_*.parquet"))])
    g = g.join(tr, on="trace_id")
    g.select(["trace_id", "gen_model", "instance_id", "task_type", "success", "direct", "grade_source", "extracted"]).write_parquet(V2 / "perf" / "success_grades_freex.parquet")
    old = pl.read_parquet(V2 / "perf" / "success_grades.parquet", columns=["trace_id", "success"]).rename({"success": "canonical"})
    c = g.join(old, on="trace_id", how="left")
    cmp_ = (c.group_by(["gen_model", "task_type"])
             .agg(pl.len().alias("n"), pl.col("canonical").mean().alias("canonical_acc"), pl.col("success").mean().alias("freex_acc"),
                  (pl.col("success") != pl.col("canonical")).mean().alias("changed"),
                  (pl.col("grade_source") == "extracted").mean().alias("share_extracted"))
             .with_columns((pl.col("freex_acc") - pl.col("canonical_acc")).alias("delta"))
             .sort(["task_type", "gen_model"]))
    cmp_.write_csv(BASE / "compare.csv")
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
    print(cmp_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "grade"])
    a = ap.parse_args()
    stage() if a.cmd == "stage" else grade()


if __name__ == "__main__":
    main()
