"""Re-graded math accuracy per run and arm, paired against each run's unsteered arm (D-47 follow-up for the older runs).

Reads a staged extraction set (XF_IN with traces_<run>__<cond>.parquet, XF_OUT/grades.parquet from `extract_free grade`)
and prints, per run and arm: n, answered, accuracy with the extraction grades and with the old direct grader, and the
paired change vs the run's baseline (per-task mean over seeds, task bootstrap). -> XF_OUT/report.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xf-in", default=os.environ.get("XF_IN", "mode_control/data/extract_in2"))
    ap.add_argument("--xf-out", default=os.environ.get("XF_OUT", "mode_control/data/extract_free2"))
    a = ap.parse_args()
    IN, OUT = Path(a.xf_in), Path(a.xf_out)
    tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "instance_id", "seed", "task_type", "answer_text"]).with_columns(
        pl.lit(p.stem.removeprefix("traces_").split("__")[0]).alias("run"), pl.lit(p.stem.split("__")[1]).alias("arm"))
        for p in sorted(IN.glob("traces_*.parquet"))], how="diagonal_relaxed").filter(pl.col("task_type") == "math")
    g = pl.read_parquet(OUT / "grades.parquet").select(["trace_id", "success", "direct", "grade_source"]).unique("trace_id")
    d = tr.join(g, on="trace_id", how="left").with_columns((pl.col("answer_text").fill_null("").str.len_chars() > 0).cast(pl.Float64).alias("answered"))
    rng = np.random.default_rng(0); rows = []
    for (run,), dr in d.group_by(["run"]):
        task = dr.group_by(["arm", "instance_id"]).agg(pl.col("success").mean(), pl.col("direct").mean(), pl.col("answered").mean())
        base = task.filter(pl.col("arm") == "baseline").drop("arm")
        for (arm,), ga in task.group_by(["arm"]):
            rec = {"run": run, "arm": arm, "n_tasks": ga.height, "n_traces": dr.filter(pl.col("arm") == arm).height,
                   "answered": float(ga["answered"].mean()), "acc": float(ga["success"].mean()), "acc_direct_grader": float(ga["direct"].mean())}
            if arm != "baseline" and base.height:
                j = ga.join(base, on="instance_id", suffix="_b")
                for m in ("success", "direct"):
                    x = (j[m] - j[f"{m}_b"]).to_numpy().astype(float)
                    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
                    k = "d_acc" if m == "success" else "d_acc_direct_grader"
                    rec[k], rec[f"{k}_lo"], rec[f"{k}_hi"] = float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
            rows.append(rec)
    df = pl.DataFrame(rows).sort(["run", "arm"]); df.write_csv(OUT / "report.csv")
    pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(240)
    print(df.select(["run", "arm", "n_tasks", "answered", "acc", "acc_direct_grader", "d_acc", "d_acc_lo", "d_acc_hi", "d_acc_direct_grader"]))
    print(d.group_by("grade_source").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
