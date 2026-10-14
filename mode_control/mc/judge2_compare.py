"""Second-judge comparison (D-65b): does the behaviour story depend on the judge?

For each run and arm that both judges have labelled (gemma-4-31B-it in judge/trackB_* and the second judge in
judge/trackB2_*), this computes
  1. segment-level agreement: per behaviour, Cohen's kappa between the two judges on the same segments, and each judge's
     positive rate;
  2. trace-level behaviour rates over the 8k window (the same windowed rates as tx_analyze) under each judge, the paired
     shift of every steered arm against the run's baseline under each judge, the cosine between the two 8-dim shift
     vectors, and sign agreement on the behaviours whose gemma CI excludes zero.
Pass rule from D-65b: cosine above 0.8 and the same signs on subgoal, Q&A and verification.

Usage: python -m mode_control.mc.judge2_compare --runs p1_budget,tx_idea,cd_countdown [--judge2 Qwen/Qwen3.5-27B]
Output: data/judge2/agreement.csv, data/judge2/shifts.csv, data/judge2/summary.csv (printed).
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from .common import BEH, DATA, JUDGE_MODEL
from .tx_analyze import rates

OUT = DATA / "judge2"


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load_tb(run: str, judge: str, sub: str) -> pl.DataFrame:
    tag = judge.replace("/", "_")
    paths = sorted((DATA / "runs" / run / "judge").rglob(f"{sub}*/trackB_full__{tag}*.parquet"))
    if not paths:
        return pl.DataFrame()
    tb = pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")
    if "judge_model" in tb.columns:
        tb = tb.filter(pl.col("judge_model") == judge)
    if "context_mode" in tb.columns:
        tb = tb.filter(pl.col("context_mode") == "full")
    return tb.unique(["trace_id", "seg_idx"], keep="last")


def traces_for(run: str, trace_ids: list[str]) -> pl.DataFrame:
    rd = DATA / "runs" / run
    tr = pl.concat([pl.read_parquet(p) for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed")
    return tr.filter(pl.col("trace_id").is_in(trace_ids))


def paired_shift(r: pl.DataFrame, base_cond: str, seed: int = 0):
    """Per-condition mean of (rate - baseline rate) over instance_id, with a 2000-draw task bootstrap."""
    rng = np.random.default_rng(seed)
    b = r.filter(pl.col("condition") == base_cond).select(["instance_id"] + [pl.col(f"{x}__rate").alias(f"b_{x}") for x in BEH])
    rows = []
    for cond in sorted(set(r["condition"].to_list()) - {base_cond}):
        j = r.filter(pl.col("condition") == cond).join(b, on="instance_id")
        rec = {"condition": cond, "n_pairs": j.height}
        for x in BEH:
            d = (j[f"{x}__rate"] - j[f"b_{x}"]).to_numpy() * 100
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)] if len(d) else [np.nan]
            rec[f"d_{x}"] = float(d.mean()) if len(d) else np.nan
            rec[f"lo_{x}"] = float(np.quantile(bs, 0.025)); rec[f"hi_{x}"] = float(np.quantile(bs, 0.975))
        rows.append(rec)
    return pl.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="p1_budget,tx_idea,cd_countdown")
    ap.add_argument("--judge2", default="Qwen/Qwen3.5-27B")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    agree, shifts, summary = [], [], []
    for run in a.runs.split(","):
        tb1, tb2 = load_tb(run, JUDGE_MODEL, "trackB"), load_tb(run, a.judge2, "trackB2")
        if tb2.height == 0:
            print(f"[judge2] {run}: no second-judge labels yet"); continue
        for col in ("kim_parsed", "gandhi_parsed"):
            if col in tb1.columns:
                tb1 = tb1.filter(pl.col(col))
            if col in tb2.columns:
                tb2 = tb2.filter(pl.col(col))
        common = tb1.join(tb2, on=["trace_id", "seg_idx"], suffix="_j2").drop_nulls(BEH + [f"{x}_j2" for x in BEH])
        tr = traces_for(run, common["trace_id"].unique().to_list())
        cond_of = dict(zip(tr["trace_id"].to_list(), tr["condition"].to_list()))
        common = common.with_columns(pl.col("trace_id").replace_strict(cond_of, default=None).alias("condition"))
        print(f"[judge2] {run}: {common.height} segments labelled by both judges, {common['trace_id'].n_unique()} traces")
        for (cond,), g in common.group_by(["condition"]):
            for x in BEH:
                j1, j2 = g[x].to_numpy().astype(int), g[f"{x}_j2"].to_numpy().astype(int)
                agree.append({"run": run, "condition": cond, "behaviour": x, "n_segments": len(j1), "kappa": kappa(j1, j2),
                              "pos_rate_gemma": float(j1.mean()), "pos_rate_j2": float(j2.mean()), "agreement": float((j1 == j2).mean())})
        # trace-level rates under each judge, restricted to traces both judged
        ids = common["trace_id"].unique().to_list()
        tr2 = tr.filter(pl.col("trace_id").is_in(ids))
        r1 = rates(tr2, tb1.filter(pl.col("trace_id").is_in(ids))).join(tr2.select(["trace_id", "instance_id", "condition"]), on="trace_id")
        r2 = rates(tr2, tb2.filter(pl.col("trace_id").is_in(ids))).join(tr2.select(["trace_id", "instance_id", "condition"]), on="trace_id")
        s1, s2 = paired_shift(r1, "baseline"), paired_shift(r2, "baseline")
        for c in s1["condition"].to_list():
            v1 = np.array([s1.filter(pl.col("condition") == c)[f"d_{x}"][0] for x in BEH])
            v2 = np.array([s2.filter(pl.col("condition") == c)[f"d_{x}"][0] for x in BEH])
            cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            sig = [x for x in BEH if not (s1.filter(pl.col("condition") == c)[f"lo_{x}"][0] <= 0 <= s1.filter(pl.col("condition") == c)[f"hi_{x}"][0])]
            same = sum(np.sign(s1.filter(pl.col("condition") == c)[f"d_{x}"][0]) == np.sign(s2.filter(pl.col("condition") == c)[f"d_{x}"][0]) for x in sig)
            key3 = {x: (np.sign(s1.filter(pl.col("condition") == c)[f"d_{x}"][0]) == np.sign(s2.filter(pl.col("condition") == c)[f"d_{x}"][0])) for x in ("subgoal", "Question_and_Answering", "verification")}
            summary.append({"run": run, "condition": c, "cosine_shift_vectors": cos, "n_sig_gemma": len(sig), "same_sign_among_sig": int(same),
                            "same_sign_subgoal_qa_verif": int(sum(key3.values())), "size_ratio_j2_over_gemma": float(np.linalg.norm(v2) / (np.linalg.norm(v1) + 1e-9)),
                            "pass": bool(cos > 0.8 and all(key3.values()))})
            for x in BEH:
                shifts.append({"run": run, "condition": c, "behaviour": x,
                               "d_gemma": float(s1.filter(pl.col("condition") == c)[f"d_{x}"][0]), "lo_gemma": float(s1.filter(pl.col("condition") == c)[f"lo_{x}"][0]), "hi_gemma": float(s1.filter(pl.col("condition") == c)[f"hi_{x}"][0]),
                               "d_j2": float(s2.filter(pl.col("condition") == c)[f"d_{x}"][0]), "lo_j2": float(s2.filter(pl.col("condition") == c)[f"lo_{x}"][0]), "hi_j2": float(s2.filter(pl.col("condition") == c)[f"hi_{x}"][0])})
    if agree:
        A = pl.DataFrame(agree); A.write_csv(OUT / "agreement.csv")
        print(A.group_by("behaviour").agg([pl.col("kappa").mean(), pl.col("pos_rate_gemma").mean(), pl.col("pos_rate_j2").mean(), pl.col("n_segments").sum()]).sort("behaviour"))
    if shifts:
        pl.DataFrame(shifts).write_csv(OUT / "shifts.csv")
    if summary:
        S = pl.DataFrame(summary); S.write_csv(OUT / "summary.csv"); print(S)


if __name__ == "__main__":
    main()
