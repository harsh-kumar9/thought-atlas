"""Archetype targets for the signature controller (D-33, stage 4).

Whole-trace signature per corpus trace: overall rate of each behavior, rate in the first half and in the
second half of the judged window (24 dims). Restrict to high-quality traces (completed, no loop flag; Math:
graded correct), whiten within domain, cluster with k-medoids (PAM on Euclidean distance in whitened
space), and write the medoid traces as targets: data/sigctrl/archetypes.json with, per archetype, the
controller target phi_star (16 dims = overall rates for the 'future' block and for the 'next4' block).
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import polars as pl

from .common import BEH, DATA, JUDGE_MODEL

SIG = DATA / "sigctrl"


def whole_signature(tb: pl.DataFrame) -> pl.DataFrame:
    tb = tb.filter(pl.col("kim_parsed") & pl.col("gandhi_parsed"))
    n = tb.group_by("trace_id").agg(pl.col("seg_idx").max().alias("mx"))
    tb = tb.join(n, on="trace_id").with_columns((pl.col("seg_idx") <= pl.col("mx") / 2).alias("early"))
    overall = tb.group_by("trace_id").agg([pl.col(b).mean().alias(f"all_{b}") for b in BEH] + [pl.len().alias("n_valid")])
    early = tb.filter(pl.col("early")).group_by("trace_id").agg([pl.col(b).mean().alias(f"early_{b}") for b in BEH])
    late = tb.filter(~pl.col("early")).group_by("trace_id").agg([pl.col(b).mean().alias(f"late_{b}") for b in BEH])
    return overall.join(early, on="trace_id", how="left").join(late, on="trace_id", how="left")


def pam(X: np.ndarray, k: int, seed: int = 0, iters: int = 50):
    rng = np.random.default_rng(seed); n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    med = list(rng.choice(n, k, replace=False))
    best = D[:, med].min(1).sum()
    for _ in range(iters):
        improved = False
        for i in range(k):
            for cand in range(n):
                if cand in med: continue
                trial = med.copy(); trial[i] = cand
                cost = D[:, trial].min(1).sum()
                if cost < best - 1e-9:
                    med, best, improved = trial, cost, True
        if not improved: break
    assign = D[:, med].argmin(1)
    return med, assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--k", type=int, default=4); ap.add_argument("--min-segments", type=int, default=20)
    a = ap.parse_args()
    from src.perf.answer_grading import grade_math_answer
    tag = JUDGE_MODEL.replace("/", "_")
    out = {}
    trs, tbs = [], []
    for run in a.runs.split(","):
        rd = DATA / "runs" / run
        trs.append(pl.concat([pl.read_parquet(p, columns=["trace_id", "instance_id", "task_type", "seed", "completed", "hard_loop", "soft_loop", "answer_text", "reference_answer", "n_new_tokens"]) for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed"))
        tbs.append(pl.concat([pl.read_parquet(p) for p in sorted((rd / "judge").rglob(f"trackB_full__{tag}*.parquet"))], how="diagonal_relaxed"))
    tr = pl.concat(trs, how="diagonal_relaxed"); tb = pl.concat(tbs, how="diagonal_relaxed").unique(["trace_id", "seg_idx"], keep="last")
    sig = whole_signature(tb).join(tr, on="trace_id", how="inner")
    sig = sig.with_columns(pl.Series("correct", [int(bool(grade_math_answer((x or "").replace("**", ""), r).get("success"))) if (t == "math" and x) else None
                                              for x, r, t in zip(sig["answer_text"].to_list(), sig["reference_answer"].to_list(), sig["task_type"].to_list())]))
    cols = [f"{p}_{b}" for p in ("all", "early", "late") for b in BEH]
    summary = []
    for dom in sorted(sig["task_type"].unique().to_list()):
        g = sig.filter((pl.col("task_type") == dom) & pl.col("completed") & ~pl.col("hard_loop").fill_null(False) & (pl.col("n_valid") >= a.min_segments))
        if dom == "math": g = g.filter(pl.col("correct") == 1)
        X = g.select(cols).fill_null(0.0).to_numpy().astype(float)
        mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
        if len(Z) > 600:
            idx = np.random.default_rng(0).choice(len(Z), 600, replace=False); Zs, gs = Z[idx], g[idx.tolist()]
        else:
            Zs, gs = Z, g
        med, assign = pam(Zs, a.k)
        for j, m in enumerate(med):
            row = gs.row(m, named=True)
            phi_all = [row[f"all_{b}"] for b in BEH]
            name = f"{dom}_A{j + 1}"
            out[name] = {"domain": dom, "trace_id": row["trace_id"], "instance_id": row["instance_id"], "size": int((assign == j).sum()), "n_pool": int(len(Zs)),
                         "phi": phi_all + phi_all, "signature": {c: float(row[c] or 0.0) for c in cols}}
            summary.append({"archetype": name, "size": int((assign == j).sum()), **{b: round(row[f"all_{b}"], 3) for b in BEH},
                            **{f"late-early_{b}": round((row[f"late_{b}"] or 0) - (row[f"early_{b}"] or 0), 3) for b in BEH}})
        # domain centroid and the pool mean for reference
        out[f"{dom}_centroid"] = {"domain": dom, "phi": list(X.mean(0)[:8]) + list(X.mean(0)[:8]), "n_pool": int(len(X))}
    SIG.mkdir(parents=True, exist_ok=True)
    (SIG / "archetypes.json").write_text(json.dumps(out, indent=1))
    pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(40)
    print(pl.DataFrame(summary))
    print("->", SIG / "archetypes.json")


if __name__ == "__main__":
    main()
