"""Step 0 of the repertoire line: do different schedules make different mistakes?

On the multi-seed run (40 tasks x 5 seeds per arm): per-task success rates, pairwise error correlation
between arms, unique solves, and mixed-portfolio pass@K (k1 samples from arm A + k2 from arm B, exact
average over subsets, independent draws) against homogeneous pass@K at the same sample count.
Idea: coverage of feasible ideas (Vendi of bge-m3 embeddings) for mixed vs homogeneous pools.
"""
from __future__ import annotations

import argparse
import glob
from itertools import combinations
from math import comb

import numpy as np
import polars as pl

from .common import DATA


def p_none_correct(n: int, c: int, k: int) -> float:
    """P(no correct among k draws without replacement from n samples with c correct)."""
    if k > n: return float("nan")
    if n - c < k: return 0.0
    return comb(n - c, k) / comb(n, k)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="sched1_div", help="run name, or comma list of runs to pool (outputs go to the first)"); ap.add_argument("--no-embed", action="store_true"); ap.add_argument("--arms", default="baseline,sus_idea_L19,sus_math_L19,idea_then_math_L19,math_then_idea_L19")
    a = ap.parse_args()
    runs = a.run.split(","); rd = DATA / "runs" / runs[0]
    from src.perf.answer_grading import grade_math_answer
    tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "condition", "instance_id", "task_type", "seed", "answer_text", "reference_answer", "n_new_tokens"]) for r in runs for p in sorted((DATA / "runs" / r).glob("traces_*.parquet"))], how="diagonal_relaxed")
    from .budget import load_grades
    _g = load_grades()   # D-47 extraction grades; direct grader only for traces without one
    _mt = tr.filter(pl.col("task_type") == "math")
    m = _mt.with_columns(pl.Series("correct", [int(_g[t]) if _g.get(t) is not None else (int(bool(grade_math_answer(x.replace("**", ""), r).get("success"))) if x else 0)
                                               for t, x, r in zip(_mt["trace_id"].to_list(), _mt["answer_text"].to_list(), _mt["reference_answer"].to_list())]))
    arms = a.arms.split(",")
    tasks = sorted(m["instance_id"].unique().to_list()); seeds = sorted(m["seed"].unique().to_list())
    # correctness tensor [arm, task, seed]
    C = np.full((len(arms), len(tasks), len(seeds)), np.nan)
    for i, c in enumerate(arms):
        g = m.filter(pl.col("condition") == c)
        for r in g.iter_rows(named=True):
            C[i, tasks.index(r["instance_id"]), seeds.index(r["seed"])] = r["correct"]
    rate = np.nanmean(C, axis=2)                       # [arm, task] per-task success rate
    tok = {c: float(m.filter(pl.col("condition") == c)["n_new_tokens"].mean()) for c in arms}
    print("== per-arm: pass@1, tokens/sample, tasks solved at least once")
    for i, c in enumerate(arms):
        print(f"  {c:22s} pass@1={np.nanmean(C[i]):.3f}  tokens={tok[c]:.0f}  solved>=1: {int((np.nansum(C[i], axis=1) > 0).sum())}/{len(tasks)}")
    print("== error correlation between arms (Pearson r of per-task success rates; sample-level phi in brackets)")
    for i, j in combinations(range(len(arms)), 2):
        r_task = np.corrcoef(rate[i], rate[j])[0, 1]
        x, y = C[i].ravel(), C[j].ravel(); ok = ~(np.isnan(x) | np.isnan(y)); phi = np.corrcoef(x[ok], y[ok])[0, 1]
        print(f"  {arms[i]:20s} vs {arms[j]:20s}  r={r_task:+.2f}  [phi={phi:+.2f}]")
    print("== unique solves: tasks solved (>=1 of 5) by row arm but never by column arm")
    solved = np.nansum(C, axis=2) > 0
    hdr = "".join(f"{c[:12]:>14s}" for c in arms); print(" " * 22 + hdr)
    for i in range(len(arms)):
        print(f"  {arms[i]:20s}" + "".join(f"{int((solved[i] & ~solved[j]).sum()):>14d}" for j in range(len(arms))))
    # mixed-portfolio pass@K: k1 from baseline + k2 from arm B vs K from baseline (and K from arm B)
    n_s = len(seeds)
    print("== portfolio pass@K (mean over tasks): homogeneous unsteered vs mixed (half unsteered, half arm)")
    ci = np.nansum(C, axis=2).astype(int)              # correct counts per [arm, task]
    def pass_k_hom(i, k): return float(np.mean([1 - p_none_correct(n_s, ci[i, t], k) for t in range(len(tasks))]))
    def pass_k_mix(i, j, k1, k2): return float(np.mean([1 - p_none_correct(n_s, ci[i, t], k1) * p_none_correct(n_s, ci[j, t], k2) for t in range(len(tasks))]))
    rng = np.random.default_rng(0)
    rows = []
    for K in [2, 4]:
        k1 = K // 2
        hom = pass_k_hom(0, K)
        line = f"  K={K}: unsteered x{K} = {hom:.3f}"
        for j in range(1, len(arms)):
            mix = pass_k_mix(0, j, k1, K - k1); homj = pass_k_hom(j, K)
            # paired bootstrap over tasks for mix - hom
            per_task_mix = np.array([1 - p_none_correct(n_s, ci[0, t], k1) * p_none_correct(n_s, ci[j, t], K - k1) for t in range(len(tasks))])
            per_task_hom = np.array([1 - p_none_correct(n_s, ci[0, t], K) for t in range(len(tasks))])
            d = per_task_mix - per_task_hom; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
            line += f" | {arms[j][:14]}: mix {mix:.3f} (Δ{d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]) hom {homj:.3f}"
            rows.append({"K": K, "arm": arms[j], "hom_unsteered": hom, "mixed": mix, "hom_arm": homj, "delta_mix_vs_unsteered": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))})
        print(line)
    pl.DataFrame(rows).write_csv(rd / "analysis" / ("complementarity_math.csv" if len(runs) == 1 else "complementarity_math_pooled.csv"))
    # Idea coverage: feasible ideas (judge feasibility >= 7) -> Vendi of embeddings for 4-sample pools
    if not a.no_embed:
        qp = [q for r in runs for q in glob.glob(str(DATA / "runs" / r / "judge/quality__*.parquet"))]
        if qp:
            from .diversity import embed, vendi
            q = pl.concat([pl.read_parquet(x) for x in qp], how="diagonal_relaxed").select(["trace_id", "axis_feasibility", "axis_originality"])
            it = tr.filter(pl.col("task_type") == "idea").join(q, on="trace_id", how="left")
            itasks = sorted(it["instance_id"].unique().to_list())
            print("== Idea: coverage of feasible ideas (feasibility >= 7), Vendi of bge-m3 embeddings, 4-sample pools (mean over tasks, paired vs unsteered x4)")
            def pool_vendi(rows_):
                txt = [r["answer_text"] for r in rows_ if r["answer_text"] and (r["axis_feasibility"] or 0) >= 7]
                if len(txt) < 2: return float(len(txt))
                return vendi(embed(txt, max_len=1024))
            for j in range(1, len(arms)):
                deltas = []
                for t in itasks:
                    b = it.filter((pl.col("condition") == "baseline") & (pl.col("instance_id") == t)).sort("seed").to_dicts()
                    o = it.filter((pl.col("condition") == arms[j]) & (pl.col("instance_id") == t)).sort("seed").to_dicts()
                    if len(b) < 4 or len(o) < 2: continue
                    hom = pool_vendi(b[:4]); mix = pool_vendi(b[:2] + o[:2]); deltas.append(mix - hom)
                d = np.array(deltas); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
                print(f"  2 unsteered + 2 {arms[j][:16]:16s} vs 4 unsteered: Δ Vendi {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] n={len(d)}")


if __name__ == "__main__":
    main()
