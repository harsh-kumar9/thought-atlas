"""Robust CoT similarity: within-arm vs between-arm distances, exact over all pairs, per task, paired.

For arm pairs (unsteered, B) and subsets {all, correct-only}:
  within_A(t)  = mean distance over all pairs of arm-A traces of task t
  within_B(t)  = same for arm B
  between(t)   = mean distance over all A x B pairs
  excess(t)    = between - (within_A + within_B)/2          (0 = arms indistinguishable)
  ratio(t)     = between / within_A
Aggregation: mean over tasks with a task bootstrap CI, plus a permutation p-value (arm labels shuffled
within task, 2000 rounds) and a leave-one-task-out logistic AUROC for "can you tell the arm from the trace".
Representations: semantic (bge-m3 embedding of the first 4096 reasoning tokens, cosine distance),
process (transition profile over behavior labels incl. 'none', Jensen-Shannon distance), lexical
(token 3-gram Jaccard distance). Output: data/runs/<run>/analysis/similarity_*.csv
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import polars as pl

from .common import BEH, DATA, JUDGE_MODEL, load_tokenizer
from .diversity import behavior_sequences, embed_cached

STATES = ["none"] + BEH


def transition_profile(seq: list[str]) -> np.ndarray:
    """Occupancy (9) + row-normalised transition matrix (9x9) over dominant single labels, flattened."""
    def key(s):
        if s == "none": return 0
        first = s.split("+")[0]
        return STATES.index(first) if first in STATES else 0
    idx = [key(s) for s in seq]
    occ = np.bincount(idx, minlength=len(STATES)).astype(float); occ /= max(occ.sum(), 1)
    T = np.zeros((len(STATES), len(STATES)))
    for a, b in zip(idx[:-1], idx[1:]): T[a, b] += 1
    T = T / np.maximum(T.sum(1, keepdims=True), 1)
    return np.concatenate([occ, T.ravel() / len(STATES)])


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12); m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def ngram_set(text: str, n: int = 3) -> set:
    tk = text.lower().split()
    return {tuple(tk[i:i + n]) for i in range(len(tk) - n + 1)}


def jaccard_distance(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return 1.0 - len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="sched1_div", help="run name, or comma list of runs to pool (outputs go to the first)"); ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--arms", default="idea_then_math_L19,math_then_idea_L19,sus_idea_L19,sus_math_L19")
    a = ap.parse_args()
    runs = a.run.split(","); rd = DATA / "runs" / runs[0]; out = rd / "analysis"
    from src.perf.answer_grading import grade_math_answer
    tr = pl.concat([pl.read_parquet(p) for r in runs for p in sorted((DATA / "runs" / r).glob("traces_*.parquet"))], how="diagonal_relaxed").filter(pl.col("task_type") == "math")
    tr = tr.with_columns(pl.Series("correct", [int(bool(grade_math_answer(x.replace("**", ""), r).get("success"))) if x else 0 for x, r in zip(tr["answer_text"].to_list(), tr["reference_answer"].to_list())]))
    tag = JUDGE_MODEL.replace("/", "_")
    tb = pl.concat([pl.read_parquet(p) for r in runs for p in sorted((DATA / "runs" / r / "judge").rglob(f"trackB_full__{tag}*.parquet"))], how="diagonal_relaxed").unique(["trace_id", "seg_idx"], keep="last")
    seqs = behavior_sequences(tb.filter(pl.col("trace_id").is_in(tr["trace_id"].to_list())))
    ids = tr["trace_id"].to_list()
    reps = {}
    reps["process"] = {t: transition_profile(seqs[t]) for t in ids if t in seqs}
    reps["lexical"] = {t: ngram_set(x or "") for t, x in zip(ids, tr["reasoning_text_for_analysis"].to_list())}
    if not a.no_embed:
        E = embed_cached(ids, tr["reasoning_text_for_analysis"].fill_null("").to_list(), out / "embeddings_reasoning_bge_m3.npz", max_len=4096)
        reps["semantic"] = {t: E[i] for i, t in enumerate(ids)}
    def dist(rep, x, y):
        if rep == "semantic": return float(1.0 - np.dot(x, y))
        if rep == "process": return js_distance(x, y)
        return jaccard_distance(x, y)
    def mean_pair(rep, R, A, B=None):
        pairs = list(combinations(A, 2)) if B is None else [(x, y) for x in A for y in B]
        pairs = [(x, y) for x, y in pairs if x in R and y in R]
        return float(np.mean([dist(rep, R[x], R[y]) for x, y in pairs])) if pairs else np.nan
    rng = np.random.default_rng(0)
    rows = []
    for rep, R in reps.items():
        for arm in a.arms.split(","):
            for subset in ["all", "correct"]:
                per = []
                for t in sorted(tr["instance_id"].unique().to_list()):
                    g = tr.filter(pl.col("instance_id") == t)
                    if subset == "correct": g = g.filter(pl.col("correct") == 1)
                    A = g.filter(pl.col("condition") == "baseline")["trace_id"].to_list(); B = g.filter(pl.col("condition") == arm)["trace_id"].to_list()
                    if len(A) < 2 or len(B) < 2: continue
                    wA, wB, bt = mean_pair(rep, R, A), mean_pair(rep, R, B), mean_pair(rep, R, A, B)
                    # permutation within task: shuffle arm labels, recompute excess
                    pool = A + B; perm = []
                    for _ in range(200):
                        rng.shuffle(pool); pA, pB = pool[:len(A)], pool[len(A):]
                        perm.append(mean_pair(rep, R, pA, pB) - 0.5 * (mean_pair(rep, R, pA) + mean_pair(rep, R, pB)))
                    per.append({"task": t, "nA": len(A), "nB": len(B), "within_A": wA, "within_B": wB, "between": bt, "excess": bt - 0.5 * (wA + wB), "excess_perm_mean": float(np.mean(perm)), "excess_perm_sd": float(np.std(perm))})
                if len(per) < 5: continue
                P = pl.DataFrame(per)
                ex = P["excess"].to_numpy(); z = ((P["excess"] - P["excess_perm_mean"]) / P["excess_perm_sd"].clip(1e-9)).to_numpy()
                bs = [ex[rng.integers(0, len(ex), len(ex))].mean() for _ in range(3000)]
                # permutation p: compare observed mean excess to the distribution of mean permuted excess
                # (approximate: per-task permutation means are independent, so use the normal approx of the sum)
                pz = float(np.mean(z)) * np.sqrt(len(z))
                from math import erf, sqrt
                p_perm = 1.0 - 0.5 * (1 + erf(pz / sqrt(2)))
                rows.append({"representation": rep, "arm": arm, "subset": subset, "n_tasks": len(per), "mean_nA": float(P["nA"].mean()), "mean_nB": float(P["nB"].mean()),
                             "within_unsteered": float(P["within_A"].mean()), "within_arm": float(P["within_B"].mean()), "between": float(P["between"].mean()),
                             "excess": float(ex.mean()), "excess_lo": float(np.percentile(bs, 2.5)), "excess_hi": float(np.percentile(bs, 97.5)),
                             "ratio_between_within_unsteered": float((P["between"] / P["within_A"]).mean()), "z_vs_permutation": float(np.mean(z)), "p_perm_one_sided": p_perm})
    suffix = "" if len(runs) == 1 else "_pooled"
    res = pl.DataFrame(rows); res.write_csv(out / f"similarity_between_within{suffix}.csv")
    # leave-one-task-out classifier: arm vs unsteered from the representation (correct-only and all)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    crow = []
    for rep in [r for r in reps if r != "lexical"]:
        R = reps[rep]
        for arm in a.arms.split(","):
            for subset in ["all", "correct"]:
                g = tr.filter(pl.col("condition").is_in(["baseline", arm]))
                if subset == "correct": g = g.filter(pl.col("correct") == 1)
                g = g.filter(pl.col("trace_id").is_in(list(R.keys())))
                if g.height < 20: continue
                X = np.stack([R[t] for t in g["trace_id"].to_list()]); y = (g["condition"] == arm).to_numpy().astype(int); tasks = g["instance_id"].to_numpy()
                if len(np.unique(y)) < 2: continue
                scores = np.zeros(len(y))
                for t in np.unique(tasks):
                    tr_m = tasks != t
                    if len(np.unique(y[tr_m])) < 2: continue
                    clf = LogisticRegression(C=0.5, max_iter=2000).fit(X[tr_m], y[tr_m]); scores[~tr_m] = clf.decision_function(X[~tr_m])
                auc = roc_auc_score(y, scores)
                bs = []
                ut = np.unique(tasks)
                for _ in range(1000):
                    pick = rng.choice(ut, len(ut), replace=True); m = np.concatenate([np.where(tasks == p)[0] for p in pick])
                    if len(np.unique(y[m])) < 2: continue
                    bs.append(roc_auc_score(y[m], scores[m]))
                crow.append({"representation": rep, "arm": arm, "subset": subset, "n": int(len(y)), "auroc_loto": float(auc), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))})
    cl = pl.DataFrame(crow); cl.write_csv(out / f"similarity_classifier{suffix}.csv")
    pl.Config.set_tbl_width_chars(240); pl.Config.set_tbl_rows(80); pl.Config.set_fmt_float("mixed")
    print(res.with_columns(pl.col(pl.Float64).round(4)).sort(["representation", "subset", "arm"]))
    print(cl.with_columns(pl.col(pl.Float64).round(3)).sort(["representation", "subset", "arm"]))


if __name__ == "__main__":
    main()
