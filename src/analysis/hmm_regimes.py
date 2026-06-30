"""src/analysis/hmm_regimes.py — HMM regimes over per-sentence behavior sequences.

Each trace -> a symbolic sequence (per sentence, the dominant behavior, or 'none').
Fit HMMs per domain over the pre-registered regime-count range; select by BIC.
Tests whether selected regime count differs by domain. Needs hmmlearn.
"""
from __future__ import annotations
import numpy as np
import polars as pl

BEHAVIORS = ["Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives",
             "Reconciliation", "verification", "backtracking", "subgoal", "backward_chaining"]


def dominant_symbol(seg_row: dict) -> int:
    """Map a sentence's multi-label presence to a single symbol id (0=none, else first present)."""
    for k, b in enumerate(BEHAVIORS, start=1):
        if seg_row.get(b):
            return k
    return 0


def sequences_for_domain(segments: pl.DataFrame, task_type: str):
    seqs = []
    seg = segments.filter(pl.col("task_type") == task_type).sort(["trace_id", "seg_idx"])
    for tid, g in seg.group_by("trace_id"):
        syms = [dominant_symbol(r) for r in g.sort("seg_idx").iter_rows(named=True)]
        if len(syms) >= 5:
            seqs.append(np.array(syms).reshape(-1, 1))
    return seqs


def select_regimes(segments: pl.DataFrame, task_type: str, k_range=(2, 5), seed=0):
    try:
        from hmmlearn import hmm
    except Exception as e:  # noqa
        return {"task_type": task_type, "error": f"hmmlearn unavailable: {e}"}
    seqs = sequences_for_domain(segments, task_type)
    if len(seqs) < 5:
        return {"task_type": task_type, "error": "too few sequences"}
    X = np.concatenate(seqs); lengths = [len(s) for s in seqs]
    n_sym = int(X.max()) + 1
    best = None
    for k in range(k_range[0], k_range[1] + 1):
        try:
            m = hmm.CategoricalHMM(n_components=k, n_iter=50, random_state=seed)
            m.n_features = n_sym
            m.fit(X, lengths)
            ll = m.score(X, lengths)
            n_params = k * (k - 1) + k * (n_sym - 1) + (k - 1)
            bic = -2 * ll + n_params * np.log(sum(lengths))
            if best is None or bic < best["bic"]:
                best = {"k": k, "bic": float(bic), "loglik": float(ll)}
        except Exception:
            continue
    return {"task_type": task_type, "selected": best, "k_range": list(k_range), "n_seqs": len(seqs)}
