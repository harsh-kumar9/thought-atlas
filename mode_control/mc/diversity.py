"""Diversity of solutions, reasoning text, and reasoning process, per arm.

Per-input (within a task, across sampling seeds; needs a multi-seed run) and across-input (across tasks)
diversity, following Kirk et al. 2024. Metrics:
  outcome   pass@k (Chen et al. 2021 unbiased estimator), majority-vote accuracy (Wang et al. 2022),
            distinct final answers per task
  lexical   self-BLEU (Zhu et al. 2018, sacrebleu sentence BLEU of each text against the others),
            distinct-n (Li et al. 2016) and expectation-adjusted distinct-n, EAD (Liu et al. 2022)
  semantic  mean pairwise cosine distance of sentence embeddings (BAAI/bge-m3), Vendi score
            (Friedman & Dieng 2023) of the cosine kernel
  process   entropy of behavior-label transitions and distinct behavior 3-grams from the Track B labels
  strategy  distinct judge-labelled solution methods per task (Math; see method_judge)
Outputs data/runs/<run>/analysis/diversity_*.csv
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from .common import BEH, DATA, JUDGE_MODEL, load_tokenizer

EMB_MODEL = "BAAI/bge-m3"


# ---------------------------------------------------------------- outcome
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _norm_answer(a: str | None) -> str | None:
    if not a:
        return None
    s = a.replace("**", "").strip()
    m = re.findall(r"\\boxed\{([^{}]*)\}", s)
    if m:
        s = m[-1]
    return re.sub(r"\s+", "", s.lower())


# ---------------------------------------------------------------- lexical
def _toks(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", (text or "").lower())


def distinct_n(texts: list[str], n: int) -> tuple[float, float]:
    """distinct-n over the pooled tokens of `texts`, and EAD (Liu et al. 2022) using the pooled vocabulary size."""
    grams = Counter()
    total = 0
    vocab = set()
    for t in texts:
        tk = _toks(t)
        vocab.update(tk)
        for i in range(len(tk) - n + 1):
            grams[tuple(tk[i:i + n])] += 1
            total += 1
    if total == 0:
        return float("nan"), float("nan")
    d = len(grams) / total
    V = max(len(vocab), 2) ** n
    ead = len(grams) / (V * (1 - ((V - 1) / V) ** total))
    return d, ead


def self_bleu(texts: list[str], max_chars: int = 6000) -> float:
    """Mean sentence-BLEU of each text against the other texts as references (Zhu et al. 2018)."""
    import sacrebleu
    ts = [t[:max_chars] for t in texts if t]
    if len(ts) < 2:
        return float("nan")
    vals = []
    for i, t in enumerate(ts):
        refs = [r for j, r in enumerate(ts) if j != i]
        vals.append(sacrebleu.sentence_bleu(t, refs).score / 100.0)
    return float(np.mean(vals))


# ---------------------------------------------------------------- semantic
_EMB = None


def embed_cached(ids: list[str], texts: list[str], cache_path, max_len: int = 4096) -> np.ndarray:
    """Embed with an on-disk cache keyed by trace_id (npz: ids, E)."""
    cache_path = Path(cache_path)
    have = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True); have = dict(zip(z["ids"].tolist(), z["E"]))
    todo = [i for i, t in enumerate(ids) if t not in have]
    if todo:
        E_new = embed([texts[i] for i in todo], max_len=max_len)
        for k, i in enumerate(todo): have[ids[i]] = E_new[k]
        np.savez(cache_path, ids=np.array(list(have.keys())), E=np.stack(list(have.values())))
    return np.stack([have[t] for t in ids])


def embed(texts: list[str], max_len: int = 4096) -> np.ndarray:
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        import torch
        _EMB = SentenceTransformer(EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
        _EMB.max_seq_length = max_len
    e = _EMB.encode([t or "" for t in texts], batch_size=8, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(e, dtype=np.float64)


def pairwise_cos_distance(E: np.ndarray) -> float:
    if E.shape[0] < 2:
        return float("nan")
    S = E @ E.T
    iu = np.triu_indices(E.shape[0], 1)
    return float(np.mean(1.0 - S[iu]))


def vendi(E: np.ndarray) -> float:
    """Vendi score: exp(entropy of the eigenvalues of K/n), K = cosine kernel (Friedman & Dieng 2023)."""
    n = E.shape[0]
    if n < 2:
        return float("nan")
    K = E @ E.T
    w = np.linalg.eigvalsh(K / n)
    w = w[w > 1e-12]
    return float(np.exp(-np.sum(w * np.log(w))))


# ---------------------------------------------------------------- process
def behavior_sequences(tb: pl.DataFrame) -> dict[str, list[str]]:
    """Per trace: the sequence of dominant behavior labels per segment ('none' when no label fires)."""
    out = {}
    cols = [b for b in BEH if b in tb.columns]
    for tid, g in tb.sort("seg_idx").group_by("trace_id", maintain_order=True):
        tid = tid[0] if isinstance(tid, (tuple, list)) else tid
        seq = []
        for r in g.iter_rows(named=True):
            on = [b for b in cols if r.get(b)]
            seq.append("+".join(sorted(on)) if on else "none")
        out[tid] = seq
    return out


def transition_entropy(seq: list[str]) -> float:
    if len(seq) < 2:
        return float("nan")
    c = Counter(zip(seq[:-1], seq[1:]))
    p = np.array(list(c.values()), float) / (len(seq) - 1)
    return float(-(p * np.log2(p)).sum())


def distinct_ngrams(seq: list[str], n: int = 3) -> float:
    if len(seq) < n:
        return float("nan")
    g = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return len(set(g)) / len(g)


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--max-texts-per-group", type=int, default=60, help="cap for across-input lexical/semantic metrics")
    a = ap.parse_args()
    runs = a.run.split(","); rd = DATA / "runs" / runs[0]; out = rd / "analysis"; out.mkdir(parents=True, exist_ok=True)
    tr = pl.concat([pl.read_parquet(p) for r in runs for p in sorted((DATA / "runs" / r).glob("traces_*.parquet"))], how="diagonal_relaxed")
    tag = JUDGE_MODEL.replace("/", "_")
    tbp = [p for r in runs for p in sorted((DATA / "runs" / r / "judge").rglob(f"trackB_full__{tag}*.parquet"))]
    tb = pl.concat([pl.read_parquet(p) for p in tbp], how="diagonal_relaxed").unique(["trace_id", "seg_idx"], keep="last") if tbp else None
    from src.perf.answer_grading import grade_math_answer
    tr = tr.with_columns([
        pl.Series("answer_norm", [_norm_answer(x) for x in tr["answer_text"].to_list()]),
        pl.Series("correct", [int(bool(grade_math_answer(x.replace("**", ""), r).get("success"))) if (t == "math" and x) else (0 if t == "math" else None)
                              for x, r, t in zip(tr["answer_text"].to_list(), tr["reference_answer"].to_list(), tr["task_type"].to_list())]),
    ])
    seeds = sorted(tr["seed"].unique().to_list()); multi = len(seeds) > 1
    conds = sorted(tr["condition"].unique().to_list())
    rng = np.random.default_rng(0)
    # ---- per-input (multi-seed) metrics
    rows = []
    if multi:
        for (cond, dom), g in tr.group_by(["condition", "task_type"]):
            for iid, t in g.group_by("instance_id"):
                iid = iid[0] if isinstance(iid, (tuple, list)) else iid
                n = t.height; texts_think = t["think_text"].fill_null("").to_list(); texts_ans = t["answer_text"].fill_null("").to_list()
                r = {"condition": cond, "task_type": dom, "instance_id": iid, "n": n,
                     "self_bleu_think": self_bleu(texts_think), "self_bleu_answer": self_bleu([x for x in texts_ans if x]),
                     "distinct2_think": distinct_n(texts_think, 2)[0], "ead2_think": distinct_n(texts_think, 2)[1],
                     "n_distinct_answers": len({x for x in t["answer_norm"].to_list() if x}), "n_answered": int(sum(1 for x in texts_ans if x))}
                if dom == "math":
                    c = int(t["correct"].sum())
                    for k in range(1, n + 1):
                        r[f"pass@{k}"] = pass_at_k(n, c, k)
                    votes = Counter([x for x in t["answer_norm"].to_list() if x])
                    if votes:
                        top = votes.most_common(1)[0][0]
                        row = t.filter(pl.col("answer_norm") == top).row(0, named=True)
                        r["majority_correct"] = int(bool(grade_math_answer(row["answer_text"].replace("**", ""), row["reference_answer"]).get("success")))
                    else:
                        r["majority_correct"] = 0
                if tb is not None:
                    seqs = behavior_sequences(tb.filter(pl.col("trace_id").is_in(t["trace_id"].to_list())))
                    r["transition_entropy"] = float(np.nanmean([transition_entropy(s) for s in seqs.values()])) if seqs else float("nan")
                    r["distinct_beh3"] = float(np.nanmean([distinct_ngrams(s) for s in seqs.values()])) if seqs else float("nan")
                if not a.no_embed:
                    E = embed(texts_think); r["cos_dist_think"] = pairwise_cos_distance(E); r["vendi_think"] = vendi(E)
                    ans = [x for x in texts_ans if x]
                    if len(ans) >= 2:
                        Ea = embed(ans); r["cos_dist_answer"] = pairwise_cos_distance(Ea); r["vendi_answer"] = vendi(Ea)
                rows.append(r)
        per = pl.DataFrame(rows); per.write_parquet(out / ("diversity_per_input.parquet" if len(runs) == 1 else "diversity_per_input_pooled.parquet"))
        # arm-level means with bootstrap CIs over tasks
        agg = []
        for (cond, dom), g in per.group_by(["condition", "task_type"]):
            r = {"condition": cond, "task_type": dom, "n_tasks": g.height}
            for m in [c for c in per.columns if c not in ("condition", "task_type", "instance_id", "n")]:
                x = g[m].cast(pl.Float64, strict=False).drop_nulls().to_numpy()
                if len(x) == 0: continue
                bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
                r[m] = float(np.nanmean(x)); r[f"{m}_lo"], r[f"{m}_hi"] = np.nanpercentile(bs, [2.5, 97.5])
            agg.append(r)
        pl.DataFrame(agg).sort(["task_type", "condition"]).write_csv(out / ("diversity_per_input_summary.csv" if len(runs) == 1 else "diversity_per_input_summary_pooled.csv"))
        print(pl.DataFrame(agg).select([c for c in pl.DataFrame(agg).columns if not c.endswith(("_lo", "_hi"))]))
    # ---- across-input metrics (any run): one text per task, same task set for every arm
    rows = []
    for (cond, dom), g in tr.group_by(["condition", "task_type"]):
        g = g.filter(pl.col("seed") == seeds[0]).sort("instance_id")
        idx = rng.choice(g.height, size=min(a.max_texts_per_group, g.height), replace=False) if g.height > a.max_texts_per_group else np.arange(g.height)
        g = g[sorted(int(i) for i in idx)]
        think = g["think_text"].fill_null("").to_list(); ans = [x for x in g["answer_text"].fill_null("").to_list() if x]
        r = {"condition": cond, "task_type": dom, "n": g.height, "self_bleu_think": self_bleu(think), "self_bleu_answer": self_bleu(ans),
             "distinct2_think": distinct_n(think, 2)[0], "ead2_think": distinct_n(think, 2)[1], "distinct2_answer": distinct_n(ans, 2)[0]}
        if tb is not None:
            seqs = behavior_sequences(tb.filter(pl.col("trace_id").is_in(g["trace_id"].to_list())))
            r["transition_entropy"] = float(np.nanmean([transition_entropy(s) for s in seqs.values()])) if seqs else float("nan")
            r["distinct_beh3"] = float(np.nanmean([distinct_ngrams(s) for s in seqs.values()])) if seqs else float("nan")
        if not a.no_embed:
            E = embed(think); r["cos_dist_think"] = pairwise_cos_distance(E); r["vendi_think"] = vendi(E)
            if len(ans) >= 2:
                Ea = embed(ans); r["cos_dist_answer"] = pairwise_cos_distance(Ea); r["vendi_answer"] = vendi(Ea)
        rows.append(r)
    ac = pl.DataFrame(rows).sort(["task_type", "condition"]); ac.write_csv(out / ("diversity_across_input.csv" if len(runs) == 1 else "diversity_across_input_pooled.csv")); print(ac)


if __name__ == "__main__":
    main()
