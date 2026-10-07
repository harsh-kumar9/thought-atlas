"""Frozen Math–Idea phenotype: windowed behavior amounts + frozen mode scores.

Amount = fraction of valid labeled segments (both taxonomies parsed) that carry a behavior,
restricted to the segments that start within the first `window` reasoning tokens. This is the
Thought Atlas amount contract applied to a fixed-length prefix window so that baseline traces
(65k budget) and intervention traces (capped budget) are phenotyped on the same footing.

Scores: ridge logistic regression (C=1, as in the paper's fixed-L2 models) on standardized
amounts, positive = Idea-like. Three variants: cog (4 cognitive), conv (4 conversational),
all (8). Baseline traces get out-of-fold scores on the paper's prompt-disjoint folds; the
frozen coefficients (fit on all baseline traces) score new traces.

CLI:
  python -m mc.phenotype fit --window 4096          -> data/phenotype/baseline_w4096.parquet, frozen_w4096.json
  python -m mc.phenotype score --traces X.parquet --trackb Y.parquet --frozen F.json --out Z.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from .common import (BEH, DATA, FAMILIES, GEN_MODEL_KEY, KIM, GAN, TRACES_BASELINE, TRACKB_BASELINE,
                     load_tokenizer, stable_fold)
from src.segment.segmenter import segment_text  # noqa: E402  (repo module)


def _segments_with_offsets(tok, rtext: str, n_expected: int | None) -> list[int] | None:
    """Cumulative token offset at the START of each segment, in the judge's segmentation."""
    segs = segment_text(f"<think>{rtext}</think>")
    if n_expected is not None and len(segs) != n_expected:
        return None
    lens = [len(x) for x in tok([s["sentence"] for s in segs], add_special_tokens=False)["input_ids"]]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)
    return starts.tolist()


def windowed_amounts(traces: pl.DataFrame, trackb: pl.DataFrame, *, window: int,
                     tok=None) -> pl.DataFrame:
    """Per-trace amounts over segments starting before `window` tokens. Requires the Track B
    per-segment table (kim_parsed, gandhi_parsed, seg_idx, n_segments, 8 label columns)."""
    tok = tok or load_tokenizer()
    tb = trackb.filter(pl.col("trace_id").is_in(traces["trace_id"].to_list()))
    n_seg = dict(zip(tb["trace_id"].to_list(), tb["n_segments"].to_list())) if tb.height else {}
    rows = []
    keep = set()
    for r in traces.select(["trace_id", "reasoning_text_for_analysis"]).iter_rows(named=True):
        tid = r["trace_id"]
        if tid not in n_seg:
            continue
        starts = _segments_with_offsets(tok, r["reasoning_text_for_analysis"] or "", n_seg[tid])
        if starts is None:
            rows.append({"trace_id": tid, "seg_idx": -1, "in_window": None})
            continue
        keep.add(tid)
        for i, s in enumerate(starts):
            rows.append({"trace_id": tid, "seg_idx": i, "in_window": bool(s < window)})
    if not rows:
        return pl.DataFrame()
    win = pl.DataFrame(rows).filter(pl.col("seg_idx") >= 0)
    lab = tb.join(win, on=["trace_id", "seg_idx"], how="inner")
    valid = lab.filter(pl.col("kim_parsed") & pl.col("gandhi_parsed") & pl.col("in_window"))
    amt = valid.group_by("trace_id").agg(
        [pl.len().alias("n_valid_segments")] + [pl.col(b).mean().alias(f"{b}__rate") for b in BEH])
    tot = lab.group_by("trace_id").agg(pl.col("in_window").sum().alias("n_window_segments"),
                                       pl.col("n_segments").first())
    out = amt.join(tot, on="trace_id", how="left")
    mismatched = set(n_seg) - keep
    if mismatched:
        print(f"[phenotype] {len(mismatched)} traces dropped: re-segmentation mismatch")
    return out


def phase_amounts(traces: pl.DataFrame, trackb: pl.DataFrame, split_tokens: dict, *, window: int,
                  tok=None) -> pl.DataFrame:
    """Per-trace behavior rates in two phases: segments starting before split_tokens[trace_id] ("early")
    and segments starting at or after it but before `window` ("late")."""
    tok = tok or load_tokenizer()
    tb = trackb.filter(pl.col("trace_id").is_in(traces["trace_id"].to_list()))
    n_seg = dict(zip(tb["trace_id"].to_list(), tb["n_segments"].to_list())) if tb.height else {}
    rows = []
    for r in traces.select(["trace_id", "reasoning_text_for_analysis"]).iter_rows(named=True):
        tid = r["trace_id"]
        if tid not in n_seg or tid not in split_tokens:
            continue
        starts = _segments_with_offsets(tok, r["reasoning_text_for_analysis"] or "", n_seg[tid])
        if starts is None:
            continue
        sp = int(split_tokens[tid])
        for i, st in enumerate(starts):
            if st >= window: continue
            rows.append({"trace_id": tid, "seg_idx": i, "phase": "early" if st < sp else "late"})
    if not rows:
        return pl.DataFrame()
    lab = tb.join(pl.DataFrame(rows), on=["trace_id", "seg_idx"], how="inner")
    valid = lab.filter(pl.col("kim_parsed") & pl.col("gandhi_parsed"))
    return valid.group_by(["trace_id", "phase"]).agg(
        [pl.len().alias("n_valid_segments")] + [pl.col(b).mean().alias(f"{b}__rate") for b in BEH])


def binned_amounts(traces: pl.DataFrame, trackb: pl.DataFrame, edges: list[int], *, tok=None) -> pl.DataFrame:
    """Per-trace behavior rates in token bins [edges[i], edges[i+1]) by segment start."""
    tok = tok or load_tokenizer()
    tb = trackb.filter(pl.col("trace_id").is_in(traces["trace_id"].to_list()))
    n_seg = dict(zip(tb["trace_id"].to_list(), tb["n_segments"].to_list())) if tb.height else {}
    rows = []
    for r in traces.select(["trace_id", "reasoning_text_for_analysis"]).iter_rows(named=True):
        tid = r["trace_id"]
        if tid not in n_seg: continue
        starts = _segments_with_offsets(tok, r["reasoning_text_for_analysis"] or "", n_seg[tid])
        if starts is None: continue
        for i, st in enumerate(starts):
            b = int(np.searchsorted(np.asarray(edges), st, side="right") - 1)
            if 0 <= b < len(edges) - 1:
                rows.append({"trace_id": tid, "seg_idx": i, "bin": b})
    if not rows: return pl.DataFrame()
    lab = tb.join(pl.DataFrame(rows), on=["trace_id", "seg_idx"], how="inner")
    valid = lab.filter(pl.col("kim_parsed") & pl.col("gandhi_parsed"))
    return valid.group_by(["trace_id", "bin"]).agg([pl.len().alias("n_valid_segments")] + [pl.col(b).mean().alias(f"{b}__rate") for b in BEH])


def _fit_ridge_logit(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1.0, max_iter=5000)
    m.fit(X, y)
    return m


def fit_frozen(pheno: pl.DataFrame, *, n_folds: int = 5) -> tuple[pl.DataFrame, dict]:
    """OOF scores on prompt-disjoint folds + frozen full-data coefficients. y = 1 for idea."""
    from sklearn.metrics import roc_auc_score
    df = pheno.with_columns(
        pl.col("instance_id").map_elements(lambda s: stable_fold(s, n_folds=n_folds), return_dtype=pl.Int64)
        .alias("prompt_fold"))
    y = (df["task_type"] == "idea").to_numpy().astype(int)
    folds = df["prompt_fold"].to_numpy()
    frozen = {"n_traces": int(df.height), "n_folds": n_folds, "families": {}}
    scores = {}
    for fam, cols in FAMILIES.items():
        feats = [f"{b}__rate" for b in cols]
        X = df.select(feats).to_numpy().astype(float)
        oof = np.zeros(len(y))
        for k in range(n_folds):
            tr, te = folds != k, folds == k
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = _fit_ridge_logit((X[tr] - mu) / sd, y[tr])
            oof[te] = m.decision_function((X[te] - mu) / sd)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        m = _fit_ridge_logit((X - mu) / sd, y)
        scores[f"s_{fam}"] = oof
        frozen["families"][fam] = {
            "features": feats, "mean": mu.tolist(), "sd": sd.tolist(),
            "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
            "oof_auroc": float(roc_auc_score(y, oof)),
            "oof_mean_math": float(oof[y == 0].mean()), "oof_mean_idea": float(oof[y == 1].mean()),
            "oof_sd_math": float(oof[y == 0].std()), "oof_sd_idea": float(oof[y == 1].std()),
        }
    df = df.with_columns([pl.Series(k, v) for k, v in scores.items()])
    return df, frozen


def score_with_frozen(amounts: pl.DataFrame, frozen: dict) -> pl.DataFrame:
    cols = {}
    for fam, spec in frozen["families"].items():
        X = amounts.select(spec["features"]).to_numpy().astype(float)
        z = (X - np.array(spec["mean"])) / np.array(spec["sd"])
        cols[f"s_{fam}"] = z @ np.array(spec["coef"]) + spec["intercept"]
    return amounts.with_columns([pl.Series(k, v) for k, v in cols.items()])


def load_baseline_traces() -> pl.DataFrame:
    tr = pl.read_parquet(TRACES_BASELINE)
    return tr.filter(pl.col("task_type").is_in(["math", "idea"]) &
                     pl.col("reasoning_text_for_analysis").is_not_null() &
                     (pl.col("reasoning_text_for_analysis").str.len_chars() > 20))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--window", type=int, default=4096)
    s = sub.add_parser("score")
    s.add_argument("--traces", required=True); s.add_argument("--trackb", required=True)
    s.add_argument("--frozen", required=True); s.add_argument("--out", required=True)
    s.add_argument("--window", type=int, default=4096)
    a = ap.parse_args()
    tok = load_tokenizer()
    if a.cmd == "fit":
        tr = load_baseline_traces()
        tb = pl.scan_parquet(TRACKB_BASELINE).filter(pl.col("trace_id").is_in(tr["trace_id"].to_list())).collect()
        amt = windowed_amounts(tr, tb, window=a.window, tok=tok)
        pheno = amt.join(tr.select(["trace_id", "instance_id", "task_type", "n_new_tokens", "completed"]), on="trace_id")
        pheno, frozen = fit_frozen(pheno)
        out = DATA / "phenotype"; out.mkdir(parents=True, exist_ok=True)
        pheno.write_parquet(out / f"baseline_w{a.window}.parquet")
        frozen["window"] = a.window; frozen["gen_model"] = GEN_MODEL_KEY
        (out / f"frozen_w{a.window}.json").write_text(json.dumps(frozen, indent=2))
        print(pheno.group_by("task_type").agg([pl.len()] + [pl.col(f"{b}__rate").mean().round(4) for b in BEH]
                                              + [pl.col(c).mean().round(3) for c in ["s_cog", "s_conv", "s_all"]]))
        for fam, spec in frozen["families"].items():
            print(f"{fam}: OOF AUROC={spec['oof_auroc']:.4f} math={spec['oof_mean_math']:.2f}±{spec['oof_sd_math']:.2f} "
                  f"idea={spec['oof_mean_idea']:.2f}±{spec['oof_sd_idea']:.2f} coef={np.round(spec['coef'], 2).tolist()}")
    else:
        tr = pl.read_parquet(a.traces)
        tb = pl.read_parquet(a.trackb)
        amt = windowed_amounts(tr, tb, window=a.window, tok=tok)
        frozen = json.loads(Path(a.frozen).read_text())
        out = score_with_frozen(amt, frozen)
        out.write_parquet(a.out); print(f"wrote {out.height} scored traces -> {a.out}")


if __name__ == "__main__":
    main()
