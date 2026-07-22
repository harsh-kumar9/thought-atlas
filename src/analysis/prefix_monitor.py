"""Prefix-based monitorability analysis for Track B behavior traces.

This module asks whether early behavior annotations predict final success before
the answer is available. It is intentionally predictive and associative:
features use only prefix behavior/metadata, outcomes are final grades, and the
main comparison is out-of-sample performance of nested logistic monitors.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from src.utils.io import resolve_trace_paths


KIM = [
    "Question_and_Answering",
    "Perspective_Shift",
    "Conflict_of_Perspectives",
    "Reconciliation",
]
GANDHI = ["verification", "backtracking", "subgoal", "backward_chaining"]
BEHAVIORS = GANDHI + KIM

PREFIXES = [0.25, 0.5, 0.75, 1.0]
TEMPORAL_BINS = 4
N_FOLDS = 5
BOOT = 100
L2 = 1.0

FEATURE_ORDER = ["metadata", "length", "counts", "temporal", "time_shuffled"]
FEATURE_LABELS = {
    "metadata": "Metadata",
    "length": "Metadata + length",
    "counts": "Metadata + length + counts",
    "temporal": "Metadata + length + temporal bins",
    "time_shuffled": "Counts repeated across time",
}


def _glob_parquet(pattern: str) -> pl.DataFrame:
    paths = resolve_trace_paths(pattern) if "traces_" in pattern else sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet files matched {pattern}")
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")


def _first_non_null(name: str) -> pl.Expr:
    return pl.col(name).drop_nulls().first().alias(name)


def coalesced_grades(paths: list[str]) -> pl.DataFrame:
    frames = [pl.read_parquet(p) for p in paths if p and Path(p).exists()]
    if not frames:
        return pl.DataFrame({"trace_id": []})
    grades = pl.concat(frames, how="diagonal_relaxed")
    for col, dtype in [
        ("success", pl.Float64),
        ("quality_score", pl.Float64),
        ("parsed", pl.Boolean),
        ("task_type", pl.Utf8),
        ("difficulty_raw", pl.Utf8),
    ]:
        if col not in grades.columns:
            grades = grades.with_columns(pl.lit(None).cast(dtype).alias(col))
    return grades.group_by("trace_id").agg(
        _first_non_null("success"),
        _first_non_null("quality_score"),
        _first_non_null("parsed"),
        _first_non_null("task_type"),
        _first_non_null("difficulty_raw"),
    )


def add_binary_outcomes(traces: pl.DataFrame, grades: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    """Attach `outcome_y` where 1 means solved/high-quality and 0 otherwise."""
    df = traces.join(grades, on="trace_id", how="left", suffix="_grade")
    if "task_type_grade" in df.columns:
        df = df.drop("task_type_grade")
    if "difficulty_raw_grade" in df.columns and "difficulty_raw" in df.columns:
        df = df.with_columns(pl.coalesce(["difficulty_raw", "difficulty_raw_grade"]).alias("difficulty_raw")).drop("difficulty_raw_grade")
    med = (
        df.filter(pl.col("quality_score").is_not_null())
        .group_by(["gen_model", "task_type"])
        .agg(pl.col("quality_score").median().alias("_quality_median"))
    )
    df = df.join(med, on=["gen_model", "task_type"], how="left")
    quality_split = {
        f"{r['gen_model']}|{r['task_type']}": float(r["_quality_median"])
        for r in med.iter_rows(named=True)
        if r.get("_quality_median") is not None
    }
    out = df.with_columns(
        pl.when(pl.col("success").is_not_null())
        .then((pl.col("success") >= 1).cast(pl.Int8))
        .when(pl.col("quality_score").is_not_null() & pl.col("_quality_median").is_not_null())
        .then((pl.col("quality_score") >= pl.col("_quality_median")).cast(pl.Int8))
        .otherwise(pl.lit(None).cast(pl.Int8))
        .alias("outcome_y")
    )
    return out, quality_split


def _prefix_bin_expr(prefix: float, bins: int) -> pl.Expr:
    denom = max(prefix, 1e-9)
    raw = ((pl.col("norm_pos") / denom) * bins).floor().cast(pl.Int32)
    return (
        pl.when(raw < 0)
        .then(pl.lit(0))
        .when(raw >= bins)
        .then(pl.lit(bins - 1))
        .otherwise(raw)
        .alias("prefix_bin")
    )


def build_prefix_dataset(
    track_b: pl.DataFrame,
    traces_with_outcomes: pl.DataFrame,
    *,
    prefix: float,
    temporal_bins: int = TEMPORAL_BINS,
    behaviors: list[str] | None = None,
) -> pl.DataFrame:
    """Build one row per trace for a retrospective percentage prefix."""
    behaviors = [b for b in (behaviors or BEHAVIORS) if b in track_b.columns]
    if not behaviors:
        return pl.DataFrame()
    meta_cols = [
        "trace_id",
        "instance_id",
        "gen_model",
        "task_type",
        "difficulty_raw",
        "decode_temperature",
        "n_new_tokens",
        "outcome_y",
    ]
    meta = traces_with_outcomes.select([c for c in meta_cols if c in traces_with_outcomes.columns]).filter(pl.col("outcome_y").is_not_null())
    seg = (
        track_b.select(["trace_id", "norm_pos"] + behaviors)
        .join(meta.select(["trace_id"]), on="trace_id", how="inner")
        .filter(pl.col("norm_pos") <= min(1.0, prefix))
    )
    if seg.is_empty():
        return pl.DataFrame()

    summary = seg.group_by("trace_id").agg(
        pl.len().alias("prefix_segments"),
        *[pl.col(b).sum().cast(pl.Float64).alias(f"count_{b}") for b in behaviors],
    )
    denom = pl.when(pl.col("prefix_segments") > 0).then(pl.col("prefix_segments")).otherwise(pl.lit(1))
    summary = summary.with_columns([((pl.col(f"count_{b}") / denom).cast(pl.Float64)).alias(f"rate_{b}") for b in behaviors])

    binned = seg.with_columns(_prefix_bin_expr(prefix, temporal_bins))
    temporal = []
    grouped = binned.group_by(["trace_id", "prefix_bin"]).agg([pl.col(b).mean().cast(pl.Float64).alias(b) for b in behaviors])
    for b in behaviors:
        for k in range(temporal_bins):
            temporal.append(
                grouped.filter(pl.col("prefix_bin") == k)
                .select(["trace_id", pl.col(b).alias(f"bin{k}_{b}")])
            )
    wide = summary
    for part in temporal:
        wide = wide.join(part, on="trace_id", how="left")
    wide = wide.fill_null(0.0)
    return meta.join(wide, on="trace_id", how="inner").with_columns(pl.lit(prefix).alias("prefix"))


def _one_hot(values: list[Any], categories: list[str]) -> np.ndarray:
    lookup = {cat: idx for idx, cat in enumerate(categories)}
    out = np.zeros((len(values), len(categories)), dtype=float)
    for i, value in enumerate(values):
        idx = lookup.get(str(value))
        if idx is not None:
            out[i, idx] = 1.0
    return out


def design_matrix(df: pl.DataFrame, feature_set: str, *, temporal_bins: int = TEMPORAL_BINS) -> tuple[np.ndarray, list[str]]:
    """Return a dense matrix for one nested monitor family."""
    n = df.height
    blocks = [np.ones((n, 1), dtype=float)]
    names = ["intercept"]

    for col in ["gen_model", "task_type", "difficulty_raw"]:
        if col in df.columns:
            cats = sorted(str(v) for v in df[col].fill_null("__missing__").unique().to_list())
            mat = _one_hot(df[col].fill_null("__missing__").to_list(), cats)
            if mat.shape[1] > 1:
                mat = mat[:, 1:]
                cat_names = cats[1:]
            else:
                cat_names = cats
            blocks.append(mat)
            names.extend([f"{col}={cat}" for cat in cat_names])

    if feature_set in ("length", "counts", "temporal", "time_shuffled"):
        cols = [c for c in ["prefix_segments", "decode_temperature"] if c in df.columns]
        if cols:
            blocks.append(df.select(cols).to_numpy().astype(float))
            names.extend(cols)

    if feature_set in ("counts", "temporal", "time_shuffled"):
        cols = [f"rate_{b}" for b in BEHAVIORS if f"rate_{b}" in df.columns]
        if cols:
            blocks.append(df.select(cols).to_numpy().astype(float))
            names.extend(cols)

    if feature_set == "temporal":
        cols = [f"bin{k}_{b}" for b in BEHAVIORS for k in range(temporal_bins) if f"bin{k}_{b}" in df.columns]
        if cols:
            blocks.append(df.select(cols).to_numpy().astype(float))
            names.extend(cols)

    if feature_set == "time_shuffled":
        cols = []
        mats = []
        for b in BEHAVIORS:
            rate_col = f"rate_{b}"
            if rate_col not in df.columns:
                continue
            values = df[rate_col].to_numpy().astype(float).reshape(-1, 1)
            for k in range(temporal_bins):
                mats.append(values)
                cols.append(f"shuffled_bin{k}_{b}")
        if mats:
            blocks.append(np.hstack(mats))
            names.extend(cols)

    X = np.hstack(blocks)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def make_folds(df: pl.DataFrame, split: str, *, n_folds: int = N_FOLDS, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = df.height
    if split == "random_trace":
        y = df["outcome_y"].to_numpy().astype(int)
        folds = [[] for _ in range(n_folds)]
        for cls in [0, 1]:
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            for i, row_idx in enumerate(idx):
                folds[i % n_folds].append(int(row_idx))
        out = []
        all_idx = np.arange(n)
        for fold in folds:
            test = np.array(sorted(fold), dtype=int)
            train = np.setdiff1d(all_idx, test, assume_unique=False)
            out.append((train, test))
        return out

    if split != "prompt_disjoint":
        raise ValueError(f"unknown split: {split}")
    group_col = "instance_id" if "instance_id" in df.columns else "trace_id"
    groups = defaultdict(list)
    for idx, group in enumerate(df[group_col].fill_null(df["trace_id"]).to_list()):
        groups[str(group)].append(idx)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    fold_groups = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=int)
    for group, idxs in sorted(group_items, key=lambda kv: len(kv[1]), reverse=True):
        target = int(np.argmin(fold_sizes))
        fold_groups[target].extend(idxs)
        fold_sizes[target] += len(idxs)
    all_idx = np.arange(n)
    out = []
    for fold in fold_groups:
        test = np.array(sorted(fold), dtype=int)
        train = np.setdiff1d(all_idx, test, assume_unique=False)
        out.append((train, test))
    return out


def _standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xtr = X_train.copy().astype(float)
    Xte = X_test.copy().astype(float)
    if Xtr.shape[1] <= 1:
        return Xtr, Xte
    mu = Xtr[:, 1:].mean(axis=0)
    sd = Xtr[:, 1:].std(axis=0)
    sd[sd == 0] = 1.0
    Xtr[:, 1:] = (Xtr[:, 1:] - mu) / sd
    Xte[:, 1:] = (Xte[:, 1:] - mu) / sd
    return Xtr, Xte


def fit_logistic_l2(X: np.ndarray, y: np.ndarray, *, l2: float = L2, max_iter: int = 100, tol: float = 1e-7) -> np.ndarray:
    beta = np.zeros(X.shape[1], dtype=float)
    penalty = np.full(X.shape[1], float(l2), dtype=float)
    penalty[0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -35, 35)
        p = 1 / (1 + np.exp(-eta))
        grad = X.T @ (p - y) + penalty * beta
        w = np.maximum(p * (1 - p), 1e-6)
        hess = X.T @ (X * w[:, None]) + np.diag(penalty)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta -= step
        if float(np.max(np.abs(step))) < tol:
            break
    return beta


def predict_logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    eta = np.clip(X @ beta, -35, 35)
    return 1 / (1 + np.exp(-eta))


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return ranks


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = float(_ranks(score)[y == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p).astype(float), 1e-6, 1 - 1e-6)
    y = np.asarray(y).astype(float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    return float(np.mean((p - y) ** 2))


def ece(y: np.ndarray, p: np.ndarray, *, bins: int = 10) -> float:
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def score_predictions(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auroc": auroc(y, p),
        "auprc": average_precision(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
    }


def _bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    *,
    metric: str,
    boot: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    if boot <= 0:
        return None, None
    unique = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in unique}
    vals = []
    for _ in range(boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled])
        if metric == "auroc":
            val = auroc(y[idx], p[idx])
        elif metric == "auprc":
            val = average_precision(y[idx], p[idx])
        elif metric == "log_loss":
            val = log_loss(y[idx], p[idx])
        elif metric == "brier":
            val = brier(y[idx], p[idx])
        elif metric == "ece":
            val = ece(y[idx], p[idx])
        else:
            raise ValueError(f"unknown bootstrap metric: {metric}")
        if val is not None and math.isfinite(float(val)):
            vals.append(float(val))
    if not vals:
        return None, None
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _bootstrap_delta_ci(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    *,
    metric: str,
    boot: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    if boot <= 0:
        return None, None
    unique = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in unique}
    vals = []
    for _ in range(boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled])
        if metric == "auroc":
            s_a = auroc(y[idx], p_a[idx])
            s_b = auroc(y[idx], p_b[idx])
        elif metric == "auprc":
            s_a = average_precision(y[idx], p_a[idx])
            s_b = average_precision(y[idx], p_b[idx])
        elif metric == "log_loss":
            s_a = log_loss(y[idx], p_a[idx])
            s_b = log_loss(y[idx], p_b[idx])
        elif metric == "brier":
            s_a = brier(y[idx], p_a[idx])
            s_b = brier(y[idx], p_b[idx])
        elif metric == "ece":
            s_a = ece(y[idx], p_a[idx])
            s_b = ece(y[idx], p_b[idx])
        else:
            raise ValueError(f"unknown bootstrap metric: {metric}")
        if s_a is not None and s_b is not None and math.isfinite(float(s_a)) and math.isfinite(float(s_b)):
            vals.append(float(s_a) - float(s_b))
    if not vals:
        return None, None
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def evaluate_prefix(
    df: pl.DataFrame,
    *,
    prefix: float,
    split: str,
    feature_sets: list[str],
    temporal_bins: int = TEMPORAL_BINS,
    n_folds: int = N_FOLDS,
    seed: int = 0,
    l2: float = L2,
    boot: int = BOOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    y = df["outcome_y"].to_numpy().astype(int)
    if "instance_id" in df.columns:
        group_values = np.array([g if g is not None else tid for g, tid in zip(df["instance_id"].to_list(), df["trace_id"].to_list())])
    else:
        group_values = np.array(df["trace_id"].to_list())
    folds = make_folds(df, split, n_folds=n_folds, seed=seed)
    metrics_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    pred_by_set: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed + int(round(prefix * 1000)) + (17 if split == "prompt_disjoint" else 0))

    for feature_set in feature_sets:
        X, names = design_matrix(df, feature_set, temporal_bins=temporal_bins)
        pred = np.full(df.height, np.nan, dtype=float)
        fold_n = 0
        for train_idx, test_idx in folds:
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            Xtr, Xte = _standardize_train_test(X[train_idx], X[test_idx])
            beta = fit_logistic_l2(Xtr, y[train_idx], l2=l2)
            pred[test_idx] = predict_logistic(Xte, beta)
            fold_n += 1
        mask = np.isfinite(pred)
        scores = score_predictions(y[mask], pred[mask])
        lo, hi = _bootstrap_ci(y[mask], pred[mask], group_values[mask], metric="auroc", boot=boot, rng=rng)
        metrics_rows.append(
            {
                "split": split,
                "prefix": prefix,
                "feature_set": feature_set,
                "feature_label": FEATURE_LABELS.get(feature_set, feature_set),
                "n": int(mask.sum()),
                "positive_rate": float(y[mask].mean()) if mask.any() else None,
                "folds": fold_n,
                "auroc": scores["auroc"],
                "auroc_ci_low": lo,
                "auroc_ci_high": hi,
                "auprc": scores["auprc"],
                "log_loss": scores["log_loss"],
                "brier": scores["brier"],
                "ece": scores["ece"],
            }
        )
        pred_by_set[feature_set] = pred

        Xs, _ = _standardize_train_test(X, X)
        if len(np.unique(y)) == 2:
            beta_full = fit_logistic_l2(Xs, y, l2=l2)
            order = np.argsort(-np.abs(beta_full[1:]))[:20] + 1
            for idx in order:
                coef_rows.append(
                    {
                        "split": split,
                        "prefix": prefix,
                        "feature_set": feature_set,
                        "feature": names[idx],
                        "coef": float(beta_full[idx]),
                    }
                )

    delta_rows: list[dict[str, Any]] = []
    for feature_set, baseline in [("length", "metadata"), ("counts", "length"), ("temporal", "counts"), ("time_shuffled", "counts"), ("temporal", "time_shuffled")]:
        if feature_set not in pred_by_set or baseline not in pred_by_set:
            continue
        mask = np.isfinite(pred_by_set[feature_set]) & np.isfinite(pred_by_set[baseline])
        if not mask.any():
            continue
        s_a = score_predictions(y[mask], pred_by_set[feature_set][mask])
        s_b = score_predictions(y[mask], pred_by_set[baseline][mask])
        dlo, dhi = _bootstrap_delta_ci(
            y[mask],
            pred_by_set[feature_set][mask],
            pred_by_set[baseline][mask],
            group_values[mask],
            metric="auroc",
            boot=boot,
            rng=rng,
        )
        delta_rows.append(
            {
                "split": split,
                "prefix": prefix,
                "feature_set": feature_set,
                "baseline": baseline,
                "delta_auroc": s_a["auroc"] - s_b["auroc"],
                "delta_auroc_ci_low": dlo,
                "delta_auroc_ci_high": dhi,
                "delta_log_loss": s_a["log_loss"] - s_b["log_loss"],
                "delta_brier": s_a["brier"] - s_b["brier"],
            }
        )
    return metrics_rows, delta_rows, coef_rows


def run_analysis(
    track_b: pl.DataFrame,
    traces: pl.DataFrame,
    grades: pl.DataFrame,
    *,
    prefixes: list[float] = PREFIXES,
    temporal_bins: int = TEMPORAL_BINS,
    splits: list[str] | None = None,
    feature_sets: list[str] | None = None,
    seed: int = 0,
    n_folds: int = N_FOLDS,
    boot: int = BOOT,
    l2: float = L2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    splits = splits or ["prompt_disjoint", "random_trace"]
    feature_sets = feature_sets or FEATURE_ORDER
    traces_out, quality_split = add_binary_outcomes(traces, grades)
    metrics_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    n_by_prefix = {}

    for prefix in prefixes:
        df = build_prefix_dataset(track_b, traces_out, prefix=prefix, temporal_bins=temporal_bins)
        df = df.filter(pl.col("prefix_segments") > 0)
        n_by_prefix[str(prefix)] = df.height
        for split in splits:
            m, d, c = evaluate_prefix(
                df,
                prefix=prefix,
                split=split,
                feature_sets=feature_sets,
                temporal_bins=temporal_bins,
                n_folds=n_folds,
                seed=seed,
                l2=l2,
                boot=boot,
            )
            metrics_rows.extend(m)
            delta_rows.extend(d)
            coef_rows.extend(c)

    meta = {
        "prefixes": prefixes,
        "temporal_bins": temporal_bins,
        "splits": splits,
        "feature_sets": feature_sets,
        "feature_labels": FEATURE_LABELS,
        "n_folds": n_folds,
        "boot": boot,
        "l2": l2,
        "n_by_prefix": n_by_prefix,
        "quality_split": quality_split,
        "notes": [
            "Outcome is solved for deterministic domains and per-model/domain median quality split for moral/idea.",
            "Features use metadata, observed prefix length, behavior counts, and behavior timing within the prefix; final answer/completion fields are not features.",
            "Prompt-disjoint folds group all model attempts of the same instance_id into the same fold.",
            "Percentage prefixes are retrospective and should not be treated as online intervention budgets.",
        ],
    }
    return metrics_rows, delta_rows, coef_rows, meta


def _safe_float(value: Any, digits: int | None = None) -> Any:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(v):
        return None
    return round(v, digits) if digits is not None else v


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dashboard_payload(metrics_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    metrics = []
    for row in metrics_rows:
        metrics.append(
            {
                "split": row["split"],
                "prefix": _safe_float(row["prefix"], 2),
                "feature_set": row["feature_set"],
                "feature_label": row.get("feature_label"),
                "n": int(row.get("n") or 0),
                "positive_rate": _safe_float(row.get("positive_rate"), 4),
                "folds": int(row.get("folds") or 0),
                "auroc": _safe_float(row.get("auroc"), 4),
                "auroc_ci_low": _safe_float(row.get("auroc_ci_low"), 4),
                "auroc_ci_high": _safe_float(row.get("auroc_ci_high"), 4),
                "auprc": _safe_float(row.get("auprc"), 4),
                "log_loss": _safe_float(row.get("log_loss"), 4),
                "brier": _safe_float(row.get("brier"), 4),
                "ece": _safe_float(row.get("ece"), 4),
            }
        )
    deltas = []
    for row in delta_rows:
        deltas.append(
            {
                "split": row["split"],
                "prefix": _safe_float(row["prefix"], 2),
                "feature_set": row["feature_set"],
                "baseline": row["baseline"],
                "delta_auroc": _safe_float(row.get("delta_auroc"), 4),
                "delta_auroc_ci_low": _safe_float(row.get("delta_auroc_ci_low"), 4),
                "delta_auroc_ci_high": _safe_float(row.get("delta_auroc_ci_high"), 4),
                "delta_log_loss": _safe_float(row.get("delta_log_loss"), 4),
                "delta_brier": _safe_float(row.get("delta_brier"), 4),
            }
        )
    compact_meta = dict(meta)
    compact_meta["quality_split"] = {k: _safe_float(v, 4) for k, v in compact_meta.get("quality_split", {}).items()}
    return {"meta": compact_meta, "metrics": metrics, "deltas": deltas}


def write_dashboard_json(path: Path, metrics_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dashboard_payload(metrics_rows, delta_rows, meta), ensure_ascii=False, separators=(",", ":")))


def rows_from_csv(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackB", default="data/judge/prod/trackB_full__google_gemma-4-31B-it.parquet")
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--grades", nargs="*", default=["data/perf/success_grades.parquet", "data/perf/code_grades.parquet"])
    ap.add_argument("--quality", default="data/judge/prod/quality__google_gemma-4-31B-it.parquet")
    ap.add_argument("--out-dir", default="data/analysis/prefix_monitor")
    ap.add_argument("--prefixes", nargs="*", type=float, default=PREFIXES)
    ap.add_argument("--splits", nargs="*", default=["prompt_disjoint", "random_trace"])
    ap.add_argument("--feature-sets", nargs="*", default=FEATURE_ORDER)
    ap.add_argument("--temporal-bins", type=int, default=TEMPORAL_BINS)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--l2", type=float, default=L2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    traces = _glob_parquet(args.traces_glob)
    track_b = pl.read_parquet(args.trackB)
    grade_paths = list(args.grades or [])
    if args.quality:
        grade_paths.append(args.quality)
    grades = coalesced_grades(grade_paths)
    metrics, deltas, coefs, meta = run_analysis(
        track_b,
        traces,
        grades,
        prefixes=args.prefixes,
        temporal_bins=args.temporal_bins,
        splits=args.splits,
        feature_sets=args.feature_sets,
        seed=args.seed,
        n_folds=args.folds,
        boot=args.boot,
        l2=args.l2,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "metrics.csv", metrics)
    write_csv(out_dir / "deltas.csv", deltas)
    write_csv(out_dir / "coefficients.csv", coefs)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"prefix-monitor metrics={len(metrics)} deltas={len(deltas)} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
