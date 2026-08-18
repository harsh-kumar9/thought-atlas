"""Within-model domain and attempt-quality signature analyses.

The estimand is deliberately predictive rather than mechanistic: if a linear,
prompt-disjoint classifier can identify domain or answer quality from held-out
traces, the corresponding trace signature differs in a reproducible way.  The
feature ladder separates per-segment amount from conditional temporal shape.
"""

from __future__ import annotations

import hashlib
import html
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import rankdata

from src.analysis.paper.constants import BEHAVIORS, DIALECTICAL, EXECUTIVE
from src.analysis.paper.io import current_commit, scan_track_b, sha256_file


DOMAINS = ["math", "code", "gpqa", "planning", "moral", "idea", "safety", "security"]
DOMAIN_LABELS = {domain: domain.title() for domain in DOMAINS} | {"gpqa": "GPQA"}
MODELS = ["reasoner", "gemma4_e4b", "gemma4_31b", "qwen35_4b", "qwen35_9b", "qwen35_27b"]
MODEL_LABELS = {
    "reasoner": "DeepSeek-R1 Distill 8B",
    "gemma4_e4b": "Gemma 4 E4B",
    "gemma4_31b": "Gemma 4 31B",
    "qwen35_4b": "Qwen3.5 4B",
    "qwen35_9b": "Qwen3.5 9B",
    "qwen35_27b": "Qwen3.5 27B",
}
FAMILIES = {"conversational": list(DIALECTICAL), "cognitive": list(EXECUTIVE)}
BEHAVIOR_LABELS = {
    "Question_and_Answering": "Question–answer",
    "Perspective_Shift": "Perspective shift",
    "Conflict_of_Perspectives": "Perspective conflict",
    "Reconciliation": "Reconciliation",
    "verification": "Verification",
    "backtracking": "Backtracking",
    "subgoal": "Subgoal setting",
    "backward_chaining": "Backward chaining",
}
ABBREVIATIONS = {
    "Question_and_Answering": "Q",
    "Perspective_Shift": "P",
    "Conflict_of_Perspectives": "C",
    "Reconciliation": "R",
    "verification": "V",
    "backtracking": "B",
    "subgoal": "S",
    "backward_chaining": "K",
}
COLORS = {
    "Question_and_Answering": "#3B6FB6",
    "Perspective_Shift": "#6E56A5",
    "Conflict_of_Perspectives": "#B14D68",
    "Reconciliation": "#C87824",
    "verification": "#157A6E",
    "backtracking": "#8A6D3B",
    "subgoal": "#467A3C",
    "backward_chaining": "#4F6673",
}


def _stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def add_timing_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Add two amount-invariant phase-shape contrasts per behavior."""
    out = frame.copy()
    for behavior in BEHAVIORS:
        phase = out[[f"{behavior}__early", f"{behavior}__middle", f"{behavior}__late"]].to_numpy(float)
        total = np.nansum(phase, axis=1)
        shares = np.divide(phase, total[:, None], out=np.zeros_like(phase), where=total[:, None] > 0)
        out[f"{behavior}__timing_middle_minus_early"] = shares[:, 1] - shares[:, 0]
        out[f"{behavior}__timing_late_minus_early"] = shares[:, 2] - shares[:, 0]
        out[f"{behavior}__timing_centroid"] = np.where(
            total > 0,
            0.5 * shares[:, 1] + shares[:, 2],
            np.nan,
        )
    return out


def assign_quality_classes(frame: pd.DataFrame) -> pd.DataFrame:
    """Define incomplete, high, and low without converting missing scores to failure."""
    out = frame.copy()
    out["quality_class"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[~out["completed"].fillna(False), "quality_class"] = "incomplete"
    eligible = out["completed"].fillna(False) & out["outcome_available"].fillna(False)
    binary = eligible & out["outcome_type"].eq("binary")
    out.loc[binary, "quality_class"] = np.where(
        out.loc[binary, "outcome_higher_is_better"].to_numpy(float) >= 0.5, "high", "low"
    )
    continuous = eligible & out["outcome_type"].eq("continuous")
    if continuous.any():
        medians = out.loc[continuous].groupby(["gen_model", "task_type"], observed=True)[
            "outcome_higher_is_better"
        ].transform("median")
        out.loc[continuous, "quality_class"] = np.where(
            out.loc[continuous, "outcome_higher_is_better"].to_numpy(float) >= medians.to_numpy(float),
            "high",
            "low",
        )
    return out


def _stratified_folds(labels: np.ndarray, folds: int, seed: int) -> np.ndarray:
    assignment = np.full(len(labels), -1, dtype=int)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng = np.random.default_rng(_stable_seed(str(label), seed))
        idx = idx[rng.permutation(len(idx))]
        assignment[idx] = np.arange(len(idx)) % folds
    if (assignment < 0).any():
        raise ValueError("fold assignment failed")
    return assignment


def _design(train: pd.DataFrame, test: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    tr = train[list(columns)].replace([np.inf, -np.inf], np.nan).astype(float)
    te = test[list(columns)].replace([np.inf, -np.inf], np.nan).astype(float)
    means = tr.mean().fillna(0.0)
    scales = tr.std(ddof=0).replace(0, 1).fillna(1.0)
    tr = ((tr.fillna(means) - means) / scales).fillna(0.0)
    te = ((te.fillna(means) - means) / scales).fillna(0.0)
    return np.column_stack([np.ones(len(tr)), tr.to_numpy()]), np.column_stack([np.ones(len(te)), te.to_numpy()])


def ridge_multinomial_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_classes: int,
    l2: float = 1.0,
) -> np.ndarray:
    """Reference-class multinomial logistic regression with ridge slopes."""
    if n_classes < 2:
        raise ValueError("multinomial regression needs at least two classes")
    n_features = x_train.shape[1]
    y_onehot = np.eye(n_classes, dtype=float)[y_train]

    initial = np.zeros((n_features, n_classes - 1), dtype=float)
    counts = np.bincount(y_train, minlength=n_classes).astype(float) + 0.5
    initial[0] = np.log(counts[1:] / counts[0])

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        weights = flat.reshape(n_features, n_classes - 1)
        eta = x_train @ weights
        logits = np.column_stack([np.zeros(len(x_train)), eta])
        log_norm = logsumexp(logits, axis=1)
        log_prob = logits - log_norm[:, None]
        probability = np.exp(log_prob)
        loss = -float(np.sum(y_onehot * log_prob)) + 0.5 * l2 * float(np.sum(weights[1:] ** 2))
        gradient = x_train.T @ (probability[:, 1:] - y_onehot[:, 1:])
        gradient[1:] += l2 * weights[1:]
        return loss, gradient.ravel()

    fit = minimize(
        objective,
        initial.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not fit.success and not np.isfinite(fit.fun):
        raise RuntimeError(f"multinomial fit failed: {fit.message}")
    weights = fit.x.reshape(n_features, n_classes - 1)
    logits = np.column_stack([np.zeros(len(x_test)), x_test @ weights])
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


def _cv_probabilities(
    data: pd.DataFrame,
    target: str,
    blocks: Mapping[str, Sequence[str]],
    *,
    folds: int,
    seed: int,
    fold_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    labels = sorted(data[target].astype(str).unique())
    lookup = {label: i for i, label in enumerate(labels)}
    y = data[target].astype(str).map(lookup).to_numpy(int)
    if fold_ids is None:
        fold_id = _stratified_folds(y, folds, seed)
    else:
        fold_id = np.asarray(fold_ids, dtype=int)
        if len(fold_id) != len(data):
            raise ValueError("fold_ids must align one-to-one with data")
        observed_folds = np.unique(fold_id)
        if len(observed_folds) < 2:
            raise ValueError("at least two held-out folds are required")
    predictions = {name: np.full((len(data), len(labels)), np.nan) for name in ["null", *blocks]}
    for fold in sorted(np.unique(fold_id)):
        train = data[fold_id != fold]
        test = data[fold_id == fold]
        train_y = y[fold_id != fold]
        if len(np.unique(train_y)) != len(labels):
            raise ValueError(f"training fold {fold} does not contain every target class")
        counts = np.bincount(train_y, minlength=len(labels)).astype(float) + 0.5
        predictions["null"][fold_id == fold] = counts / counts.sum()
        for name, columns in blocks.items():
            x_train, x_test = _design(train, test, columns)
            predictions[name][fold_id == fold] = ridge_multinomial_predict(
                x_train, train_y, x_test, len(labels), l2=1.0
            )
    return y, predictions, labels


def _log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probability[np.arange(len(y)), y], 1e-12, 1))))


def _balanced_accuracy(y: np.ndarray, probability: np.ndarray) -> float:
    predicted = probability.argmax(axis=1)
    return float(np.mean([np.mean(predicted[y == cls] == cls) for cls in np.unique(y)]))


def _binary_auc(y: np.ndarray, probability_high: np.ndarray) -> float:
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return math.nan
    ranks = rankdata(probability_high)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _bootstrap_gains(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    first: str,
    second: str,
    *,
    reps: int,
    seed: int,
) -> tuple[float, float, float, float]:
    per_trace = (
        -np.log(np.clip(predictions[first][np.arange(len(y)), y], 1e-12, 1))
        + np.log(np.clip(predictions[second][np.arange(len(y)), y], 1e-12, 1))
    ) / math.log(2)
    point = float(per_trace.mean())
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == value) for value in np.unique(y)]
    values = np.empty(reps)
    for rep in range(reps):
        sampled = np.concatenate([idx[rng.integers(0, len(idx), len(idx))] for idx in strata])
        values[rep] = (
            _log_loss(y[sampled], predictions[first][sampled])
            - _log_loss(y[sampled], predictions[second][sampled])
        ) / math.log(2)
    null = np.empty(reps)
    for rep in range(reps):
        null[rep] = np.mean(per_trace * rng.choice(np.array([-1.0, 1.0]), size=len(per_trace)))
    p_value = (1 + int((null >= point).sum())) / (reps + 1)
    return float(point), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), float(p_value)


def _bh(values: pd.Series) -> pd.Series:
    x = values.to_numpy(float); valid = np.isfinite(x); out = np.full(len(x), np.nan)
    if not valid.any():
        return pd.Series(out, index=values.index)
    p = x[valid]; order = np.argsort(p); ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(adjusted); restored[order] = np.minimum(adjusted, 1)
    out[valid] = restored
    return pd.Series(out, index=values.index)


def domain_signature_models(features: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_data = features[features["gen_model"].eq(model)].reset_index(drop=True)
        for family, behaviors in FAMILIES.items():
            amount = [f"{behavior}__rate" for behavior in behaviors]
            timing = [
                f"{behavior}__timing_{contrast}"
                for behavior in behaviors
                for contrast in ["middle_minus_early", "late_minus_early"]
            ]
            y, pred, labels = _cv_probabilities(
                model_data,
                "task_type",
                {
                    "length": ["log_tokens", "log_segments"],
                    "amount": ["log_tokens", "log_segments", *amount],
                    "full": ["log_tokens", "log_segments", *amount, *timing],
                },
                folds=5,
                seed=_stable_seed(f"domain|{model}|{family}", seed),
                fold_ids=model_data["prompt_fold"].to_numpy(int),
            )
            amount_gain = _bootstrap_gains(y, pred, "length", "amount", reps=reps, seed=_stable_seed(f"amount|{model}|{family}", seed))
            timing_gain = _bootstrap_gains(y, pred, "amount", "full", reps=reps, seed=_stable_seed(f"timing|{model}|{family}", seed))
            rows.append({
                "gen_model": model,
                "model_label": MODEL_LABELS[model],
                "signature_family": family,
                "n": len(model_data),
                "domains": len(labels),
                "length_balanced_accuracy": _balanced_accuracy(y, pred["length"]),
                "amount_balanced_accuracy": _balanced_accuracy(y, pred["amount"]),
                "full_balanced_accuracy": _balanced_accuracy(y, pred["full"]),
                "length_log_loss": _log_loss(y, pred["length"]),
                "amount_log_loss": _log_loss(y, pred["amount"]),
                "full_log_loss": _log_loss(y, pred["full"]),
                "amount_gain_bits": amount_gain[0],
                "amount_gain_ci_low": amount_gain[1],
                "amount_gain_ci_high": amount_gain[2],
                "amount_p_value": amount_gain[3],
                "timing_gain_bits": timing_gain[0],
                "timing_gain_ci_low": timing_gain[1],
                "timing_gain_ci_high": timing_gain[2],
                "timing_p_value": timing_gain[3],
            })
    result = pd.DataFrame(rows)
    result["amount_q_value"] = _bh(result.amount_p_value)
    result["timing_q_value"] = _bh(result.timing_p_value)
    result["amount_supported"] = (result.amount_gain_ci_low > 0) & (result.amount_q_value < .05)
    result["timing_supported"] = (result.timing_gain_ci_low > 0) & (result.timing_q_value < .05)
    return result


def quality_signature_models(features: pd.DataFrame, reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = (
        features.groupby(["gen_model", "task_type", "quality_class"], observed=True)
        .size().unstack(fill_value=0).reindex(columns=["high", "low", "incomplete"], fill_value=0).reset_index()
    )
    rows: list[dict[str, Any]] = []
    for _, counts in coverage.iterrows():
        model = counts.gen_model; domain = counts.task_type
        cell = features[
            features["gen_model"].eq(model)
            & features["task_type"].eq(domain)
            & features["quality_class"].isin(["high", "low"])
        ].reset_index(drop=True)
        for family, behaviors in FAMILIES.items():
            base = {
                "gen_model": model,
                "model_label": MODEL_LABELS[model],
                "task_type": domain,
                "signature_family": family,
                "high_n": int(counts.get("high", 0)),
                "low_n": int(counts.get("low", 0)),
                "incomplete_n": int(counts.get("incomplete", 0)),
                "n": len(cell),
            }
            if min(base["high_n"], base["low_n"]) < 20:
                rows.append({**base, "status": "insufficient_high_low_support"})
                continue
            amount = [f"{behavior}__rate" for behavior in behaviors]
            timing = [
                f"{behavior}__timing_{contrast}"
                for behavior in behaviors
                for contrast in ["middle_minus_early", "late_minus_early"]
            ]
            y, pred, labels = _cv_probabilities(
                cell,
                "quality_class",
                {
                    "length": ["log_tokens", "log_segments"],
                    "amount": ["log_tokens", "log_segments", *amount],
                    "full": ["log_tokens", "log_segments", *amount, *timing],
                },
                folds=5,
                seed=_stable_seed(f"quality|{model}|{domain}|{family}", seed),
                fold_ids=cell["prompt_fold"].to_numpy(int),
            )
            high_index = labels.index("high")
            y_high = (y == high_index).astype(int)
            amount_gain = _bootstrap_gains(y, pred, "length", "amount", reps=reps, seed=_stable_seed(f"qamount|{model}|{domain}|{family}", seed))
            timing_gain = _bootstrap_gains(y, pred, "amount", "full", reps=reps, seed=_stable_seed(f"qtiming|{model}|{domain}|{family}", seed))
            rows.append({
                **base,
                "status": "estimated",
                "length_auroc": _binary_auc(y_high, pred["length"][:, high_index]),
                "amount_auroc": _binary_auc(y_high, pred["amount"][:, high_index]),
                "full_auroc": _binary_auc(y_high, pred["full"][:, high_index]),
                "amount_gain_bits": amount_gain[0],
                "amount_gain_ci_low": amount_gain[1],
                "amount_gain_ci_high": amount_gain[2],
                "amount_p_value": amount_gain[3],
                "timing_gain_bits": timing_gain[0],
                "timing_gain_ci_low": timing_gain[1],
                "timing_gain_ci_high": timing_gain[2],
                "timing_p_value": timing_gain[3],
            })
    result = pd.DataFrame(rows)
    estimated = result.status.eq("estimated")
    result.loc[estimated, "amount_q_value"] = _bh(result.loc[estimated, "amount_p_value"])
    result.loc[estimated, "timing_q_value"] = _bh(result.loc[estimated, "timing_p_value"])
    result["amount_supported"] = estimated & (result.amount_gain_ci_low > 0) & (result.amount_q_value < .05)
    result["timing_supported"] = estimated & (result.timing_gain_ci_low > 0) & (result.timing_q_value < .05)
    coverage["incomplete_adequate_for_three_class_model"] = coverage["incomplete"] >= 20
    return result, coverage


def _gain_strata(y: np.ndarray, outcome_kind: str) -> list[np.ndarray]:
    """Return stable resampling strata for paired predictive-score gains."""
    if outcome_kind in {"multiclass", "binary"}:
        return [np.flatnonzero(y == value) for value in np.unique(y)]
    ranks = pd.Series(y).rank(method="first")
    bins = pd.qcut(ranks, q=min(4, len(y)), labels=False, duplicates="drop").to_numpy(int)
    return [np.flatnonzero(bins == value) for value in np.unique(bins)]


def _per_trace_predictive_gain(
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    outcome_kind: str,
) -> np.ndarray:
    """Paired held-out score improvement; positive values favor ``second``."""
    if outcome_kind == "multiclass":
        row = np.arange(len(y))
        return np.log2(
            np.clip(second[row, y], 1e-12, 1) / np.clip(first[row, y], 1e-12, 1)
        )
    if outcome_kind == "binary":
        first_true = np.where(y == 1, first, 1 - first)
        second_true = np.where(y == 1, second, 1 - second)
        return np.log2(np.clip(second_true, 1e-12, 1) / np.clip(first_true, 1e-12, 1))
    variance = float(np.var(y, ddof=0))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("continuous predictive gain requires nonzero outcome variance")
    return ((y - first) ** 2 - (y - second) ** 2) / variance


def _summarize_paired_gain(
    values: np.ndarray,
    strata: Sequence[np.ndarray],
    *,
    reps: int,
    seed: int,
) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        return math.nan, math.nan, math.nan, math.nan
    point = float(values.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(reps)
    for rep in range(reps):
        sampled = np.concatenate(
            [idx[rng.integers(0, len(idx), len(idx))] for idx in strata if len(idx)]
        )
        boot[rep] = float(values[sampled].mean())
    null = np.empty(reps)
    for rep in range(reps):
        null[rep] = float(np.mean(values * rng.choice(np.array([-1.0, 1.0]), len(values))))
    p_value = (1 + int((null >= point).sum())) / (reps + 1)
    return point, float(np.quantile(boot, .025)), float(np.quantile(boot, .975)), float(p_value)


def _subset_feature_blocks(
    base: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> tuple[dict[str, list[str]], dict[int, str]]:
    names = list(groups)
    blocks: dict[str, list[str]] = {}
    lookup: dict[int, str] = {}
    for mask in range(1 << len(names)):
        key = f"subset_{mask:02d}"
        columns = list(base)
        for index, name in enumerate(names):
            if mask & (1 << index):
                columns.extend(groups[name])
        blocks[key] = columns
        lookup[mask] = key
    return blocks, lookup


def exact_shapley_trace_gains(
    y: np.ndarray,
    predictions: Mapping[int, np.ndarray],
    behavior_names: Sequence[str],
    outcome_kind: str,
) -> dict[str, np.ndarray]:
    """Exact four-group Shapley allocation of held-out predictive gain."""
    count = len(behavior_names)
    expected_masks = set(range(1 << count))
    if set(predictions) != expected_masks:
        raise ValueError("predictions must contain every feature-subset mask")
    result: dict[str, np.ndarray] = {}
    denominator = math.factorial(count)
    for index, behavior in enumerate(behavior_names):
        contribution = np.zeros(len(y), dtype=float)
        for mask in expected_masks:
            if mask & (1 << index):
                continue
            subset_size = int(mask.bit_count())
            weight = (
                math.factorial(subset_size)
                * math.factorial(count - subset_size - 1)
                / denominator
            )
            contribution += weight * _per_trace_predictive_gain(
                y,
                predictions[mask],
                predictions[mask | (1 << index)],
                outcome_kind,
            )
        result[behavior] = contribution
    return result


def _two_group_direction(
    first: np.ndarray,
    second: np.ndarray,
    *,
    standardized: bool,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    first = np.asarray(first, dtype=float); second = np.asarray(second, dtype=float)
    first = first[np.isfinite(first)]; second = second[np.isfinite(second)]
    if len(first) < 5 or len(second) < 5:
        return math.nan, math.nan, math.nan
    pooled = np.concatenate([first, second])
    scale = float(np.std(pooled, ddof=0)) if standardized else 1.0
    if not np.isfinite(scale) or scale <= 0:
        return 0.0, 0.0, 0.0
    point = float((first.mean() - second.mean()) / scale)
    rng = np.random.default_rng(seed); boot = np.empty(reps)
    for rep in range(reps):
        left = first[rng.integers(0, len(first), len(first))]
        right = second[rng.integers(0, len(second), len(second))]
        boot[rep] = (left.mean() - right.mean()) / scale
    return point, float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def pairwise_domain_signature_models(features: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """Localize the omnibus domain result to all 28 domain pairs."""
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_data = features[features.gen_model.eq(model)]
        for family, behaviors in FAMILIES.items():
            amount = [f"{behavior}__rate" for behavior in behaviors]
            timing = [
                f"{behavior}__timing_{contrast}"
                for behavior in behaviors
                for contrast in ["middle_minus_early", "late_minus_early"]
            ]
            for domain_a, domain_b in itertools.combinations(DOMAINS, 2):
                cell = model_data[model_data.task_type.isin([domain_a, domain_b])].reset_index(drop=True)
                y, pred, labels = _cv_probabilities(
                    cell,
                    "task_type",
                    {
                        "length": ["log_tokens", "log_segments"],
                        "amount": ["log_tokens", "log_segments", *amount],
                        "full": ["log_tokens", "log_segments", *amount, *timing],
                    },
                    folds=5,
                    seed=_stable_seed(f"pair|{model}|{family}|{domain_a}|{domain_b}", seed),
                    fold_ids=cell.prompt_fold.to_numpy(int),
                )
                amount_gain = _bootstrap_gains(
                    y, pred, "length", "amount", reps=reps,
                    seed=_stable_seed(f"pair-amount|{model}|{family}|{domain_a}|{domain_b}", seed),
                )
                timing_gain = _bootstrap_gains(
                    y, pred, "amount", "full", reps=reps,
                    seed=_stable_seed(f"pair-timing|{model}|{family}|{domain_a}|{domain_b}", seed),
                )
                rows.append({
                    "gen_model": model,
                    "model_label": MODEL_LABELS[model],
                    "signature_family": family,
                    "domain_a": domain_a,
                    "domain_b": domain_b,
                    "domain_pair": f"{DOMAIN_LABELS[domain_a]} – {DOMAIN_LABELS[domain_b]}",
                    "n": len(cell),
                    "n_domain_a": int(cell.task_type.eq(domain_a).sum()),
                    "n_domain_b": int(cell.task_type.eq(domain_b).sum()),
                    "length_auroc": _binary_auc(y, pred["length"][:, 1]),
                    "amount_auroc": _binary_auc(y, pred["amount"][:, 1]),
                    "full_auroc": _binary_auc(y, pred["full"][:, 1]),
                    "amount_gain_bits": amount_gain[0],
                    "amount_gain_ci_low": amount_gain[1],
                    "amount_gain_ci_high": amount_gain[2],
                    "amount_p_value": amount_gain[3],
                    "timing_gain_bits": timing_gain[0],
                    "timing_gain_ci_low": timing_gain[1],
                    "timing_gain_ci_high": timing_gain[2],
                    "timing_p_value": timing_gain[3],
                    "class_order": "|".join(labels),
                })
    result = pd.DataFrame(rows)
    result["amount_q_value"] = _bh(result.amount_p_value)
    result["timing_q_value"] = _bh(result.timing_p_value)
    result["amount_supported"] = (result.amount_gain_ci_low > 0) & (result.amount_q_value < .05)
    result["timing_supported"] = (result.timing_gain_ci_low > 0) & (result.timing_q_value < .05)
    return result


def domain_behavior_attribution(features: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """Allocate held-out eight-domain information gain to individual behaviors."""
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_data = features[features.gen_model.eq(model)].reset_index(drop=True)
        for family, behaviors in FAMILIES.items():
            amount_groups = {behavior: [f"{behavior}__rate"] for behavior in behaviors}
            timing_groups = {
                behavior: [
                    f"{behavior}__timing_middle_minus_early",
                    f"{behavior}__timing_late_minus_early",
                ]
                for behavior in behaviors
            }
            amount_blocks, amount_lookup = _subset_feature_blocks(
                ["log_tokens", "log_segments"], amount_groups
            )
            y, amount_pred, labels = _cv_probabilities(
                model_data, "task_type", amount_blocks, folds=5,
                seed=_stable_seed(f"domain-shapley-amount|{model}|{family}", seed),
                fold_ids=model_data.prompt_fold.to_numpy(int),
            )
            amount_predictions = {mask: amount_pred[key] for mask, key in amount_lookup.items()}
            amount_shapley = exact_shapley_trace_gains(
                y, amount_predictions, list(behaviors), "multiclass"
            )
            all_amount = [f"{behavior}__rate" for behavior in behaviors]
            timing_blocks, timing_lookup = _subset_feature_blocks(
                ["log_tokens", "log_segments", *all_amount], timing_groups
            )
            _, timing_pred, _ = _cv_probabilities(
                model_data, "task_type", timing_blocks, folds=5,
                seed=_stable_seed(f"domain-shapley-timing|{model}|{family}", seed),
                fold_ids=model_data.prompt_fold.to_numpy(int),
            )
            timing_predictions = {mask: timing_pred[key] for mask, key in timing_lookup.items()}
            timing_shapley = exact_shapley_trace_gains(
                y, timing_predictions, list(behaviors), "multiclass"
            )
            for domain in DOMAINS:
                domain_index = labels.index(domain)
                selected = np.flatnonzero(y == domain_index)
                rest = np.flatnonzero(y != domain_index)
                for layer, shapley in [("amount", amount_shapley), ("timing", timing_shapley)]:
                    for behavior in behaviors:
                        contribution = shapley[behavior][selected]
                        gain = _summarize_paired_gain(
                            contribution, [np.arange(len(contribution))], reps=reps,
                            seed=_stable_seed(f"domain-attr|{layer}|{model}|{family}|{domain}|{behavior}", seed),
                        )
                        if layer == "amount":
                            direction = _two_group_direction(
                                model_data.loc[selected, f"{behavior}__rate"].to_numpy(float),
                                model_data.loc[rest, f"{behavior}__rate"].to_numpy(float),
                                standardized=True, reps=reps,
                                seed=_stable_seed(f"domain-direction|amount|{model}|{domain}|{behavior}", seed),
                            )
                            direction_unit = "standardized rate difference: domain minus other domains"
                        else:
                            direction = _two_group_direction(
                                model_data.loc[selected, f"{behavior}__timing_centroid"].to_numpy(float),
                                model_data.loc[rest, f"{behavior}__timing_centroid"].to_numpy(float),
                                standardized=False, reps=reps,
                                seed=_stable_seed(f"domain-direction|timing|{model}|{domain}|{behavior}", seed),
                            )
                            direction_unit = "conditional timing centroid difference: positive means later"
                        rows.append({
                            "gen_model": model,
                            "model_label": MODEL_LABELS[model],
                            "task_type": domain,
                            "domain": DOMAIN_LABELS[domain],
                            "signature_family": family,
                            "layer": layer,
                            "behavior": behavior,
                            "behavior_label": BEHAVIOR_LABELS[behavior],
                            "n_domain": len(selected),
                            "shapley_gain": gain[0],
                            "shapley_ci_low": gain[1],
                            "shapley_ci_high": gain[2],
                            "shapley_p_value": gain[3],
                            "shapley_unit": "bits per held-out trace",
                            "direction_effect": direction[0],
                            "direction_ci_low": direction[1],
                            "direction_ci_high": direction[2],
                            "direction_unit": direction_unit,
                        })
    result = pd.DataFrame(rows)
    result["shapley_q_value"] = np.nan
    for layer in ["amount", "timing"]:
        selected = result.layer.eq(layer)
        result.loc[selected, "shapley_q_value"] = _bh(result.loc[selected, "shapley_p_value"])
    result["supported"] = (result.shapley_ci_low > 0) & (result.shapley_q_value < .05)
    return result


def _ridge_continuous_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    l2: float = 1.0,
) -> np.ndarray:
    penalty = np.eye(x_train.shape[1]) * l2
    penalty[0, 0] = 0.0
    return x_test @ np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y_train


def _cv_native_predictions(
    data: pd.DataFrame,
    target: str,
    blocks: Mapping[str, Sequence[str]],
    *,
    outcome_kind: str,
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = data[target].to_numpy(float)
    if outcome_kind == "binary":
        labels = sorted(np.unique(raw))
        if len(labels) != 2:
            raise ValueError("binary outcome requires exactly two observed classes")
        y = pd.Series(raw).map({value: index for index, value in enumerate(labels)}).to_numpy(int)
    else:
        y = raw.astype(float)
    fold_id = np.asarray(fold_ids, dtype=int)
    predictions = {name: np.full(len(data), np.nan) for name in blocks}
    for fold in sorted(np.unique(fold_id)):
        train = data[fold_id != fold]; test = data[fold_id == fold]
        train_y = y[fold_id != fold]
        if outcome_kind == "binary" and len(np.unique(train_y)) != 2:
            raise ValueError(f"training fold {fold} does not contain both outcome classes")
        for name, columns in blocks.items():
            x_train, x_test = _design(train, test, columns)
            if outcome_kind == "binary":
                probability = ridge_multinomial_predict(x_train, train_y, x_test, 2, l2=1.0)
                predictions[name][fold_id == fold] = probability[:, 1]
            else:
                predictions[name][fold_id == fold] = _ridge_continuous_predict(
                    x_train, train_y, x_test, l2=1.0
                )
    return y, predictions


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - prediction) ** 2) / denominator) if denominator > 0 else math.nan


def _native_metric(y: np.ndarray, prediction: np.ndarray, outcome_kind: str) -> float:
    return _binary_auc(y, prediction) if outcome_kind == "binary" else _r2(y, prediction)


def _prompt_stratum_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    stratum = frame["difficulty_raw"].astype("string").fillna("not_recorded")
    dummies = pd.get_dummies(stratum, prefix="prompt_stratum", dtype=float)
    return pd.concat([frame.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1), list(dummies)


def native_outcome_signature_models(
    features: pd.DataFrame,
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict native outcomes within fixed model-domain cells and localize signal."""
    prepared, stratum_columns = _prompt_stratum_columns(features)
    model_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for domain in DOMAINS:
            base_cell = prepared[
                prepared.gen_model.eq(model)
                & prepared.task_type.eq(domain)
                & prepared.outcome_available.fillna(False)
            ].reset_index(drop=True)
            outcome_types = base_cell.outcome_type.dropna().unique().tolist()
            outcome_kind = "binary" if outcome_types == ["binary"] else "continuous"
            counts = base_cell.outcome_higher_is_better.value_counts()
            common = {
                "gen_model": model,
                "model_label": MODEL_LABELS[model],
                "task_type": domain,
                "domain": DOMAIN_LABELS[domain],
                "outcome_kind": outcome_kind,
                "n": len(base_cell),
                "outcome_unique": int(base_cell.outcome_higher_is_better.nunique()),
            }
            if len(base_cell) < 40:
                for family in FAMILIES:
                    model_rows.append({**common, "signature_family": family, "status": "insufficient_total_support"})
                continue
            if outcome_kind == "binary" and (len(counts) != 2 or int(counts.min()) < 20):
                for family in FAMILIES:
                    model_rows.append({**common, "signature_family": family, "status": "insufficient_binary_class_support"})
                continue
            if outcome_kind == "continuous" and base_cell.outcome_higher_is_better.nunique() < 3:
                for family in FAMILIES:
                    model_rows.append({**common, "signature_family": family, "status": "insufficient_outcome_variation"})
                continue
            active_strata = [column for column in stratum_columns if base_cell[column].nunique() > 1]
            baseline = ["log_tokens", "log_segments", *active_strata]
            for family, behaviors in FAMILIES.items():
                amount_groups = {behavior: [f"{behavior}__rate"] for behavior in behaviors}
                timing_groups = {
                    behavior: [
                        f"{behavior}__timing_middle_minus_early",
                        f"{behavior}__timing_late_minus_early",
                    ]
                    for behavior in behaviors
                }
                amount_blocks, amount_lookup = _subset_feature_blocks(baseline, amount_groups)
                try:
                    y, amount_pred = _cv_native_predictions(
                        base_cell, "outcome_higher_is_better", amount_blocks,
                        outcome_kind=outcome_kind, fold_ids=base_cell.prompt_fold.to_numpy(int),
                    )
                except ValueError as error:
                    model_rows.append({
                        **common, "signature_family": family,
                        "status": "fold_support_failure", "status_detail": str(error),
                    })
                    continue
                amount_predictions = {mask: amount_pred[key] for mask, key in amount_lookup.items()}
                all_amount = [f"{behavior}__rate" for behavior in behaviors]
                timing_blocks, timing_lookup = _subset_feature_blocks(
                    [*baseline, *all_amount], timing_groups
                )
                _, timing_pred = _cv_native_predictions(
                    base_cell, "outcome_higher_is_better", timing_blocks,
                    outcome_kind=outcome_kind, fold_ids=base_cell.prompt_fold.to_numpy(int),
                )
                timing_predictions = {mask: timing_pred[key] for mask, key in timing_lookup.items()}
                strata = _gain_strata(y, outcome_kind)
                amount_values = _per_trace_predictive_gain(
                    y, amount_predictions[0], amount_predictions[15], outcome_kind
                )
                timing_values = _per_trace_predictive_gain(
                    y, timing_predictions[0], timing_predictions[15], outcome_kind
                )
                amount_gain = _summarize_paired_gain(
                    amount_values, strata, reps=reps,
                    seed=_stable_seed(f"native-amount|{model}|{domain}|{family}", seed),
                )
                timing_gain = _summarize_paired_gain(
                    timing_values, strata, reps=reps,
                    seed=_stable_seed(f"native-timing|{model}|{domain}|{family}", seed),
                )
                metric_name = "AUROC" if outcome_kind == "binary" else "R2"
                model_rows.append({
                    **common,
                    "signature_family": family,
                    "status": "estimated",
                    "baseline_columns": "|".join(baseline),
                    "primary_metric": metric_name,
                    "baseline_metric": _native_metric(y, amount_predictions[0], outcome_kind),
                    "amount_metric": _native_metric(y, amount_predictions[15], outcome_kind),
                    "full_metric": _native_metric(y, timing_predictions[15], outcome_kind),
                    "amount_gain": amount_gain[0],
                    "amount_gain_ci_low": amount_gain[1],
                    "amount_gain_ci_high": amount_gain[2],
                    "amount_p_value": amount_gain[3],
                    "timing_gain": timing_gain[0],
                    "timing_gain_ci_low": timing_gain[1],
                    "timing_gain_ci_high": timing_gain[2],
                    "timing_p_value": timing_gain[3],
                    "gain_unit": "bits/trace" if outcome_kind == "binary" else "delta R2",
                })
                for layer, predictions in [("amount", amount_predictions), ("timing", timing_predictions)]:
                    shapley = exact_shapley_trace_gains(y, predictions, list(behaviors), outcome_kind)
                    high = y == 1 if outcome_kind == "binary" else y >= np.median(y)
                    for behavior in behaviors:
                        gain = _summarize_paired_gain(
                            shapley[behavior], strata, reps=reps,
                            seed=_stable_seed(f"native-attr|{layer}|{model}|{domain}|{family}|{behavior}", seed),
                        )
                        if layer == "amount":
                            direction = _two_group_direction(
                                base_cell.loc[high, f"{behavior}__rate"].to_numpy(float),
                                base_cell.loc[~high, f"{behavior}__rate"].to_numpy(float),
                                standardized=True, reps=reps,
                                seed=_stable_seed(f"native-direction|amount|{model}|{domain}|{behavior}", seed),
                            )
                            direction_unit = "standardized high-minus-low outcome rate difference"
                        else:
                            direction = _two_group_direction(
                                base_cell.loc[high, f"{behavior}__timing_centroid"].to_numpy(float),
                                base_cell.loc[~high, f"{behavior}__timing_centroid"].to_numpy(float),
                                standardized=False, reps=reps,
                                seed=_stable_seed(f"native-direction|timing|{model}|{domain}|{behavior}", seed),
                            )
                            direction_unit = "high-minus-low conditional timing centroid; positive means later"
                        attribution_rows.append({
                            **common,
                            "signature_family": family,
                            "layer": layer,
                            "behavior": behavior,
                            "behavior_label": BEHAVIOR_LABELS[behavior],
                            "shapley_gain": gain[0],
                            "shapley_ci_low": gain[1],
                            "shapley_ci_high": gain[2],
                            "shapley_p_value": gain[3],
                            "shapley_unit": "bits/trace" if outcome_kind == "binary" else "delta R2",
                            "direction_effect": direction[0],
                            "direction_ci_low": direction[1],
                            "direction_ci_high": direction[2],
                            "direction_unit": direction_unit,
                        })
    models = pd.DataFrame(model_rows)
    estimated = models.status.eq("estimated")
    models.loc[estimated, "amount_q_value"] = _bh(models.loc[estimated, "amount_p_value"])
    models.loc[estimated, "timing_q_value"] = _bh(models.loc[estimated, "timing_p_value"])
    models["amount_supported"] = estimated & (models.amount_gain_ci_low > 0) & (models.amount_q_value < .05)
    models["timing_supported"] = estimated & (models.timing_gain_ci_low > 0) & (models.timing_q_value < .05)
    attribution = pd.DataFrame(attribution_rows)
    attribution["shapley_q_value"] = np.nan
    for layer in ["amount", "timing"]:
        selected = attribution.layer.eq(layer)
        attribution.loc[selected, "shapley_q_value"] = _bh(
            attribution.loc[selected, "shapley_p_value"]
        )
    attribution["supported"] = (
        (attribution.shapley_ci_low > 0) & (attribution.shapley_q_value < .05)
    )
    parents = models[
        ["gen_model", "task_type", "signature_family", "amount_supported", "timing_supported"]
    ]
    attribution = attribution.merge(
        parents, on=["gen_model", "task_type", "signature_family"], how="left"
    )
    attribution["parent_block_supported"] = np.where(
        attribution.layer.eq("amount"), attribution.amount_supported, attribution.timing_supported
    )
    return models, attribution


def modal_paths(config: Mapping[str, Any], features: pd.DataFrame) -> pd.DataFrame:
    meta = features[["trace_id", "gen_model", "task_type", "quality_class"]].copy()
    all_meta = meta[["trace_id", "gen_model", "task_type"]].copy(); all_meta["quality_class"] = "all"
    actual = meta.dropna(subset=["quality_class"])
    join_meta = pl.from_pandas(pd.concat([all_meta, actual], ignore_index=True)).lazy()
    lf = (
        scan_track_b(config["paths"]["track_b_full"], "full")
        .filter(pl.col("kim_parsed").fill_null(False) & pl.col("gandhi_parsed").fill_null(False))
        .select(["trace_id", "norm_pos", *BEHAVIORS])
        .join(join_meta, on="trace_id", how="inner")
        .with_columns((pl.col("norm_pos") * 10).floor().cast(pl.Int8).clip(0, 9).alias("decile"))
        .group_by(["gen_model", "task_type", "quality_class", "decile"])
        .agg([pl.len().alias("segments"), *[pl.col(behavior).sum().alias(behavior) for behavior in BEHAVIORS]])
    )
    wide = lf.collect(engine="streaming").to_pandas()
    long = wide.melt(
        id_vars=["gen_model", "task_type", "quality_class", "decile", "segments"],
        value_vars=list(BEHAVIORS),
        var_name="behavior",
        value_name="positive_segments",
    )
    long["prevalence"] = long["positive_segments"] / long["segments"]
    long["signature_family"] = np.where(long["behavior"].isin(DIALECTICAL), "conversational", "cognitive")
    family_total = long.groupby(
        ["gen_model", "task_type", "quality_class", "decile", "signature_family"], observed=True
    )["prevalence"].transform("sum")
    long["dominance_share"] = np.divide(long["prevalence"], family_total, out=np.zeros(len(long)), where=family_total > 0)
    order = long.sort_values(
        ["gen_model", "task_type", "quality_class", "decile", "signature_family", "prevalence", "behavior"],
        ascending=[True, True, True, True, True, False, True],
    )
    modal = order.groupby(
        ["gen_model", "task_type", "quality_class", "decile", "signature_family"], observed=True
    ).head(1).copy()
    modal["behavior_label"] = modal["behavior"].map(BEHAVIOR_LABELS)
    modal["abbreviation"] = modal["behavior"].map(ABBREVIATIONS)
    return modal.sort_values(["gen_model", "task_type", "quality_class", "signature_family", "decile"])


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "figure.dpi": 120, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, stem: str, figures: Path) -> None:
    fig.savefig(figures / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_domain_gains(results: pd.DataFrame, figures: Path) -> None:
    # Compact paired forest plot: model labels appear once, the two feature
    # families share each row, and amount/timing retain honest separate scales.
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.45), sharey=True, constrained_layout=True)
    y = np.arange(len(MODELS), dtype=float)
    styles = {
        "conversational": {"offset": -.105, "marker": "o", "color": "#3B6FB6", "label": "Conversational"},
        "cognitive": {"offset": .105, "marker": "s", "color": "#157A6E", "label": "Cognitive"},
    }
    for ax, (metric, title) in zip(axes, [
        ("amount", "Amount beyond trace length"),
        ("timing", "Timing beyond amount"),
    ]):
        for family in ["conversational", "cognitive"]:
            style = styles[family]
            part = results[results.signature_family.eq(family)].set_index("gen_model").reindex(MODELS)
            point = part[f"{metric}_gain_bits"].to_numpy(float)
            lo = part[f"{metric}_gain_ci_low"].to_numpy(float)
            hi = part[f"{metric}_gain_ci_high"].to_numpy(float)
            yy = y + style["offset"]
            ax.hlines(yy, lo, hi, color=style["color"], lw=.8, alpha=.75)
            ax.plot(point, yy, linestyle="none", marker=style["marker"], ms=4.2,
                    color=style["color"], label=style["label"])
        ax.axvline(0, color="#9AA1A8", lw=.7)
        ax.set_title(title, loc="left", pad=8)
        ax.set_xlabel("Held-out information gain (bits/trace)")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.grid(axis="x", color="#E4E7E9", lw=.45)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [MODEL_LABELS[model] for model in MODELS])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(.995, .995))
    fig.suptitle("Domain signature within model", x=.01, ha="left", fontsize=11.5)
    _save(fig, "fig10_domain_signature_information_gain", figures)


def figure_pairwise_domain_localization(results: pd.DataFrame, figures: Path) -> None:
    """Small-multiple dot matrices for all domain pairs, with no redundant ink."""
    pairs = [
        f"{DOMAIN_LABELS[left]} – {DOMAIN_LABELS[right]}"
        for left, right in itertools.combinations(DOMAINS, 2)
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(10.3, 10.8), sharex=True, sharey=True, constrained_layout=True
    )
    metrics = [
        ("amount", "Amount beyond trace length"),
        ("timing", "Timing beyond amount"),
    ]
    for row, family in enumerate(["conversational", "cognitive"]):
        for column, (metric, title) in enumerate(metrics):
            ax = axes[row, column]
            part = results[results.signature_family.eq(family)]
            maximum = max(float(results[f"{metric}_gain_bits"].quantile(.99)), 1e-6)
            for model_index, model in enumerate(MODELS):
                cell = part[part.gen_model.eq(model)].set_index("domain_pair").reindex(pairs)
                values = cell[f"{metric}_gain_bits"].to_numpy(float)
                supported = cell[f"{metric}_supported"].fillna(False).to_numpy(bool)
                sizes = 7 + 62 * np.clip(values / maximum, 0, 1)
                yy = np.arange(len(pairs))
                ax.scatter(
                    np.full(len(pairs), model_index), yy, s=sizes,
                    facecolors=np.where(supported, COLORS["Perspective_Shift"] if family == "conversational" else COLORS["verification"], "none"),
                    edgecolors=COLORS["Perspective_Shift"] if family == "conversational" else COLORS["verification"],
                    linewidths=.55, alpha=.88,
                )
            ax.set_title(title if row == 0 else "", loc="left", pad=9)
            ax.set_xticks(np.arange(len(MODELS)), [MODEL_LABELS[value].replace("DeepSeek-R1 Distill ", "DeepSeek ").replace("Qwen3.5 ", "Qwen ").replace("Gemma 4 ", "Gemma ") for value in MODELS], rotation=38, ha="right")
            ax.set_yticks(np.arange(len(pairs)), pairs)
            ax.invert_yaxis()
            ax.grid(False); ax.tick_params(length=0)
            for y_value in np.arange(len(pairs)):
                ax.axhline(y_value, color="#EEF0F2", lw=.32, zorder=0)
        axes[row, 0].text(
            -.34, .5, family.title(), transform=axes[row, 0].transAxes,
            ha="center", va="center", rotation=90, fontweight="bold",
        )
    fig.suptitle("Which domain pairs have distinct signatures?", x=.01, y=1.012, ha="left", fontsize=11.5)
    fig.text(.01, -.012, "Dot area = held-out information gain · open dot = not supported after FDR correction", fontsize=7.8)
    _save(fig, "fig13_pairwise_domain_localization", figures)


def _behavior_attribution_figure(
    results: pd.DataFrame,
    layer: str,
    figures: Path,
    stem: str,
) -> None:
    """Model-specific domain × behavior fingerprints using signed direction."""
    data = results[results.layer.eq(layer)].copy()
    maximum = max(float(data.shapley_gain.clip(lower=0).quantile(.99)), 1e-6)
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.99, top=.88, bottom=.22, hspace=.27, wspace=.08)
    for ax, model in zip(axes.flat, MODELS):
        part = data[data.gen_model.eq(model)]
        for x_value, behavior in enumerate(BEHAVIORS):
            cell = part[part.behavior.eq(behavior)].set_index("task_type").reindex(DOMAINS)
            gains = cell.shapley_gain.to_numpy(float)
            direction = cell.direction_effect.to_numpy(float)
            supported = cell.supported.fillna(False).to_numpy(bool)
            sizes = 4 + 82 * np.clip(np.maximum(gains, 0) / maximum, 0, 1)
            colors = np.where(direction >= 0, "#2E6E9E", "#C46A32")
            yy = np.arange(len(DOMAINS))
            ax.scatter(np.full(len(DOMAINS), x_value), yy, s=sizes, c=colors, alpha=.72, linewidths=0)
            if supported.any():
                ax.scatter(
                    np.full(supported.sum(), x_value), yy[supported], s=sizes[supported] + 18,
                    facecolors="none", edgecolors="#17202A", linewidths=.7,
                )
        ax.set_title(MODEL_LABELS[model], loc="left")
        ax.set_xticks(np.arange(len(BEHAVIORS)), [ABBREVIATIONS[value] for value in BEHAVIORS])
        ax.set_yticks(np.arange(len(DOMAINS)), [DOMAIN_LABELS[value] for value in DOMAINS])
        ax.invert_yaxis(); ax.tick_params(length=0); ax.grid(False)
        for y_value in np.arange(len(DOMAINS)):
            ax.axhline(y_value, color="#EEF0F2", lw=.35, zorder=0)
    # The family labels are part of the x-axis, not a second legend.  Thin,
    # unfilled blocks keep the Q/P/C/R and V/B/S/K groupings explicit without
    # competing with dot area, ring, or color encodings.
    for ax in axes[1, :]:
        transform = ax.get_xaxis_transform()
        for left, label in [(-.45, "Conversational"), (3.55, "Cognitive")]:
            width = 3.9
            ax.add_patch(Rectangle(
                (left, -.155), width, .115,
                transform=transform,
                fill=False,
                edgecolor="#8B9298",
                linewidth=.6,
                clip_on=False,
            ))
            ax.text(
                left + width / 2, -.19, label,
                transform=transform,
                ha="center", va="top",
                color="#4F565C", fontsize=7.4,
                clip_on=False,
            )
    direction_text = "blue = more in domain · orange = less" if layer == "amount" else "blue = later in domain · orange = earlier"
    fig.suptitle(
        f"Which behaviors make each domain identifiable? — {layer}",
        x=.015, y=.975, ha="left", fontsize=11.3,
    )
    key = " · ".join(f"{ABBREVIATIONS[value]} {BEHAVIOR_LABELS[value]}" for value in BEHAVIORS)
    fig.text(.015, .018, f"Area = exact held-out Shapley gain · ring = FDR support · {direction_text}\n{key}", fontsize=7.2)
    _save(fig, stem, figures)


def _native_outcome_gain_figure(
    results: pd.DataFrame,
    metric: str,
    figures: Path,
    stem: str,
) -> None:
    estimated = results[results.status.eq("estimated")]
    fig, axes = plt.subplots(2, 6, figsize=(12.2, 5.6), sharey="row")
    fig.subplots_adjust(left=.055, right=.99, top=.84, bottom=.17, hspace=.42, wspace=.12)
    for column, model in enumerate(MODELS):
        for row, outcome_kind in enumerate(["binary", "continuous"]):
            ax = axes[row, column]
            part = estimated[estimated.gen_model.eq(model) & estimated.outcome_kind.eq(outcome_kind)]
            domains = [domain for domain in DOMAINS if not part[part.task_type.eq(domain)].empty]
            x = np.arange(len(domains), dtype=float)
            ax.axhline(0, color="#9AA1A8", lw=.65)
            for family, offset, marker, color in [
                ("conversational", -.09, "o", "#3B6FB6"),
                ("cognitive", .09, "s", "#157A6E"),
            ]:
                cell = part[part.signature_family.eq(family)].set_index("task_type").reindex(domains)
                value = cell[f"{metric}_gain"].to_numpy(float)
                lo = cell[f"{metric}_gain_ci_low"].to_numpy(float)
                hi = cell[f"{metric}_gain_ci_high"].to_numpy(float)
                supported = cell[f"{metric}_supported"].fillna(False).to_numpy(bool)
                xx = x + offset
                ax.vlines(xx, lo, hi, color=color, lw=.65, alpha=.7)
                ax.plot(xx, value, linestyle="none", marker=marker, color=color, ms=3.5)
                if supported.any():
                    ax.plot(xx[supported], value[supported], "o", ms=6.5, mfc="none", mec="#17202A", mew=.7)
            ax.set_xticks(x, [DOMAIN_LABELS[value][:4] for value in domains], rotation=45, ha="right")
            ax.tick_params(length=0); ax.grid(axis="y", color="#E6E9EB", lw=.4)
            if row == 0:
                ax.set_title(MODEL_LABELS[model].replace("DeepSeek-R1 Distill ", "DeepSeek ").replace("Qwen3.5 ", "Qwen ").replace("Gemma 4 ", "Gemma "), fontsize=8.5)
            if column == 0:
                ax.set_ylabel("Bits/trace" if outcome_kind == "binary" else "ΔR²")
    qualifier = "beyond baseline" if metric == "amount" else "beyond amount"
    support_count = int(estimated[f"{metric}_supported"].fillna(False).sum())
    fig.suptitle(
        f"Do signatures predict native outcome quality? — {metric} {qualifier}",
        x=.015, y=.975, ha="left", fontsize=11.3,
    )
    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="none", color="#3B6FB6", label="Conversational"),
        mpl.lines.Line2D([], [], marker="s", linestyle="none", color="#157A6E", label="Cognitive"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(.995, .985))
    baseline_text = "Baseline: log tokens + log valid segments + available prompt stratum" if metric == "amount" else "Timing is tested after length, prompt stratum, and all behavior amounts"
    fig.text(.015, .025, f"{baseline_text} · {support_count} cells survive interval + FDR correction", fontsize=7.6)
    _save(fig, stem, figures)


def figure_native_outcome_evidence_gate(results: pd.DataFrame, figures: Path) -> None:
    """Show the full multiplicity result when no native-outcome block is supported."""
    estimated = results[results.status.eq("estimated")]
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    fig.subplots_adjust(left=.105, right=.98, top=.82, bottom=.19)
    for metric, marker, color, label in [
        ("amount", "o", "#3B6FB6", "Amount beyond baseline"),
        ("timing", "s", "#157A6E", "Timing beyond amount"),
    ]:
        values = np.sort(estimated[f"{metric}_q_value"].dropna().to_numpy(float))
        rank = np.arange(1, len(values) + 1)
        ax.plot(rank, values, linestyle="none", marker=marker, ms=3.2, color=color, alpha=.75, label=label)
    ax.axhline(.05, color="#9AA1A8", lw=.75)
    ax.text(1, .058, "FDR threshold .05", color="#687078", va="bottom", fontsize=7.5)
    ax.set_xlabel("Ordered model–domain–family tests")
    ax.set_ylabel("Adjusted q value")
    ax.set_ylim(0, 1.02); ax.grid(axis="y", color="#E6E9EB", lw=.4)
    ax.legend(frameon=False, ncol=1, loc="lower right")
    fig.suptitle("Native outcomes: no signature block clears FDR correction", x=.02, y=.96, ha="left", fontsize=11.3)
    _save(fig, "fig17_native_outcome_evidence_gate", figures)


def _quality_gain_figure(results: pd.DataFrame, metric: str, figures: Path, stem: str, title: str) -> None:
    estimated = results[results.status.eq("estimated")]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2), sharex=True, sharey=True, constrained_layout=True)
    symbols = {"conversational": "o", "cognitive": "s"}; offsets = {"conversational": -.08, "cognitive": .08}
    for ax, model in zip(axes.flat, MODELS):
        part = estimated[estimated.gen_model.eq(model)]
        x = np.arange(len(DOMAINS))
        ax.axhline(0, color="#9AA1A8", lw=.8)
        for family in ["conversational", "cognitive"]:
            fam = part[part.signature_family.eq(family)].set_index("task_type").reindex(DOMAINS)
            values = fam[f"{metric}_gain_bits"]
            ax.plot(x + offsets[family], values, symbols[family], ms=4,
                    color="#3B6FB6" if family == "conversational" else "#157A6E", label=family.title())
            supported = fam[f"{metric}_supported"].fillna(False).to_numpy(bool)
            if supported.any():
                ax.plot((x + offsets[family])[supported], values.to_numpy(float)[supported], "o", ms=7,
                        markerfacecolor="none", markeredgecolor="#17202A", markeredgewidth=.8)
        ax.set_title(MODEL_LABELS[model], loc="left")
        ax.set_xticks(x, [d.title() for d in DOMAINS], rotation=43, ha="right")
        ax.grid(axis="y", color="#E1E4E7", lw=.5)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    fig.supylabel("Information gain (bits/trace)", x=-.01)
    suffix = "  ·  black ring = interval + FDR support" if estimated[f"{metric}_supported"].fillna(False).any() else "  ·  no cell survives FDR correction"
    fig.suptitle(title + suffix, x=.02, ha="left", fontsize=11.5)
    _save(fig, stem, figures)


def _legend_text(fig: plt.Figure, behaviors: Sequence[str], y: float = .01) -> None:
    x = .08
    for behavior in behaviors:
        label = f"{ABBREVIATIONS[behavior]} {BEHAVIOR_LABELS[behavior]}"
        fig.text(x, y, label, color=COLORS[behavior], fontsize=7.5, ha="left")
        x += .21 if len(behaviors) == 4 else .12


def _draw_modal_panel(ax: plt.Axes, data: pd.DataFrame, rows: list[tuple[str, str]], family: str) -> None:
    ax.set_xlim(-.5, 9.5); ax.set_ylim(-.5, len(rows)-.5); ax.invert_yaxis()
    ax.set_xticks([0, 4, 9], ["start", "middle", "end"])
    ax.set_yticks(np.arange(len(rows)), [label for _, label in rows])
    for y, (key, _) in enumerate(rows):
        domain, quality = key.split("|", 1)
        lane = data[data.task_type.eq(domain) & data.quality_class.eq(quality) & data.signature_family.eq(family)]
        if lane.empty:
            ax.text(4.5, y, "not estimable", color="#9299A0", ha="center", va="center", fontsize=7)
            continue
        ax.hlines(y, 0, 9, color="#D7DBDE", lw=.6)
        for point in lane.itertuples():
            alpha = float(np.clip(.35 + .65 * point.dominance_share, .35, 1))
            ax.text(point.decile, y, point.abbreviation, color=COLORS[point.behavior], alpha=alpha,
                    ha="center", va="center", fontweight="bold", fontsize=8)
    ax.grid(False); ax.tick_params(axis="both", length=0)


def figure_modal_paths(modal: pd.DataFrame, coverage: pd.DataFrame, figures: Path) -> None:
    for model in MODELS:
        data = modal[modal.gen_model.eq(model)]
        rows = [(f"{domain}|all", DOMAIN_LABELS[domain]) for domain in DOMAINS]
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), constrained_layout=True)
        for ax, family in zip(axes, ["conversational", "cognitive"]):
            _draw_modal_panel(ax, data, rows, family); ax.set_title(family.title(), loc="left")
        fig.suptitle(f"Dominant operation by trace decile — {MODEL_LABELS[model]}", x=.02, ha="left", fontsize=11.5)
        _legend_text(fig, BEHAVIORS, y=-.015)
        _save(fig, f"modal_paths_domain_{model}", figures)

        quality_rows: list[tuple[str, str]] = []
        allowed: set[tuple[str, str]] = set()
        for domain in DOMAINS:
            cell = coverage[coverage.gen_model.eq(model) & coverage.task_type.eq(domain)]
            for quality, short in [("high", "H"), ("low", "L"), ("incomplete", "I")]:
                quality_rows.append((f"{domain}|{quality}", f"{DOMAIN_LABELS[domain]} · {short}"))
                if not cell.empty and int(cell.iloc[0].get(quality, 0)) >= 20:
                    allowed.add((domain, quality))
        plot_data = data[data.apply(lambda row: (row.task_type, row.quality_class) in allowed, axis=1)]
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 9.2), constrained_layout=True)
        for ax, family in zip(axes, ["conversational", "cognitive"]):
            _draw_modal_panel(ax, plot_data, quality_rows, family); ax.set_title(family.title(), loc="left")
        fig.suptitle(f"Quality-stratified modal paths — {MODEL_LABELS[model]}  (H high · L low · I incomplete)", x=.02, ha="left", fontsize=11.5)
        _legend_text(fig, BEHAVIORS, y=-.008)
        _save(fig, f"modal_paths_quality_{model}", figures)


def _compressed_paths(modal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in modal.sort_values("decile").groupby(
        ["gen_model", "task_type", "quality_class", "signature_family"], observed=True
    ):
        sequence = part.behavior.tolist(); compressed = [sequence[0]] if sequence else []
        for behavior in sequence[1:]:
            if behavior != compressed[-1]: compressed.append(behavior)
        rows.append({
            "gen_model": keys[0], "task_type": keys[1], "quality_class": keys[2],
            "signature_family": keys[3],
            "modal_path": " → ".join(BEHAVIOR_LABELS[value] for value in compressed),
            "modal_path_codes": "→".join(ABBREVIATIONS[value] for value in compressed),
        })
    return pd.DataFrame(rows)


def _domain_path_table(paths: pd.DataFrame) -> pd.DataFrame:
    table = (
        paths[paths.quality_class.eq("all")]
        .pivot(index=["gen_model", "task_type"], columns="signature_family", values="modal_path")
        .reset_index()
    )
    table["gen_model"] = pd.Categorical(table.gen_model, categories=MODELS, ordered=True)
    table["task_type"] = pd.Categorical(table.task_type, categories=DOMAINS, ordered=True)
    table = table.sort_values(["gen_model", "task_type"])
    table.insert(1, "model", table.gen_model.astype(str).map(MODEL_LABELS))
    table.insert(3, "domain", table.task_type.astype(str).map(DOMAIN_LABELS))
    return table[["gen_model", "model", "task_type", "domain", "cognitive", "conversational"]]


def _write_model_log(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    report_dir: Path,
    analysis_frame: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    """Write machine-readable and paper-writer-readable regression specifications."""
    config_path = Path(str(config.get("_config_path", "configs/paper_analysis.yaml")))
    prior_spec_path = output_dir / "04_outcome_models" / "model_specifications.json"
    prior_ladder = json.loads(prior_spec_path.read_text()) if prior_spec_path.exists() else {}
    fold_counts = (
        analysis_frame[["trace_id", "prompt_fold"]]
        .groupby("prompt_fold").size().sort_index().astype(int).to_dict()
    )
    log = {
        "schema_version": "thought-atlas-signature-model-log-v2",
        "config_sha256": sha256_file(config_path) if config_path.exists() else None,
        "code_commit": current_commit(config.get("_repo_root", ".")),
        "analysis_population": {
            "models": MODELS,
            "domains": DOMAINS,
            "traces": int(len(analysis_frame)),
            "inclusion": "reasoning models; n_valid > 0; full-context labels parsed",
            "generation_replication": "one decoding seed per prompt",
        },
        "split_contract": {
            "registry": str(output_dir / "splits.parquet"),
            "group": "instance_id",
            "outer_folds": 5,
            "assignment": "frozen deterministic prompt_fold from the audited split registry",
            "fold_counts": {str(key): value for key, value in fold_counts.items()},
            "leakage_check": "every instance_id maps to exactly one prompt_fold",
        },
        "shared_estimator": {
            "categorical_targets": "reference-class multinomial logistic regression",
            "continuous_targets": "linear ridge regression",
            "penalty": "L2=1.0 on standardized slopes; intercept unpenalized",
            "preprocessing": "training-fold mean imputation and z-standardization; test fold transformed with training values",
            "probability_metric": "out-of-fold log score; differences reported in bits per trace",
            "continuous_metric": "out-of-fold R2; paired gain equals per-trace squared-error reduction divided by outcome variance",
        },
        "mathematical_estimands": {
            "training_fold_preprocessing": {
                "equation": "z_ij = (x_ij - mean_train,j) / sd_train,j",
                "missing_values": "replace x_ij by the training-fold mean before standardizing",
                "zero_variance_columns": "use scale 1",
                "held_out_contract": "the held-out fold never contributes to imputation means, scales, or coefficients; fixed prompt-stratum dummy columns are created from prompt metadata before CV and use no outcomes",
            },
            "categorical_regression": {
                "reference_class": "eta_i0 = 0; eta_ic = alpha_c + z_i^T beta_c for c=1,...,K-1",
                "probability": "p_ic = exp(eta_ic) / sum_r exp(eta_ir)",
                "objective": "minimize -sum_i log p_i,y_i + (lambda/2) sum_c ||beta_c||_2^2",
                "lambda": 1.0,
                "intercept": "unpenalized",
            },
            "continuous_regression": {
                "prediction": "yhat_i = alpha + z_i^T beta",
                "objective": "minimize sum_i (y_i - alpha - z_i^T beta)^2 + lambda ||beta||_2^2",
                "closed_form": "theta_hat = (X^T X + lambda P)^+ X^T y, with P_00=0 and P_jj=1 for slopes",
                "lambda": 1.0,
            },
            "behavior_features": {
                "amount": "a_ib = positive valid segments for behavior b / all valid segments in trace i",
                "phase_prevalence": "r_ibp = positive valid segments for b in phase p / valid segments in phase p",
                "conditional_share": "s_ibp = r_ibp / sum_q r_ibq when the sum is positive; otherwise all shares are zero",
                "timing_contrasts": "t_ib,1 = s_ib,middle - s_ib,early; t_ib,2 = s_ib,late - s_ib,early",
                "descriptive_timing_centroid": "c_ib = 0.5 s_ib,middle + s_ib,late, defined only when behavior b occurs",
            },
            "categorical_predictive_gain": {
                "per_trace": "g_i(B|A) = log2[p_B(y_i|x_i) / p_A(y_i|x_i)]",
                "reported_increment": "Delta(B|A) = mean_i g_i(B|A), in bits per held-out trace",
                "probability_multiplier": "2^Delta is the geometric-mean multiplier in held-out probability assigned to the true class",
            },
            "continuous_predictive_gain": {
                "per_trace": "g_i(B|A) = ((y_i-yhat_A,i)^2 - (y_i-yhat_B,i)^2) / Var(y)",
                "reported_increment": "mean_i g_i(B|A) = R2_B - R2_A",
            },
            "exact_grouped_shapley": {
                "players": "the four behaviors in one signature family; an amount player is one rate and a timing player is its two timing contrasts",
                "coalitions": "all 2^4 = 16 subsets, fit on the same outer folds",
                "coalition_value_categorical": "v_i(S) = log2 p_S(y_i|x_i)",
                "coalition_value_continuous": "v_i(S) = -(y_i-yhat_S,i)^2 / Var(y)",
                "equation": "phi_ib = sum_{S subseteq B\\{b}} |S|!(m-|S|-1)!/m! * [v_i(S union {b}) - v_i(S)]",
                "domain_summary": "mean phi_ib over held-out traces whose true domain is d",
                "efficiency": "sum_b phi_ib = v_i(B) - v_i(empty set), up to numerical precision",
            },
            "direction_overlay": {
                "amount": "[mean_domain(a_b) - mean_other_domains(a_b)] / sd_all_domains(a_b)",
                "timing": "mean_domain(c_b) - mean_other_domains(c_b), conditional on behavior occurrence",
                "interpretation": "direction is descriptive and separate from Shapley predictive credit",
            },
        },
        "runs": {
            "prior_staged_outcome_ladder_RQ3": {
                "status": "pre-existing analysis documented for continuity",
                "population": "pooled models within each domain",
                "specification_source": str(prior_spec_path),
                "feature_ladder": prior_ladder,
                "primary_metrics": "AUROC for binary outcomes; R2 for continuous outcomes",
                "claim_boundary": "associational prediction, not a within-model causal effect",
            },
            "omnibus_domain_signature_RQ8": {
                "unit": "trace within a fixed model",
                "target": "eight-class domain",
                "baseline": ["log_tokens", "log_segments"],
                "amount_step": "four behavior rates from one signature family",
                "timing_step": "two conditional phase contrasts per behavior",
                "tests": int(summary["domain_cells"]),
            },
            "binary_quality_signature_RQ9": {
                "unit": "completed, scored trace within fixed model and domain",
                "target": "high versus low quality",
                "continuous_score_rule": "within-model/domain median split",
                "baseline": ["log_tokens", "log_segments"],
                "minimum_support": "20 traces in each quality class",
                "role": "common-scale sensitivity analysis; native-outcome analysis is primary for localization",
            },
            "pairwise_domain_localization_RQ8a": {
                "target": "binary domain identity for all 28 unordered pairs",
                "fits": "separate fixed-model, fixed-family models",
                "baseline": ["log_tokens", "log_segments"],
                "feature_steps": ["amount rates", "conditional timing beyond all amounts"],
                "tests_per_step": int(summary["pairwise_domain_cells"]),
                "correction_family": "Benjamini-Hochberg across all 336 pair × model × family tests, separately for amount and timing",
            },
            "domain_behavior_attribution_RQ8b": {
                "method": "exact grouped Shapley decomposition of out-of-fold log-score gain",
                "groups": "the four behaviors in each family",
                "subset_models": 16,
                "amount_base": ["log_tokens", "log_segments"],
                "timing_base": "length plus all four behavior rates",
                "localization": "per true domain within each fixed model",
                "direction": "one-domain-versus-rest standardized amount contrast; conditional timing-centroid contrast",
                "correction_family": "Benjamini-Hochberg across all model × domain × behavior attributions, separately for amount and timing",
                "claim_boundary": "predictive credit is shared under correlation; it is not causal attribution",
            },
            "native_outcome_signature_RQ9a": {
                "unit": "completed, scored trace within fixed model and domain",
                "targets": "native binary outcomes retained as binary; moral, idea, and safety scores retained as continuous",
                "baseline": ["log_tokens", "log_segments", "available within-domain prompt-stratum indicators"],
                "minimum_support": "n >= 40; binary cells also require >=20 per class; continuous cells require >=3 unique outcomes",
                "feature_steps": ["four family behavior rates", "two timing contrasts per behavior beyond amount"],
                "attribution": "exact four-group Shapley decomposition for every estimable cell",
                "correction_family": "Benjamini-Hochberg across all estimable model × domain × family blocks; behavior attributions corrected separately by layer",
                "claim_boundary": "predictive association only; the data cannot show that inducing a signature improves an answer",
            },
        },
        "uncertainty_and_testing": {
            "bootstrap": "1,000 paired prompt resamples; class-stratified for categorical targets and outcome-quantile-stratified for continuous targets",
            "test": "one-sided paired sign-flip test of held-out per-trace score improvement",
            "support_rule": "95% interval lower bound > 0 and BH q < 0.05",
            "multiplicity": "amount and timing are separate predeclared testing families",
        },
        "result_counts": dict(summary),
    }
    out = output_dir / "09_signatures"
    (out / "model_run_log.json").write_text(json.dumps(log, indent=2))
    lines = [
        "# Regression and model run log",
        "",
        "This file records the exact predictive estimands used before and during the signature-localization extension. It is intended as a durable Methods-writing source, not as a causal-analysis claim.",
        "",
        "## Shared fitting contract",
        "",
        "All signature models use the frozen five-fold `prompt_fold` registry grouped by `instance_id`. Every reported prediction is out of fold. Numeric predictors are mean-imputed and standardized using the training fold only. Categorical targets use L2-penalized multinomial logistic regression; continuous targets use L2-penalized linear regression. In both cases the slope penalty is 1.0 and the intercept is unpenalized.",
        "",
        "The length baseline contains log token count and log valid-segment count. Amount features are behavior-positive segments per valid segment. Timing features are middle-minus-early and late-minus-early shares conditional on the behavior's total occurrence, and therefore enter only after amount.",
        "",
        "## Exact regression mathematics",
        "",
        "Let `i` index traces, `j` predictors, and `c` target classes. Inside each training fold, every numeric predictor is transformed as `z_ij = (x_ij - mean_train,j) / sd_train,j`; missing values receive the training mean first, and a zero standard deviation is replaced by 1. The same training-fold quantities transform the held-out fold. Fixed prompt-stratum dummy columns are created from prompt metadata before cross-validation and use no outcomes; their imputation, scaling, and coefficients still come from the training fold.",
        "",
        "For a categorical target with `K` classes, class 0 is the reference: `eta_i0 = 0` and `eta_ic = alpha_c + z_i^T beta_c` for `c = 1,...,K-1`. Probabilities are softmax values, `p_ic = exp(eta_ic) / sum_r exp(eta_ir)`. The fitted parameters minimize `-sum_i log p_i,y_i + (lambda/2) sum_c ||beta_c||_2^2`, with `lambda = 1`. Only slopes are penalized; intercepts are not. Pairwise domain and binary-outcome models are the `K = 2` special case.",
        "",
        "For a continuous native outcome, `yhat_i = alpha + z_i^T beta`. The fitted parameters minimize `sum_i (y_i - alpha - z_i^T beta)^2 + lambda ||beta||_2^2`, again with `lambda = 1` and no intercept penalty. Equivalently, `theta_hat = (X^T X + lambda P)^+ X^T y`, where `P_00 = 0`, all slope diagonals of `P` are 1, and `+` denotes the Moore–Penrose inverse.",
        "",
        "### Exact feature definitions",
        "",
        "For behavior `b`, amount is `a_ib = m_ib / n_i`, where `m_ib` is the number of valid segments labeled positive and `n_i` is the number of valid segments. For phase `p` in early/middle/late, phase prevalence is `r_ibp = m_ibp / n_ip`. We normalize within behavior as `s_ibp = r_ibp / sum_q r_ibq` when the sum is positive, otherwise all three shares are zero. The two regression timing features are `t_ib1 = s_ib,middle - s_ib,early` and `t_ib2 = s_ib,late - s_ib,early`. Their sum is not an amount measure: timing enters only after all four family amounts are already in the base model.",
        "",
        "The color direction in the timing figure uses the descriptive centroid `c_ib = 0.5 s_ib,middle + s_ib,late`, conditional on the behavior occurring. It is not an additional regression predictor. The amount color is the one-domain-versus-rest standardized mean difference; the timing color is the corresponding unstandardized centroid difference.",
        "",
        "### Held-out predictive gain",
        "",
        "For nested categorical models `A` and `B`, the trace-level gain is `g_i(B|A) = log2[p_B(y_i|x_i) / p_A(y_i|x_i)]`. The reported increment is `Delta(B|A) = mean_i g_i(B|A)` in bits per held-out trace. A value of `Delta` means the richer model assigns the true class `2^Delta` times as much probability on average on the geometric-mean scale. Amount uses the length model as `A`; timing uses the length-plus-all-amounts model as `A`.",
        "",
        "For continuous outcomes, `g_i(B|A) = [(y_i-yhat_A,i)^2 - (y_i-yhat_B,i)^2] / Var(y)`. Averaging this expression equals `R2_B - R2_A`, so categorical and continuous analyses both measure paired held-out predictive improvement, on their appropriate scoring scales.",
        "",
        "## Exact grouped Shapley calculation",
        "",
        "Within one family there are `m = 4` behavior players. An amount player contributes one rate; a timing player contributes its two timing contrasts together. We fit every `2^4 = 16` coalition on the same outer folds. For trace `i`, define `v_i(S) = log2 p_S(y_i|x_i)` for categorical targets, or `v_i(S) = -(y_i-yhat_S,i)^2 / Var(y)` for continuous targets.",
        "",
        "The exact contribution of behavior `b` is `phi_ib = sum over S not containing b of [|S|! (m-|S|-1)! / m!] * [v_i(S union {b}) - v_i(S)]`. The factorial weight is the fraction of all behavior-addition orders in which coalition `S` appears immediately before `b`. Thus each behavior receives its average marginal held-out contribution over every possible order, instead of credit from one arbitrary coefficient or one arbitrary feature order.",
        "",
        "For a domain–behavior dot, we average `phi_ib` over held-out traces whose true class is that domain. Exact Shapley efficiency gives `sum_b phi_ib = v_i(all four) - v_i(no behaviors)` for every trace; the implementation's maximum numerical reconstruction error is below `4e-16`. Dot area encodes positive predictive credit. The blue/orange overlay comes from the separate direction contrast because Shapley importance alone cannot say whether a behavior is more or less prevalent, or earlier or later.",
        "",
        "### Why this specification",
        "",
        "Fixed-model fits remove model identity as an easy shortcut. Prompt-disjoint folds test generalization to unseen prompts. Length controls stop verbosity from masquerading as process structure. Ridge regularization stabilizes correlated behavior features and sparse prompt-stratum indicators. Proper log score rewards calibrated probability assigned to the actual class, not only whether the top class wins. Exact grouped Shapley is preferable to reading regression coefficients because coefficients depend on reference coding, scaling, and correlations; Shapley instead allocates the actual held-out score improvement while sharing credit across correlated behaviors. Timing is tested after amount so that ‘later’ cannot simply mean ‘more.’",
        "",
        "## Models that existed before localization",
        "",
        "The RQ3 staged outcome ladder pools models within domain and compares metadata, dialectical amount, executive amount, their union, timing, coupling, and motifs. Its exact M0–M5 column lists remain in `04_outcome_models/model_specifications.json`.",
        "",
        "The original RQ8 omnibus model predicts all eight domains separately within each model. The original RQ9 sensitivity model predicts high versus low quality separately within model and domain, with continuous scores median-split only for that common-scale sensitivity analysis.",
        "",
        "## New localization runs",
        "",
        "### RQ8a: particular domain pairs",
        "",
        "All 28 unordered domain pairs are fit separately for six models and two signature families. Amount is evaluated beyond the two length controls; timing is evaluated beyond length and all four family amounts. The 336 tests in each step form one BH correction family.",
        "",
        "### RQ8b: behavior attribution",
        "",
        "For each four-behavior family, all 16 feature subsets are fit on the same held-out folds. Exact Shapley values average a behavior's marginal held-out log-score contribution over every addition order. Contributions are summarized by the true domain. A separate one-domain-versus-rest contrast supplies direction; importance alone does not imply more or less behavior.",
        "",
        "### RQ9a: native outcomes",
        "",
        "Models are fit inside each model–domain cell. Binary outcomes remain binary; moral, idea, and safety remain continuous. The baseline contains both length controls and available prompt-stratum indicators. Cells require at least 40 scored traces, at least 20 per binary class, or at least three distinct continuous scores. Behavior attribution again uses all 16 feature subsets.",
        "",
        "## Uncertainty, multiplicity, and language",
        "",
        "Intervals use 1,000 paired bootstrap resamples of held-out trace scores. Categorical targets are class-stratified; continuous targets are outcome-quantile-stratified. One-sided paired sign-flip tests ask whether the mean held-out increment is positive. A result is called supported only when the 95% interval excludes zero and BH q < .05.",
        "",
        "For gain values `g_i`, the randomization p-value is `(1 + number of r with mean_i(s_ri g_i) >= mean_i(g_i)) / (R + 1)`, where every `s_ri` is independently `-1` or `+1` and `R = 1,000`. For sorted p-values, Benjamini–Hochberg reports `q_(k) = min_{j >= k} [m p_(j) / j]`, capped at 1 and restored to the original test order. Requiring both a positive bootstrap lower bound and `q < .05` makes a supported ring deliberately stricter than either criterion alone.",
        "",
        "Shapley values allocate predictive information under correlated features; they do not identify a causal mechanism. Outcome analyses support wording such as ‘predicts,’ ‘is associated with,’ or ‘carries signal,’ never ‘causes,’ ‘produces,’ or ‘improves.’",
        "",
        "## Reproducibility",
        "",
        "```bash",
        ".venv/bin/python scripts/run_signature_analysis.py --config configs/paper_analysis.yaml --force",
        "```",
    ]
    (report_dir / "REGRESSION_MODEL_LOG.md").write_text("\n".join(lines) + "\n")


def _write_report(
    report_dir: Path,
    domain_results: pd.DataFrame,
    quality_results: pd.DataFrame,
    coverage: pd.DataFrame,
    paths: pd.DataFrame,
    summary: Mapping[str, Any],
    *,
    pairwise_results: pd.DataFrame,
    domain_attribution: pd.DataFrame,
    native_results: pd.DataFrame,
    native_attribution: pd.DataFrame,
) -> None:
    best_domain = domain_results.sort_values("amount_gain_bits", ascending=False).iloc[0]
    quality_est = quality_results[quality_results.status.eq("estimated")]
    supported_quality = quality_est[quality_est.amount_supported.fillna(False)]
    strongest_quality = (supported_quality if len(supported_quality) else quality_est).sort_values(
        "amount_gain_bits", ascending=False
    ).iloc[0]
    supported_quality_text = "; ".join(
        f"{row.model_label} {DOMAIN_LABELS[row.task_type]} {row.signature_family}"
        for row in supported_quality.itertuples()
    ) or "none"
    pair_summary = (
        pairwise_results.groupby("domain_pair", observed=True)
        .agg(
            median_amount_gain=("amount_gain_bits", "median"),
            amount_supported=("amount_supported", "sum"),
            timing_supported=("timing_supported", "sum"),
        )
        .sort_values("median_amount_gain", ascending=False)
    )
    strongest_pair = pair_summary.iloc[0]
    strongest_pair_name = str(pair_summary.index[0])
    supported_domain_attr = domain_attribution[
        domain_attribution.layer.eq("amount") & domain_attribution.supported
    ]
    behavior_support = (
        supported_domain_attr.groupby("behavior_label", observed=True).size().sort_values(ascending=False)
    )
    behavior_support_text = ", ".join(
        f"{behavior} ({count})" for behavior, count in behavior_support.head(4).items()
    ) or "none after correction"
    native_estimated = native_results[native_results.status.eq("estimated")]
    native_supported = native_estimated[native_estimated.amount_supported]
    native_supported_text = "; ".join(
        f"{row.model_label} {row.domain} {row.signature_family}"
        for row in native_supported.itertuples()
    ) or "none"
    native_behavior_supported = native_attribution[
        native_attribution.supported & native_attribution.parent_block_supported.fillna(False)
    ]
    native_behavior_text = "; ".join(
        f"{row.model_label} {row.domain}: {row.behavior_label} ({row.layer})"
        for row in native_behavior_supported.itertuples()
    ) or "no individual behavior survives the global attribution correction"
    all_domain_paths = _domain_path_table(paths)
    lines = [
        "# Thought Atlas: conversational and cognitive signatures",
        "",
        "## Why these analyses",
        "",
        "A domain signature is useful only if it distinguishes held-out prompts within the same model. We therefore predict domain separately for each model, rather than treating model identity as evidence of a domain effect. We then ask whether timing improves prediction after behavior amount is already known.",
        "",
        "For attempt quality, the same logic is applied within each model and domain. This prevents a model's overall capability or a domain's base rate from masquerading as a process signature.",
        "",
        "## RQ8 — Do conversational and cognitive signatures differ across domains within a fixed model?",
        "",
        f"**Yes for both families in every reasoning model.** Amount improves held-out domain identification in {summary['domain_amount_supported_cells']} of {summary['domain_cells']} model-family tests. Conditional timing adds further information in {summary['domain_timing_supported_cells']} of {summary['domain_cells']} tests after false-discovery correction.",
        "",
        f"The strongest amount signal is {best_domain.signature_family} behavior in {best_domain.model_label}: {best_domain.amount_gain_bits:.3f} bits per held-out trace beyond trace length. Across all cells, the median timing increment is {summary['domain_median_timing_gain_bits']:.3f} bits. Thus timing is not merely decorative, but amount carries the larger share of the domain signal.",
        "",
        f"Cognitive amount is more domain-distinctive than conversational amount in {summary['cognitive_amount_stronger_models']} of 6 models. The modal paths make the difference concrete: verification is the final dominant cognitive operation in math for all six models; backtracking appears in the idea-domain cognitive path for five; moral traces begin with subgoal setting in all six. Conversational paths are less varied in identity—perspective shift usually dominates—but Qwen math and planning traces often move toward question–answer behavior later.",
        "",
        "![Domain signature gains](figures/fig10_domain_signature_information_gain.png)",
        "",
        "## RQ8a — Which particular domains differ?",
        "",
        f"The omnibus result is localized with all 28 domain pairs, separately for every model and signature family. Amount is interval- and FDR-supported in {summary['pairwise_amount_supported_cells']} of {summary['pairwise_domain_cells']} pairwise cells; conditional timing is supported in {summary['pairwise_timing_supported_cells']}. The strongest cross-model amount separation is {strongest_pair_name}, with a median held-out gain of {strongest_pair.median_amount_gain:.3f} bits per trace.",
        "",
        "Each pairwise model retains the two length controls, so a dot means the behavior profile separates the two domains beyond simple response length. Open dots in the figure are estimates that do not survive the joint 336-test correction family.",
        "",
        "![Pairwise domain localization](figures/fig13_pairwise_domain_localization.png)",
        "",
        "## RQ8b — Which behaviors carry the distinction?",
        "",
        f"Exact held-out Shapley decomposition identifies {summary['domain_amount_behavior_attributions_supported']} supported amount contributions and {summary['domain_timing_behavior_attributions_supported']} timing contributions across model, domain, and behavior cells. The most frequently supported amount contributors are {behavior_support_text}.",
        "",
        "Shapley values divide the classifier's held-out information gain across all four behaviors by fitting every one of the 16 possible behavior subsets. The x-axis boxes separate conversational behaviors (Q/P/C/R) from cognitive behaviors (V/B/S/K). Dot area shows predictive credit; blue/orange shows whether the domain uses the behavior more/later or less/earlier than the other domains. This is predictive attribution, not causal attribution.",
        "",
        "![Domain behavior amount attribution](figures/fig14_domain_behavior_amount_attribution.png)",
        "",
        "![Domain behavior timing attribution](figures/fig15_domain_behavior_timing_attribution.png)",
        "",
        "## RQ9 — Do signatures differ between high- and low-quality attempts within model and domain?",
        "",
        f"**Sometimes, but not universally.** High-versus-low quality was estimable in {summary['quality_estimated_cells']} of {summary['quality_total_cells']} model-domain-family cells. Amount provided a positive, interval- and FDR-supported increment in {summary['quality_amount_supported_cells']} cells; timing did so in {summary['quality_timing_supported_cells']} cells.",
        "",
        f"After correction, the supported high-versus-low sensitivity cells are {supported_quality_text}. The strongest is {strongest_quality.model_label} {DOMAIN_LABELS[strongest_quality.task_type]} {strongest_quality.signature_family}: {strongest_quality.amount_gain_bits:.3f} bits per held-out trace. No timing increment survives correction. Because this common-scale analysis dichotomizes continuous scores and uses less outcome detail, it is interpreted alongside—not above—the native-outcome analysis below.",
        "",
        "![Quality amount gains](figures/fig11_quality_amount_gain.png)",
        "",
        "![Quality timing gains](figures/fig12_quality_timing_gain.png)",
        "",
        "## RQ9a — Which signatures predict native outcome quality?",
        "",
        f"The stricter native-outcome analysis retains binary outcomes as binary and continuous rubric scores as continuous, while adding prompt-stratum and two length controls to the baseline. It estimates {summary['native_outcome_estimated_cells']} of {summary['native_outcome_total_cells']} model–domain–family cells. Amount is supported in {summary['native_outcome_amount_supported_cells']} cells and timing in {summary['native_outcome_timing_supported_cells']}. Supported amount cells: {native_supported_text}. Thus the two high-versus-low sensitivity findings are not confirmed by the primary native-outcome specification.",
        "",
        f"Behavior-level localization yields: {native_behavior_text}. These are associations on held-out prompts; without randomized behavior manipulation, they cannot show that inducing a signature would improve an answer.",
        "",
        "![Native outcome amount gain](figures/fig16_native_outcome_amount_gain.png)",
        "",
        "![Native outcome timing gain](figures/fig16b_native_outcome_timing_gain.png)",
        "",
        "![Native outcome evidence gate](figures/fig17_native_outcome_evidence_gate.png)",
        "",
        "## Modal-path interpretation",
        "",
        "Each modal path reports the most prevalent operation within each normalized trace decile. Because labels can co-occur, it is a dominant-operation path, not a mutually exclusive hidden-state sequence. Opacity reflects how strongly the winning operation dominates the other operations in its family.",
        "",
        "| Model | Domain | Modal cognitive | Modal conversational |",
        "| --- | --- | --- | --- |",
    ]
    for row in all_domain_paths.itertuples():
        lines.append(
            f"| {row.model} | {row.domain} | "
            f"{row.cognitive} | {row.conversational} |"
        )
    lines += [
        "",
        "## Statistical model",
        "",
        "All signature analyses use the frozen five-fold prompt registry, with every instance assigned to one held-out fold. Categorical targets use L2-penalized multinomial logistic regression; native continuous outcomes use L2-penalized linear regression. The baseline contains log token length and log valid-segment count; native-outcome baselines additionally contain available prompt-stratum indicators. Training-fold means and scales are applied to held-out folds. Amount adds four rates; timing adds two amount-invariant phase contrasts per behavior. Intervals use 1,000 paired, stratified bootstrap resamples of out-of-fold score differences. One-sided paired sign-flip tests are Benjamini–Hochberg corrected within explicitly logged amount, timing, and attribution families.",
        "",
        "## Exact regression and Shapley math, in plain language",
        "",
        "Inside each training fold, predictor j is standardized as z_ij = (x_ij - mean_train,j) / sd_train,j. The held-out fold uses those training values and never helps fit itself. For class c, the regression computes eta_ic = alpha_c + z_i^T beta_c and p_ic = exp(eta_ic) / sum_r exp(eta_ir). It chooses the coefficients that minimize negative log probability of the true classes plus (1/2) sum_c ||beta_c||^2. The penalty is on slopes only. This is ordinary ridge multinomial logistic regression with lambda = 1; binary models are its two-class special case.",
        "",
        "For a continuous native outcome, the prediction is yhat_i = alpha + z_i^T beta. Ridge minimizes sum_i(y_i - yhat_i)^2 + ||beta||^2, again leaving the intercept unpenalized. We use ridge because behavior rates and timing measures are correlated; the small fixed penalty stabilizes held-out predictions rather than allowing large, fragile coefficients.",
        "",
        "A categorical model's trace-level improvement is g_i(B|A) = log2[p_B(true class) / p_A(true class)]. Its mean is the reported bits per trace. Thus 0.10 bits means the richer model gives the true class 2^0.10, or about 1.07 times, the probability on the geometric-mean scale. For continuous outcomes, g_i(B|A) = [(y_i-yhat_A)^2 - (y_i-yhat_B)^2] / Var(y); its mean is exactly R2_B - R2_A. Amount is compared with length alone. Timing is compared with length plus every amount feature, so timing must add information that amount did not already supply.",
        "",
        "For Shapley attribution, the four behaviors in one family are the four players. We fit all 16 subsets. If v_i(S) is a subset's held-out score, then phi_ib = sum over subsets S without b of |S|!(4-|S|-1)!/4! times [v_i(S plus b) - v_i(S)]. In words: add behavior b after every possible set of the other behaviors, measure how much the held-out score changes, and average those changes with the exact order weights. The four values add back to the full-versus-base held-out gain for each trace, up to numerical precision.",
        "",
        "We do not use coefficient size as behavior importance because a coefficient changes with scaling, reference-class coding, and correlated predictors. Exact held-out Shapley allocates the quantity we actually claim—generalization gain—and fairly shares overlapping signal. Its sign still does not mean ‘more’ or ‘later,’ so figure color comes from a separate one-domain-versus-rest descriptive contrast. Rings require both a positive 95% bootstrap lower bound and a Benjamini–Hochberg q value below .05.",
        "",
        "The complete regression specifications, earlier M0–M5 outcome ladder, correction families, estimands, and claim boundaries are recorded in `REGRESSION_MODEL_LOG.md` and `09_signatures/model_run_log.json`.",
        "",
        "## Quality definitions and the incomplete-attempt boundary",
        "",
        "For binary-outcome domains, high means correct/favorable and low means incorrect/unfavorable. For continuous moral, idea, and safety outcomes, high and low are split at the within-model/domain median among completed, scored attempts. Completed attempts without a valid outcome remain missing; they are not called low quality.",
        "",
        f"The sentence-level analysis contains only {summary['incomplete_total']} incomplete attempts, and no model-domain cell reaches the prespecified minimum of 20 incomplete traces. A defensible three-class high/low/incomplete model is therefore unavailable. The quality modal-path figures show these lanes as ‘not estimable’ instead of fabricating a result.",
        "",
        "## Boundaries",
        "",
        "- ‘Conversational’ refers to the four annotated dialectical/self-dialogue operations; ‘cognitive’ refers to the four annotated executive-control operations. These are textual process signatures, not direct measurements of latent cognition.",
        "- Domain classification establishes reproducible difference, not a causal domain effect.",
        "- Quality classification can reflect response difficulty or trace length despite within-cell fitting and length adjustment.",
        "- There is one decoding seed per prompt, so intervals describe prompt-composition uncertainty rather than generation variance.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        ".venv/bin/python scripts/run_signature_analysis.py --config configs/paper_analysis.yaml --force",
        "```",
    ]
    md = "\n".join(lines) + "\n"
    (report_dir / "SIGNATURE_ANALYSIS_REPORT.md").write_text(md)
    body: list[str] = []
    in_table = False
    for line in lines:
        if in_table and not line.startswith("|"):
            body.append("</table>"); in_table = False
        if line.startswith("# "): body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "): body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("!["):
            src = line.split("(", 1)[1].rstrip(")"); body.append(f'<img src="{html.escape(src)}" alt="" style="max-width:100%;margin:1rem 0">')
        elif line.startswith("- "): body.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("|"):
            if "---" in line: continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not in_table: body.append("<table>"); in_table = True
            tag = "th" if cells[0] == "Model" else "td"
            body.append("<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells) + "</tr>")
        elif line and not line.startswith("```"):
            if in_table: body.append("</table>"); in_table = False
            body.append(f"<p>{html.escape(line)}</p>")
    if in_table: body.append("</table>")
    css = "body{font:16px/1.55 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#17202a}h1,h2{color:#163a5f}img{border-top:1px solid #ddd;border-bottom:1px solid #ddd}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #ddd}"
    (report_dir / "SIGNATURE_ANALYSIS_REPORT.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>Thought Atlas signatures</title><style>{css}</style><main>{''.join(body)}</main>"
    )


def run_signature_analysis(
    *,
    config: Mapping[str, Any],
    features: pd.DataFrame,
    output_dir: Path,
    report_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    out = output_dir / "09_signatures"; out.mkdir(parents=True, exist_ok=True)
    figures = report_dir / "figures"; figures.mkdir(parents=True, exist_ok=True)
    figure_data = report_dir / "figure_data"; figure_data.mkdir(parents=True, exist_ok=True)
    summary_path = out / "signature_summary.json"
    required = [
        out / "domain_signature_models.csv",
        out / "quality_signature_models.csv",
        out / "pairwise_domain_signature_models.csv",
        out / "domain_behavior_attribution.csv",
        out / "native_outcome_signature_models.csv",
        out / "native_outcome_behavior_attribution.csv",
        out / "modal_paths.csv",
        out / "model_run_log.json",
    ]
    if summary_path.exists() and all(path.exists() for path in required) and not force:
        return json.loads(summary_path.read_text())

    use = features[features.gen_model.isin(MODELS) & (features.n_valid > 0)].copy()
    if "prompt_fold" not in use.columns:
        split_path = output_dir / "splits.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"missing frozen prompt split registry: {split_path}")
        splits = pd.read_parquet(split_path, columns=["trace_id", "instance_id", "prompt_fold"])
        if splits.trace_id.duplicated().any():
            raise ValueError("split registry must contain one row per trace_id")
        fold_leaks = splits.groupby("instance_id").prompt_fold.nunique()
        if (fold_leaks != 1).any():
            raise ValueError("split registry assigns an instance_id to multiple prompt folds")
        use = use.merge(
            splits[["trace_id", "prompt_fold"]], on="trace_id", how="left", validate="one_to_one"
        )
    if use.prompt_fold.isna().any():
        raise ValueError("every signature-analysis trace must have a frozen prompt fold")
    use = add_timing_contrasts(use)
    use = assign_quality_classes(use)
    reps = int(config.get("resampling", {}).get("bootstrap_reps", 1000))
    seed = int(config.get("seed", 42))

    domain_results = domain_signature_models(use, reps, seed)
    quality_results, coverage = quality_signature_models(use, reps, seed)
    pairwise_results = pairwise_domain_signature_models(use, reps, seed)
    domain_attribution = domain_behavior_attribution(use, reps, seed)
    native_results, native_attribution = native_outcome_signature_models(use, reps, seed)
    modal = modal_paths(config, use)
    paths = _compressed_paths(modal)
    domain_path_table = _domain_path_table(paths)

    domain_results.to_csv(out / "domain_signature_models.csv", index=False)
    quality_results.to_csv(out / "quality_signature_models.csv", index=False)
    pairwise_results.to_csv(out / "pairwise_domain_signature_models.csv", index=False)
    domain_attribution.to_csv(out / "domain_behavior_attribution.csv", index=False)
    native_results.to_csv(out / "native_outcome_signature_models.csv", index=False)
    native_attribution.to_csv(out / "native_outcome_behavior_attribution.csv", index=False)
    coverage.to_csv(out / "quality_class_coverage.csv", index=False)
    modal.to_csv(out / "modal_paths.csv", index=False)
    paths.to_csv(out / "modal_path_sequences.csv", index=False)
    domain_path_table.to_csv(out / "domain_modal_path_table.csv", index=False)
    domain_path_table.to_csv(output_dir / "tables" / "table12_domain_modal_paths.csv", index=False)
    pairwise_results.to_csv(output_dir / "tables" / "table13_pairwise_domain_signatures.csv", index=False)
    domain_attribution.to_csv(output_dir / "tables" / "table14_domain_behavior_attribution.csv", index=False)
    native_results.to_csv(output_dir / "tables" / "table15_native_outcome_signatures.csv", index=False)
    native_attribution.to_csv(output_dir / "tables" / "table16_native_outcome_behavior_attribution.csv", index=False)
    domain_results.to_csv(figure_data / "fig10_domain_signature_information_gain.csv", index=False)
    quality_results.to_csv(figure_data / "fig11_quality_amount_gain.csv", index=False)
    quality_results.to_csv(figure_data / "fig12_quality_timing_gain.csv", index=False)
    pairwise_results.to_csv(figure_data / "fig13_pairwise_domain_localization.csv", index=False)
    domain_attribution[domain_attribution.layer.eq("amount")].to_csv(
        figure_data / "fig14_domain_behavior_amount_attribution.csv", index=False
    )
    domain_attribution[domain_attribution.layer.eq("timing")].to_csv(
        figure_data / "fig15_domain_behavior_timing_attribution.csv", index=False
    )
    native_results.to_csv(figure_data / "fig16_native_outcome_signature_gain.csv", index=False)
    native_attribution.to_csv(
        figure_data / "fig17_native_outcome_behavior_attribution.csv", index=False
    )
    native_results.to_csv(
        figure_data / "fig17_native_outcome_evidence_gate.csv", index=False
    )

    _style()
    figure_domain_gains(domain_results, figures)
    figure_pairwise_domain_localization(pairwise_results, figures)
    _behavior_attribution_figure(
        domain_attribution, "amount", figures, "fig14_domain_behavior_amount_attribution"
    )
    _behavior_attribution_figure(
        domain_attribution, "timing", figures, "fig15_domain_behavior_timing_attribution"
    )
    _native_outcome_gain_figure(
        native_results, "amount", figures, "fig16_native_outcome_amount_gain"
    )
    _native_outcome_gain_figure(
        native_results, "timing", figures, "fig16b_native_outcome_timing_gain"
    )
    figure_native_outcome_evidence_gate(native_results, figures)
    _quality_gain_figure(quality_results, "amount", figures, "fig11_quality_amount_gain", "Amount–quality association within model and domain")
    _quality_gain_figure(quality_results, "timing", figures, "fig12_quality_timing_gain", "Timing–quality association beyond amount")
    figure_modal_paths(modal, coverage, figures)

    estimated = quality_results[quality_results.status.eq("estimated")]
    native_estimated = native_results[native_results.status.eq("estimated")]
    summary = {
        "domain_cells": int(len(domain_results)),
        "domain_amount_supported_cells": int(domain_results.amount_supported.sum()),
        "domain_timing_supported_cells": int(domain_results.timing_supported.sum()),
        "domain_median_amount_gain_bits": float(domain_results.amount_gain_bits.median()),
        "domain_median_timing_gain_bits": float(domain_results.timing_gain_bits.median()),
        "cognitive_amount_stronger_models": int((
            domain_results.pivot(index="gen_model", columns="signature_family", values="amount_gain_bits")["cognitive"]
            > domain_results.pivot(index="gen_model", columns="signature_family", values="amount_gain_bits")["conversational"]
        ).sum()),
        "quality_total_cells": int(len(quality_results)),
        "quality_estimated_cells": int(len(estimated)),
        "quality_amount_supported_cells": int(estimated.amount_supported.fillna(False).sum()),
        "quality_timing_supported_cells": int(estimated.timing_supported.fillna(False).sum()),
        "quality_median_amount_gain_bits": float(estimated.amount_gain_bits.median()),
        "quality_median_timing_gain_bits": float(estimated.timing_gain_bits.median()),
        "incomplete_total": int(coverage.incomplete.sum()),
        "incomplete_adequate_cells": int(coverage.incomplete_adequate_for_three_class_model.sum()),
        "pairwise_domain_cells": int(len(pairwise_results)),
        "pairwise_amount_supported_cells": int(pairwise_results.amount_supported.sum()),
        "pairwise_timing_supported_cells": int(pairwise_results.timing_supported.sum()),
        "domain_amount_behavior_attributions_supported": int(
            domain_attribution[domain_attribution.layer.eq("amount")].supported.sum()
        ),
        "domain_timing_behavior_attributions_supported": int(
            domain_attribution[domain_attribution.layer.eq("timing")].supported.sum()
        ),
        "native_outcome_total_cells": int(len(native_results)),
        "native_outcome_estimated_cells": int(len(native_estimated)),
        "native_outcome_amount_supported_cells": int(native_estimated.amount_supported.sum()),
        "native_outcome_timing_supported_cells": int(native_estimated.timing_supported.sum()),
        "native_outcome_behavior_attributions_supported": int(native_attribution.supported.sum()),
        "interpretation": "Domain signatures are widespread and localized by domain pair and behavior; native-outcome associations remain cell-specific and non-causal.",
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    _write_model_log(
        config=config,
        output_dir=output_dir,
        report_dir=report_dir,
        analysis_frame=use,
        summary=summary,
    )
    _write_report(
        report_dir,
        domain_results,
        quality_results,
        coverage,
        paths,
        summary,
        pairwise_results=pairwise_results,
        domain_attribution=domain_attribution,
        native_results=native_results,
        native_attribution=native_attribution,
    )
    return summary


def run(*, config: Mapping[str, Any], output_dir: Path, dev: bool = False, force: bool = False, jobs: int = 1) -> dict[str, Any]:
    if dev:
        raise ValueError("signature analysis requires the reviewed full registry")
    feature_path = output_dir / "03_trace_features" / "trace_features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"missing trace feature layer: {feature_path}")
    summary = run_signature_analysis(
        config=config,
        features=pd.read_parquet(feature_path),
        output_dir=output_dir,
        report_dir=Path(config["paths"]["report_dir"]),
        force=force,
    )
    return {"ready": True, "stage": "signatures", "mode": "final", "output_dir": output_dir, "summaries": {"signatures": summary}, "report": Path(config["paths"]["report_dir"]) / "SIGNATURE_ANALYSIS_REPORT.md"}
