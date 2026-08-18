#!/usr/bin/env python3
"""Build the Thought Atlas paper-results package from the checked-in v2 release.

The script intentionally uses only the CPU analysis dependencies already declared
by the repository.  It produces trace-level features, statistical result tables,
and vector/raster figures for the research questions proposed in the paper audit.

Interpretive guardrails:
* The Llama anchor is answer text while the other conditions are think text.  It is
  shown in coverage tables but excluded from inferential dual-system analyses.
* Track-B coverage is selected by availability of a judgeable analysis section.
  Completion/coverage is therefore reported explicitly.
* Full-context and isolated labels are paired measurements by the same judge, not
  independent ground truth.  We call their difference context sensitivity.
* Outcome/timing results are observational associations, never causal effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.errors import PerformanceWarning
from scipy.optimize import minimize
from scipy.special import expit
from statsmodels.stats.anova import anova_lm


BEHAVIORS = [
    "Question_and_Answering",
    "Perspective_Shift",
    "Conflict_of_Perspectives",
    "Reconciliation",
    "verification",
    "backtracking",
    "subgoal",
    "backward_chaining",
]
DIALECTICAL = BEHAVIORS[:4]
CONTROL = BEHAVIORS[4:]
REASONING_MODELS = [
    "gemma4_e4b",
    "gemma4_31b",
    "qwen35_4b",
    "qwen35_9b",
    "qwen35_27b",
    "reasoner",
]
DOMAINS = ["math", "code", "gpqa", "planning", "moral", "idea", "safety", "security"]
MODEL_LABELS = {
    "anchor": "Llama 3.1 8B\n(answer anchor)",
    "reasoner": "DeepSeek-R1\nDistill 8B",
    "gemma4_e4b": "Gemma 4 E4B",
    "gemma4_31b": "Gemma 4 31B",
    "qwen35_4b": "Qwen3.5 4B",
    "qwen35_9b": "Qwen3.5 9B",
    "qwen35_27b": "Qwen3.5 27B",
}
MODEL_ORDER = ["anchor", "reasoner", "gemma4_e4b", "gemma4_31b", "qwen35_4b", "qwen35_9b", "qwen35_27b"]
BEHAVIOR_LABELS = {
    "Question_and_Answering": "Question → answer",
    "Perspective_Shift": "Perspective shift",
    "Conflict_of_Perspectives": "Perspective conflict",
    "Reconciliation": "Reconciliation",
    "verification": "Verification",
    "backtracking": "Backtracking",
    "subgoal": "Subgoal setting",
    "backward_chaining": "Backward chaining",
}
MOTIFS = {
    "question_to_subgoal": ("Question_and_Answering", "subgoal"),
    "conflict_to_verification": ("Conflict_of_Perspectives", "verification"),
    "reconciliation_to_verification": ("Reconciliation", "verification"),
    "verification_to_backtracking": ("verification", "backtracking"),
    "backtracking_to_subgoal": ("backtracking", "subgoal"),
    "perspective_to_reconciliation": ("Perspective_Shift", "Reconciliation"),
}
MOTIF_LABELS = {
    "question_to_subgoal": "Question → subgoal",
    "conflict_to_verification": "Conflict → verification",
    "reconciliation_to_verification": "Reconciliation → verification",
    "verification_to_backtracking": "Verification → backtracking",
    "backtracking_to_subgoal": "Backtracking → subgoal",
    "perspective_to_reconciliation": "Perspective shift → reconciliation",
}


@dataclass
class SegmentData:
    trace_ids: np.ndarray
    codes: np.ndarray
    seg_idx: np.ndarray
    norm_pos: np.ndarray
    n_segments: np.ndarray
    valid: np.ndarray
    truncated: np.ndarray
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper_results"))
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--splits", type=int, default=200, help="Repeated split-halves for adaptive differentiation")
    parser.add_argument("--bootstrap", type=int, default=500, help="Prompt-cluster bootstrap replicates")
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def trace_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in (data_dir / "traces").glob("traces_*.parquet") if ".shard" not in p.name)


def load_trace_metadata(data_dir: Path) -> pd.DataFrame:
    cols = [
        "trace_id",
        "gen_model",
        "instance_id",
        "task_type",
        "completed",
        "finish_reason",
        "n_new_tokens",
        "thinking_style",
        "parse_status",
    ]
    parts = [pq.read_table(p, columns=cols).to_pandas() for p in trace_files(data_dir)]
    traces = pd.concat(parts, ignore_index=True)
    if traces["trace_id"].duplicated().any():
        raise ValueError("Canonical trace files contain duplicate trace IDs")
    traces["instance_id"] = traces["instance_id"].astype(str)
    return traces


def load_segments(path: Path) -> SegmentData:
    cols = [
        "trace_id",
        "seg_idx",
        "n_segments",
        "norm_pos",
        "kim_parsed",
        "gandhi_parsed",
        "judge_context_truncated",
        *BEHAVIORS,
    ]
    frame = pq.read_table(path, columns=cols).to_pandas()
    codes, uniques = pd.factorize(frame.pop("trace_id"), sort=False)
    values = frame[BEHAVIORS].to_numpy(dtype=np.int8, copy=True)
    valid = (frame["kim_parsed"].fillna(False) & frame["gandhi_parsed"].fillna(False)).to_numpy()
    out = SegmentData(
        trace_ids=np.asarray(uniques.astype(str)),
        codes=codes.astype(np.int32, copy=False),
        seg_idx=frame["seg_idx"].to_numpy(dtype=np.int32, copy=False),
        norm_pos=frame["norm_pos"].to_numpy(dtype=np.float32, copy=False),
        n_segments=frame["n_segments"].to_numpy(dtype=np.int32, copy=False),
        valid=valid,
        truncated=frame["judge_context_truncated"].fillna(False).to_numpy(dtype=bool),
        values=values,
    )
    del frame
    return out


def assert_paired(full: SegmentData, isolated: SegmentData) -> None:
    checks = [
        (len(full.codes) == len(isolated.codes), "row count"),
        (np.array_equal(full.trace_ids, isolated.trace_ids), "trace ID factorization"),
        (np.array_equal(full.codes, isolated.codes), "trace row order"),
        (np.array_equal(full.seg_idx, isolated.seg_idx), "segment row order"),
    ]
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise ValueError(f"Full/isolated Track-B tables are not paired on: {failed}")


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full(np.broadcast_shapes(np.shape(num), np.shape(den)), np.nan), where=np.asarray(den) != 0)


def bincount_sum(codes: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(codes, weights=weights, minlength=n).astype(float)


def build_trace_features(seg: SegmentData, valid_mask: np.ndarray, bins: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate labels into amount, timing, coupling, and transition features."""
    n_traces = len(seg.trace_ids)
    code = seg.codes
    valid = valid_mask.astype(bool)
    val = seg.values
    pos = seg.norm_pos
    n_valid = bincount_sum(code, valid.astype(float), n_traces)
    features = pd.DataFrame({"trace_id": seg.trace_ids, "n_valid": n_valid})

    behavior_counts: dict[str, np.ndarray] = {}
    for j, behavior in enumerate(BEHAVIORS):
        weights = val[:, j] * valid
        counts = bincount_sum(code, weights, n_traces)
        behavior_counts[behavior] = counts
        features[f"{behavior}__count"] = counts
        features[f"{behavior}__rate"] = safe_div(counts, n_valid)
        weighted_pos = bincount_sum(code, weights * pos, n_traces)
        features[f"{behavior}__centroid"] = safe_div(weighted_pos, counts)

    d_labels = val[:, :4].sum(axis=1).astype(float) * valid
    e_labels = val[:, 4:].sum(axis=1).astype(float) * valid
    d_any = (val[:, :4].any(axis=1) & valid).astype(float)
    e_any = (val[:, 4:].any(axis=1) & valid).astype(float)
    both = d_any * e_any
    d_count = bincount_sum(code, d_labels, n_traces)
    e_count = bincount_sum(code, e_labels, n_traces)
    d_present = bincount_sum(code, d_any, n_traces)
    e_present = bincount_sum(code, e_any, n_traces)
    both_count = bincount_sum(code, both, n_traces)
    features["dialectical_amount"] = safe_div(d_count, n_valid)
    features["control_amount"] = safe_div(e_count, n_valid)
    features["dialectical_presence"] = safe_div(d_present, n_valid)
    features["control_presence"] = safe_div(e_present, n_valid)
    features["coupling_presence"] = safe_div(both_count, n_valid)
    features["coupling_excess"] = features["coupling_presence"] - features["dialectical_presence"] * features["control_presence"]
    features["dialectical_centroid"] = safe_div(bincount_sum(code, d_labels * pos, n_traces), d_count)
    features["control_centroid"] = safe_div(bincount_sum(code, e_labels * pos, n_traces), e_count)

    thirds = np.minimum((pos * 3).astype(np.int8), 2)
    time_bins = np.minimum((pos * bins).astype(np.int16), bins - 1)
    for name, weights in [("dialectical", d_labels), ("control", e_labels)]:
        for third, label in enumerate(["early", "middle", "late"]):
            mask = valid & (thirds == third)
            denom = bincount_sum(code, mask.astype(float), n_traces)
            numer = bincount_sum(code, weights * mask, n_traces)
            features[f"{name}_{label}"] = safe_div(numer, denom)

    for j, behavior in enumerate(BEHAVIORS):
        weights = val[:, j] * valid
        for third, label in enumerate(["early", "middle", "late"]):
            mask = valid & (thirds == third)
            denom = bincount_sum(code, mask.astype(float), n_traces)
            numer = bincount_sum(code, weights * mask, n_traces)
            features[f"{behavior}__{label}"] = safe_div(numer, denom)

    # Equal-trace trajectory features. Missing bins remain NaN; conditional shape
    # normalizes each trace's binned label mass separately for the two families.
    group_code = code.astype(np.int64) * bins + time_bins.astype(np.int64)
    bin_den = np.bincount(group_code[valid], minlength=n_traces * bins).reshape(n_traces, bins).astype(float)
    for prefix, weights in [("D", d_labels), ("E", e_labels)]:
        numer = np.bincount(group_code, weights=weights, minlength=n_traces * bins).reshape(n_traces, bins)
        rates = safe_div(numer, bin_den)
        shape = safe_div(numer, numer.sum(axis=1, keepdims=True))
        for b in range(bins):
            features[f"{prefix}_rate_bin{b:02d}"] = rates[:, b]
            features[f"{prefix}_shape_bin{b:02d}"] = shape[:, b]

    # Adjacent-segment transitions. Only genuinely adjacent, jointly valid rows
    # within a trace count as opportunities.
    adjacent = np.zeros(len(code), dtype=bool)
    adjacent[:-1] = (code[:-1] == code[1:]) & (seg.seg_idx[1:] == seg.seg_idx[:-1] + 1) & valid[:-1] & valid[1:]
    opportunities = bincount_sum(code, adjacent.astype(float), n_traces)
    features["transition_opportunities"] = opportunities
    next_d = np.zeros_like(d_any)
    next_e = np.zeros_like(e_any)
    next_d[:-1] = d_any[1:]
    next_e[:-1] = e_any[1:]
    general_transitions = {
        "D_to_D": d_any * next_d,
        "D_to_E": d_any * next_e,
        "E_to_D": e_any * next_d,
        "E_to_E": e_any * next_e,
    }
    for name, indicator in general_transitions.items():
        count = bincount_sum(code, indicator * adjacent, n_traces)
        features[f"{name}__count"] = count
        features[f"{name}__rate"] = safe_div(count, opportunities)

    motif_rows = []
    behavior_index = {b: i for i, b in enumerate(BEHAVIORS)}
    # Per-trace/per-third counts for a position-aware analytic shuffle null.
    trace_third = code.astype(np.int64) * 3 + thirds.astype(np.int64)
    n_by_third = np.bincount(trace_third[valid], minlength=n_traces * 3).reshape(n_traces, 3).astype(float)
    same_third_adjacent = adjacent.copy()
    same_third_adjacent[:-1] &= thirds[:-1] == thirds[1:]
    for name, (source, dest) in MOTIFS.items():
        src = val[:, behavior_index[source]].astype(float)
        dst = val[:, behavior_index[dest]].astype(float)
        next_dst = np.zeros_like(dst)
        next_dst[:-1] = dst[1:]
        observed_all = bincount_sum(code, src * next_dst * adjacent, n_traces)
        observed_local = bincount_sum(code, src * next_dst * same_third_adjacent, n_traces)
        src_third = np.bincount(trace_third, weights=src * valid, minlength=n_traces * 3).reshape(n_traces, 3)
        dst_third = np.bincount(trace_third, weights=dst * valid, minlength=n_traces * 3).reshape(n_traces, 3)
        expected_local = np.nansum(
            np.maximum(n_by_third - 1, 0) * safe_div(src_third, n_by_third) * safe_div(dst_third, n_by_third),
            axis=1,
        )
        features[f"{name}__count"] = observed_all
        features[f"{name}__rate"] = safe_div(observed_all, opportunities)
        for trace_idx in range(n_traces):
            motif_rows.append(
                {
                    "trace_id": seg.trace_ids[trace_idx],
                    "motif": name,
                    "observed": observed_local[trace_idx],
                    "expected": expected_local[trace_idx],
                }
            )
    motifs = pd.DataFrame(motif_rows)
    return features, motifs


def load_endpoints(data_dir: Path, traces: pd.DataFrame) -> pd.DataFrame:
    success = pq.read_table(data_dir / "perf" / "success_grades.parquet", columns=["trace_id", "task_type", "success", "gradeable"]).to_pandas()
    code = pq.read_table(data_dir / "perf" / "code_grades.parquet", columns=["trace_id", "success", "gradeable"]).to_pandas()
    code = code.rename(columns={"success": "code_success", "gradeable": "code_gradeable"})
    quality = pq.read_table(
        data_dir / "judge" / "quality__google_gemma-4-31B-it.parquet",
        columns=["trace_id", "task_type", "quality_score", "parsed", "high_harmful_compliance", "safety_harm_score"],
    ).to_pandas()
    out = traces[["trace_id", "gen_model", "task_type", "instance_id"]].merge(
        success[["trace_id", "success", "gradeable"]], on="trace_id", how="left"
    )
    out = out.merge(code, on="trace_id", how="left").merge(
        quality.drop(columns=["task_type"]), on="trace_id", how="left"
    )
    out["favorable_endpoint"] = np.nan
    objective = out["task_type"].isin(["math", "gpqa", "planning", "security"])
    out.loc[objective & out["gradeable"].fillna(False), "favorable_endpoint"] = out.loc[
        objective & out["gradeable"].fillna(False), "success"
    ]
    code_mask = (out["task_type"] == "code") & out["code_gradeable"].fillna(False)
    out.loc[code_mask, "favorable_endpoint"] = out.loc[code_mask, "code_success"]
    for domain in ["moral", "idea"]:
        mask = (out["task_type"] == domain) & out["quality_score"].notna()
        med = out.loc[mask].groupby("gen_model")["quality_score"].transform("median")
        out.loc[mask, "favorable_endpoint"] = (out.loc[mask, "quality_score"].to_numpy() >= med.to_numpy()).astype(int)
    safety = (out["task_type"] == "safety") & out["high_harmful_compliance"].notna()
    out.loc[safety, "favorable_endpoint"] = 1 - out.loc[safety, "high_harmful_compliance"].astype(int)
    endpoint_kind = {
        "math": "exact/symbolic correctness",
        "code": "all executable tests passed",
        "gpqa": "multiple-choice correctness",
        "planning": "task correctness",
        "moral": "at or above model-specific median quality",
        "idea": "at or above model-specific median quality",
        "safety": "no high harmful compliance",
        "security": "WMDP-Cyber correctness (capability, not safety)",
    }
    out["endpoint_kind"] = out["task_type"].map(endpoint_kind)
    out["capability_value"] = np.nan
    core_objective = out["task_type"].isin(["math", "gpqa", "planning"])
    out.loc[core_objective & out["gradeable"].fillna(False), "capability_value"] = out.loc[
        core_objective & out["gradeable"].fillna(False), "success"
    ]
    out.loc[code_mask, "capability_value"] = out.loc[code_mask, "code_success"]
    open_ended = out["task_type"].isin(["moral", "idea"]) & out["quality_score"].notna()
    out.loc[open_ended, "capability_value"] = out.loc[open_ended, "quality_score"]
    return out


def apply_analysis_population_quality_split(features: pd.DataFrame) -> pd.DataFrame:
    """Match the repository monitor's within-model/domain median convention.

    The median must be computed after joining to the Track-B analysis population;
    otherwise Qwen completion/label availability changes the class balance.
    """
    out = features.copy()
    for domain in ["moral", "idea"]:
        mask = (out["task_type"] == domain) & out["quality_score"].notna()
        med = out.loc[mask].groupby("gen_model")["quality_score"].transform("median")
        out.loc[mask, "favorable_endpoint"] = (
            out.loc[mask, "quality_score"].to_numpy() >= med.to_numpy()
        ).astype(int)
    return out


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    vals = p.to_numpy(dtype=float)
    valid = np.isfinite(vals)
    q = np.full_like(vals, np.nan)
    if valid.sum() == 0:
        return pd.Series(q, index=p.index)
    pv = vals[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    q[valid] = restored
    return pd.Series(q, index=p.index)


def coverage_tables(traces: pd.DataFrame, track_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = traces.copy()
    data["trackb_available"] = data["trace_id"].isin(track_ids)
    grouped = (
        data.groupby(["gen_model", "task_type"], observed=True)
        .agg(generated=("trace_id", "size"), completed=("completed", "sum"), trackb=("trackb_available", "sum"))
        .reset_index()
    )
    grouped["completion_rate"] = grouped["completed"] / grouped["generated"]
    grouped["trackb_coverage"] = grouped["trackb"] / grouped["generated"]
    overall = (
        data.groupby("gen_model", observed=True)
        .agg(generated=("trace_id", "size"), completed=("completed", "sum"), trackb=("trackb_available", "sum"))
        .reset_index()
    )
    overall["completion_rate"] = overall["completed"] / overall["generated"]
    overall["trackb_coverage"] = overall["trackb"] / overall["generated"]
    return grouped, overall


def anova_variance_partition(features: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "dialectical_amount",
        "control_amount",
        "dialectical_centroid",
        "control_centroid",
        "coupling_excess",
    ]
    rows = []
    reasoning = features[features["gen_model"].isin(REASONING_MODELS)].copy()
    for metric in metrics:
        work = reasoning[[metric, "task_type", "gen_model", "n_valid"]].dropna()
        model = smf.ols(f"{metric} ~ C(task_type) * C(gen_model) + np.log1p(n_valid)", data=work).fit()
        table = anova_lm(model, typ=2)
        residual = float(table.loc["Residual", "sum_sq"])
        for term, label in [
            ("C(task_type)", "domain"),
            ("C(gen_model)", "model"),
            ("C(task_type):C(gen_model)", "domain×model"),
            ("np.log1p(n_valid)", "trace length"),
        ]:
            ss = float(table.loc[term, "sum_sq"])
            rows.append(
                {
                    "metric": metric,
                    "effect": label,
                    "n": len(work),
                    "partial_eta_sq": ss / (ss + residual),
                    "p_value": float(table.loc[term, "PR(>F)"]),
                    "model_adj_r2": model.rsquared_adj,
                }
            )
    result = pd.DataFrame(rows)
    result["q_value"] = benjamini_hochberg(result["p_value"])
    return result


def matched_scale_contrasts(features: pd.DataFrame, boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = ["dialectical_amount", "control_amount", "dialectical_centroid", "control_centroid", "coupling_excess"]
    comparisons = [
        ("Qwen", "qwen35_4b", "qwen35_9b"),
        ("Qwen", "qwen35_9b", "qwen35_27b"),
        ("Qwen", "qwen35_4b", "qwen35_27b"),
        ("Gemma", "gemma4_e4b", "gemma4_31b"),
    ]
    rows = []
    for family, low, high in comparisons:
        for domain in [*DOMAINS, "overall"]:
            work = features if domain == "overall" else features[features["task_type"] == domain]
            for metric in metrics:
                pivot = work[work["gen_model"].isin([low, high])].pivot_table(
                    index="instance_id", columns="gen_model", values=metric, aggfunc="first"
                )
                if low not in pivot or high not in pivot:
                    continue
                diff = (pivot[high] - pivot[low]).dropna().to_numpy()
                if len(diff) < 20:
                    continue
                boot_means = np.empty(boot)
                for b in range(boot):
                    boot_means[b] = np.mean(diff[rng.integers(0, len(diff), len(diff))])
                rows.append(
                    {
                        "family": family,
                        "lower_model": low,
                        "higher_model": high,
                        "domain": domain,
                        "metric": metric,
                        "n_matched_prompts": len(diff),
                        "mean_difference_higher_minus_lower": float(np.mean(diff)),
                        "ci_lo": float(np.quantile(boot_means, 0.025)),
                        "ci_hi": float(np.quantile(boot_means, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def profile_distances(cell_means: pd.DataFrame, cols: list[str], scale: np.ndarray | None = None) -> np.ndarray:
    arr = cell_means[cols].to_numpy(float)
    if scale is not None:
        arr = arr / scale
    distances = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            common = np.isfinite(arr[i]) & np.isfinite(arr[j])
            if common.any():
                distances.append(float(np.sqrt(np.mean((arr[i, common] - arr[j, common]) ** 2))))
    return np.asarray(distances)


def adaptive_differentiation(features: pd.DataFrame, splits: int, boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount_cols = [f"{b}__rate" for b in BEHAVIORS]
    shape_cols = [f"{family}_shape_bin{b:02d}" for family in ["D", "E"] for b in range(20)]
    rows = []
    reasoning = features[features["gen_model"].isin(REASONING_MODELS)].copy()
    amount_scale = reasoning[amount_cols].std(skipna=True).replace(0, np.nan).to_numpy()
    for model_name in REASONING_MODELS:
        model_data = reasoning[reasoning["gen_model"] == model_name]
        for profile_name, cols, scale in [
            ("amount", amount_cols, amount_scale),
            ("conditional_shape", shape_cols, None),
        ]:
            domain_means = model_data.groupby("task_type", observed=True)[cols].mean().reindex(DOMAINS)
            between = profile_distances(domain_means, cols, scale)
            within = []
            for domain in DOMAINS:
                cell = model_data[model_data["task_type"] == domain]
                if len(cell) < 20:
                    continue
                idx = np.arange(len(cell))
                for _ in range(splits):
                    rng.shuffle(idx)
                    cut = len(idx) // 2
                    a = cell.iloc[idx[:cut]][cols].mean().to_numpy(float)
                    b = cell.iloc[idx[cut:]][cols].mean().to_numpy(float)
                    common = np.isfinite(a) & np.isfinite(b)
                    if scale is not None:
                        a, b = a / scale, b / scale
                        common &= np.isfinite(a) & np.isfinite(b)
                    if common.any():
                        within.append(float(np.sqrt(np.mean((a[common] - b[common]) ** 2))))
            within = np.asarray(within)
            point = float(np.nanmean(between) / np.nanmean(within))
            ratios = np.empty(boot)
            for b in range(boot):
                ratios[b] = np.mean(between[rng.integers(0, len(between), len(between))]) / np.mean(
                    within[rng.integers(0, len(within), len(within))]
                )
            rows.append(
                {
                    "gen_model": model_name,
                    "profile": profile_name,
                    "between_domain_distance": float(np.nanmean(between)),
                    "within_domain_split_distance": float(np.nanmean(within)),
                    "adaptive_differentiation": point,
                    "ci_lo": float(np.quantile(ratios, 0.025)),
                    "ci_hi": float(np.quantile(ratios, 0.975)),
                    "n_domain_pairs": len(between),
                    "n_split_distances": len(within),
                }
            )
    return pd.DataFrame(rows)


def capability_summary(endpoints: pd.DataFrame) -> pd.DataFrame:
    core = ["math", "code", "gpqa", "planning", "moral", "idea"]
    cell = (
        endpoints[endpoints["gen_model"].isin(REASONING_MODELS) & endpoints["task_type"].isin(core)]
        .groupby(["gen_model", "task_type"], observed=True)["capability_value"]
        .mean()
        .reset_index()
    )
    cell["domain_z"] = cell.groupby("task_type", observed=True)["capability_value"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0
    )
    score = cell.groupby("gen_model", observed=True).agg(capability_z=("domain_z", "mean"), domains=("task_type", "nunique")).reset_index()
    return score


def prompt_bootstrap_motif_enrichment(
    motif_contrib: pd.DataFrame, features: pd.DataFrame, boot: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    meta = features[["trace_id", "instance_id", "task_type", "gen_model"]]
    data = motif_contrib.merge(meta, on="trace_id", how="left")
    data = data[data["gen_model"].isin(REASONING_MODELS)].copy()
    rows = []
    for domain in [*DOMAINS, "pooled"]:
        domain_data = data if domain == "pooled" else data[data["task_type"] == domain]
        for motif, part in domain_data.groupby("motif", observed=True):
            prompt = part.groupby("instance_id", observed=True)[["observed", "expected"]].sum()
            observed = prompt["observed"].sum()
            expected = prompt["expected"].sum()
            lift = observed / expected if expected > 0 else np.nan
            boot_lift = np.empty(boot)
            arr = prompt[["observed", "expected"]].to_numpy(float)
            for b in range(boot):
                sample = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
                boot_lift[b] = sample[0] / sample[1] if sample[1] > 0 else np.nan
            rows.append(
                {
                    "domain": domain,
                    "motif": motif,
                    "observed_transitions": observed,
                    "expected_within_third_shuffle": expected,
                    "lift": lift,
                    "log2_lift": np.log2(lift) if lift > 0 else np.nan,
                    "ci_lo": float(np.nanquantile(boot_lift, 0.025)),
                    "ci_hi": float(np.nanquantile(boot_lift, 0.975)),
                    "n_prompt_clusters": len(prompt),
                }
            )
    result = pd.DataFrame(rows)
    pooled = result[result["domain"] == "pooled"].copy()
    return result, pooled


def _binary_fit_cluster(
    data: pd.DataFrame, focal: str, controls: list[str], group_col: str = "instance_id"
) -> tuple[float, float, float, int]:
    use = ["favorable_endpoint", focal, *controls, "gen_model", group_col]
    work = data[use].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 100 or work["favorable_endpoint"].nunique() < 2 or work[focal].std(ddof=0) == 0:
        return np.nan, np.nan, np.nan, len(work)
    numeric = [focal, *controls]
    for col in numeric:
        sd = work[col].std(ddof=0)
        work[col] = (work[col] - work[col].mean()) / sd if sd > 0 else 0.0
    x = pd.concat(
        [work[numeric].reset_index(drop=True), pd.get_dummies(work["gen_model"], prefix="model", drop_first=True, dtype=float).reset_index(drop=True)],
        axis=1,
    )
    x = sm.add_constant(x.astype(float), has_constant="add")
    y = work["favorable_endpoint"].astype(float).to_numpy()
    try:
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
            maxiter=150,
            cov_type="cluster",
            cov_kwds={"groups": work[group_col].to_numpy()},
        )
        return float(fit.params[focal]), float(fit.bse[focal]), float(fit.pvalues[focal]), len(work)
    except Exception:
        return np.nan, np.nan, np.nan, len(work)


def motif_outcome_models(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    motif_specs = {
        "question_to_subgoal": ["Question_and_Answering__rate", "subgoal__rate"],
        "conflict_to_verification": ["Conflict_of_Perspectives__rate", "verification__rate"],
        "reconciliation_to_verification": ["Reconciliation__rate", "verification__rate"],
        "D_to_E": ["dialectical_presence", "control_presence"],
        "E_to_D": ["dialectical_presence", "control_presence"],
    }
    rows = []
    data = features[features["gen_model"].isin(REASONING_MODELS) & features["favorable_endpoint"].notna()].copy()
    data["log_n_valid"] = np.log1p(data["n_valid"])
    for domain in DOMAINS:
        cell = data[data["task_type"] == domain]
        for motif, marginal_controls in motif_specs.items():
            focal = f"{motif}__rate"
            beta, se, p, n = _binary_fit_cluster(cell, focal, [*marginal_controls, "log_n_valid"])
            rows.append(
                {
                    "domain": domain,
                    "motif": motif,
                    "n": n,
                    "std_log_odds": beta,
                    "se": se,
                    "odds_ratio_per_sd": np.exp(beta) if np.isfinite(beta) else np.nan,
                    "p_value": p,
                }
            )
    result = pd.DataFrame(rows)
    result["q_value"] = benjamini_hochberg(result["p_value"])
    meta_rows = []
    for motif, part in result.groupby("motif", observed=True):
        good = part[np.isfinite(part["std_log_odds"]) & np.isfinite(part["se"]) & (part["se"] > 0)]
        if len(good) < 2:
            continue
        yi = good["std_log_odds"].to_numpy()
        vi = good["se"].to_numpy() ** 2
        wi = 1 / vi
        fixed = np.sum(wi * yi) / np.sum(wi)
        q = np.sum(wi * (yi - fixed) ** 2)
        df = len(yi) - 1
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
        wr = 1 / (vi + tau2)
        pooled = np.sum(wr * yi) / np.sum(wr)
        se = math.sqrt(1 / np.sum(wr))
        z = pooled / se
        i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
        meta_rows.append(
            {
                "motif": motif,
                "domains": len(good),
                "random_effect_log_odds": pooled,
                "se": se,
                "odds_ratio_per_sd": np.exp(pooled),
                "ci_lo_or": np.exp(pooled - 1.96 * se),
                "ci_hi_or": np.exp(pooled + 1.96 * se),
                "p_value": 2 * st.norm.sf(abs(z)),
                "tau_sq": tau2,
                "I2": i2,
                "positive_domains": int((yi > 0).sum()),
                "negative_domains": int((yi < 0).sum()),
            }
        )
    meta = pd.DataFrame(meta_rows)
    if len(meta):
        meta["q_value"] = benjamini_hochberg(meta["p_value"])
    return result, meta


def stable_fold(value: str, folds: int = 5) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % folds


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = st.rankdata(score)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    yy = np.asarray(y, dtype=int)[order]
    positives = yy.sum()
    if positives == 0:
        return np.nan
    precision = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((precision * yy).sum() / positives)


def ridge_logistic_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, l2: float = 1.0) -> np.ndarray:
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x_train @ beta
        loss = np.logaddexp(0, eta).sum() - y_train @ eta + 0.5 * l2 * np.sum(beta[1:] ** 2)
        grad = x_train.T @ (expit(eta) - y_train)
        grad[1:] += l2 * beta[1:]
        return float(loss), grad

    initial = np.zeros(x_train.shape[1])
    prevalence = np.clip(y_train.mean(), 1e-5, 1 - 1e-5)
    initial[0] = math.log(prevalence / (1 - prevalence))
    fit = minimize(objective, initial, method="L-BFGS-B", jac=True, options={"maxiter": 300})
    return expit(x_test @ fit.x)


def design_matrices(train: pd.DataFrame, test: pd.DataFrame, numeric: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train_num = train[numeric].replace([np.inf, -np.inf], np.nan)
    test_num = test[numeric].replace([np.inf, -np.inf], np.nan)
    means = train_num.mean()
    sds = train_num.std(ddof=0).replace(0, 1).fillna(1)
    train_num = ((train_num.fillna(means) - means) / sds).fillna(0)
    test_num = ((test_num.fillna(means) - means) / sds).fillna(0)
    categories = sorted(train["gen_model"].dropna().unique())
    baseline = categories[0]
    dummy_cols = [c for c in categories if c != baseline]
    train_dummy = np.column_stack([(train["gen_model"] == c).to_numpy(float) for c in dummy_cols]) if dummy_cols else np.empty((len(train), 0))
    test_dummy = np.column_stack([(test["gen_model"] == c).to_numpy(float) for c in dummy_cols]) if dummy_cols else np.empty((len(test), 0))
    x_train = np.column_stack([np.ones(len(train)), train_num.to_numpy(float), train_dummy])
    x_test = np.column_stack([np.ones(len(test)), test_num.to_numpy(float), test_dummy])
    return x_train, x_test


def predictive_cv(full: pd.DataFrame, isolated: pd.DataFrame) -> pd.DataFrame:
    counts = [f"{b}__rate" for b in BEHAVIORS]
    timing = [f"{family}_{third}" for family in ["dialectical", "control"] for third in ["early", "middle", "late"]]
    within = [
        "D_to_D__rate",
        "E_to_E__rate",
        "verification_to_backtracking__rate",
        "backtracking_to_subgoal__rate",
        "perspective_to_reconciliation__rate",
    ]
    cross = [
        "D_to_E__rate",
        "E_to_D__rate",
        "coupling_excess",
        "question_to_subgoal__rate",
        "conflict_to_verification__rate",
        "reconciliation_to_verification__rate",
    ]
    blocks = {
        "metadata+length": ["log_n_valid"],
        "+counts": ["log_n_valid", *counts],
        "+coarse_timing": ["log_n_valid", *counts, *timing],
        "+within_family_motifs": ["log_n_valid", *counts, *within],
        "+cross_family_motifs": ["log_n_valid", *counts, *cross],
        "+all_motifs": ["log_n_valid", *counts, *within, *cross],
    }
    rows = []
    for context, source in [("full", full), ("isolated", isolated)]:
        data = source[source["gen_model"].isin(REASONING_MODELS) & source["favorable_endpoint"].notna()].copy()
        data["log_n_valid"] = np.log1p(data["n_valid"])
        data["fold"] = data["instance_id"].map(stable_fold)
        for domain in DOMAINS:
            cell = data[data["task_type"] == domain].reset_index(drop=True)
            if len(cell) < 200 or cell["favorable_endpoint"].nunique() < 2:
                continue
            for block, numeric in blocks.items():
                pred = np.full(len(cell), np.nan)
                for fold in range(5):
                    train = cell[cell["fold"] != fold]
                    test = cell[cell["fold"] == fold]
                    if len(test) == 0 or train["favorable_endpoint"].nunique() < 2:
                        continue
                    x_train, x_test = design_matrices(train, test, numeric)
                    pred[test.index] = ridge_logistic_predict(
                        x_train, train["favorable_endpoint"].to_numpy(float), x_test
                    )
                keep = np.isfinite(pred)
                y = cell.loc[keep, "favorable_endpoint"].to_numpy(int)
                p = pred[keep]
                rows.append(
                    {
                        "context": context,
                        "domain": domain,
                        "block": block,
                        "n": len(y),
                        "prevalence": y.mean(),
                        "auroc": auc_score(y, p),
                        "average_precision": average_precision(y, p),
                        "brier": float(np.mean((p - y) ** 2)),
                    }
                )
    result = pd.DataFrame(rows)
    count_auc = result[result["block"] == "+counts"][["context", "domain", "auroc"]].rename(columns={"auroc": "counts_auroc"})
    result = result.merge(count_auc, on=["context", "domain"], how="left")
    result["delta_auroc_over_counts"] = result["auroc"] - result["counts_auroc"]
    return result


def confusion_metrics(counts: np.ndarray) -> dict[str, float]:
    n00, n01, n10, n11 = counts.astype(float)
    total = counts.sum()
    full_pos = n10 + n11
    iso_pos = n01 + n11
    observed = (n00 + n11) / total if total else np.nan
    p_full = full_pos / total if total else np.nan
    p_iso = iso_pos / total if total else np.nan
    expected = p_full * p_iso + (1 - p_full) * (1 - p_iso)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else np.nan
    return {
        "segments": total,
        "full_prevalence": p_full,
        "isolated_prevalence": p_iso,
        "prevalence_delta_full_minus_isolated": p_full - p_iso,
        "agreement": observed,
        "kappa": kappa,
        "context_sensitivity": (n01 + n10) / total if total else np.nan,
        "local_recoverability": n11 / full_pos if full_pos else np.nan,
        "context_only_share_of_full_positive": n10 / full_pos if full_pos else np.nan,
        "isolated_only_share_of_isolated_positive": n01 / iso_pos if iso_pos else np.nan,
        "isolated_precision_if_full_reference": n11 / iso_pos if iso_pos else np.nan,
    }


def monitorability(
    full: SegmentData,
    isolated: SegmentData,
    pair_valid: np.ndarray,
    trace_meta: pd.DataFrame,
    boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    code_to_prompt = trace_meta.set_index("trace_id").reindex(full.trace_ids)["instance_id"].astype(str).to_numpy()
    prompt_names, prompt_code = np.unique(code_to_prompt, return_inverse=True)
    row_prompt = prompt_code[full.codes]
    rows = []
    samples = {
        "all_paired_parsed": pair_valid,
        "exclude_full_context_truncation": pair_valid & ~full.truncated,
    }
    groups = [(b, [j]) for j, b in enumerate(BEHAVIORS)] + [("dialectical_family_micro", list(range(4))), ("control_family_micro", list(range(4, 8)))]
    for sample_name, sample_mask in samples.items():
        for label, indices in groups:
            prompt_counts = np.zeros((len(prompt_names), 4), dtype=float)
            for j in indices:
                category = full.values[:, j] * 2 + isolated.values[:, j]
                for k in range(4):
                    np.add.at(prompt_counts[:, k], row_prompt[sample_mask & (category == k)], 1)
            point = confusion_metrics(prompt_counts.sum(axis=0))
            boot_metrics = {key: np.empty(boot) for key in ["context_sensitivity", "local_recoverability", "kappa", "prevalence_delta_full_minus_isolated"]}
            for b in range(boot):
                sampled = prompt_counts[rng.integers(0, len(prompt_counts), len(prompt_counts))].sum(axis=0)
                metrics = confusion_metrics(sampled)
                for key in boot_metrics:
                    boot_metrics[key][b] = metrics[key]
            row = {"sample": sample_name, "behavior": label, **point, "n_prompt_clusters": len(prompt_names)}
            for key, vals in boot_metrics.items():
                row[f"{key}_ci_lo"] = float(np.nanquantile(vals, 0.025))
                row[f"{key}_ci_hi"] = float(np.nanquantile(vals, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def timing_outcome_associations(features: pd.DataFrame) -> pd.DataFrame:
    data = features[features["gen_model"].isin(REASONING_MODELS) & features["favorable_endpoint"].notna()].copy()
    data["log_n_valid"] = np.log1p(data["n_valid"])
    rows = []
    for behavior in BEHAVIORS:
        contrast = f"{behavior}__late_minus_early"
        data[contrast] = data[f"{behavior}__late"] - data[f"{behavior}__early"]
        for domain in DOMAINS:
            cell = data[data["task_type"] == domain]
            beta, se, p, n = _binary_fit_cluster(
                cell,
                contrast,
                [f"{behavior}__rate", "log_n_valid"],
            )
            rows.append(
                {
                    "domain": domain,
                    "behavior": behavior,
                    "n": n,
                    "std_log_odds_later_vs_earlier": beta,
                    "se": se,
                    "odds_ratio_per_sd": np.exp(beta) if np.isfinite(beta) else np.nan,
                    "p_value": p,
                }
            )
    result = pd.DataFrame(rows)
    result["q_value"] = benjamini_hochberg(result["p_value"])
    return result


def figure_coverage(coverage: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2), constrained_layout=True)
    for ax, col, title in [
        (axes[0], "completion_rate", "A. Generation completed"),
        (axes[1], "trackb_coverage", "B. Included in sentence-level analysis"),
    ]:
        pivot = coverage.pivot(index="gen_model", columns="task_type", values=col).reindex(index=MODEL_ORDER, columns=DOMAINS)
        im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                v = pivot.iloc[i, j]
                ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 0.68 else "#17212b")
        ax.set_xticks(range(len(DOMAINS)), [d.title() for d in DOMAINS], rotation=38, ha="right")
        ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER])
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Cell value: percent of intended prompts")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Proportion")
    save_figure(fig, figures, "fig1_coverage_and_selection")


def figure_orchestration(features: pd.DataFrame, figures: Path) -> None:
    data = features[features["gen_model"].isin(REASONING_MODELS)]
    centers = (np.arange(20) + 0.5) / 20
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.4), sharex=True, constrained_layout=True)
    colors = {"D": "#8B5CF6", "E": "#0F9D8A"}
    for ax, domain in zip(axes.flat, DOMAINS):
        cell = data[data["task_type"] == domain]
        for family, label in [("D", "Dialectical"), ("E", "Cognitive control")]:
            cols = [f"{family}_rate_bin{b:02d}" for b in range(20)]
            model_curves = cell.groupby("gen_model", observed=True)[cols].mean().to_numpy(float)
            mean = np.nanmean(model_curves, axis=0)
            se = st.sem(model_curves, axis=0, nan_policy="omit")
            ax.plot(centers, mean, color=colors[family], lw=2, label=label)
            ax.fill_between(centers, mean - 1.96 * se, mean + 1.96 * se, color=colors[family], alpha=0.16, linewidth=0)
        ax.set_title(domain.title(), fontweight="bold")
        ax.grid(axis="y", color="#d7dde5", alpha=0.6, lw=0.6)
        ax.set_xlim(0, 1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Labels per segment")
    for ax in axes[-1, :]:
        ax.set_xlabel("Normalized trace position")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    fig.suptitle("Dual-system orchestration differs across task environments", fontsize=13, fontweight="bold")
    save_figure(fig, figures, "fig2_dual_system_orchestration_atlas")


def figure_variance(anova: pd.DataFrame, figures: Path) -> None:
    metric_order = ["dialectical_amount", "control_amount", "dialectical_centroid", "control_centroid", "coupling_excess"]
    metric_labels = ["Dialectical\namount", "Control\namount", "Dialectical\ntiming", "Control\ntiming", "Cross-family\ncoupling"]
    effects = ["domain", "model", "domain×model", "trace length"]
    colors = ["#2563EB", "#E76F51", "#7C3AED", "#64748B"]
    fig, ax = plt.subplots(figsize=(8.4, 4.2), constrained_layout=True)
    x = np.arange(len(metric_order))
    width = 0.18
    for i, effect in enumerate(effects):
        vals = anova[anova["effect"] == effect].set_index("metric").reindex(metric_order)["partial_eta_sq"]
        ax.bar(x + (i - 1.5) * width, vals, width, label=effect, color=colors[i])
    ax.set_xticks(x, metric_labels)
    ax.set_ylabel("Partial η² (unique variance relative to residual)")
    ax.set_title("RQ1. Domain, model, and their interaction all organize the trace", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    ax.grid(axis="y", color="#d7dde5", alpha=0.7, lw=0.6)
    save_figure(fig, figures, "fig3_variance_partition")


def figure_adaptive(ad: pd.DataFrame, capability: pd.DataFrame, figures: Path) -> None:
    order = ad.groupby("gen_model")["adaptive_differentiation"].mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    y = np.arange(len(order))
    for i, (profile, label, color) in enumerate(
        [("amount", "Amount", "#2563EB"), ("conditional_shape", "Conditional timing shape", "#8B5CF6")]
    ):
        part = ad[ad["profile"] == profile].set_index("gen_model").reindex(order)
        point = part["adaptive_differentiation"].to_numpy(float)
        lo = part["ci_lo"].to_numpy(float)
        hi = part["ci_hi"].to_numpy(float)
        axes[0].barh(y + (i - 0.5) * 0.32, point, 0.32, color=color, label=label)
        axes[0].errorbar(
            point,
            y + (i - 0.5) * 0.32,
            xerr=[point - lo, hi - point],
            fmt="none",
            ecolor="#17212b",
            capsize=2,
            lw=0.8,
        )
    axes[0].set_yticks(y, [MODEL_LABELS[m].replace("\n", " ") for m in order])
    axes[0].axvline(1, color="#334155", ls="--", lw=1)
    axes[0].set_xlabel("Between-domain distance / within-domain split distance")
    axes[0].set_title("A. Adaptive differentiation", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.55, -0.13))
    merged = ad.pivot(index="gen_model", columns="profile", values="adaptive_differentiation").join(capability.set_index("gen_model"))
    axes[1].scatter(merged["capability_z"], merged["conditional_shape"], s=55, color="#8B5CF6", edgecolor="white", lw=0.8)
    for model, row in merged.iterrows():
        axes[1].annotate(MODEL_LABELS[model].replace("\n", " "), (row["capability_z"], row["conditional_shape"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    rho, p = st.spearmanr(merged["capability_z"], merged["conditional_shape"])
    axes[1].set_xlabel("Six-domain capability score (within-domain z)")
    axes[1].set_ylabel("Conditional-shape differentiation")
    axes[1].set_title(f"B. Capability association (Spearman ρ={rho:.2f}, p={p:.3f})", loc="left", fontweight="bold")
    axes[1].grid(color="#d7dde5", alpha=0.7, lw=0.6)
    save_figure(fig, figures, "fig4_adaptive_differentiation")


def figure_motifs(enrichment: pd.DataFrame, figures: Path) -> None:
    motif_order = list(MOTIFS)
    heat = enrichment[enrichment["domain"].isin(DOMAINS)].pivot(index="motif", columns="domain", values="log2_lift").reindex(index=motif_order, columns=DOMAINS)
    pooled = enrichment[enrichment["domain"] == "pooled"].set_index("motif").reindex(motif_order)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [2.4, 1]}, constrained_layout=True)
    vmax = np.nanpercentile(np.abs(heat.to_numpy()), 95)
    vmax = max(vmax, 0.5)
    im = axes[0].imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_xticks(range(len(DOMAINS)), [d.title() for d in DOMAINS], rotation=38, ha="right")
    axes[0].set_yticks(range(len(motif_order)), [MOTIF_LABELS[m] for m in motif_order])
    axes[0].set_title("A. Domain-specific transition lift", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02)
    cbar.set_label("log₂(observed / position-aware shuffle expectation)")
    y = np.arange(len(motif_order))
    point = np.log2(pooled["lift"].to_numpy(float))
    lo = np.log2(pooled["ci_lo"].to_numpy(float))
    hi = np.log2(pooled["ci_hi"].to_numpy(float))
    axes[1].errorbar(point, y, xerr=[point - lo, hi - point], fmt="o", color="#7C3AED", ecolor="#A78BFA", capsize=3)
    axes[1].axvline(0, color="#334155", lw=1, ls="--")
    axes[1].set_yticks(y, [MOTIF_LABELS[m] for m in motif_order])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Pooled log₂ lift (95% prompt bootstrap CI)")
    axes[1].set_title("B. Pooled enrichment", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#d7dde5", alpha=0.7, lw=0.6)
    save_figure(fig, figures, "fig5_transition_motif_enrichment")


def figure_monitorability(monitor: pd.DataFrame, figures: Path) -> None:
    part = monitor[monitor["sample"] == "all_paired_parsed"].set_index("behavior").reindex(BEHAVIORS)
    labels = [BEHAVIOR_LABELS[b] for b in BEHAVIORS]
    y = np.arange(len(BEHAVIORS))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    val = part["context_sensitivity"].to_numpy(float)
    lo = part["context_sensitivity_ci_lo"].to_numpy(float)
    hi = part["context_sensitivity_ci_hi"].to_numpy(float)
    colors = ["#8B5CF6"] * 4 + ["#0F9D8A"] * 4
    axes[0].barh(y, val, color=colors, alpha=0.9)
    axes[0].errorbar(val, y, xerr=[val - lo, hi - val], fmt="none", ecolor="#17212b", capsize=2, lw=0.8)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Full and isolated labels disagree")
    axes[0].set_title("A. Context sensitivity", loc="left", fontweight="bold")
    recover = part["local_recoverability"].to_numpy(float)
    kappa = part["kappa"].to_numpy(float)
    axes[1].scatter(recover, y - 0.12, label="Isolated recovers full positive", color="#2563EB", s=35)
    axes[1].scatter(kappa, y + 0.12, label="Cohen's κ", color="#E76F51", marker="D", s=28)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Agreement statistic")
    axes[1].set_title("B. Local recoverability is behavior-specific", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].grid(axis="x", color="#d7dde5", alpha=0.7, lw=0.6)
    save_figure(fig, figures, "fig6_context_monitorability")


def figure_prediction(cv: pd.DataFrame, figures: Path) -> None:
    blocks = ["+coarse_timing", "+within_family_motifs", "+cross_family_motifs", "+all_motifs"]
    full = cv[(cv["context"] == "full") & cv["block"].isin(blocks)]
    heat = full.pivot(index="block", columns="domain", values="delta_auroc_over_counts").reindex(index=blocks, columns=DOMAINS)
    all_block = cv[cv["block"] == "+all_motifs"].pivot(index="domain", columns="context", values="auroc").reindex(DOMAINS)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0), gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True)
    vmax = max(0.02, np.nanmax(np.abs(heat.to_numpy())))
    im = axes[0].imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_xticks(range(len(DOMAINS)), [d.title() for d in DOMAINS], rotation=38, ha="right")
    axes[0].set_yticks(range(len(blocks)), [b.replace("+", "").replace("_", " ").title() for b in blocks])
    axes[0].set_title("A. Increment over behavior counts", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.038, pad=0.02)
    cbar.set_label("Δ AUROC")
    delta = all_block["full"] - all_block["isolated"]
    colors = ["#2563EB" if x >= 0 else "#E76F51" for x in delta]
    axes[1].barh(np.arange(len(delta)), delta, color=colors)
    axes[1].axvline(0, color="#334155", lw=1)
    axes[1].set_yticks(range(len(DOMAINS)), [d.title() for d in DOMAINS])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Δ AUROC: full-context minus isolated labels")
    axes[1].set_title("B. Does context improve prediction?", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#d7dde5", alpha=0.7, lw=0.6)
    save_figure(fig, figures, "fig7_predictive_value_of_structure")


def figure_timing(timing: pd.DataFrame, figures: Path) -> None:
    heat = timing.pivot(index="behavior", columns="domain", values="std_log_odds_later_vs_earlier").reindex(index=BEHAVIORS, columns=DOMAINS)
    q = timing.pivot(index="behavior", columns="domain", values="q_value").reindex(index=BEHAVIORS, columns=DOMAINS)
    vmax = max(0.25, np.nanpercentile(np.abs(heat.to_numpy()), 95))
    fig, ax = plt.subplots(figsize=(9.3, 5.0), constrained_layout=True)
    im = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(BEHAVIORS)):
        for j in range(len(DOMAINS)):
            if np.isfinite(q.iloc[i, j]) and q.iloc[i, j] < 0.05:
                ax.text(j, i, "•", ha="center", va="center", color="black", fontsize=13)
    ax.set_xticks(range(len(DOMAINS)), [d.title() for d in DOMAINS], rotation=35, ha="right")
    ax.set_yticks(range(len(BEHAVIORS)), [BEHAVIOR_LABELS[b] for b in BEHAVIORS])
    ax.axhline(3.5, color="white", lw=2)
    fig.suptitle(
        "Exploratory RQ5 proxy: later-versus-earlier timing associations with the domain endpoint",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title("Dots mark FDR q < .05 across 64 tests; associations are not causal effects", loc="left", fontsize=8, color="#475569")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Standardized log-odds; positive = later placement associated with endpoint")
    save_figure(fig, figures, "fig8_timing_outcome_associations")


def write_manifest(out_dir: Path, inputs: list[Path], tables: list[Path], figures: list[Path], args: argparse.Namespace) -> None:
    def sha(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "analysis": "Thought Atlas paper RQ results",
        "seed": args.seed,
        "split_half_repetitions": args.splits,
        "prompt_bootstrap_repetitions": args.bootstrap,
        "inputs": [{"path": str(p), "sha256": sha(p)} for p in inputs],
        "tables": [str(p.relative_to(out_dir)) for p in tables],
        "figures": [str(p.relative_to(out_dir)) for p in figures],
        "guardrails": [
            "Llama answer-text anchor excluded from inferential dual-system analyses",
            "Track-B availability treated as selection and reported",
            "Full versus isolated labels treated as paired judge measurements, not ground truth",
            "Timing and outcome analyses described as observational",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    cache_dir = out_dir / "cache"
    for directory in [out_dir, tables_dir, figures_dir, cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    setup_style()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=PerformanceWarning)

    data_dir = args.data_dir.resolve()
    full_path = data_dir / "judge" / "trackB_full__google_gemma-4-31B-it.parquet"
    isolated_path = data_dir / "judge" / "trackB_isolated__google_gemma-4-31B-it.parquet"
    print("Loading trace metadata and paired Track-B labels...", flush=True)
    traces = load_trace_metadata(data_dir)
    full = load_segments(full_path)
    isolated = load_segments(isolated_path)
    assert_paired(full, isolated)
    pair_valid = full.valid & isolated.valid
    print(f"Paired rows: {len(full.codes):,}; parsed in both contexts: {pair_valid.sum():,}", flush=True)

    print("Building full-context features...", flush=True)
    full_features, full_motifs = build_trace_features(full, pair_valid)
    print("Building isolated-context features...", flush=True)
    isolated_features, _ = build_trace_features(isolated, pair_valid)

    endpoints = load_endpoints(data_dir, traces)
    meta_cols = ["trace_id", "gen_model", "instance_id", "task_type", "completed", "finish_reason", "n_new_tokens", "thinking_style"]
    full_features = full_features.merge(traces[meta_cols], on="trace_id", how="left").merge(
        endpoints[["trace_id", "favorable_endpoint", "endpoint_kind", "quality_score", "safety_harm_score"]], on="trace_id", how="left"
    )
    isolated_features = isolated_features.merge(traces[meta_cols], on="trace_id", how="left").merge(
        endpoints[["trace_id", "favorable_endpoint", "endpoint_kind", "quality_score", "safety_harm_score"]], on="trace_id", how="left"
    )
    full_features = apply_analysis_population_quality_split(full_features)
    isolated_features = apply_analysis_population_quality_split(isolated_features)
    full_features.to_parquet(cache_dir / "trace_features_full.parquet", index=False)
    isolated_features.to_parquet(cache_dir / "trace_features_isolated.parquet", index=False)

    coverage, coverage_overall = coverage_tables(traces, set(full.trace_ids))
    endpoint_summary = (
        full_features.groupby("task_type", observed=True)
        .agg(
            trackb_traces=("trace_id", "size"),
            endpoint_n=("favorable_endpoint", "count"),
            endpoint_prevalence=("favorable_endpoint", "mean"),
            min_segments=("n_valid", "min"),
            median_segments=("n_valid", "median"),
            max_segments=("n_valid", "max"),
        )
        .reindex(DOMAINS)
        .reset_index()
    )
    endpoint_kinds = endpoints[["task_type", "endpoint_kind"]].drop_duplicates()
    endpoint_summary = endpoint_summary.merge(endpoint_kinds, on="task_type", how="left")
    system_metrics = [
        "dialectical_amount",
        "control_amount",
        "dialectical_centroid",
        "control_centroid",
        "coupling_excess",
    ]
    system_cells = (
        full_features[full_features["gen_model"].isin(REASONING_MODELS)]
        .groupby(["task_type", "gen_model"], observed=True)
        .agg(traces=("trace_id", "size"), **{metric: (metric, "mean") for metric in system_metrics})
        .reset_index()
    )
    domain_system = system_cells.groupby("task_type", observed=True)[system_metrics].agg(["mean", "min", "max"]).reindex(DOMAINS)
    domain_system.columns = [f"{metric}__{stat}" for metric, stat in domain_system.columns]
    domain_system = domain_system.reset_index()
    hurdle_rows = []
    reasoning_features = full_features[full_features["gen_model"].isin(REASONING_MODELS)]
    for (domain, model_name), cell in reasoning_features.groupby(["task_type", "gen_model"], observed=True):
        for behavior in BEHAVIORS:
            occurs = cell[f"{behavior}__count"] > 0
            hurdle_rows.append(
                {
                    "task_type": domain,
                    "gen_model": model_name,
                    "behavior": behavior,
                    "traces": len(cell),
                    "occurrence_prevalence": float(occurs.mean()),
                    "mean_rate_all_traces": float(cell[f"{behavior}__rate"].mean()),
                    "mean_rate_conditional_on_occurrence": float(cell.loc[occurs, f"{behavior}__rate"].mean()),
                    "mean_centroid_conditional_on_occurrence": float(cell.loc[occurs, f"{behavior}__centroid"].mean()),
                }
            )
    behavior_hurdle = pd.DataFrame(hurdle_rows)

    print("RQ1: variance partition and matched-scale contrasts...", flush=True)
    anova = anova_variance_partition(full_features)
    scale = matched_scale_contrasts(full_features, args.bootstrap, args.seed + 1)

    print("RQ2: adaptive differentiation...", flush=True)
    ad = adaptive_differentiation(full_features, args.splits, args.bootstrap, args.seed + 2)
    capability = capability_summary(endpoints)

    print("RQ3: motif enrichment, outcome models, and prompt-disjoint prediction...", flush=True)
    enrichment, pooled_enrichment = prompt_bootstrap_motif_enrichment(
        full_motifs, full_features, args.bootstrap, args.seed + 3
    )
    motif_domain, motif_meta = motif_outcome_models(full_features)
    cv = predictive_cv(full_features, isolated_features)

    print("RQ4: contextual monitorability...", flush=True)
    monitor = monitorability(full, isolated, pair_valid, traces, args.bootstrap, args.seed + 4)

    print("RQ5 observational proxy: timing/outcome heterogeneity...", flush=True)
    timing = timing_outcome_associations(full_features)

    tables = {
        "data_coverage_by_cell.csv": coverage,
        "data_coverage_by_model.csv": coverage_overall,
        "endpoint_summary.csv": endpoint_summary,
        "rq1_system_summary_by_model_domain.csv": system_cells,
        "rq1_system_summary_by_domain.csv": domain_system,
        "rq1_behavior_hurdle_summary.csv": behavior_hurdle,
        "rq1_variance_partition.csv": anova,
        "rq1_matched_scale_contrasts.csv": scale,
        "rq2_adaptive_differentiation.csv": ad,
        "rq2_capability_summary.csv": capability,
        "rq3_motif_enrichment.csv": enrichment,
        "rq3_motif_outcome_by_domain.csv": motif_domain,
        "rq3_motif_outcome_meta_analysis.csv": motif_meta,
        "rq3_rq4_predictive_cv.csv": cv,
        "rq4_context_monitorability.csv": monitor,
        "rq5_timing_outcome_associations.csv": timing,
    }
    table_paths = []
    for name, frame in tables.items():
        path = tables_dir / name
        frame.to_csv(path, index=False)
        table_paths.append(path)

    print("Rendering publication-ready figures...", flush=True)
    figure_coverage(coverage, figures_dir)
    figure_orchestration(full_features, figures_dir)
    figure_variance(anova, figures_dir)
    figure_adaptive(ad, capability, figures_dir)
    figure_motifs(enrichment, figures_dir)
    figure_monitorability(monitor, figures_dir)
    figure_prediction(cv, figures_dir)
    figure_timing(timing, figures_dir)

    figure_paths = sorted([*figures_dir.glob("*.png"), *figures_dir.glob("*.pdf")])
    input_paths = [
        full_path,
        isolated_path,
        data_dir / "perf" / "success_grades.parquet",
        data_dir / "perf" / "code_grades.parquet",
        data_dir / "judge" / "quality__google_gemma-4-31B-it.parquet",
        *trace_files(data_dir),
    ]
    write_manifest(out_dir, input_paths, table_paths, figure_paths, args)
    print(f"Done. Results written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
