"""Timing-vs-level decomposition for Track B behavior gaps.

This module asks whether outcome-linked behavior differences are explained by
overall behavior rate alone, or whether the solved/high-quality minus failed/
low-quality gap changes over normalized trace position. The result is
associational and retrospective: normalized bins require the final trace length.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from src.utils.io import resolve_trace_paths
from scipy.stats import chi2, norm


K = 24
MIN_SEG_PER_BIN = 30
MIN_BINS = 12
B_BOOT = 1000
BH_Q = 0.05

KIM = [
    "Question_and_Answering",
    "Perspective_Shift",
    "Conflict_of_Perspectives",
    "Reconciliation",
]
GANDHI = ["verification", "backtracking", "subgoal", "backward_chaining"]
BEHAVIORS = GANDHI + KIM
FAMILIES = {b: "cognitive" for b in GANDHI} | {b: "conversational" for b in KIM}


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
        ("completed", pl.Boolean),
        ("task_type", pl.Utf8),
        ("grade_method", pl.Utf8),
    ]:
        if col not in grades.columns:
            grades = grades.with_columns(pl.lit(None).cast(dtype).alias(col))
    return grades.group_by("trace_id").agg(
        _first_non_null("success"),
        _first_non_null("quality_score"),
        _first_non_null("parsed"),
        _first_non_null("completed"),
        _first_non_null("task_type"),
        _first_non_null("grade_method"),
    )


def add_outcome_classes(traces: pl.DataFrame, grades: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    """Attach good/bad outcome classes and per-cell quality split thresholds."""
    df = traces.join(grades, on="trace_id", how="left", suffix="_grade")
    if "task_type_grade" in df.columns:
        df = df.drop("task_type_grade")
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
        .then(pl.when(pl.col("success") >= 1).then(pl.lit("good")).otherwise(pl.lit("bad")))
        .when(pl.col("quality_score").is_not_null() & pl.col("_quality_median").is_not_null())
        .then(
            pl.when(pl.col("quality_score") >= pl.col("_quality_median"))
            .then(pl.lit("good"))
            .otherwise(pl.lit("bad"))
        )
        .otherwise(pl.lit(None))
        .alias("outcome_class")
    )
    return out, quality_split


def _bin_expr(bins: int) -> pl.Expr:
    # Match the repository's heartbeat export so per-bin gaps line up with the
    # public dashboard curves.
    raw = (pl.col("norm_pos") * (bins - 1)).round().cast(pl.Int32)
    return (
        pl.when(raw < 0)
        .then(pl.lit(0))
        .when(raw >= bins)
        .then(pl.lit(bins - 1))
        .otherwise(raw)
        .alias("bin")
    )


def compute_trace_bin_table(
    track_b: pl.DataFrame,
    traces_with_outcomes: pl.DataFrame,
    *,
    behaviors: list[str] | None = None,
    bins: int = K,
) -> pl.DataFrame:
    """Return per-trace per-bin X/n counts for each behavior.

    The trace-level table is the bootstrap workhorse. `compute_bin_table` below
    collapses this to the public aggregate table described in the analysis spec.
    """
    behaviors = [b for b in (behaviors or BEHAVIORS) if b in track_b.columns]
    if not behaviors:
        return pl.DataFrame()

    meta = (
        traces_with_outcomes.filter(pl.col("outcome_class").is_in(["good", "bad"]))
        .select(["trace_id", "gen_model", "task_type", "outcome_class"])
    )
    seg = (
        track_b.select(["trace_id", "norm_pos"] + behaviors)
        .join(meta, on="trace_id", how="inner")
        .with_columns(_bin_expr(bins))
    )
    keys = ["gen_model", "task_type", "outcome_class", "trace_id", "bin"]
    parts = []
    for behavior in behaviors:
        parts.append(
            seg.group_by(keys)
            .agg(
                pl.col(behavior).cast(pl.Int64).sum().alias("X"),
                pl.len().cast(pl.Int64).alias("n"),
            )
            .with_columns(pl.lit(behavior).alias("behavior"))
            .select(["gen_model", "task_type", "outcome_class", "behavior", "trace_id", "bin", "X", "n"])
        )
    return pl.concat(parts, how="diagonal_relaxed").sort(["gen_model", "task_type", "behavior", "outcome_class", "trace_id", "bin"])


def compute_bin_table(
    track_b: pl.DataFrame,
    traces_with_outcomes: pl.DataFrame,
    *,
    behaviors: list[str] | None = None,
    bins: int = K,
) -> pl.DataFrame:
    """Aggregate Track B to (model, domain, outcome, behavior, bin, X, n)."""
    trace_bins = compute_trace_bin_table(track_b, traces_with_outcomes, behaviors=behaviors, bins=bins)
    if trace_bins.is_empty():
        return trace_bins
    return (
        trace_bins.group_by(["gen_model", "task_type", "outcome_class", "behavior", "bin"])
        .agg(pl.col("X").sum(), pl.col("n").sum())
        .sort(["gen_model", "task_type", "behavior", "outcome_class", "bin"])
    )


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _two_sided_z_p(z: float) -> float:
    return float(2 * norm.sf(abs(z)))


def _chi_p(q: float, df: int) -> float:
    return float(chi2.sf(max(0.0, q), max(1, df)))


def _i2(q: float, df: int) -> float:
    if q <= 0:
        return 0.0
    return float(max(0.0, (q - df) / q))


def _stack_trace_arrays(trace_rows: dict[str, tuple[np.ndarray, np.ndarray]], bins: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids = sorted(trace_rows)
    if not ids:
        return np.zeros((0, bins)), np.zeros((0, bins)), []
    x = np.stack([trace_rows[tid][0] for tid in ids]).astype(float)
    n = np.stack([trace_rows[tid][1] for tid in ids]).astype(float)
    return x, n, ids


def _bootstrap_gaps(
    x_good: np.ndarray,
    n_good: np.ndarray,
    x_bad: np.ndarray,
    n_bad: np.ndarray,
    *,
    boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_g = x_good.shape[0]
    n_b = x_bad.shape[0]
    wg = rng.multinomial(n_g, np.full(n_g, 1 / n_g), size=boot)
    wb = rng.multinomial(n_b, np.full(n_b, 1 / n_b), size=boot)
    xg = wg @ x_good
    ng = wg @ n_good
    xb = wb @ x_bad
    nb = wb @ n_bad
    pg = np.divide(xg, ng, out=np.zeros_like(xg, dtype=float), where=ng > 0)
    pb = np.divide(xb, nb, out=np.zeros_like(xb, dtype=float), where=nb > 0)
    return pg - pb


def _regularized_inverse(cov: np.ndarray, v: np.ndarray, n_good: int, n_bad: int) -> np.ndarray:
    k = cov.shape[0]
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    if k == 1:
        return np.array([[1.0 / max(float(cov[0, 0]), float(v[0]), 1e-10)]])
    if not np.isfinite(cov).all() or np.trace(cov) <= 0:
        cov = np.diag(np.maximum(v, 1e-10))
    if k > min(n_good, n_bad) / 3:
        alpha = min(0.85, max(0.25, 1 - min(n_good, n_bad) / max(1, 3 * k)))
        cov = (1 - alpha) * cov + alpha * np.diag(np.diag(cov))
    trace_mean = float(np.trace(cov) / k)
    ridge = 1e-3 * trace_mean if trace_mean > 0 else 1e-10
    return np.linalg.pinv(cov + ridge * np.eye(k), rcond=1e-10)


def _decompose_pair(
    good_rows: dict[str, tuple[np.ndarray, np.ndarray]],
    bad_rows: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    bins: int,
    min_seg_per_bin: int,
    min_bins: int,
    boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    x_good, n_good, good_ids = _stack_trace_arrays(good_rows, bins)
    x_bad, n_bad, bad_ids = _stack_trace_arrays(bad_rows, bins)
    n_traces = {"good": len(good_ids), "bad": len(bad_ids)}
    empty = {
        "status": "insufficient",
        "n_good": n_traces["good"],
        "n_bad": n_traces["bad"],
        "bins_used": 0,
        "d_pp": [None] * bins,
        "d_se_pp": [None] * bins,
        "std_profile": [None] * bins,
        "thirds_early": 0.0,
        "thirds_mid": 0.0,
        "thirds_late": 0.0,
    }
    if n_traces["good"] == 0 or n_traces["bad"] == 0:
        return empty

    xg = x_good.sum(axis=0)
    ng = n_good.sum(axis=0)
    xb = x_bad.sum(axis=0)
    nb = n_bad.sum(axis=0)
    mask = (ng >= min_seg_per_bin) & (nb >= min_seg_per_bin)
    if int(mask.sum()) < min_bins:
        out = dict(empty)
        out["bins_used"] = int(mask.sum())
        return out

    p_good = np.divide(xg, ng, out=np.zeros_like(xg, dtype=float), where=ng > 0)
    p_bad = np.divide(xb, nb, out=np.zeros_like(xb, dtype=float), where=nb > 0)
    d_full = p_good - p_bad
    pt_good = np.divide(xg + 0.5, ng + 1, out=np.zeros_like(xg, dtype=float), where=ng >= 0)
    pt_bad = np.divide(xb + 0.5, nb + 1, out=np.zeros_like(xb, dtype=float), where=nb >= 0)
    v_full = pt_good * (1 - pt_good) / np.maximum(ng, 1) + pt_bad * (1 - pt_bad) / np.maximum(nb, 1)
    v_full = np.maximum(v_full, 1e-12)

    bins_idx = np.where(mask)[0]
    d = d_full[mask]
    v = v_full[mask]
    w = 1 / v
    w_sum = float(w.sum())
    dbar = float((w * d).sum() / w_sum)
    se = math.sqrt(1 / w_sum)
    z = dbar / se if se > 0 else 0.0
    q = float((w * (d - dbar) ** 2).sum())
    df = int(len(d) - 1)

    boot_d = _bootstrap_gaps(x_good[:, mask], n_good[:, mask], x_bad[:, mask], n_bad[:, mask], boot=boot, rng=rng)
    cov = np.cov(boot_d, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    # The trace bootstrap captures between-trace dependence, but observed
    # per-trace bin counts are fixed inside each replicate. Keep the analytic
    # binomial diagonal as a conservative within-bin floor rather than letting
    # sparse trace resamples imply unrealistically sharp contrasts.
    cov = cov + np.diag(v)
    sinv = _regularized_inverse(cov, v, n_traces["good"], n_traces["bad"])
    ones = np.ones(len(d))
    denom = float(ones @ sinv @ ones)
    if denom <= 0 or not math.isfinite(denom):
        sinv = np.diag(1 / np.maximum(np.diag(cov), v, 1e-10))
        denom = float(ones @ sinv @ ones)
    dbar_gls = float((ones @ sinv @ d) / denom)
    se_gls = math.sqrt(1 / denom)
    z_gls = dbar_gls / se_gls if se_gls > 0 else 0.0
    resid = d - dbar_gls
    q_gls = float(resid @ sinv @ resid)

    d_se = np.std(boot_d, axis=0, ddof=1)
    std = (d - dbar_gls) / np.sqrt(v)
    contrib = w * (d - dbar_gls) ** 2
    total_contrib = float(contrib.sum())
    thirds = {}
    for name, lo, hi in [("early", 0, bins // 3), ("mid", bins // 3, 2 * bins // 3), ("late", 2 * bins // 3, bins)]:
        part = contrib[(bins_idx >= lo) & (bins_idx < hi)].sum()
        thirds[name] = float(part / total_contrib) if total_contrib > 0 else 0.0

    d_pp = [None] * bins
    d_se_pp = [None] * bins
    std_profile = [None] * bins
    for pos, bin_idx in enumerate(bins_idx):
        d_pp[int(bin_idx)] = float(d[pos] * 100)
        d_se_pp[int(bin_idx)] = float(d_se[pos] * 100)
        std_profile[int(bin_idx)] = float(std[pos])

    return {
        "status": "tested",
        "n_good": n_traces["good"],
        "n_bad": n_traces["bad"],
        "bins_used": int(len(d)),
        "level_dbar_pp": dbar * 100,
        "level_z": z,
        "level_p": _two_sided_z_p(z),
        "level_dbar_robust_pp": dbar_gls * 100,
        "level_z_robust": z_gls,
        "level_p_robust": _two_sided_z_p(z_gls),
        "timing_Q": q,
        "timing_df": df,
        "timing_I2": _i2(q, df),
        "timing_p": _chi_p(q, df),
        "timing_Q_gls": q_gls,
        "timing_I2_robust": _i2(q_gls, df),
        "timing_p_robust": _chi_p(q_gls, df),
        "thirds_early": thirds["early"],
        "thirds_mid": thirds["mid"],
        "thirds_late": thirds["late"],
        "d_pp": d_pp,
        "d_se_pp": d_se_pp,
        "std_profile": std_profile,
    }


def _bh_qvalues(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(np.asarray(pvals, dtype=float))
    q = np.empty(n, dtype=float)
    running = 1.0
    for rank_idx in range(n - 1, -1, -1):
        idx = order[rank_idx]
        rank = rank_idx + 1
        running = min(running, pvals[idx] * n / rank)
        q[idx] = min(1.0, running)
    return q.tolist()


def _apply_multiplicity(rows: list[dict[str, Any]], bh_q: float) -> None:
    tested = [row for row in rows if row["status"] == "tested"]
    level_q = _bh_qvalues([row["level_p_robust"] for row in tested])
    timing_q = _bh_qvalues([row["timing_p_robust"] for row in tested])
    level_q_naive = _bh_qvalues([row["level_p"] for row in tested])
    timing_q_naive = _bh_qvalues([row["timing_p"] for row in tested])
    for row, lq, tq, lnq, tnq in zip(tested, level_q, timing_q, level_q_naive, timing_q_naive):
        row["level_q_robust"] = lq
        row["timing_q_robust"] = tq
        row["level_q"] = lnq
        row["timing_q"] = tnq
        level_sig = lq <= bh_q
        timing_sig = tq <= bh_q
        if level_sig and timing_sig:
            cls = "both"
        elif level_sig:
            cls = "level_only"
        elif timing_sig:
            cls = "timing_only"
        else:
            cls = "neither"
        row["class"] = cls
    for row in rows:
        if row["status"] != "tested":
            row["class"] = "insufficient"
            row["level_q_robust"] = None
            row["timing_q_robust"] = None
            row["level_q"] = None
            row["timing_q"] = None


def decompose(
    trace_bin_table: pl.DataFrame,
    *,
    bins: int = K,
    min_seg_per_bin: int = MIN_SEG_PER_BIN,
    min_bins: int = MIN_BINS,
    boot: int = B_BOOT,
    seed: int = 0,
    bh_q: float = BH_Q,
    quality_split: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Compute level and timing tests for every (model, domain, behavior) pair."""
    quality_split = quality_split or {}
    grouped: dict[tuple[str, str, str], dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = defaultdict(lambda: {"good": {}, "bad": {}})
    for row in trace_bin_table.iter_rows(named=True):
        key = (row["gen_model"], row["task_type"], row["behavior"])
        outcome = row["outcome_class"]
        trace_id = row["trace_id"]
        if outcome not in ("good", "bad"):
            continue
        bucket = grouped[key][outcome].setdefault(trace_id, (np.zeros(bins, dtype=float), np.zeros(bins, dtype=float)))
        idx = int(row["bin"])
        bucket[0][idx] += float(row["X"] or 0)
        bucket[1][idx] += float(row["n"] or 0)

    rng = np.random.default_rng(seed)
    rows = []
    for gen_model, task_type, behavior in sorted(grouped):
        stats = _decompose_pair(
            grouped[(gen_model, task_type, behavior)]["good"],
            grouped[(gen_model, task_type, behavior)]["bad"],
            bins=bins,
            min_seg_per_bin=min_seg_per_bin,
            min_bins=min_bins,
            boot=boot,
            rng=rng,
        )
        stats.update(
            {
                "gen_model": gen_model,
                "task_type": task_type,
                "behavior": behavior,
                "family": FAMILIES.get(behavior, "unknown"),
                "K": bins,
                "min_seg_per_bin": min_seg_per_bin,
                "min_bins": min_bins,
                "boot": boot,
                "bh_q": bh_q,
                "quality_threshold": quality_split.get(f"{gen_model}|{task_type}"),
            }
        )
        rows.append(stats)
    _apply_multiplicity(rows, bh_q)
    return rows


def _round_or_none(value: Any, digits: int) -> Any:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(v):
        return None
    return round(v, digits)


def _round_list(values: list[Any] | None, digits: int) -> list[Any]:
    return [_round_or_none(v, digits) for v in (values or [])]


def dashboard_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Shape decomposition rows into the compact GitHub Pages JSON."""
    if rows:
        first = rows[0]
        meta = {
            "K": int(first.get("K") or K),
            "min_seg_per_bin": int(first.get("min_seg_per_bin") or MIN_SEG_PER_BIN),
            "min_bins": int(first.get("min_bins") or MIN_BINS),
            "boot": int(first.get("boot") or B_BOOT),
            "bh_q": _round_or_none(first.get("bh_q", BH_Q), 3),
            "quality_split": {
                f"{row['gen_model']}|{row['task_type']}": _round_or_none(row.get("quality_threshold"), 4)
                for row in rows
                if row.get("quality_threshold") is not None
            },
        }
    else:
        meta = {"K": K, "min_seg_per_bin": MIN_SEG_PER_BIN, "min_bins": MIN_BINS, "boot": B_BOOT, "bh_q": BH_Q, "quality_split": {}}

    pairs = []
    for row in sorted(rows, key=lambda r: (r["gen_model"], r["task_type"], r["behavior"])):
        pairs.append(
            {
                "gen_model": row["gen_model"],
                "task_type": row["task_type"],
                "behavior": row["behavior"],
                "family": row.get("family"),
                "status": row.get("status"),
                "class": row.get("class"),
                "bins_used": int(row.get("bins_used") or 0),
                "n_traces": {"good": int(row.get("n_good") or 0), "bad": int(row.get("n_bad") or 0)},
                "level": {
                    "dbar_pp": _round_or_none(row.get("level_dbar_pp"), 2),
                    "z": _round_or_none(row.get("level_z"), 2),
                    "p": _round_or_none(row.get("level_p"), 4),
                    "q": _round_or_none(row.get("level_q"), 4),
                    "dbar_robust_pp": _round_or_none(row.get("level_dbar_robust_pp"), 2),
                    "z_robust": _round_or_none(row.get("level_z_robust"), 2),
                    "p_robust": _round_or_none(row.get("level_p_robust"), 4),
                    "q_robust": _round_or_none(row.get("level_q_robust"), 4),
                },
                "timing": {
                    "Q": _round_or_none(row.get("timing_Q"), 2),
                    "df": int(row.get("timing_df") or 0),
                    "I2": _round_or_none(row.get("timing_I2"), 2),
                    "p": _round_or_none(row.get("timing_p"), 4),
                    "q": _round_or_none(row.get("timing_q"), 4),
                    "Q_gls": _round_or_none(row.get("timing_Q_gls"), 2),
                    "I2_robust": _round_or_none(row.get("timing_I2_robust"), 2),
                    "p_robust": _round_or_none(row.get("timing_p_robust"), 4),
                    "q_robust": _round_or_none(row.get("timing_q_robust"), 4),
                },
                "d_pp": _round_list(row.get("d_pp"), 2),
                "d_se_pp": _round_list(row.get("d_se_pp"), 2),
                "std_profile": _round_list(row.get("std_profile"), 2),
                "thirds": {
                    "early": _round_or_none(row.get("thirds_early"), 2),
                    "mid": _round_or_none(row.get("thirds_mid"), 2),
                    "late": _round_or_none(row.get("thirds_late"), 2),
                },
            }
        )
    return {"meta": meta, "pairs": pairs}


def write_dashboard_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dashboard_payload(rows), ensure_ascii=False, separators=(",", ":")))


def rows_from_parquet(path: str | Path) -> list[dict[str, Any]]:
    return pl.read_parquet(path).to_dicts()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackB", default="data/judge/prod/trackB_full__google_gemma-4-31B-it.parquet")
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--grades", nargs="*", default=["data/perf/success_grades.parquet", "data/perf/code_grades.parquet"])
    ap.add_argument("--quality", default="data/judge/prod/quality__google_gemma-4-31B-it.parquet")
    ap.add_argument("--out", default="data/analysis/timing_level.parquet")
    ap.add_argument("--boot", type=int, default=B_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=K)
    ap.add_argument("--min-seg-per-bin", type=int, default=MIN_SEG_PER_BIN)
    ap.add_argument("--min-bins", type=int, default=MIN_BINS)
    args = ap.parse_args()

    traces = _glob_parquet(args.traces_glob)
    track_b = pl.read_parquet(args.trackB)
    grade_paths = list(args.grades or [])
    if args.quality:
        grade_paths.append(args.quality)
    grades = coalesced_grades(grade_paths)
    with_outcomes, quality_split = add_outcome_classes(traces, grades)
    trace_bins = compute_trace_bin_table(track_b, with_outcomes, bins=args.bins)
    rows = decompose(
        trace_bins,
        bins=args.bins,
        min_seg_per_bin=args.min_seg_per_bin,
        min_bins=args.min_bins,
        boot=args.boot,
        seed=args.seed,
        quality_split=quality_split,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out)
    print(f"timing-level rows={len(rows)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
