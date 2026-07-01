"""Export compact JSON artifacts for the GitHub Pages dashboard.

The research dataset stays in parquet under data/. This script builds small,
browser-friendly summaries under docs/data/ so the dashboard can be hosted by
GitHub Pages without asking the browser to read parquet.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from src.segment.thinkarm_vendored import process_response_to_sentences


KIM = [
    "Question_and_Answering",
    "Perspective_Shift",
    "Conflict_of_Perspectives",
    "Reconciliation",
]
GANDHI = ["verification", "backtracking", "subgoal", "backward_chaining"]
BEHAVIORS = GANDHI + KIM
DOMAINS = ["math", "code", "gpqa", "planning", "moral", "idea"]
OUTCOME_GROUPS = {
    "all": ["solved", "failed", "high_quality", "low_quality", "unknown"],
    "positive": ["solved", "high_quality"],
    "negative": ["failed", "low_quality"],
    "solved": ["solved"],
    "failed": ["failed"],
    "high_quality": ["high_quality"],
    "low_quality": ["low_quality"],
    "unknown": ["unknown"],
}


def _read_many(paths: list[str], *, required: bool = True) -> pl.DataFrame:
    existing = [p for p in paths if Path(p).exists()]
    if not existing and required:
        raise FileNotFoundError(f"no input files found: {paths}")
    if not existing:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(p) for p in existing], how="diagonal_relaxed")


def _glob_parquet(pattern: str) -> pl.DataFrame:
    paths = sorted(glob.glob(pattern))
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


def add_outcomes(traces: pl.DataFrame, grades: pl.DataFrame) -> pl.DataFrame:
    df = traces.join(grades, on="trace_id", how="left", suffix="_grade")
    if "task_type_grade" in df.columns:
        df = df.drop("task_type_grade")
    med = (
        df.group_by("task_type")
        .agg(pl.col("quality_score").median().alias("_quality_median"))
    )
    df = df.join(med, on="task_type", how="left")
    return df.with_columns(
        pl.when(pl.col("success").is_not_null())
        .then(pl.when(pl.col("success") >= 1).then(pl.lit("solved")).otherwise(pl.lit("failed")))
        .when(pl.col("quality_score").is_not_null())
        .then(
            pl.when(pl.col("quality_score") >= pl.col("_quality_median"))
            .then(pl.lit("high_quality"))
            .otherwise(pl.lit("low_quality"))
        )
        .otherwise(pl.lit("unknown"))
        .alias("outcome")
    )


def _safe(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, dict):
        return {k: _safe(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_safe(val) for val in v]
    return v


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(data), indent=2, ensure_ascii=False))


def clipped(text: str | None, max_chars: int) -> dict[str, Any]:
    text = text or ""
    was = len(text) > max_chars
    if was:
        half = max_chars // 2
        text = text[:half] + "\n\n...[middle clipped for dashboard]...\n\n" + text[-half:]
    return {
        "text": text,
        "chars": len(text),
        "truncated": was,
        "tokens_est": round(len(text) / 4),
    }


def segment_annotations(
    full_text: str | None,
    labels_by_idx: dict[int, list[str]],
    *,
    max_items: int = 80,
    max_chars: int = 260,
) -> list[dict[str, Any]]:
    """Reconstruct sampled sentence labels for raw-text inspection.

    The judge output stores behavior labels by segment index. The dashboard export
    keeps this compact by reconstructing sentence text only for sampled traces.
    """
    if not labels_by_idx:
        return []
    segments = process_response_to_sentences(full_text or "", apply_merging=True)
    n = len(segments)
    rows = []
    for idx, seg in enumerate(segments):
        behaviors = labels_by_idx.get(idx)
        if not behaviors:
            continue
        text = seg.get("sentence") or ""
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        rows.append(
            {
                "seg_idx": idx,
                "norm_pos": (idx / (n - 1)) if n > 1 else 0.0,
                "section_type": seg.get("type"),
                "text": text,
                "behaviors": behaviors,
            }
        )
        if len(rows) >= max_items:
            break
    return rows


def export_manifest(traces: pl.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    models = (
        traces.group_by(["gen_model", "gen_model_id"])
        .agg(pl.len().alias("n_traces"))
        .sort("gen_model")
        .to_dicts()
    )
    domains = (
        traces.group_by("task_type")
        .agg(pl.len().alias("n_traces"))
        .sort("task_type")
        .to_dicts()
    )
    write_json(
        out_dir / "manifest.json",
        {
            "project": "Thought Atlas",
            "experiment": "society-task-exp2",
            "source_paper": "https://arxiv.org/abs/2601.10825",
            "bins": args.bins,
            "samples_per_cell": args.samples_per_cell,
            "max_text_chars": args.max_text_chars,
            "models": models,
            "domains": domains,
            "behaviors": [
                {"key": b, "family": "cognitive" if b in GANDHI else "conversational"}
                for b in BEHAVIORS
            ],
            "notes": [
                "Raw full-fidelity traces remain in data/traces/*.parquet.",
                "Dashboard trace text is sampled and clipped for browser performance.",
                "Split thinking/answer token counts are estimates; n_new_tokens is the model-reported total generation length.",
            ],
        },
    )


def export_summary(
    traces: pl.DataFrame, track_a: pl.DataFrame, with_outcomes: pl.DataFrame, out_dir: Path
) -> None:
    count_cols = [c for c in BEHAVIORS if c in track_a.columns]
    counts = track_a.with_columns(
        pl.sum_horizontal([pl.col(c) for c in count_cols]).alias("behavior_total")
    )
    df = with_outcomes.join(counts, on="trace_id", how="left")
    summary = (
        df.group_by(["gen_model", "task_type"])
        .agg(
            pl.len().alias("n_traces"),
            pl.col("completed").mean().round(4).alias("completed_rate"),
            (pl.col("answer_text").str.len_chars() > 0).mean().round(4).alias("has_answer_rate"),
            pl.col("n_new_tokens").median().round(1).alias("median_new_tokens"),
            pl.col("n_new_tokens").mean().round(1).alias("mean_new_tokens"),
            pl.col("success").mean().round(4).alias("success_rate"),
            pl.col("quality_score").mean().round(4).alias("quality_score"),
            pl.col("behavior_total").mean().round(3).alias("mean_behavior_count"),
        )
        .sort(["gen_model", "task_type"])
    )
    behavior = []
    for b in count_cols:
        part = (
            df.group_by(["gen_model", "task_type"])
            .agg(pl.col(b).mean().round(4).alias("mean_count"))
            .with_columns(pl.lit(b).alias("behavior"))
            .select(["gen_model", "task_type", "behavior", "mean_count"])
        )
        behavior.append(part)
    write_json(
        out_dir / "summary.json",
        {
            "cells": summary.to_dicts(),
            "behavior_counts": pl.concat(behavior).sort(["gen_model", "task_type", "behavior"]).to_dicts()
            if behavior
            else [],
        },
    )


def _track_a_aggregate(df: pl.DataFrame, value_col: str, *, key_col: str, key_value: str) -> pl.DataFrame:
    part = (
        df.group_by(["gen_model", "task_type"])
        .agg(
            pl.len().alias("n_traces"),
            pl.col(value_col).sum().round(4).alias("total_count"),
            pl.col(value_col).cast(pl.Float64).mean().alias("mean_count"),
            pl.col(value_col).cast(pl.Float64).std().fill_null(0).alias("_sd_count"),
            (pl.col(value_col) > 0).cast(pl.Float64).mean().alias("presence_rate"),
        )
        .with_columns(
            (pl.col("_sd_count") / pl.col("n_traces").cast(pl.Float64).sqrt()).alias("_se_count"),
            ((pl.col("presence_rate") * (1 - pl.col("presence_rate")) / pl.col("n_traces")).sqrt()).alias("_se_presence"),
        )
        .with_columns(
            pl.lit(key_value).alias(key_col),
            pl.max_horizontal(pl.lit(0.0), pl.col("mean_count") - 1.96 * pl.col("_se_count")).round(4).alias("lower_count"),
            (pl.col("mean_count") + 1.96 * pl.col("_se_count")).round(4).alias("upper_count"),
            pl.max_horizontal(pl.lit(0.0), pl.col("presence_rate") - 1.96 * pl.col("_se_presence")).round(4).alias("lower_presence"),
            pl.min_horizontal(pl.lit(1.0), pl.col("presence_rate") + 1.96 * pl.col("_se_presence")).round(4).alias("upper_presence"),
            pl.col("mean_count").round(4),
            pl.col("presence_rate").round(4),
        )
    )
    return part


def export_track_a(track_a: pl.DataFrame, with_outcomes: pl.DataFrame, out_dir: Path) -> None:
    """Export whole-trace Track A counts for non-temporal dashboard views."""
    count_cols = [c for c in BEHAVIORS if c in track_a.columns]
    meta = with_outcomes.select(["trace_id", "gen_model", "task_type", "outcome"])
    df = track_a.select(["trace_id"] + count_cols).join(meta, on="trace_id", how="inner")
    cells: list[pl.DataFrame] = []
    families: list[pl.DataFrame] = []
    family_specs = {
        "cognitive": [b for b in GANDHI if b in count_cols],
        "conversational": [b for b in KIM if b in count_cols],
    }

    for outcome_group, outcomes in OUTCOME_GROUPS.items():
        sub = df.filter(pl.col("outcome").is_in(outcomes))
        if sub.height == 0:
            continue
        for behavior in count_cols:
            cells.append(
                _track_a_aggregate(sub, behavior, key_col="behavior", key_value=behavior)
                .with_columns(pl.lit(outcome_group).alias("outcome_group"))
                .select(
                    [
                        "gen_model",
                        "task_type",
                        "outcome_group",
                        "behavior",
                        "n_traces",
                        "total_count",
                        "mean_count",
                        "lower_count",
                        "upper_count",
                        "presence_rate",
                        "lower_presence",
                        "upper_presence",
                    ]
                )
            )
        for family, behaviors in family_specs.items():
            if not behaviors:
                continue
            family_df = sub.with_columns(pl.sum_horizontal([pl.col(b) for b in behaviors]).alias("_family_count"))
            families.append(
                _track_a_aggregate(family_df, "_family_count", key_col="family", key_value=family)
                .with_columns(pl.lit(outcome_group).alias("outcome_group"))
                .select(
                    [
                        "gen_model",
                        "task_type",
                        "outcome_group",
                        "family",
                        "n_traces",
                        "total_count",
                        "mean_count",
                        "lower_count",
                        "upper_count",
                        "presence_rate",
                        "lower_presence",
                        "upper_presence",
                    ]
                )
            )

    write_json(
        out_dir / "trackA.json",
        {
            "description": "Track A whole-trace behavior counts, stratified by model, domain, and outcome group.",
            "cells": pl.concat(cells).sort(["gen_model", "task_type", "outcome_group", "behavior"]).to_dicts() if cells else [],
            "families": pl.concat(families).sort(["gen_model", "task_type", "outcome_group", "family"]).to_dicts() if families else [],
        },
    )


def export_heartbeat(
    track_b: pl.DataFrame, with_outcomes: pl.DataFrame, out_dir: Path, bins: int
) -> None:
    meta = with_outcomes.select(["trace_id", "gen_model", "task_type", "outcome", "completed"])
    binned = (
        track_b.join(meta, on="trace_id", how="inner")
        .with_columns((pl.col("norm_pos") * (bins - 1)).round().cast(pl.Int32).alias("bin"))
    )
    parts = []
    for behavior in [b for b in BEHAVIORS if b in binned.columns]:
        parts.append(
            binned.group_by(["gen_model", "task_type", "outcome", "bin"])
            .agg(
                pl.col(behavior).mean().round(5).alias("freq"),
                pl.len().alias("n_segments"),
                pl.col("trace_id").n_unique().alias("n_traces"),
            )
            .with_columns(pl.lit(behavior).alias("behavior"))
            .select(["gen_model", "task_type", "outcome", "behavior", "bin", "freq", "n_segments", "n_traces"])
        )
    heartbeat = pl.concat(parts).sort(["gen_model", "task_type", "outcome", "behavior", "bin"])
    boundaries = (
        track_b.filter(pl.col("section_type") == "answer")
        .group_by("trace_id")
        .agg(pl.col("norm_pos").min().alias("answer_start"))
        .join(meta, on="trace_id", how="inner")
        .group_by(["gen_model", "task_type", "outcome"])
        .agg(
            pl.col("answer_start").quantile(0.25).round(5).alias("q25"),
            pl.col("answer_start").median().round(5).alias("median"),
            pl.col("answer_start").quantile(0.75).round(5).alias("q75"),
            pl.len().alias("n_traces"),
        )
        .sort(["gen_model", "task_type", "outcome"])
    )
    write_json(
        out_dir / "heartbeat.json",
        {"bins": bins, "curves": heartbeat.to_dicts(), "answer_boundaries": boundaries.to_dicts()},
    )


def export_trace_samples(
    with_outcomes: pl.DataFrame,
    track_a: pl.DataFrame,
    track_b: pl.DataFrame,
    out_dir: Path,
    samples_per_cell: int,
    max_text_chars: int,
) -> None:
    counts = {
        r["trace_id"]: {b: r.get(b, 0) for b in BEHAVIORS}
        for r in track_a.select(["trace_id"] + [b for b in BEHAVIORS if b in track_a.columns]).to_dicts()
    }
    sample_records = []
    for model in sorted(with_outcomes["gen_model"].unique().to_list()):
        for domain in DOMAINS:
            sub = with_outcomes.filter((pl.col("gen_model") == model) & (pl.col("task_type") == domain))
            if sub.height == 0:
                continue
            sample = sub.sort("trace_id").head(samples_per_cell)
            sample_records.extend(sample.iter_rows(named=True))

    label_cols = [b for b in BEHAVIORS if b in track_b.columns]
    sample_ids = [r["trace_id"] for r in sample_records]
    labels_by_trace: dict[str, dict[int, list[str]]] = {}
    if sample_ids and label_cols and track_b.height:
        label_rows = (
            track_b.filter(pl.col("trace_id").is_in(sample_ids))
            .select(["trace_id", "seg_idx"] + label_cols)
            .to_dicts()
        )
        for row in label_rows:
            labels = [b for b in label_cols if row.get(b)]
            if labels:
                labels_by_trace.setdefault(row["trace_id"], {})[int(row["seg_idx"])] = labels

    rows = []
    for r in sample_records:
        think = clipped(r.get("think_text"), max_text_chars)
        answer = clipped(r.get("answer_text"), max_text_chars)
        prompt = clipped(r.get("prompt"), max_text_chars // 2)
        rows.append(
            {
                "trace_id": r["trace_id"],
                "gen_model": r["gen_model"],
                "gen_model_id": r.get("gen_model_id"),
                "task_type": r["task_type"],
                "outcome": r.get("outcome"),
                "instance_id": r.get("instance_id"),
                "completed": r.get("completed"),
                "finish_reason": r.get("finish_reason"),
                "failure_mode": r.get("failure_mode"),
                "n_new_tokens": r.get("n_new_tokens"),
                "success": r.get("success"),
                "quality_score": r.get("quality_score"),
                "prompt": prompt,
                "thinking": think,
                "answer": answer,
                "behavior_counts": counts.get(r["trace_id"], {}),
                "annotations": segment_annotations(r.get("full_text"), labels_by_trace.get(r["trace_id"], {})),
            }
        )
    write_json(out_dir / "trace_samples.json", {"traces": rows})


def export_distances(distance_dir: Path, out_dir: Path) -> None:
    payload: dict[str, Any] = {"shape": [], "magnitude": [], "noise": {}}
    for kind, key in [("shape", "shape"), ("mag", "magnitude")]:
        csv_path = distance_dir / f"model_distance_{kind}.csv"
        if csv_path.exists():
            payload[key] = pl.read_csv(csv_path).to_dicts()
        noise_path = distance_dir / f"noise_floor_{kind}.json"
        if noise_path.exists():
            payload["noise"][key] = json.loads(noise_path.read_text())
    write_json(out_dir / "distance.json", payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--trackA", default="data/judge/prod/trackA_counts__google_gemma-4-31B-it.parquet")
    ap.add_argument("--trackB", default="data/judge/prod/trackB_full__google_gemma-4-31B-it.parquet")
    ap.add_argument("--grades", nargs="*", default=["data/perf/success_grades.parquet", "data/perf/code_grades.parquet"])
    ap.add_argument("--quality", default="data/judge/prod/quality__google_gemma-4-31B-it.parquet")
    ap.add_argument("--distance-dir", default="data/analysis/cross_model")
    ap.add_argument("--out-dir", default="docs/data")
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--samples-per-cell", type=int, default=12)
    ap.add_argument("--max-text-chars", type=int, default=12000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    traces = _glob_parquet(args.traces_glob)
    track_a = _read_many([args.trackA])
    track_b = _read_many([args.trackB])
    grade_paths = list(args.grades or [])
    if args.quality:
        grade_paths.append(args.quality)
    grades = coalesced_grades(grade_paths)
    with_outcomes = add_outcomes(traces, grades)

    export_manifest(traces, out_dir, args)
    export_summary(traces, track_a, with_outcomes, out_dir)
    export_track_a(track_a, with_outcomes, out_dir)
    export_heartbeat(track_b, with_outcomes, out_dir, args.bins)
    export_trace_samples(with_outcomes, track_a, track_b, out_dir, args.samples_per_cell, args.max_text_chars)
    export_distances(Path(args.distance_dir), out_dir)
    print(f"dashboard data -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
