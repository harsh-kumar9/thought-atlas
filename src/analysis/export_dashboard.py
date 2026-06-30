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


KIM = [
    "Question_and_Answering",
    "Perspective_Shift",
    "Conflict_of_Perspectives",
    "Reconciliation",
]
GANDHI = ["verification", "backtracking", "subgoal", "backward_chaining"]
BEHAVIORS = GANDHI + KIM
DOMAINS = ["math", "code", "gpqa", "planning", "moral", "idea"]


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
    write_json(out_dir / "heartbeat.json", {"bins": bins, "curves": heartbeat.to_dicts()})


def export_trace_samples(
    with_outcomes: pl.DataFrame,
    track_a: pl.DataFrame,
    out_dir: Path,
    samples_per_cell: int,
    max_text_chars: int,
) -> None:
    counts = {
        r["trace_id"]: {b: r.get(b, 0) for b in BEHAVIORS}
        for r in track_a.select(["trace_id"] + [b for b in BEHAVIORS if b in track_a.columns]).to_dicts()
    }
    rows = []
    for model in sorted(with_outcomes["gen_model"].unique().to_list()):
        for domain in DOMAINS:
            sub = with_outcomes.filter((pl.col("gen_model") == model) & (pl.col("task_type") == domain))
            if sub.height == 0:
                continue
            sample = sub.sort("trace_id").head(samples_per_cell)
            for r in sample.iter_rows(named=True):
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
    export_heartbeat(track_b, with_outcomes, out_dir, args.bins)
    export_trace_samples(with_outcomes, track_a, out_dir, args.samples_per_cell, args.max_text_chars)
    export_distances(Path(args.distance_dir), out_dir)
    print(f"dashboard data -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
