"""Phase-1 data audit and readiness manifest for paper analyses."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl

from src.analysis.paper.constants import AUDIT_FILENAMES, BEHAVIORS
from src.analysis.paper.io import (
    assert_parquet_files,
    build_full_segment_table,
    build_paired_context_table,
    expand_parquet_paths,
    scan_track_b,
    write_analysis_manifest,
)


def _collect(frame: pl.LazyFrame) -> pl.DataFrame:
    return frame.collect(engine="streaming")


def _safe_rate(numerator: pl.Expr, denominator: pl.Expr, name: str) -> pl.Expr:
    return (
        pl.when(denominator > 0)
        .then(numerator.cast(pl.Float64) / denominator)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias(name)
    )


def trace_coverage(trace_index: pl.DataFrame) -> pl.DataFrame:
    return (
        trace_index.group_by(["gen_model", "task_type"])
        .agg(
            pl.len().alias("n_traces"),
            pl.col("instance_id").n_unique().alias("n_instances"),
            pl.col("seed").n_unique().alias("n_sample_seeds"),
            pl.col("completed").fill_null(False).sum().alias("n_completed"),
            pl.col("n_new_tokens").min().alias("tokens_min"),
            pl.col("n_new_tokens").median().alias("tokens_median"),
            pl.col("n_new_tokens").max().alias("tokens_max"),
            pl.col("analysis_source").drop_nulls().unique().sort().str.join("|").alias("analysis_sources"),
            pl.col("configured_token_budget").drop_nulls().unique().sort().cast(pl.String).str.join("|").alias("configured_token_budgets"),
        )
        .with_columns(_safe_rate(pl.col("n_completed"), pl.col("n_traces"), "completion_rate"))
        .sort(["gen_model", "task_type"])
    )


def completion_by_cell(
    trace_index: pl.DataFrame, full_segments: pl.LazyFrame
) -> pl.DataFrame:
    per_trace_segments = _collect(
        full_segments.group_by("trace_id").agg(
            pl.col("n_segments").max().alias("n_segments"),
            pl.len().alias("n_segment_rows"),
        )
    )
    base = trace_index.join(per_trace_segments, on="trace_id", how="left", validate="1:1")
    return (
        base.group_by(["gen_model", "task_type"])
        .agg(
            pl.len().alias("n_traces"),
            pl.col("completed").fill_null(False).sum().alias("n_completed"),
            pl.col("n_new_tokens").min().alias("tokens_min"),
            pl.col("n_new_tokens").quantile(0.25).alias("tokens_q25"),
            pl.col("n_new_tokens").median().alias("tokens_median"),
            pl.col("n_new_tokens").quantile(0.75).alias("tokens_q75"),
            pl.col("n_new_tokens").max().alias("tokens_max"),
            pl.col("n_segments").min().alias("segments_min"),
            pl.col("n_segments").median().alias("segments_median"),
            pl.col("n_segments").max().alias("segments_max"),
            (pl.col("failure_mode") != "completed").fill_null(True).sum().alias("n_noncompleted_failure_mode"),
        )
        .with_columns(_safe_rate(pl.col("n_completed"), pl.col("n_traces"), "completion_rate"))
        .sort(["gen_model", "task_type"])
    )


def _segment_coverage_one(segments: pl.LazyFrame, context: str) -> pl.DataFrame:
    schema = segments.collect_schema()
    null_expr = pl.any_horizontal(
        [pl.col(behavior).is_null() for behavior in BEHAVIORS if behavior in schema]
    )
    valid_expr = pl.col("kim_parsed").fill_null(False) & pl.col(
        "gandhi_parsed"
    ).fill_null(False)
    truncated = (
        pl.col("judge_context_truncated").fill_null(False)
        if "judge_context_truncated" in schema
        else pl.lit(False)
    )
    result = _collect(
        segments.group_by(["gen_model", "task_type"])
        .agg(
            pl.len().alias("n_rows"),
            pl.col("trace_id").n_unique().alias("n_traces"),
            valid_expr.sum().alias("n_parsed_rows"),
            (~valid_expr).sum().alias("n_parse_failure_rows"),
            null_expr.sum().alias("n_rows_with_null_behavior"),
            truncated.sum().alias("n_truncated_context_rows"),
            pl.col("n_segments").median().alias("median_n_segments"),
        )
        .with_columns(pl.lit(context).alias("context_mode"))
    )
    return result.select(
        "context_mode",
        "gen_model",
        "task_type",
        "n_rows",
        "n_traces",
        "n_parsed_rows",
        "n_parse_failure_rows",
        "n_rows_with_null_behavior",
        "n_truncated_context_rows",
        "median_n_segments",
    ).sort(["context_mode", "gen_model", "task_type"])


def segment_coverage(full: pl.LazyFrame, isolated: pl.LazyFrame) -> pl.DataFrame:
    return pl.concat(
        [_segment_coverage_one(full, "full"), _segment_coverage_one(isolated, "isolated")],
        how="diagonal_relaxed",
    ).sort(["context_mode", "gen_model", "task_type"])


def _key_duplicates(segments: pl.LazyFrame) -> int:
    return int(
        _collect(
            segments.group_by(["trace_id", "seg_idx"])
            .len()
            .filter(pl.col("len") > 1)
            .select(pl.len().alias("n"))
        )["n"][0]
    )


def context_pair_coverage(full: pl.LazyFrame, isolated: pl.LazyFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    keys = ["trace_id", "seg_idx"]
    cell = ["gen_model", "task_type"]
    full_keys = full.select(keys + cell)
    isolated_keys = isolated.select(keys + cell)
    paired = full_keys.join(isolated_keys.select(keys), on=keys, how="inner", validate="1:1")
    full_only = full_keys.join(isolated_keys.select(keys), on=keys, how="anti")
    isolated_only = isolated_keys.join(full_keys.select(keys), on=keys, how="anti")

    def counts(frame: pl.LazyFrame, name: str) -> pl.DataFrame:
        return _collect(frame.group_by(cell).agg(pl.len().alias(name)))

    coverage = counts(full_keys, "full_rows")
    for frame, name in [
        (isolated_keys, "isolated_rows"),
        (paired, "paired_rows"),
        (full_only, "full_only_rows"),
        (isolated_only, "isolated_only_rows"),
    ]:
        coverage = coverage.join(counts(frame, name), on=cell, how="full", coalesce=True)
    count_columns = [
        "full_rows",
        "isolated_rows",
        "paired_rows",
        "full_only_rows",
        "isolated_only_rows",
    ]
    coverage = coverage.with_columns(
        [pl.col(column).fill_null(0).cast(pl.Int64) for column in count_columns]
    )

    paired_labels = build_paired_context_table(full, isolated)
    paired_schema = paired_labels.collect_schema()
    diagnostics: list[pl.Expr] = []
    for prefix in ["full", "isolated"]:
        for behavior in BEHAVIORS:
            column = f"{prefix}__{behavior}"
            if column in paired_schema:
                diagnostics.append(pl.col(column).is_null().sum().alias(f"null__{column}"))
        for parser in ["kim_parsed", "gandhi_parsed"]:
            column = f"{prefix}__{parser}"
            if column in paired_schema:
                diagnostics.append((~pl.col(column).fill_null(False)).sum().alias(f"fail__{column}"))
    if diagnostics:
        diag = _collect(paired_labels.group_by(cell).agg(diagnostics))
        coverage = coverage.join(diag, on=cell, how="left")

    totals = {
        "full_rows": int(coverage["full_rows"].sum()),
        "isolated_rows": int(coverage["isolated_rows"].sum()),
        "paired_rows": int(coverage["paired_rows"].sum()),
        "full_only_rows": int(coverage["full_only_rows"].sum()),
        "isolated_only_rows": int(coverage["isolated_only_rows"].sum()),
        "full_duplicate_keys": _key_duplicates(full),
        "isolated_duplicate_keys": _key_duplicates(isolated),
    }
    return coverage.sort(cell), totals


def outcome_coverage(outcomes: pl.DataFrame) -> pl.DataFrame:
    return (
        outcomes.group_by(
            [
                "gen_model",
                "task_type",
                "outcome_name",
                "outcome_type",
                "outcome_direction",
            ]
        )
        .agg(
            pl.len().alias("n_traces"),
            pl.col("outcome_available").sum().alias("n_outcome_available"),
            pl.col("outcome_raw").min().alias("outcome_raw_min"),
            pl.col("outcome_raw").median().alias("outcome_raw_median"),
            pl.col("outcome_raw").max().alias("outcome_raw_max"),
        )
        .with_columns(
            _safe_rate(
                pl.col("n_outcome_available"), pl.col("n_traces"), "outcome_coverage_rate"
            )
        )
        .sort(["gen_model", "task_type"])
    )


def analysis_source_by_model(trace_index: pl.DataFrame) -> pl.DataFrame:
    return (
        trace_index.group_by("gen_model")
        .agg(
            pl.len().alias("n_traces"),
            pl.col("analysis_source").drop_nulls().unique().sort().str.join("|").alias("analysis_source"),
            pl.col("configured_token_budget").drop_nulls().unique().sort().cast(pl.String).str.join("|").alias("configured_token_budget"),
            pl.col("model_family").drop_nulls().unique().sort().str.join("|").alias("model_family"),
            pl.col("seed").n_unique().alias("n_sample_seeds"),
        )
        .sort("gen_model")
    )


def run_upstream_strict_audit(paths: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/audit_pipeline.py",
        "--tasks-dir",
        str(paths["tasks_dir"]),
        "--traces-glob",
        str(paths["traces_glob"]),
        "--extractions-glob",
        str(paths["answer_extractions"]),
        "--grades-glob",
        "data/v2/perf/*_grades.parquet",
        "--quality-glob",
        str(paths["quality"]),
        "--judge-glob",
        "data/v2/judge/trackB_*__google_gemma-4-31B-it.parquet",
        "--strict-v2",
    ]
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        report = {"ok": False, "issues": ["strict audit did not emit valid JSON"]}
    return {
        "ok": bool(report.get("ok")) and result.returncode == 0,
        "returncode": result.returncode,
        "issues": report.get("issues", []),
        "report": report,
        "stderr": result.stderr.strip(),
        "command": command,
    }


def adjudicate_upstream_audit(
    upstream: Mapping[str, Any], accepted_exceptions: Iterable[str]
) -> dict[str, Any]:
    """Apply only exact, version-controlled exception strings to a strict audit.

    An empty list preserves the literal strict-v2 result. This function does not
    infer or broaden exceptions; every emitted issue must be listed verbatim.
    """
    result = dict(upstream)
    accepted = set(accepted_exceptions)
    issues = [str(issue) for issue in result.get("issues", [])]
    unaccepted = [issue for issue in issues if issue not in accepted]
    result["strict_ok"] = bool(result.get("ok"))
    result["unaccepted_issues"] = unaccepted
    result["effective_ok"] = bool(result.get("ok")) or bool(issues and not unaccepted)
    return result


def _flag(
    flag_id: str,
    severity: str,
    blocking: bool,
    detail: str,
    *,
    observed: Any = None,
    expected: Any = None,
) -> dict[str, Any]:
    return {
        "flag_id": flag_id,
        "severity": severity,
        "blocking": blocking,
        "observed": None if observed is None else str(observed),
        "expected": None if expected is None else str(expected),
        "detail": detail,
    }


def build_quality_flags(
    trace_index: pl.DataFrame,
    segment_table: pl.DataFrame,
    pair_totals: Mapping[str, int],
    outcome_table: pl.DataFrame,
    upstream: Mapping[str, Any],
    model_metadata: Mapping[str, Mapping[str, Any]],
) -> pl.DataFrame:
    flags: list[dict[str, Any]] = []
    effective_ok = bool(upstream.get("effective_ok", upstream["ok"]))
    strict_ok = bool(upstream.get("strict_ok", upstream["ok"]))
    flags.append(
        _flag(
            "upstream_strict_v2",
            "info" if strict_ok else ("warning" if effective_ok else "error"),
            not effective_ok,
            (
                "The literal strict-v2 audit failed, but every issue is covered by the version-controlled exception list."
                if effective_ok and not strict_ok
                else "Scientific stages remain gated until the repository strict-v2 audit passes or a written exception policy is approved."
            ),
            observed=strict_ok,
            expected=True,
        )
    )
    observed_models = set(trace_index["gen_model"].unique().to_list())
    configured_models = set(model_metadata)
    missing_models = sorted(configured_models - observed_models)
    if missing_models:
        flags.append(
            _flag(
                "configured_models_absent",
                "warning",
                False,
                "Configured comparisons are intentions only; absent cells are not fabricated.",
                observed=missing_models,
                expected=sorted(configured_models),
            )
        )

    anchor_source = trace_index.filter(pl.col("gen_model") == "anchor")["analysis_source"].unique().to_list()
    flags.append(
        _flag(
            "anchor_analysis_channel",
            "warning",
            False,
            "The non-reasoning anchor is analyzed from answer_text and is channel-confounded with reasoning models.",
            observed=anchor_source,
            expected="answer_text",
        )
    )
    reasoning_sources = (
        trace_index.filter(pl.col("gen_model") != "anchor")["analysis_source"].unique().to_list()
    )
    flags.append(
        _flag(
            "reasoning_analysis_channel",
            "info",
            False,
            "Reasoning-model analysis uses think_text/reasoning_text_for_analysis.",
            observed=reasoning_sources,
            expected="think_text",
        )
    )
    budgets = {
        row["gen_model"]: row["configured_token_budget"]
        for row in analysis_source_by_model(trace_index).iter_rows(named=True)
    }
    flags.append(
        _flag(
            "unequal_generation_budgets",
            "warning",
            False,
            "Anchor and reasoning models have different configured token ceilings; cross-channel comparisons need matched-budget sensitivity.",
            observed=budgets,
            expected="matched budget/channel for causal comparison",
        )
    )
    qwen4 = trace_index.filter(pl.col("gen_model") == "qwen35_4b")
    qwen4_rate = float(qwen4["completed"].fill_null(False).mean()) if qwen4.height else math.nan
    flags.append(
        _flag(
            "qwen35_4b_completion_selection",
            "warning",
            False,
            "Completed-only Qwen-4B results are selection-conditioned and require all-valid/censoring sensitivity.",
            observed=None if math.isnan(qwen4_rate) else round(qwen4_rate, 6),
            expected="report completed-only and censoring sensitivities",
        )
    )
    seed_count = trace_index["seed"].n_unique()
    flags.append(
        _flag(
            "single_sample_seed",
            "warning",
            False,
            "Only one sampled generation is available per model-prompt condition; uncertainty cannot estimate within-prompt decoding variation.",
            observed=seed_count,
            expected=">1 for multi-sample robustness",
        )
    )
    flags.extend(
        [
            _flag(
                "posthoc_extension_domains",
                "warning",
                False,
                "Safety and security were added after the frozen H1-H4 preregistration and must be labeled exploratory extensions.",
                observed="safety|security",
                expected="explicit post-hoc labeling",
            ),
            _flag(
                "protect_sensitive_raw_text",
                "warning",
                False,
                "Raw safety/security prompts, answers, and reasoning must never enter public outputs; this audit writes aggregates only.",
                observed="aggregate-only audit outputs",
                expected="no raw sensitive text",
            ),
        ]
    )

    parse_failures = int(segment_table["n_parse_failure_rows"].sum())
    if parse_failures:
        flags.append(
            _flag(
                "track_b_parse_failures",
                "warning",
                False,
                "Invalid judge rows remain null and must be excluded with missingness reported, never converted to zero.",
                observed=parse_failures,
                expected=0,
            )
        )
    mismatch = pair_totals["full_only_rows"] + pair_totals["isolated_only_rows"]
    duplicate_keys = pair_totals["full_duplicate_keys"] + pair_totals["isolated_duplicate_keys"]
    if mismatch:
        flags.append(
            _flag(
                "paired_context_join_loss",
                "error",
                True,
                "Full/isolated rows do not pair exactly; losses must be explained before context analysis.",
                observed=mismatch,
                expected=0,
            )
        )
    if duplicate_keys:
        flags.append(
            _flag(
                "track_b_duplicate_keys",
                "error",
                True,
                "Track B trace_id+seg_idx keys are not unique.",
                observed=duplicate_keys,
                expected=0,
            )
        )
    unknown_outcomes = outcome_table.filter(pl.col("outcome_name").is_null()).height
    if unknown_outcomes:
        flags.append(
            _flag(
                "unregistered_outcome_domains",
                "error",
                True,
                "Every observed domain must have an explicit outcome definition and direction.",
                observed=unknown_outcomes,
                expected=0,
            )
        )
    return pl.DataFrame(flags).sort(["blocking", "severity", "flag_id"], descending=[True, False, False])


def _write_csv(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)


def audit_input_paths(paths: Mapping[str, Any]) -> list[Path]:
    inputs: list[Path] = []
    inputs.extend(expand_parquet_paths(paths["traces_glob"], traces=True))
    inputs.extend(sorted(Path(paths["tasks_dir"]).glob("*.parquet")))
    task_manifest = Path(paths["tasks_dir"]) / "manifest.json"
    if task_manifest.exists():
        inputs.append(task_manifest)
    experiment_config = paths.get("experiment_config")
    if experiment_config and Path(experiment_config).exists():
        inputs.append(Path(experiment_config))
    inputs.extend(
        [
            Path(paths["answer_extractions"]),
            Path(paths["track_b_full"]),
            Path(paths["track_b_isolated"]),
            Path(paths["quality"]),
        ]
    )
    inputs.extend(Path(value) for value in paths["grades"])
    manifests: list[Path] = []
    for source in inputs:
        candidate = source.with_suffix(".manifest.json")
        if candidate.exists():
            manifests.append(candidate)
    return sorted(set(inputs + manifests))


def run_audit(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    trace_index: pl.DataFrame,
    outcomes: pl.DataFrame,
    output_dir: Path,
    repo_root: Path,
    force: bool = False,
    upstream_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate every required Phase-1 audit artifact."""
    audit_dir = output_dir / "00_audit"
    expected = [audit_dir / name for name in AUDIT_FILENAMES]
    if not force and any(path.exists() for path in expected):
        raise FileExistsError(
            f"audit outputs already exist under {audit_dir}; pass --force to replace them"
        )
    audit_dir.mkdir(parents=True, exist_ok=True)

    paths = config["paths"]
    inputs = audit_input_paths(paths)
    assert_parquet_files(path for path in inputs if path.suffix == ".parquet")
    upstream = adjudicate_upstream_audit(
        upstream_report or run_upstream_strict_audit(paths, repo_root=repo_root),
        config.get("audit", {}).get("accepted_release_exceptions", []),
    )

    full_raw = scan_track_b(paths["track_b_full"], "full")
    isolated_raw = scan_track_b(paths["track_b_isolated"], "isolated")
    full = build_full_segment_table(full_raw, trace_index)
    isolated = build_full_segment_table(isolated_raw, trace_index)

    trace_cov = trace_coverage(trace_index)
    segment_cov = segment_coverage(full, isolated)
    pair_cov, pair_totals = context_pair_coverage(full, isolated)
    completion = completion_by_cell(trace_index, full)
    outcome_cov = outcome_coverage(outcomes)
    source_cov = analysis_source_by_model(trace_index)
    flags = build_quality_flags(
        trace_index,
        segment_cov,
        pair_totals,
        outcomes,
        upstream,
        config.get("model_metadata", {}),
    )
    tables = {
        "trace_coverage.csv": trace_cov,
        "segment_coverage.csv": segment_cov,
        "context_pair_coverage.csv": pair_cov,
        "completion_by_cell.csv": completion,
        "outcome_coverage.csv": outcome_cov,
        "analysis_source_by_model.csv": source_cov,
        "data_quality_flags.csv": flags,
    }
    for name, table in tables.items():
        _write_csv(table, audit_dir / name)

    blocking_flags = flags.filter(pl.col("blocking")).height
    ready = bool(upstream["effective_ok"]) and blocking_flags == 0
    metadata = {
        "mode": "dev" if output_dir.name.endswith("_dev") else "final",
        "paper_analysis_ready": ready,
        "strict_v2_ok": bool(upstream["strict_ok"]),
        "effective_audit_ok": bool(upstream["effective_ok"]),
        "strict_v2_returncode": upstream.get("returncode"),
        "strict_v2_issues": upstream.get("issues", []),
        "unaccepted_audit_issues": upstream.get("unaccepted_issues", []),
        "blocking_flag_count": blocking_flags,
        "trace_count": trace_index.height,
        "paired_context_totals": pair_totals,
        "accepted_release_exceptions": config.get("audit", {}).get(
            "accepted_release_exceptions", []
        ),
    }
    output_paths = [audit_dir / name for name in tables]
    output_paths.extend(
        path
        for path in [
            output_dir / "trace_index.parquet",
            output_dir / "outcomes.parquet",
            output_dir / "splits.parquet",
        ]
        if path.exists()
    )
    write_analysis_manifest(
        audit_dir / "analysis_manifest.json",
        config_path=config_path,
        input_paths=inputs,
        output_paths=output_paths,
        metadata=metadata,
        repo_root=repo_root,
    )
    return {
        "ready": ready,
        "audit_dir": audit_dir,
        "flags": flags,
        "pair_totals": pair_totals,
        "upstream": upstream,
    }
