"""Canonical lazy data access for the paper-analysis pipeline."""

from __future__ import annotations

import glob
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl

from src.analysis.paper.constants import BEHAVIORS
from src.utils.io import resolve_trace_paths


TRACE_INDEX_REQUIRED = (
    "trace_id",
    "instance_id",
    "seed",
    "gen_model",
    "gen_model_id",
    "task_type",
    "difficulty_raw",
    "n_new_tokens",
    "completed",
    "finish_reason",
    "failure_mode",
)
TRACE_PROVENANCE_COLUMNS = (
    "generation_kind",
    "generation_fingerprint",
    "generation_version",
    "task_fingerprint",
    "task_set_fingerprint",
    "sampling_seed",
    "sampling_params",
    "decode_temperature",
    "decode_top_p",
    "decode_top_k",
    "thinking_style",
    "parse_status",
    "answer_source",
    "close_tag_count",
)
TEXT_COLUMNS = ("reasoning_text_for_analysis",)
PAIR_KEY = ("trace_id", "seg_idx")


def _string_paths(paths: Sequence[str | Path]) -> list[str]:
    return [str(Path(path)) for path in paths]


def expand_parquet_paths(paths: list[str] | str | Path, *, traces: bool = False) -> list[Path]:
    """Resolve concrete Parquet inputs deterministically.

    Trace globs use the repository's validated resolver so canonical files and
    their retained shards cannot be loaded together.
    """
    if isinstance(paths, (str, Path)):
        raw = str(paths)
        if traces:
            resolved = resolve_trace_paths(raw)
        elif any(char in raw for char in "*?["):
            resolved = [Path(value) for value in sorted(glob.glob(raw))]
        else:
            resolved = [Path(raw)]
    else:
        resolved = [Path(value) for value in paths]
    if not resolved:
        raise FileNotFoundError(f"no Parquet files matched {paths}")
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Parquet inputs: {missing}")
    return sorted(resolved)


def has_parquet_magic(path: str | Path) -> bool:
    """Return whether a file has Parquet's leading and trailing PAR1 bytes."""
    source = Path(path)
    if not source.is_file() or source.stat().st_size < 8:
        return False
    with source.open("rb") as handle:
        head = handle.read(4)
        handle.seek(-4, 2)
        tail = handle.read(4)
    return head == b"PAR1" and tail == b"PAR1"


def assert_parquet_files(paths: Iterable[str | Path]) -> None:
    bad = [str(path) for path in paths if not has_parquet_magic(path)]
    if bad:
        raise ValueError(f"files are not materialized Parquet objects: {bad}")


def scan_traces(paths: list[str] | str) -> pl.LazyFrame:
    """Lazily scan canonical trace files without double-loading shards."""
    resolved = expand_parquet_paths(paths, traces=True)
    assert_parquet_files(resolved)
    frames = [pl.scan_parquet(path) for path in resolved]
    return pl.concat(frames, how="diagonal_relaxed")


def scan_track_b(path: str, context_mode: str) -> pl.LazyFrame:
    """Lazily scan one Track-B context and enforce its requested mode."""
    if context_mode not in {"full", "isolated"}:
        raise ValueError("context_mode must be 'full' or 'isolated'")
    source = Path(path)
    assert_parquet_files([source])
    frame = pl.scan_parquet(source)
    schema = frame.collect_schema()
    if "context_mode" not in schema:
        raise ValueError(f"Track B file lacks context_mode: {source}")
    return frame.filter(pl.col("context_mode") == context_mode)


def _mapping_expr(
    field: str,
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    fallback: pl.Expr,
    dtype: pl.DataType,
) -> pl.Expr:
    expression = fallback
    for model, values in reversed(list(metadata.items())):
        if field in values:
            expression = (
                pl.when(pl.col("gen_model") == model)
                .then(pl.lit(values[field], dtype=dtype))
                .otherwise(expression)
            )
    return expression.alias(field)


def _collect_streaming(frame: pl.LazyFrame) -> pl.DataFrame:
    return frame.collect(engine="streaming")


def load_trace_index(
    paths: list[str] | str,
    *,
    model_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    include_reasoning_text: bool = True,
    dev: bool = False,
    dev_max_per_cell: int = 200,
) -> pl.DataFrame:
    """Create a unique, deterministic one-row-per-trace index.

    In dev mode traces are selected by sorted deterministic trace ID within
    model x domain cells. Text is joined only after that selection, which keeps
    the workstation audit memory-bounded.
    """
    metadata = {
        model: {
            **values,
            "model_family": values.get("model_family", values.get("family", model)),
        }
        for model, values in (model_metadata or {}).items()
    }
    traces = scan_traces(paths)
    schema = traces.collect_schema()
    missing = [column for column in TRACE_INDEX_REQUIRED if column not in schema]
    if missing:
        raise ValueError(f"trace inputs lack required columns: {missing}")

    columns = list(TRACE_INDEX_REQUIRED)
    columns.extend(column for column in TRACE_PROVENANCE_COLUMNS if column in schema)
    index = _collect_streaming(traces.select(columns))
    duplicated = index.group_by("trace_id").len().filter(pl.col("len") > 1)
    if duplicated.height:
        raise ValueError(f"trace_id is not unique ({duplicated.height} duplicated IDs)")

    source_fallback = (
        pl.when(pl.col("generation_kind") == "non_reasoning")
        .then(pl.lit("answer_text"))
        .otherwise(pl.lit("think_text"))
        if "generation_kind" in index.columns
        else pl.lit("think_text")
    )
    index = index.with_columns(
        _mapping_expr(
            "analysis_source", metadata, fallback=source_fallback, dtype=pl.String
        ),
        _mapping_expr(
            "configured_token_budget",
            metadata,
            fallback=pl.lit(None, dtype=pl.Int64),
            dtype=pl.Int64,
        ),
        _mapping_expr(
            "model_family",
            metadata,
            fallback=pl.col("gen_model"),
            dtype=pl.String,
        ),
    )

    if dev:
        if dev_max_per_cell <= 0:
            raise ValueError("dev_max_per_cell must be positive")
        index = (
            index.sort(["gen_model", "task_type", "trace_id"])
            .group_by(["gen_model", "task_type"], maintain_order=True)
            .head(dev_max_per_cell)
            .sort("trace_id")
        )

    if include_reasoning_text:
        if "reasoning_text_for_analysis" not in schema:
            raise ValueError("trace inputs lack reasoning_text_for_analysis")
        selected_ids = index["trace_id"].to_list()
        text_scan = traces.select(["trace_id", "reasoning_text_for_analysis"])
        if dev:
            text_scan = text_scan.filter(pl.col("trace_id").is_in(selected_ids))
        text = _collect_streaming(text_scan)
        index = index.join(text, on="trace_id", how="left", validate="1:1")

    return index.sort(["gen_model", "task_type", "instance_id", "seed"])


def _as_lazy(frame_or_path: pl.DataFrame | pl.LazyFrame | str, context_mode: str) -> pl.LazyFrame:
    if isinstance(frame_or_path, pl.LazyFrame):
        return frame_or_path
    if isinstance(frame_or_path, pl.DataFrame):
        return frame_or_path.lazy()
    return scan_track_b(frame_or_path, context_mode)


def build_full_segment_table(
    track_b_full: pl.DataFrame | pl.LazyFrame | str,
    trace_index: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    """Join full-context segment labels to trace metadata lazily."""
    segments = _as_lazy(track_b_full, "full")
    metadata = trace_index.lazy() if isinstance(trace_index, pl.DataFrame) else trace_index
    keep = [
        "trace_id",
        "instance_id",
        "seed",
        "gen_model",
        "gen_model_id",
        "model_family",
        "task_type",
        "difficulty_raw",
        "n_new_tokens",
        "completed",
        "finish_reason",
        "failure_mode",
        "analysis_source",
        "configured_token_budget",
    ]
    available = metadata.collect_schema()
    return segments.join(
        metadata.select([column for column in keep if column in available]),
        on="trace_id",
        how="inner",
        validate="m:1",
    )


def _context_projection(frame: pl.LazyFrame, prefix: str, *, include_metadata: bool) -> pl.LazyFrame:
    schema = frame.collect_schema()
    shared = ["n_segments", "norm_pos", "section_type", "context_mode"]
    diagnostics = [
        "kim_parsed",
        "gandhi_parsed",
        "judge_context_truncated",
        "judge_model",
        "score_version",
    ]
    metadata = [
        "instance_id",
        "seed",
        "gen_model",
        "gen_model_id",
        "model_family",
        "task_type",
        "completed",
        "analysis_source",
        "configured_token_budget",
    ]
    columns = list(PAIR_KEY)
    expressions: list[pl.Expr] = []
    for column in shared + diagnostics + list(BEHAVIORS):
        if column in schema:
            expressions.append(pl.col(column).alias(f"{prefix}__{column}"))
    if include_metadata:
        expressions.extend(pl.col(column) for column in metadata if column in schema)
    return frame.select(columns + expressions)


def build_paired_context_table(
    full_segments: pl.DataFrame | pl.LazyFrame | str,
    isolated_segments: pl.DataFrame | pl.LazyFrame | str,
    trace_index: pl.DataFrame | pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Inner-join paired contexts on exact trace and segment keys.

    Coverage losses must be computed separately before this table is used; the
    audit stage does so and writes full-only/isolated-only counts.
    """
    full = _as_lazy(full_segments, "full")
    isolated = _as_lazy(isolated_segments, "isolated")
    if trace_index is not None:
        full = build_full_segment_table(full, trace_index)
    left = _context_projection(full, "full", include_metadata=True)
    right = _context_projection(isolated, "isolated", include_metadata=False)
    return left.join(right, on=list(PAIR_KEY), how="inner", validate="1:1")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def current_commit(repo_root: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_analysis_manifest(
    path: str | Path,
    *,
    config_path: str | Path,
    input_paths: Iterable[str | Path],
    output_paths: Iterable[str | Path] = (),
    metadata: Mapping[str, Any] | None = None,
    repo_root: str | Path = ".",
) -> None:
    """Write a reproducibility manifest with hashes for every concrete input."""
    destination = Path(path)
    inputs = sorted({Path(value) for value in input_paths})
    outputs = sorted({Path(value) for value in output_paths if Path(value).exists()})
    payload = {
        "schema_version": "paper-analysis-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(repo_root),
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "inputs": [
            {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
            for source in inputs
        ],
        "outputs": [
            {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            }
            for output in outputs
        ],
        "metadata": dict(metadata or {}),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
