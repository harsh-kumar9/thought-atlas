"""Domain-specific outcome registry with explicit direction and missingness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import polars as pl


@dataclass(frozen=True)
class OutcomeSpec:
    domain: str
    name: str
    outcome_type: str
    raw_column: str
    better_direction: str
    interpretation: str


OUTCOME_REGISTRY: dict[str, OutcomeSpec] = {
    "math": OutcomeSpec("math", "success", "binary", "success", "higher", "deterministic correctness"),
    "code": OutcomeSpec("code", "code_success", "binary", "code_success", "higher", "sandboxed execution success"),
    "gpqa": OutcomeSpec("gpqa", "success", "binary", "success", "higher", "deterministic correctness"),
    "planning": OutcomeSpec("planning", "success", "binary", "success", "higher", "deterministic correctness"),
    "moral": OutcomeSpec("moral", "quality_score", "continuous", "quality_score", "higher", "answer-only rubric quality"),
    "idea": OutcomeSpec("idea", "quality_score", "continuous", "quality_score", "higher", "answer-only rubric quality"),
    "safety": OutcomeSpec(
        "safety",
        "safety_harm_score",
        "continuous",
        "safety_harm_score",
        "lower",
        "harmful compliance under direct harmful requests",
    ),
    "security": OutcomeSpec(
        "security",
        "success",
        "binary",
        "success",
        "higher_capability",
        "hazardous cybersecurity knowledge; higher is not safer",
    ),
}


def get_outcome_spec(domain: str) -> OutcomeSpec:
    try:
        return OUTCOME_REGISTRY[domain]
    except KeyError as exc:
        raise KeyError(f"no registered primary outcome for domain '{domain}'") from exc


def registry_frame() -> pl.DataFrame:
    return pl.DataFrame([asdict(spec) for spec in OUTCOME_REGISTRY.values()])


def _read_frame(source: str | Path | pl.DataFrame) -> pl.DataFrame:
    if isinstance(source, pl.DataFrame):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def _first_non_null(column: str) -> pl.Expr:
    return pl.col(column).drop_nulls().first().alias(column)


def _coalesce_grade_sources(
    grade_sources: Iterable[str | Path | pl.DataFrame],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    objective: list[pl.DataFrame] = []
    code: list[pl.DataFrame] = []
    for source in grade_sources:
        frame = _read_frame(source)
        if "trace_id" not in frame.columns or "success" not in frame.columns:
            raise ValueError("grade source must contain trace_id and success")
        is_code = "tests_passed" in frame.columns or (
            not isinstance(source, pl.DataFrame) and "code_grade" in Path(source).name
        )
        selected = frame.select(
            "trace_id", pl.col("success").cast(pl.Float64, strict=False).alias("value")
        )
        (code if is_code else objective).append(selected)

    def collapse(frames: list[pl.DataFrame], name: str) -> pl.DataFrame:
        if not frames:
            return pl.DataFrame(
                {"trace_id": pl.Series([], dtype=pl.String), name: pl.Series([], dtype=pl.Float64)}
            )
        return (
            pl.concat(frames, how="diagonal_relaxed")
            .rename({"value": name})
            .group_by("trace_id")
            .agg(_first_non_null(name))
        )

    return collapse(objective, "success"), collapse(code, "code_success")


def _quality_table(source: str | Path | pl.DataFrame | None) -> pl.DataFrame:
    columns = {
        "quality_score": pl.Float64,
        "safety_harm_score": pl.Float64,
        "high_harmful_compliance": pl.Boolean,
    }
    if source is None:
        return pl.DataFrame(
            {"trace_id": pl.Series([], dtype=pl.String)}
            | {name: pl.Series([], dtype=dtype) for name, dtype in columns.items()}
        )
    frame = _read_frame(source)
    if "trace_id" not in frame.columns:
        raise ValueError("quality source must contain trace_id")
    for name, dtype in columns.items():
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(name))
    return frame.group_by("trace_id").agg(*[_first_non_null(name) for name in columns])


def build_outcome_table(
    trace_index: pl.DataFrame,
    grade_sources: Iterable[str | Path | pl.DataFrame],
    quality_source: str | Path | pl.DataFrame | None,
) -> pl.DataFrame:
    """Attach the registered primary outcome to every trace.

    ``outcome_raw`` keeps the domain's native direction. The normalized
    ``outcome_higher_is_better`` reverses safety harm only; security remains a
    higher-capability endpoint and is labeled separately.
    """
    required = {"trace_id", "task_type"}
    if not required.issubset(trace_index.columns):
        raise ValueError(f"trace_index must contain {sorted(required)}")
    if trace_index["trace_id"].n_unique() != trace_index.height:
        raise ValueError("trace_index trace_id must be unique")

    objective, code = _coalesce_grade_sources(grade_sources)
    quality = _quality_table(quality_source)
    joined = (
        trace_index.join(objective, on="trace_id", how="left", validate="1:1")
        .join(code, on="trace_id", how="left", validate="1:1")
        .join(quality, on="trace_id", how="left", validate="1:1")
    )
    for name, dtype in [
        ("success", pl.Float64),
        ("code_success", pl.Float64),
        ("quality_score", pl.Float64),
        ("safety_harm_score", pl.Float64),
        ("high_harmful_compliance", pl.Boolean),
    ]:
        if name not in joined.columns:
            joined = joined.with_columns(pl.lit(None, dtype=dtype).alias(name))

    raw = (
        pl.when(pl.col("task_type") == "code")
        .then(pl.col("code_success"))
        .when(pl.col("task_type").is_in(["moral", "idea"]))
        .then(pl.col("quality_score"))
        .when(pl.col("task_type") == "safety")
        .then(pl.col("safety_harm_score"))
        .when(pl.col("task_type").is_in(["math", "gpqa", "planning", "security"]))
        .then(pl.col("success"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    higher = (
        pl.when(pl.col("task_type") == "safety")
        .then(1.0 - pl.col("safety_harm_score"))
        .otherwise(raw)
    )
    outcome_type = (
        pl.when(pl.col("task_type").is_in(["moral", "idea", "safety"]))
        .then(pl.lit("continuous"))
        .when(pl.col("task_type").is_in(["math", "code", "gpqa", "planning", "security"]))
        .then(pl.lit("binary"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    name = (
        pl.when(pl.col("task_type") == "code")
        .then(pl.lit("code_success"))
        .when(pl.col("task_type").is_in(["moral", "idea"]))
        .then(pl.lit("quality_score"))
        .when(pl.col("task_type") == "safety")
        .then(pl.lit("safety_harm_score"))
        .when(pl.col("task_type").is_in(["math", "gpqa", "planning", "security"]))
        .then(pl.lit("success"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    direction = (
        pl.when(pl.col("task_type") == "safety")
        .then(pl.lit("lower_raw_is_better"))
        .when(pl.col("task_type") == "security")
        .then(pl.lit("higher_hazardous_capability_not_safer"))
        .when(pl.col("task_type").is_in(list(OUTCOME_REGISTRY)))
        .then(pl.lit("higher_is_better"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    return joined.with_columns(
        raw.cast(pl.Float64).alias("outcome_raw"),
        higher.cast(pl.Float64).alias("outcome_higher_is_better"),
        outcome_type.alias("outcome_type"),
        name.alias("outcome_name"),
        direction.alias("outcome_direction"),
    ).with_columns(pl.col("outcome_raw").is_not_null().alias("outcome_available"))

