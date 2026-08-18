"""Deterministic prompt-grouped split assignments and leakage checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator, Mapping

import polars as pl


def stable_fold(value: str, *, n_folds: int = 5, seed: int = 42) -> int:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def infer_model_family(model: str) -> str:
    if model.startswith("qwen"):
        return "qwen35"
    if model.startswith("gemma"):
        return "gemma4"
    if model == "anchor":
        return "llama_anchor"
    if model == "reasoner":
        return "deepseek_distill"
    return model


def make_split_registry(
    trace_index: pl.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 42,
    model_families: Mapping[str, str] | None = None,
    include_model_disjoint: bool = True,
) -> pl.DataFrame:
    """Return one deterministic assignment row per trace.

    Family/model transfer evaluation is nested inside the prompt fold: held-out
    family traces in fold ``k`` are tested against other-family traces outside
    fold ``k``. This preserves both transfer and prompt-disjointness.
    """
    required = {"trace_id", "instance_id", "gen_model", "task_type"}
    if not required.issubset(trace_index.columns):
        raise ValueError(f"trace_index must contain {sorted(required)}")
    base = trace_index.select(sorted(required)).unique()
    if base["trace_id"].n_unique() != base.height:
        raise ValueError("trace_id must be unique before creating splits")

    family_map = dict(model_families or {})
    rows = []
    for row in base.sort("trace_id").iter_rows(named=True):
        instance = str(row["instance_id"])
        model = str(row["gen_model"])
        family = family_map.get(model, infer_model_family(model))
        assignment = {
            **row,
            "prompt_fold": stable_fold(instance, n_folds=n_folds, seed=seed),
            "domain_holdout": row["task_type"],
            "model_family": family,
            "model_family_holdout": family,
        }
        if include_model_disjoint:
            assignment["model_holdout"] = model
        rows.append(assignment)
    registry = pl.DataFrame(rows)
    assert_prompt_fold_integrity(registry)
    return registry.sort("trace_id")


def assert_prompt_fold_integrity(registry: pl.DataFrame) -> None:
    required = {"instance_id", "prompt_fold"}
    if not required.issubset(registry.columns):
        raise ValueError(f"split registry must contain {sorted(required)}")
    leaks = (
        registry.group_by("instance_id")
        .agg(pl.col("prompt_fold").n_unique().alias("n_folds"))
        .filter(pl.col("n_folds") != 1)
    )
    if leaks.height:
        raise AssertionError(
            f"{leaks.height} instance_id values were assigned to multiple prompt folds"
        )


def split_masks(
    registry: pl.DataFrame,
    *,
    split_type: str,
    holdout: str | int,
    prompt_fold: int | None = None,
) -> tuple[pl.Series, pl.Series]:
    """Build train/test masks for one registered split and assert no leakage."""
    if split_type == "prompt_disjoint":
        test = registry["prompt_fold"] == int(holdout)
        train = ~test
    elif split_type == "leave_one_domain_out":
        test = registry["domain_holdout"] == str(holdout)
        test_instances = registry.filter(test)["instance_id"].unique().to_list()
        train = (registry["domain_holdout"] != str(holdout)) & ~registry[
            "instance_id"
        ].is_in(test_instances)
    elif split_type in {"leave_one_model_family_out", "model_disjoint"}:
        column = (
            "model_family_holdout"
            if split_type == "leave_one_model_family_out"
            else "model_holdout"
        )
        if prompt_fold is None:
            raise ValueError(f"{split_type} requires a nested prompt_fold")
        test = (registry[column] == str(holdout)) & (
            registry["prompt_fold"] == prompt_fold
        )
        train = (registry[column] != str(holdout)) & (
            registry["prompt_fold"] != prompt_fold
        )
    else:
        raise ValueError(f"unknown split type: {split_type}")
    assert_no_instance_leakage(registry, train, test)
    return train, test


def assert_no_instance_leakage(
    registry: pl.DataFrame, train_mask: pl.Series, test_mask: pl.Series
) -> None:
    train_instances = set(registry.filter(train_mask)["instance_id"].to_list())
    test_instances = set(registry.filter(test_mask)["instance_id"].to_list())
    overlap = train_instances & test_instances
    if overlap:
        examples = sorted(str(value) for value in overlap)[:5]
        raise AssertionError(
            f"instance_id leakage between train and test: {len(overlap)} overlaps; examples={examples}"
        )


def iter_split_definitions(
    registry: pl.DataFrame,
) -> Iterator[tuple[str, str | int, int | None]]:
    for fold in sorted(registry["prompt_fold"].unique().to_list()):
        yield "prompt_disjoint", int(fold), None
    for domain in sorted(registry["domain_holdout"].unique().to_list()):
        yield "leave_one_domain_out", str(domain), None
    folds = sorted(registry["prompt_fold"].unique().to_list())
    for family in sorted(registry["model_family_holdout"].unique().to_list()):
        for fold in folds:
            yield "leave_one_model_family_out", str(family), int(fold)
    if "model_holdout" in registry.columns:
        for model in sorted(registry["model_holdout"].unique().to_list()):
            for fold in folds:
                yield "model_disjoint", str(model), int(fold)


def write_splits(registry: pl.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    registry.write_parquet(destination)
