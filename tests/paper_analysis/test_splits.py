import polars as pl
import pytest

from src.analysis.paper.splits import (
    assert_prompt_fold_integrity,
    iter_split_definitions,
    make_split_registry,
    split_masks,
)


def _index():
    rows = []
    for prompt in range(40):
        domain = "math" if prompt < 20 else "idea"
        for model in ["qwen35_4b", "gemma4_e4b", "anchor"]:
            rows.append(
                {
                    "trace_id": f"t-{prompt}-{model}",
                    "instance_id": f"p-{prompt}",
                    "gen_model": model,
                    "task_type": domain,
                }
            )
    return pl.DataFrame(rows)


def test_split_registry_is_deterministic_and_prompt_grouped():
    first = make_split_registry(_index(), n_folds=5, seed=42)
    second = make_split_registry(_index().reverse(), n_folds=5, seed=42)
    assert first.equals(second)
    assert_prompt_fold_integrity(first)
    assert set(first["model_family_holdout"]) == {"qwen35", "gemma4", "llama_anchor"}


def test_every_materialized_split_has_no_instance_leakage():
    registry = make_split_registry(_index(), n_folds=5, seed=7)
    checked = 0
    for split_type, holdout, prompt_fold in iter_split_definitions(registry):
        train, test = split_masks(
            registry,
            split_type=split_type,
            holdout=holdout,
            prompt_fold=prompt_fold,
        )
        if train.sum() and test.sum():
            train_ids = set(registry.filter(train)["instance_id"])
            test_ids = set(registry.filter(test)["instance_id"])
            assert train_ids.isdisjoint(test_ids)
            checked += 1
    assert checked > 10


def test_integrity_check_detects_tampered_prompt_assignment():
    registry = make_split_registry(_index(), n_folds=5, seed=42)
    prompt = registry["instance_id"][0]
    rows = registry.with_row_index("row_idx").with_columns(
        pl.when((pl.col("instance_id") == prompt) & (pl.col("row_idx") == 0))
        .then((pl.col("prompt_fold") + 1) % 5)
        .otherwise(pl.col("prompt_fold"))
        .alias("prompt_fold")
    )
    with pytest.raises(AssertionError, match="multiple prompt folds"):
        assert_prompt_fold_integrity(rows)

