import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.prefix_monitor import (  # noqa: E402
    auroc,
    average_precision,
    build_prefix_dataset,
    design_matrix,
    evaluate_prefix,
    make_folds,
)


def test_prompt_disjoint_folds_keep_instances_together():
    df = pl.DataFrame(
        {
            "trace_id": [f"t{i}" for i in range(24)],
            "instance_id": [f"p{i // 3}" for i in range(24)],
            "outcome_y": [i % 2 for i in range(24)],
        }
    )
    folds = make_folds(df, "prompt_disjoint", n_folds=4, seed=3)
    for train_idx, test_idx in folds:
        train_groups = set(df[train_idx]["instance_id"].to_list())
        test_groups = set(df[test_idx]["instance_id"].to_list())
        assert train_groups.isdisjoint(test_groups)


def test_metrics_are_order_sensitive():
    y = np.array([0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    bad = 1 - good
    assert auroc(y, good) == 1.0
    assert auroc(y, bad) == 0.0
    assert average_precision(y, good) == 1.0


def test_prefix_dataset_uses_only_prefix_segments():
    traces = pl.DataFrame(
        {
            "trace_id": ["a", "b"],
            "instance_id": ["p1", "p2"],
            "gen_model": ["m", "m"],
            "task_type": ["math", "math"],
            "difficulty_raw": ["easy", "easy"],
            "decode_temperature": [0.6, 0.6],
            "n_new_tokens": [100, 100],
            "outcome_y": [1, 0],
        }
    )
    track_b = pl.DataFrame(
        {
            "trace_id": ["a", "a", "a", "b", "b", "b"],
            "norm_pos": [0.1, 0.4, 0.9, 0.1, 0.4, 0.9],
            "verification": [1, 1, 1, 0, 1, 1],
        }
    )
    df = build_prefix_dataset(track_b, traces, prefix=0.5, temporal_bins=2, behaviors=["verification"])
    row_a = df.filter(pl.col("trace_id") == "a").row(0, named=True)
    row_b = df.filter(pl.col("trace_id") == "b").row(0, named=True)
    assert row_a["prefix_segments"] == 2
    assert row_a["count_verification"] == 2
    assert row_b["prefix_segments"] == 2
    assert row_b["count_verification"] == 1


def test_temporal_monitor_beats_counts_on_order_signal():
    rng = np.random.default_rng(4)
    rows = []
    for i in range(160):
        y = i % 2
        early = rng.normal(0.75 if y else 0.25, 0.05)
        late = 1.0 - early
        rows.append(
            {
                "trace_id": f"t{i}",
                "instance_id": f"p{i}",
                "gen_model": "m",
                "task_type": "math",
                "difficulty_raw": "same",
                "decode_temperature": 0.6,
                "prefix_segments": 40,
                "rate_verification": 0.5,
                "bin0_verification": float(early),
                "bin1_verification": float(late),
                "outcome_y": y,
            }
        )
    df = pl.DataFrame(rows)
    metrics, deltas, _ = evaluate_prefix(
        df,
        prefix=0.5,
        split="prompt_disjoint",
        feature_sets=["counts", "temporal"],
        temporal_bins=2,
        n_folds=5,
        seed=5,
        boot=0,
    )
    by_set = {row["feature_set"]: row for row in metrics}
    assert by_set["temporal"]["auroc"] > 0.95
    assert by_set["temporal"]["auroc"] - by_set["counts"]["auroc"] > 0.35
    assert any(row["feature_set"] == "temporal" and row["baseline"] == "counts" for row in deltas)


def test_design_matrix_includes_shuffled_time_placebo():
    df = pl.DataFrame(
        {
            "trace_id": ["a", "b"],
            "gen_model": ["m", "m"],
            "task_type": ["math", "math"],
            "difficulty_raw": ["easy", "hard"],
            "prefix_segments": [10, 20],
            "rate_verification": [0.2, 0.8],
            "outcome_y": [0, 1],
        }
    )
    X, names = design_matrix(df, "time_shuffled", temporal_bins=3)
    assert X.shape[0] == 2
    assert "shuffled_bin0_verification" in names
    assert "shuffled_bin2_verification" in names
