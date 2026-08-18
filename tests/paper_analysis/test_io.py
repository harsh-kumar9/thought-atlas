from pathlib import Path

import polars as pl
import pytest

from src.analysis.paper.constants import BEHAVIORS
from src.analysis.paper.io import (
    build_full_segment_table,
    build_paired_context_table,
    has_parquet_magic,
    load_trace_index,
    scan_track_b,
    scan_traces,
)


def _trace_rows():
    rows = []
    for idx in range(6):
        rows.append(
            {
                "trace_id": f"t{idx}",
                "instance_id": f"p{idx // 2}",
                "seed": 42,
                "gen_model": "anchor" if idx < 3 else "reasoner",
                "gen_model_id": "synthetic/model",
                "generation_kind": "non_reasoning" if idx < 3 else "reasoning",
                "task_type": "math",
                "difficulty_raw": "synthetic",
                "n_new_tokens": 10 + idx,
                "completed": idx != 5,
                "finish_reason": "stop" if idx != 5 else "length",
                "failure_mode": None if idx != 5 else "length",
                "reasoning_text_for_analysis": f"synthetic reasoning {idx}",
                "generation_fingerprint": "fp",
            }
        )
    return rows


def _segments(context: str):
    rows = []
    for trace_id in ["t0", "t1"]:
        for seg_idx in range(2):
            rows.append(
                {
                    "trace_id": trace_id,
                    "seg_idx": seg_idx,
                    "n_segments": 2,
                    "norm_pos": float(seg_idx),
                    "section_type": "reasoning",
                    "context_mode": context,
                    "kim_parsed": True,
                    "gandhi_parsed": True,
                    "judge_context_truncated": False,
                    "judge_model": "synthetic-judge",
                    "score_version": "synthetic-v1",
                    **{behavior: int(seg_idx == 0) for behavior in BEHAVIORS},
                }
            )
    return rows


def test_lazy_scans_dev_index_and_analysis_source(tmp_path):
    trace_path = tmp_path / "traces_synthetic.parquet"
    pl.DataFrame(_trace_rows()).write_parquet(trace_path)
    lazy = scan_traces(str(tmp_path / "traces_*.parquet"))
    assert isinstance(lazy, pl.LazyFrame)
    index = load_trace_index(
        str(tmp_path / "traces_*.parquet"),
        model_metadata={
            "anchor": {
                "analysis_source": "answer_text",
                "configured_token_budget": 4096,
                "family": "llama_anchor",
            },
            "reasoner": {
                "analysis_source": "think_text",
                "configured_token_budget": 65536,
                "family": "deepseek_distill",
            },
        },
        dev=True,
        dev_max_per_cell=2,
    )
    assert index.height == 4
    assert set(index.filter(pl.col("gen_model") == "anchor")["analysis_source"]) == {
        "answer_text"
    }
    assert set(index.filter(pl.col("gen_model") == "reasoner")["model_family"]) == {
        "deepseek_distill"
    }
    assert "reasoning_text_for_analysis" in index.columns


def test_full_and_paired_tables_use_exact_keys(tmp_path):
    full_path = tmp_path / "full.parquet"
    isolated_path = tmp_path / "isolated.parquet"
    pl.DataFrame(_segments("full")).write_parquet(full_path)
    isolated_rows = _segments("isolated")[:-1]
    pl.DataFrame(isolated_rows).write_parquet(isolated_path)
    index = pl.DataFrame(
        [
            {
                "trace_id": trace_id,
                "instance_id": f"p-{trace_id}",
                "gen_model": "m",
                "task_type": "math",
                "completed": True,
            }
            for trace_id in ["t0", "t1"]
        ]
    )
    full = build_full_segment_table(scan_track_b(str(full_path), "full"), index)
    assert isinstance(full, pl.LazyFrame)
    assert full.collect().height == 4
    paired = build_paired_context_table(
        scan_track_b(str(full_path), "full"),
        scan_track_b(str(isolated_path), "isolated"),
        index,
    ).collect()
    assert paired.height == 3
    assert "full__verification" in paired.columns
    assert "isolated__verification" in paired.columns


def test_lfs_pointer_is_rejected(tmp_path):
    pointer = tmp_path / "traces_pointer.parquet"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    assert not has_parquet_magic(pointer)
    with pytest.raises(ValueError, match="not materialized Parquet"):
        scan_traces(str(pointer))

