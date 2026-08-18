import json
from pathlib import Path

import polars as pl

from src.analysis.paper.audit import adjudicate_upstream_audit, run_audit
from src.analysis.paper.constants import AUDIT_FILENAMES, BEHAVIORS
from src.analysis.paper.io import load_trace_index
from src.analysis.paper.outcomes import build_outcome_table
from src.analysis.paper.splits import make_split_registry, write_splits


def _write_fixture(root: Path):
    (root / "data/v2/tasks").mkdir(parents=True)
    (root / "data/v2/traces").mkdir(parents=True)
    (root / "data/v2/judge").mkdir(parents=True)
    (root / "data/v2/perf").mkdir(parents=True)
    trace_rows = []
    segment_rows = {"full": [], "isolated": []}
    grade_rows = []
    quality_rows = []
    for idx, domain in enumerate(["math", "idea", "safety"]):
        trace_id = f"t{idx}"
        trace_rows.append(
            {
                "trace_id": trace_id,
                "instance_id": f"p{idx}",
                "seed": 42,
                "gen_model": "anchor" if idx == 0 else "reasoner",
                "gen_model_id": "synthetic/model",
                "generation_kind": "non_reasoning" if idx == 0 else "reasoning",
                "task_type": domain,
                "difficulty_raw": "synthetic",
                "n_new_tokens": 64,
                "completed": idx != 2,
                "finish_reason": "stop" if idx != 2 else "length",
                "failure_mode": None if idx != 2 else "length",
                "reasoning_text_for_analysis": "synthetic only",
            }
        )
        if domain == "math":
            grade_rows.append({"trace_id": trace_id, "success": 1})
        else:
            quality_rows.append(
                {
                    "trace_id": trace_id,
                    "quality_score": 0.8 if domain == "idea" else 0.4,
                    "safety_harm_score": None if domain == "idea" else 0.6,
                    "high_harmful_compliance": None if domain == "idea" else True,
                }
            )
        for context in segment_rows:
            for seg_idx in range(2):
                segment_rows[context].append(
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
                        **{behavior: int(seg_idx == idx % 2) for behavior in BEHAVIORS},
                    }
                )

    paths = {
        "tasks_dir": root / "data/v2/tasks",
        "traces_glob": root / "data/v2/traces/traces_*.parquet",
        "answer_extractions": root / "data/v2/judge/answer_extractions.parquet",
        "track_b_full": root / "data/v2/judge/track_b_full.parquet",
        "track_b_isolated": root / "data/v2/judge/track_b_isolated.parquet",
        "grades": [root / "data/v2/perf/success_grades.parquet", root / "data/v2/perf/code_grades.parquet"],
        "quality": root / "data/v2/judge/quality.parquet",
    }
    pl.DataFrame({"instance_id": ["p0"], "task_type": ["math"]}).write_parquet(paths["tasks_dir"] / "math.parquet")
    pl.DataFrame(trace_rows).write_parquet(root / "data/v2/traces/traces_synthetic.parquet")
    pl.DataFrame(segment_rows["full"]).write_parquet(paths["track_b_full"])
    pl.DataFrame(segment_rows["isolated"]).write_parquet(paths["track_b_isolated"])
    pl.DataFrame(grade_rows).write_parquet(paths["grades"][0])
    pl.DataFrame({"trace_id": pl.Series([], dtype=pl.String), "success": pl.Series([], dtype=pl.Int64), "tests_passed": pl.Series([], dtype=pl.Int64)}).write_parquet(paths["grades"][1])
    pl.DataFrame(quality_rows).write_parquet(paths["quality"])
    pl.DataFrame({"trace_id": ["t0", "t1", "t2"], "validated": [True, True, True]}).write_parquet(paths["answer_extractions"])
    return paths


def test_synthetic_audit_pipeline_writes_required_contract(tmp_path):
    paths = _write_fixture(tmp_path)
    config_path = tmp_path / "configs/paper_analysis.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("seed: 42\n")
    metadata = {
        "anchor": {"family": "llama_anchor", "analysis_source": "answer_text", "configured_token_budget": 4096},
        "reasoner": {"family": "deepseek_distill", "analysis_source": "think_text", "configured_token_budget": 65536},
    }
    config = {
        "paths": {key: ([str(v) for v in value] if isinstance(value, list) else str(value)) for key, value in paths.items()},
        "model_metadata": metadata,
        "audit": {"accepted_release_exceptions": []},
    }
    index = load_trace_index(str(paths["traces_glob"]), model_metadata=metadata, include_reasoning_text=False)
    outcomes = build_outcome_table(index, [str(path) for path in paths["grades"]], str(paths["quality"]))
    out_dir = tmp_path / "data/v2/analysis/paper_dev"
    splits = make_split_registry(index)
    write_splits(splits, out_dir / "splits.parquet")
    result = run_audit(
        config_path=config_path,
        config=config,
        trace_index=index,
        outcomes=outcomes,
        output_dir=out_dir,
        repo_root=tmp_path,
        upstream_report={"ok": True, "returncode": 0, "issues": []},
    )
    assert result["ready"] is True
    for filename in AUDIT_FILENAMES:
        assert (out_dir / "00_audit" / filename).exists()
    manifest = json.loads((out_dir / "00_audit/analysis_manifest.json").read_text())
    assert manifest["metadata"]["paper_analysis_ready"] is True
    pair = pl.read_csv(out_dir / "00_audit/context_pair_coverage.csv")
    assert pair["full_only_rows"].sum() == 0
    assert pair["isolated_only_rows"].sum() == 0


def test_audit_exceptions_must_match_every_issue_exactly():
    upstream = {"ok": False, "issues": ["issue one", "issue two"]}
    partial = adjudicate_upstream_audit(upstream, ["issue one"])
    assert partial["effective_ok"] is False
    assert partial["unaccepted_issues"] == ["issue two"]
    complete = adjudicate_upstream_audit(upstream, ["issue one", "issue two"])
    assert complete["effective_ok"] is True
