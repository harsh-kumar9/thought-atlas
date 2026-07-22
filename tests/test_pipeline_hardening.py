import json
from pathlib import Path

import polars as pl
import pytest


def test_reasoning_parser_recovers_stopped_direct_answer_and_uses_last_close():
    from src.utils.parse import parse_generation_detailed
    direct = parse_generation_detailed("Final answer: 7", "reasoning", finish_reason="stop")
    assert direct["answer_text"] == "Final answer: 7"
    assert direct["parse_status"] == "direct_answer_no_close"
    truncated = parse_generation_detailed("still thinking", "reasoning", finish_reason="length")
    assert truncated["answer_text"] == ""
    multi = parse_generation_detailed(
        "first thought</think>draft<think>revised</think>Final answer: B",
        "reasoning", finish_reason="stop")
    assert multi["answer_text"] == "Final answer: B"
    assert multi["close_tag_count"] == 2


def test_math_and_mcq_extractors_choose_actual_final_answer():
    from src.perf.answer_grading import grade_math_answer, grade_mcq_answer
    math = grade_math_answer(r"work... \boxed{\frac{1}{2}}", r"\frac{1}{2}")
    assert math["success"] == 1 and math["prediction"] == r"\frac{1}{2}"
    answer = "**A.** tempting\n**D.** also possible\nAfter checking, Final Answer: (B)"
    mcq = grade_mcq_answer(answer, "B")
    assert mcq["success"] == 1 and mcq["prediction"] == "B"
    assert grade_mcq_answer("No final selection was made.", "A")["success"] is None


def test_acp_option_cleanup_and_exact_stratified_sample():
    from src.utils.data_loaders import _filter_and_sample, _strip_embedded_mcq_options
    question = "What is reachable? A. old one B. old two C. old three D. old four"
    assert _strip_embedded_mcq_options(question, list("ABCD")) == "What is reachable?"
    df = pl.DataFrame({"prompt": [f"prompt {i}" for i in range(13)],
                       "stratum": ["a"] * 8 + ["b"] * 3 + ["c"] * 2})
    out = _filter_and_sample(df, n=7, seed=42, strata_col="stratum")
    assert out.height == 7


def test_signed_moral_score_is_bounded_and_inverts_penalties():
    from src.judge.run_quality import moral_weighted_score
    # Positive criterion satisfied, undesirable negative criterion absent -> perfect.
    assert moral_weighted_score([2, 0], [2, -3]) == 1.0
    assert moral_weighted_score([0, 2], [2, -3]) == 0.0
    assert moral_weighted_score([1, 1], [2, -3]) == 0.5


def test_sentence_judge_requires_complete_unique_indices():
    from src.judge.run_judge import _analysis_segment_text, _parse_sentence_batch
    good = json.dumps({"sentences": [{"index": 1, "behaviors": ["verification"]},
                                      {"index": 2, "behaviors": []}]})
    assert _parse_sentence_batch(good, 2, {"verification"}) == {
        1: ["verification"], 2: []}
    duplicate = json.dumps({"sentences": [{"index": 1, "behaviors": []},
                                           {"index": 1, "behaviors": []}]})
    assert _parse_sentence_batch(duplicate, 2, {"verification"}) is None
    assert _analysis_segment_text({"generation_kind": "reasoning",
                                   "reasoning_text_for_analysis": "work"}) == "<think>work</think>"


def test_code_language_function_and_string_output_handling():
    from src.perf.grade_code_exec import extract_submission, derive_fn_name, _value_equal
    code, language = extract_submission("```cpp\n#include <bits/stdc++.h>\nint main(){}\n```")
    assert code and language == "cpp"
    assert derive_fn_name({"starter_code": "class Solution:\n    def reverseDegree(self, s: str):\n        pass"}) == "reverseDegree"
    assert _value_equal("cccc", '"cccc"')
    assert _value_equal([0, 1], "[0, 1]")


def test_trace_resolver_excludes_shards_when_canonical_exists_and_rejects_gaps(tmp_path):
    from src.utils.io import resolve_trace_paths
    canonical = tmp_path / "traces_m.parquet"; canonical.touch()
    (tmp_path / "traces_m.shard00of02.parquet").touch()
    (tmp_path / "traces_m.shard01of02.parquet").touch()
    assert resolve_trace_paths(str(tmp_path / "traces_*.parquet")) == [canonical]
    canonical.unlink(); (tmp_path / "traces_m.shard01of02.parquet").unlink()
    with pytest.raises(ValueError, match="incomplete shard"):
        resolve_trace_paths(str(tmp_path / "traces_*.parquet"))


def test_trace_merger_validates_overlap_and_writes_manifest(tmp_path):
    from src.generate.merge_shards import merge_trace_shards
    base = {"gen_model": ["m"], "instance_id": ["i0"], "seed": [42],
            "trace_id": ["t0"], "task_type": ["math"],
            "generation_fingerprint": ["fp"]}
    pl.DataFrame(base).write_parquet(tmp_path / "traces_m.shard00of02.parquet")
    second = {**base, "instance_id": ["i1"], "trace_id": ["t1"]}
    pl.DataFrame(second).write_parquet(tmp_path / "traces_m.shard01of02.parquet")
    out = merge_trace_shards("m", tmp_path, 2)
    assert pl.read_parquet(out).height == 2
    assert out.with_suffix(".manifest.json").exists()


def test_judge_merger_requires_every_disjoint_shard(tmp_path):
    from src.judge.merge_shards import merge_judge_kind
    for idx in range(2):
        pl.DataFrame({"trace_id": [f"t{idx}"], "kim_parsed": [True],
                      "gandhi_parsed": [True], "score_version": ["behavior-v2"],
                      "judge_model": ["judge"]}).write_parquet(
            tmp_path / f"trackA_counts__judge.shard{idx:02d}of02.parquet")
    out = merge_judge_kind("trackA_counts", "judge", tmp_path, 2)
    assert pl.read_parquet(out).height == 2
    assert out.with_suffix(".manifest.json").exists()
