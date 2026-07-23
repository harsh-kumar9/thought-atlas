import json
import sys
import types
from pathlib import Path

import polars as pl
import pytest


def test_reasoning_parser_recovers_stopped_direct_answer_and_uses_last_close():
    from src.utils.parse import parse_generation_detailed
    direct = parse_generation_detailed("Final answer: 7", "reasoning", finish_reason="stop")
    assert direct["answer_text"] == "Final answer: 7"
    assert direct["think_text"] is None
    assert direct["parse_status"] == "direct_answer_no_close"
    truncated = parse_generation_detailed("still thinking", "reasoning", finish_reason="length")
    assert truncated["answer_text"] == ""
    multi = parse_generation_detailed(
        "first thought</think>draft<think>revised</think>Final answer: B",
        "reasoning", finish_reason="stop")
    assert multi["answer_text"] == "Final answer: B"
    assert multi["close_tag_count"] == 2


def test_sampling_can_preserve_generation_special_tokens(monkeypatch):
    from src.judge.vllm_engine import make_sampling

    class SamplingParams:
        def __init__(self, temperature=0, max_tokens=0, top_p=1, top_k=-1,
                     seed=None, skip_special_tokens=True):
            self.skip_special_tokens = skip_special_tokens

    monkeypatch.setitem(
        sys.modules, "vllm", types.SimpleNamespace(SamplingParams=SamplingParams))
    sampling = make_sampling(skip_special_tokens=False)
    assert sampling.skip_special_tokens is False


def test_math_and_mcq_extractors_choose_actual_final_answer():
    from src.perf.answer_grading import grade_math_answer, grade_mcq_answer
    math = grade_math_answer(r"work... \boxed{\frac{1}{2}}", r"\frac{1}{2}")
    assert math["success"] == 1 and math["prediction"] == r"\frac{1}{2}"
    answer = "**A.** tempting\n**D.** also possible\nAfter checking, Final Answer: (B)"
    mcq = grade_mcq_answer(answer, "B")
    assert mcq["success"] == 1 and mcq["prediction"] == "B"
    assert grade_mcq_answer("No final selection was made.", "A")["success"] is None


def test_objective_extractors_do_not_rescue_or_override_final_answers():
    from src.perf.answer_grading import grade_math_answer, grade_mcq_answer
    math = grade_math_answer(
        r"We know the target was $12$. Computing gives \boxed{15}.", "12")
    assert math["prediction"] == "15"
    assert math["success"] == 0
    recap = "Final answer: B\n\nOption recap:\nC. Third option words\nD."
    mcq = grade_mcq_answer(recap, "B")
    assert mcq["prediction"] == "B"
    assert mcq["success"] == 1
    assert grade_mcq_answer("Therefore, the answer must be (D).", "D")["success"] == 1


def test_reference_blind_answer_extraction_validation_and_consumption():
    from src.judge.run_answer_extraction import (
        build_prompt, code_candidates, validate_extraction)
    from src.utils.answer_extractions import (
        ANSWER_EXTRACTION_VERSION, extraction_input_sha256, select_answer,
        text_sha256)

    math_trace = {
        "trace_id": "math-1", "task_type": "math",
        "prompt": "Compute the quantity.", "reference_answer": "SECRET_REFERENCE",
        "full_text": "draft 12</think>After checking, final answer: 15.",
        "answer_text": "After checking, final answer: 15.",
    }
    parsed = {
        "status": "ok", "final_answer": "15",
        "evidence": "final answer: 15.", "answer_start_quote": "",
        "answer_end_quote": "", "code_block_index": -1, "confidence": "high",
    }
    valid = validate_extraction(parsed, math_trace)
    assert valid["validated"] and valid["extracted_answer"] == "15"
    parsed["evidence"] = "draft 12"
    assert validate_extraction(parsed, math_trace)["validation_status"] == "evidence_in_reasoning"

    class CharTokenizer:
        def __call__(self, value, add_special_tokens=False):
            return {"input_ids": [ord(ch) for ch in value]}

        def decode(self, ids):
            return "".join(chr(item) for item in ids)

    prompt, _ = build_prompt(math_trace, CharTokenizer(), 10000)
    assert "SECRET_REFERENCE" not in prompt

    code_text = (
        "draft\n```python\nprint('wrong')\n```\n</think>final\n"
        "```python\nprint('right')\n```")
    code_trace = {"trace_id": "code-1", "task_type": "code",
                  "full_text": code_text, "answer_text": ""}
    candidates = code_candidates(code_text)
    code_parsed = {
        "status": "ok", "final_answer": "", "evidence": "print('right')",
        "answer_start_quote": "", "answer_end_quote": "",
        "code_block_index": 1, "confidence": "high",
    }
    code_valid = validate_extraction(code_parsed, code_trace, candidates)
    assert code_valid["validated"]
    assert code_valid["extracted_answer"] == "print('right')"

    row = {
        "trace_id": "math-1", "score_version": ANSWER_EXTRACTION_VERSION,
        "validated": True, "extraction_status": "ok",
        "validation_status": "ok", "extracted_answer": "15",
        "extracted_answer_sha256": text_sha256("15"),
        "source_response_sha256": text_sha256(math_trace["full_text"]),
        "extraction_input_sha256": extraction_input_sha256(math_trace),
        "confidence": "high", "judge_model": "extractor", "evidence": "final answer: 15.",
    }
    selected = select_answer(math_trace, {"math-1": row})
    assert selected["used"] and selected["answer"] == "15"
    stale = {**math_trace, "full_text": math_trace["full_text"] + " changed"}
    assert select_answer(stale, {"math-1": row})["selection_status"] == "stale_response_hash"


def test_subjective_extraction_uses_exact_post_reasoning_span():
    from src.judge.run_answer_extraction import validate_extraction
    full = "private draft</think>Final recommendation: help them.\nExplain why."
    trace = {"trace_id": "m1", "task_type": "moral",
             "full_text": full, "answer_text": ""}
    parsed = {
        "status": "ok", "final_answer": "", "evidence": "",
        "answer_start_quote": "Final recommendation:",
        "answer_end_quote": "Explain why.", "code_block_index": -1,
        "confidence": "high",
    }
    result = validate_extraction(parsed, trace)
    assert result["validated"]
    assert result["extracted_answer"] == "Final recommendation: help them.\nExplain why."


def test_objective_grading_uses_validated_extraction_without_reference_leakage():
    from src.perf.grade import grade_traces
    from src.utils.answer_extractions import (
        ANSWER_EXTRACTION_VERSION, extraction_input_sha256, text_sha256)
    full = "Reasoning without a delimiter. The final answer is B."
    traces = pl.DataFrame([{
        "trace_id": "t1", "task_type": "gpqa", "answer_text": "",
        "full_text": full, "reference_answer": "B",
        "prompt": "A. one\nB. two\nC. three\nD. four",
        "completed": True, "difficulty_raw": "hard",
    }])
    extraction = {
        "trace_id": "t1", "score_version": ANSWER_EXTRACTION_VERSION,
        "validated": True, "validation_status": "ok", "extraction_status": "ok",
        "extracted_answer": "B", "extracted_answer_sha256": text_sha256("B"),
        "source_response_sha256": text_sha256(full),
        "extraction_input_sha256": extraction_input_sha256(traces.to_dicts()[0]),
        "confidence": "high",
        "judge_model": "extractor", "evidence": "final answer is B",
    }
    grade = grade_traces(traces, {"t1": extraction}).to_dicts()[0]
    assert grade["success"] == 1
    assert grade["extraction_used"]
    assert grade["grade_method"].startswith("llm_extraction/")

    incomplete = traces.with_columns(pl.lit(False).alias("completed"))
    incomplete_grade = grade_traces(incomplete, {"t1": extraction}).to_dicts()[0]
    assert incomplete_grade["success"] is None
    assert not incomplete_grade["gradeable"]
    assert incomplete_grade["grade_status"] == "incomplete_generation"


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

    for idx in range(2):
        pl.DataFrame({
            "trace_id": [f"e{idx}"], "validated": [True],
            "score_version": ["answer-extraction-v1"],
            "judge_model": ["judge"],
            "extraction_fingerprint": ["fingerprint"],
        }).write_parquet(
            tmp_path / f"answer_extractions__judge.shard{idx:02d}of02.parquet")
    extracted = merge_judge_kind("answer_extractions", "judge", tmp_path, 2)
    assert pl.read_parquet(extracted).height == 2
