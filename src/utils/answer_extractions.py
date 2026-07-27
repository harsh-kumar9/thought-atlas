"""Validated consumption of LLM answer-extraction artifacts.

The extractor never receives reference answers.  Consumers still treat its
parquet as untrusted: a row is usable only when it was evidence-validated, was
produced by the current extraction contract, and hashes the exact trace text
being graded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import polars as pl


ANSWER_EXTRACTION_VERSION = "answer-extraction-v1"


def text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def extraction_input_sha256(trace: dict) -> str:
    """Hash every reference-free field that can affect extraction."""
    payload = json.dumps({
        "task_type": trace.get("task_type"),
        "prompt": trace.get("prompt"),
        "full_text": trace.get("full_text"),
        "generation_kind": trace.get("generation_kind"),
        "parse_status": trace.get("parse_status"),
        "finish_reason": trace.get("finish_reason"),
        "completed": trace.get("completed"),
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return text_sha256(payload)


def load_extraction_index(path: Optional[str]) -> dict[str, dict]:
    if not path:
        return {}
    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(f"answer-extraction artifact not found: {artifact}")
    frame = pl.read_parquet(artifact)
    if "trace_id" not in frame.columns:
        raise ValueError(f"{artifact} has no trace_id column")
    if frame["trace_id"].n_unique() != frame.height:
        raise ValueError(f"{artifact} contains duplicate trace_id rows")
    return {str(row["trace_id"]): row for row in frame.iter_rows(named=True)}


def select_answer(trace: dict, extraction_index: dict[str, dict]) -> dict:
    """Choose a validated extracted answer, otherwise retain the stored answer.

    ``selection_status`` makes every fallback explicit.  In particular, an LLM
    ``no_answer``/``ambiguous`` opinion does not erase a deterministic answer;
    only an evidence-validated ``ok`` row can replace it.
    """
    original = str(trace.get("answer_text") or "")
    trace_id = str(trace.get("trace_id") or "")
    row = extraction_index.get(trace_id)
    base = {
        "answer": original,
        "used": False,
        "selection_status": "no_extraction_row" if row is None else "not_selected",
        "extractor_status": row.get("extraction_status") if row else None,
        "extractor_confidence": row.get("confidence") if row else None,
        "extractor_model": row.get("judge_model") if row else None,
        "extracted_answer": row.get("extracted_answer") if row else None,
        "evidence": row.get("evidence") if row else None,
        "selected_code_language": row.get("selected_code_language") if row else None,
    }
    if row is None:
        return base
    if row.get("score_version") != ANSWER_EXTRACTION_VERSION:
        base["selection_status"] = "incompatible_extraction_version"
        return base
    if not bool(row.get("validated")) or row.get("extraction_status") != "ok":
        base["selection_status"] = str(row.get("validation_status") or
                                       row.get("extraction_status") or "invalid_extraction")
        return base
    if row.get("source_response_sha256") != text_sha256(str(trace.get("full_text") or "")):
        base["selection_status"] = "stale_response_hash"
        return base
    if row.get("extraction_input_sha256") != extraction_input_sha256(trace):
        base["selection_status"] = "stale_extraction_input_hash"
        return base
    extracted = row.get("extracted_answer")
    if not isinstance(extracted, str) or not extracted.strip():
        base["selection_status"] = "blank_extracted_answer"
        return base
    if row.get("extracted_answer_sha256") != text_sha256(extracted):
        base["selection_status"] = "extracted_answer_hash_mismatch"
        return base
    full_text = str(trace.get("full_text") or "")
    task_type = str(trace.get("task_type") or "")
    evidence = row.get("evidence")
    if task_type in {"math", "gpqa", "planning", "security", "code"}:
        if not isinstance(evidence, str) or not evidence or evidence not in full_text:
            base["selection_status"] = "evidence_contract_failed"
            return base
        last_close = full_text.rfind("</think>")
        if (task_type in {"math", "gpqa", "planning", "security"} and last_close >= 0 and
                full_text.rfind(evidence) < last_close + len("</think>")):
            base["selection_status"] = "evidence_in_reasoning"
            return base
    if task_type == "math":
        from src.perf.answer_grading import grade_math_answer
        if grade_math_answer(evidence, extracted).get("success") != 1:
            base["selection_status"] = "math_evidence_mismatch"
            return base
    if task_type in {"gpqa", "planning", "security"}:
        from src.perf.answer_grading import extract_choice
        if extract_choice(evidence, str(trace.get("prompt") or ""))[0] != extracted.upper():
            base["selection_status"] = "mcq_evidence_mismatch"
            return base
    if task_type in {"code", "safety", "moral", "idea"} and extracted not in full_text:
        base["selection_status"] = "extracted_answer_not_verbatim"
        return base
    if task_type == "code" and evidence not in extracted:
        base["selection_status"] = "code_evidence_not_in_answer"
        return base
    if task_type in {"safety", "moral", "idea"}:
        last_close = full_text.rfind("</think>")
        if last_close >= 0 and full_text.rfind(extracted) < last_close + len("</think>"):
            base["selection_status"] = "extracted_answer_in_reasoning"
            return base
        if last_close >= 0 and extracted.strip() != full_text[
                last_close + len("</think>"):].strip():
            base["selection_status"] = "incomplete_post_reasoning_answer"
            return base
        if (last_close < 0 and trace.get("generation_kind") == "non_reasoning" and
                extracted.strip() != full_text.strip()):
            base["selection_status"] = "incomplete_non_reasoning_answer"
            return base
    base.update({"answer": extracted, "used": True, "selection_status": "validated_extraction"})
    return base
