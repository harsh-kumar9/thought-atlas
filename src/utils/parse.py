"""src/utils/parse.py — Per-model think/answer parsing for the two Llama-line models.

Cross-model design (not Qwen thinking on/off):
  * DeepSeek-R1-Distill-Llama-8B (`reasoner` key): native <think>...</think>. The
    chat template opens <think>; the model emits </think> then the final answer.
    We split on </think>: prefix=think_text, suffix=answer_text. Missing </think>
    => truncated/looped (completed=False).
  * Llama-3.1-8B-Instruct (`anchor` key): no think block. Whole generation is the answer;
    think_text=None. This is the non-reasoning control.

`reasoning_text_for_analysis` = think_text for DeepSeek-R1-Distill-Llama-8B,
answer_text for Llama-3.1-8B-Instruct (config: gen_models.*.analysis_source).
"""

from __future__ import annotations

from typing import Optional


def parse_generation_detailed(generation_text: str, kind: str, *,
                              finish_reason: Optional[str] = None) -> dict:
    """Parse a generation while preserving delimiter diagnostics.

    Reasoning templates often inject the opening tag into the prompt, so generated
    text may legitimately contain only closing tags.  Multiple closing tags denote
    repeated think/final cycles; the suffix after the *last* close is the final
    visible answer.  A stopped response with no close is treated as a direct-answer
    fallback, while a length-truncated response remains thought-only.
    """
    text = generation_text or ""
    if kind == "non_reasoning":
        answer = text.strip()
        return {"think_text": None, "answer_text": answer,
                "parse_status": "non_reasoning", "answer_source": "full_generation",
                "close_tag_count": 0}

    close_count = text.count("</think>")
    if close_count == 0:
        thought = text.replace("<think>", " ").strip()
        if finish_reason == "stop" and thought:
            # This may be a direct answer or a response whose special delimiter
            # was lost. Preserve it for grading, but do not contaminate
            # reasoning-only analyses with unseparated answer prose.
            return {"think_text": None, "answer_text": thought,
                    "parse_status": "direct_answer_no_close",
                    "answer_source": "full_generation_fallback", "close_tag_count": 0}
        return {"think_text": thought, "answer_text": "",
                "parse_status": "missing_close_truncated",
                "answer_source": "none", "close_tag_count": 0}

    pre, post = text.rsplit("</think>", 1)
    # Multiple cycles can leave earlier canonical delimiters inside the prefix.
    # They are diagnostics, not content; flatten them before downstream segmentation.
    thought = pre.replace("<think>", " ").replace("</think>", " ").strip()
    answer = post.strip()
    status = "single_close" if close_count == 1 else "multiple_close_last_suffix"
    return {"think_text": thought, "answer_text": answer,
            "parse_status": status, "answer_source": "after_last_close",
            "close_tag_count": close_count}


def parse_generation(generation_text: str, kind: str, *,
                     finish_reason: Optional[str] = None) -> tuple[Optional[str], str]:
    """kind in {'reasoning','non_reasoning'} -> (think_text|None, answer_text)."""
    result = parse_generation_detailed(generation_text, kind, finish_reason=finish_reason)
    return result["think_text"], result["answer_text"]


def is_completed(*, kind: str, finish_reason: str, has_close_tag: bool, has_answer: bool) -> bool:
    if kind == "reasoning":
        return has_close_tag and has_answer and finish_reason == "stop"
    return has_answer and finish_reason == "stop"


def reasoning_text_for_analysis(*, analysis_source: str, think_text: Optional[str],
                                answer_text: str) -> str:
    return (think_text or "") if analysis_source == "think_text" else answer_text
