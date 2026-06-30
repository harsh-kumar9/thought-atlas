"""src/utils/parse.py — Per-model think/answer parsing for the two generators.

Cross-model design (not Qwen thinking on/off):
  * reasoner (DeepSeek-R1-Distill-Llama-8B): native <think>...</think>. The chat template
    opens <think>; the model emits </think> then the final answer. We split on </think>:
    prefix=think_text, suffix=answer_text. Missing </think> => truncated/looped (completed=False).
  * anchor (Llama-3.1-8B-Instruct): no think block. Whole generation is the answer;
    think_text=None. This is the non-reasoning control.

`reasoning_text_for_analysis` = think_text for the reasoner, answer_text for the anchor
(config: gen_models.*.analysis_source).
"""

from __future__ import annotations

from typing import Optional


def parse_generation(generation_text: str, kind: str) -> tuple[Optional[str], str]:
    """kind in {'reasoning','non_reasoning'} -> (think_text|None, answer_text)."""
    if kind == "non_reasoning":
        return None, (generation_text or "").strip()
    # reasoning
    if "</think>" not in generation_text:
        return (generation_text or "").strip(), ""   # truncated/looped: keep think, empty answer
    pre, post = generation_text.split("</think>", 1)
    return pre.replace("<think>", "").strip(), post.strip()


def is_completed(*, kind: str, finish_reason: str, has_close_tag: bool, has_answer: bool) -> bool:
    if kind == "reasoning":
        return has_close_tag and has_answer and finish_reason == "stop"
    return finish_reason == "stop"


def reasoning_text_for_analysis(*, analysis_source: str, think_text: Optional[str],
                                answer_text: str) -> str:
    return (think_text or "") if analysis_source == "think_text" else answer_text
