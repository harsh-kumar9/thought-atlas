"""Canonical, auditable answer extraction and grading.

This module is the single implementation used by both the production grader and
the convenience metric API.  Every result preserves the extracted prediction,
method, and status so parse failures cannot silently become model failures.
"""
from __future__ import annotations

import re
from typing import Optional


GRADE_VERSION = "objective-v2"


def _balanced_boxed(text: str) -> list[str]:
    out: list[str] = []
    start = 0
    while True:
        idx = text.find(r"\boxed", start)
        if idx < 0:
            break
        left = text.find("{", idx)
        if left < 0:
            break
        depth = 0
        for right in range(left, len(text)):
            if text[right] == "{":
                depth += 1
            elif text[right] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[left + 1:right].strip())
                    start = right + 1
                    break
        else:
            break
    return out


def _clean_math_candidate(value: str) -> str:
    value = value.strip().strip("` ")
    if value.startswith("**") and value.endswith("**") and len(value) > 4:
        value = value[2:-2].strip()
    if value.startswith("$") and value.endswith("$") and len(value) > 2:
        value = value[1:-1].strip()
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]")):
        if value.startswith(left) and value.endswith(right):
            value = value[len(left):-len(right)].strip()
    value = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*", "", value,
                   flags=re.I)
    return value.strip()


def extract_math_candidates(answer_text: str) -> list[tuple[str, str]]:
    """Return final-answer-focused candidates ordered strongest first."""
    text = answer_text or ""
    candidates: list[tuple[str, str]] = []
    boxes = _balanced_boxed(text)
    if boxes:
        candidates.append((_clean_math_candidate(boxes[-1]), "last_boxed"))

    marker = re.compile(
        r"(?im)(?:^|\n)\s*(?:\*\*)?(?:final\s+answer|answer)(?:\*\*)?\s*"
        r"(?:is|:|=)?\s*(.+?)\s*$")
    marked = list(marker.finditer(text))
    if marked:
        candidates.append((_clean_math_candidate(marked[-1].group(1)), "final_answer_line"))

    # Last displayed/inline math expression is often the answer even when the
    # model omits an explicit label.
    math_spans = re.findall(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]", text, flags=re.S)
    if math_spans:
        value = next((part for part in math_spans[-1] if part), "")
        candidates.append((_clean_math_candidate(value), "last_math_span"))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and len(lines[-1]) <= 300:
        last = _clean_math_candidate(lines[-1])
        # A generic prose line is not evidence that an answer was parsed.  Keep
        # this fallback only for compact mathematical-looking terminal lines.
        words = re.findall(r"[A-Za-z]+", last)
        looks_math = (bool(re.search(r"\d|\\(?:frac|sqrt|pi|infty)|[=<>±]", last))
                      and len(words) <= 8)
        if looks_math:
            candidates.append((last, "last_math_line"))

    seen = set()
    out = []
    for value, method in candidates:
        if value and value not in seen:
            seen.add(value)
            out.append((value, method))
    return out


def _norm_math(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    x = value.strip().lower()
    x = x.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    x = x.replace(r"\left", "").replace(r"\right", "")
    x = re.sub(r"\\(?:text|mbox)\{[^{}]*\}", "", x)
    x = x.replace("$", "").replace(r"\!", "").replace(r"\,", "")
    x = x.replace(r"\%", "%").replace("°", r"^\circ")
    x = re.sub(r"\s+", "", x).rstrip(".")
    return x


def _candidate_equal(prediction: str, reference: str) -> bool:
    p, r = _norm_math(prediction), _norm_math(reference)
    if not p or not r:
        return False
    if p == r:
        return True
    try:
        return abs(float(p) - float(r)) < 1e-9
    except Exception:
        return False


def _math_verify(answer_text: str, reference: str) -> tuple[Optional[bool], bool, str]:
    try:
        from math_verify import parse, verify
    except Exception:
        return None, False, "math_verify_unavailable"
    try:
        gold = parse(reference)
        pred = parse(answer_text)
        parsed = bool(pred)
        if not parsed or not gold:
            return None, parsed, "math_verify_parse_fail"
        return bool(verify(gold, pred)), True, "math_verify"
    except Exception as exc:
        return None, False, f"math_verify_error:{type(exc).__name__}"


def grade_math_answer(answer_text: str, reference: Optional[str]) -> dict:
    if not reference:
        return {"prediction": None, "parsed": False, "success": None,
                "parse_method": None, "status": "no_reference"}

    candidates = extract_math_candidates(answer_text)
    for prediction, method in candidates:
        if _candidate_equal(prediction, reference):
            return {"prediction": prediction, "parsed": True, "success": 1,
                    "parse_method": method, "status": "exact_normalized"}

    verified, mv_parsed, mv_status = _math_verify(answer_text or "", reference)
    prediction = candidates[0][0] if candidates else None
    method = candidates[0][1] if candidates else ("math_verify" if mv_parsed else None)
    parsed = bool(candidates) or mv_parsed
    if verified is True:
        return {"prediction": prediction, "parsed": True, "success": 1,
                "parse_method": method, "status": mv_status}
    if parsed:
        return {"prediction": prediction, "parsed": True, "success": 0,
                "parse_method": method, "status": mv_status if mv_parsed else "parsed_not_equivalent"}
    return {"prediction": None, "parsed": False, "success": None,
            "parse_method": None, "status": mv_status}


_MCQ_EXPLICIT = re.compile(
    r"(?i)\b(?:final\s+answer|correct\s+answer|answer|choice)\s*"
    r"(?:is\s*)?(?::|=)?\s*(?:option\s*)?(?:\\boxed\{\s*)?"
    r"(?:\*\*)?\(?([A-E])\b")
_MCQ_BOXED = re.compile(r"(?i)\\boxed\{\s*([A-E])\s*\}")
_MCQ_TEXT_BOXED = re.compile(r"(?i)\\boxed\{\s*\\text\{\s*([A-E])\s*\}\s*\}")
_MCQ_TERMINAL = re.compile(r"(?im)^\s*(?:\*\*)?\(?([A-E])\)?(?:\*\*)?[.)]?\s*$")
_MCQ_CONCLUSION = re.compile(
    r"(?i)\b(?:therefore|thus|hence|so)\b[^\n]{0,100}?\b(?:option|choice|answer)\s*"
    r"(?:is\s*)?(?:\*\*)?\(?([A-E])\b")


def _prompt_options(prompt: str) -> dict[str, str]:
    matches = re.findall(r"(?m)^\s*([A-E])\.\s+(.+?)\s*$", prompt or "")
    # If malformed legacy data has multiple option blocks, the last occurrence
    # for each letter is the block nearest the answer-format instruction.
    return {letter.upper(): text.strip() for letter, text in matches}


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def extract_choice(answer_text: str, prompt: str = "") -> tuple[Optional[str], Optional[str]]:
    text = answer_text or ""
    hits: list[tuple[int, int, str, str]] = []
    for match in _MCQ_EXPLICIT.finditer(text):
        hits.append((match.start(), 3, match.group(1).upper(), "explicit_final_marker"))
    for match in _MCQ_BOXED.finditer(text):
        hits.append((match.start(), 2, match.group(1).upper(), "boxed_letter"))
    for match in _MCQ_TEXT_BOXED.finditer(text):
        hits.append((match.start(), 2, match.group(1).upper(), "boxed_text_letter"))
    for match in _MCQ_CONCLUSION.finditer(text):
        hits.append((match.start(), 2, match.group(1).upper(), "conclusion_marker"))
    for match in _MCQ_TERMINAL.finditer(text[-1200:]):
        hits.append((len(text) - min(len(text), 1200) + match.start(), 1,
                     match.group(1).upper(), "terminal_letter"))
    if hits:
        _, _, letter, method = max(hits, key=lambda item: (item[0], item[1]))
        return letter, method

    options = _prompt_options(prompt)
    tail = _norm_text(text[-1600:])
    semantic = []
    for letter, option in options.items():
        normalized = _norm_text(option)
        if len(normalized) >= 12 and normalized in tail:
            semantic.append(letter)
    if len(semantic) == 1:
        return semantic[0], "option_text_match"
    return None, None


def grade_mcq_answer(answer_text: str, reference: Optional[str], prompt: str = "") -> dict:
    if not reference:
        return {"prediction": None, "parsed": False, "success": None,
                "parse_method": None, "status": "no_reference"}
    prediction, method = extract_choice(answer_text, prompt)
    if prediction is None:
        return {"prediction": None, "parsed": False, "success": None,
                "parse_method": None, "status": "parse_fail"}
    correct = reference.strip().upper()
    return {"prediction": prediction, "parsed": True,
            "success": int(prediction == correct), "parse_method": method, "status": "ok"}
