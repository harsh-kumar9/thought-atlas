"""src/perf/metrics.py — Per-task performance metrics dispatcher.

Performance is a covariate / bifurcator (correct vs incorrect), NOT a mediator.
Each task type maps to one metric. Returns a dict merged onto the trace row; the
canonical boolean is `is_correct` (None where unassessable) plus a continuous
`perf_score` where one exists (code pass_rate, plan goal, rubric adherence).

  math      -> math-verify symbolic equivalence (fallback: normalized string match)
  code      -> sandboxed execution pass@1 (src/perf/code_exec.py)
  planning  -> ACPBench MCQ letter accuracy (grade_mcq)
  gpqa      -> GPQA 4-way MCQ letter accuracy (grade_mcq)
  moral     -> rubric adherence, LLM-graded (needs a judge callable; held-out family)
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from .code_exec import grade_code


# ----------------------------- math -----------------------------
def _norm_math(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = re.sub(r"[\$\\]|\\left|\\right|\\,", "", s)
    s = re.sub(r"\s+", "", s)
    return s.lower().rstrip(".")


def grade_math(answer_text: str, reference: Optional[str]) -> dict:
    if not reference:
        return {"is_correct": None, "perf_score": None, "status": "no_reference"}
    # Prefer math-verify (sympy equivalence) if installed
    try:
        from math_verify import parse, verify
        gold = parse(reference)
        pred = parse(answer_text)
        ok = bool(verify(gold, pred))
        return {"is_correct": ok, "perf_score": float(ok), "status": "math_verify"}
    except Exception:
        # Fallback: last \boxed{} / "answer is X" vs normalized reference
        boxes = re.findall(r"\\boxed\{([^}]*)\}", answer_text or "")
        cand = boxes[-1] if boxes else None
        if cand is None:
            m = re.search(r"(?:answer\s+is|=)\s*([^\n.]{1,80})\s*$", answer_text or "", re.IGNORECASE)
            cand = m.group(1) if m else None
        if cand is None:
            return {"is_correct": None, "perf_score": None, "status": "parse_fail"}
        ok = _norm_math(cand) == _norm_math(reference)
        return {"is_correct": ok, "perf_score": float(ok), "status": "string_fallback"}


# ----------------------------- MCQ (gpqa, planning/acp) -----------------------------
def grade_mcq(answer_text: str, reference_letter: Optional[str]) -> dict:
    """4-way (or N-way) letter MCQ. reference_letter is the canonical correct letter (A-D),
    set by the loader after seeded option-shuffle. We extract the model's final letter robustly:
    prefer an explicit 'answer: X' / 'final answer X' / boxed/parenthesized letter near the end,
    else the last standalone capital letter A-D in the tail.
    """
    if not reference_letter:
        return {"is_correct": None, "perf_score": None, "status": "no_reference"}
    text = answer_text or ""
    correct = reference_letter.strip().upper()
    tail = text[-400:]
    pick = None
    # strongest signals first
    for pat in (r"(?:final\s+answer|answer|choice)\s*(?:is|:)?\s*\(?\*?\*?([A-D])\b",
                r"\\boxed\{\s*([A-D])\s*\}",
                r"\b([A-D])\b\s*$"):
        m = re.findall(pat, tail, re.IGNORECASE)
        if m:
            pick = m[-1].upper(); break
    if pick is None:
        caps = re.findall(r"\b([A-D])\b", tail)
        pick = caps[-1].upper() if caps else None
    if pick is None:
        return {"is_correct": None, "perf_score": None, "status": "parse_fail"}
    ok = pick == correct
    return {"is_correct": ok, "perf_score": float(ok), "status": "ok", "picked": pick}


# ----------------------------- moral (rubric, LLM-graded) -----------------------------
RUBRIC_PROMPT = """You are grading a model's response to a moral dilemma against a rubric.
Return ONLY JSON: {{"items_met": <int>, "items_total": <int>, "adherence": <float 0..1>}}.

[Dilemma]
{dilemma}
[Rubric — list of criteria the response should satisfy]
{rubric}
[Response to grade]
{response}
"""


def grade_moral(answer_text: str, metadata: dict, *,
                judge_fn: Optional[Callable[[str], dict]] = None) -> dict:
    """LLM-graded rubric adherence. judge_fn(prompt)->dict (a guided-JSON judge call).

    Use a judge from a DIFFERENT family than the generator (R1-Distill-Llama is Llama-lineage).
    If no judge_fn supplied (e.g. dry run), returns status=needs_judge.
    """
    rubric = metadata.get("rubric")
    if not rubric:
        return {"is_correct": None, "perf_score": None, "status": "no_rubric"}
    if judge_fn is None:
        return {"is_correct": None, "perf_score": None, "status": "needs_judge"}
    prompt = RUBRIC_PROMPT.format(
        dilemma=metadata.get("context") or "", rubric=json.dumps(rubric, ensure_ascii=False),
        response=answer_text or "")
    out = judge_fn(prompt) or {}
    adh = out.get("adherence")
    return {"is_correct": (adh is not None and adh >= 0.5) if adh is not None else None,
            "perf_score": adh, "status": "ok" if adh is not None else "judge_parse_fail"}


# ----------------------------- idea (LiveIdeaBench, judge-scored) -----------------------------
IDEA_PROMPT = """You are a strict scientific reviewer scoring a proposed research idea on two axes,
each 1-10. Return ONLY JSON: {{"originality": <int 1-10>, "feasibility": <int 1-10>}}.

[Keyword]
{keyword}
[Proposed idea]
{response}
"""


def grade_idea(answer_text: str, metadata: dict, *,
               judge_fn: Optional[Callable[[str], dict]] = None) -> dict:
    """LiveIdeaBench originality+feasibility, LLM-judged (NOT ground-truth; held-out family).
    perf_score = mean(originality, feasibility)/10; is_correct = perf_score >= 0.6 (above-bar idea).
    """
    if judge_fn is None:
        return {"is_correct": None, "perf_score": None, "status": "needs_judge"}
    prompt = IDEA_PROMPT.format(keyword=metadata.get("keyword") or "", response=answer_text or "")
    out = judge_fn(prompt) or {}
    o, f = out.get("originality"), out.get("feasibility")
    if o is None or f is None:
        return {"is_correct": None, "perf_score": None, "status": "judge_parse_fail"}
    score = (float(o) + float(f)) / 20.0
    return {"is_correct": score >= 0.6, "perf_score": score, "status": "ok",
            "originality": o, "feasibility": f}


# ----------------------------- dispatch -----------------------------
def assess(*, task_type: str, answer_text: str, reference_answer: Optional[str],
           metadata: dict, judge_fn: Optional[Callable[[str], dict]] = None,
           code_timeout_s: float = 6.0) -> dict:
    if task_type == "math":
        return grade_math(answer_text, reference_answer)
    if task_type == "code":
        r = grade_code(answer_text, metadata, timeout_s=code_timeout_s)
        return {"is_correct": r.get("passed_all"), "perf_score": r.get("pass_rate"),
                "status": r.get("status"), "n_tests": r.get("n_tests"), "n_passed": r.get("n_passed")}
    if task_type in ("planning", "gpqa"):       # ACP + GPQA are both letter-MCQ
        return grade_mcq(answer_text, reference_answer)
    if task_type == "moral":
        return grade_moral(answer_text, metadata, judge_fn=judge_fn)
    if task_type == "idea":                     # LiveIdeaBench: judge-scored originality+feasibility
        return grade_idea(answer_text, metadata, judge_fn=judge_fn)
    return {"is_correct": None, "perf_score": None, "status": "unknown_task"}