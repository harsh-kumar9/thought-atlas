"""CPU-runnable smoke/logic tests (no GPU, no external model). Run: python -m pytest tests/ -q
Covers: perf metrics, IRR math, per-model parse, and segmenter determinism.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import polars as pl


def test_math_grader():
    from src.perf.metrics import grade_math
    assert grade_math(r"so \boxed{42}", "42")["is_correct"] is True
    assert grade_math(r"answer is 7", "42")["is_correct"] is False


def test_mcq_grader_parens():
    from src.perf.metrics import grade_mcq
    assert grade_mcq("After checking, final answer: (B)", "B")["is_correct"] is True
    assert grade_mcq("Final answer: A", "B")["is_correct"] is False


def test_irr_math():
    from src.judge.bakeoff import _krippendorff_binary, _cohen_kappa
    a = np.array([1, 0, 1, 1, 0, 0])
    assert abs(_krippendorff_binary(a, a) - 1.0) < 1e-9
    assert _cohen_kappa(a, np.array([0, 1, 0, 0, 1, 1])) < 0  # anti-correlated


def test_parse_per_model():
    from src.utils.parse import parse_generation
    th, ans = parse_generation("reasoning here</think>final answer", "reasoning")
    assert th == "reasoning here" and ans == "final answer"
    th2, ans2 = parse_generation("just an answer", "non_reasoning")
    assert th2 is None and ans2 == "just an answer"


def test_segmenter_deterministic():
    from src.segment.segmenter import segment_text
    txt = "First I compute 2+2=4. Wait, that's wrong. Let me recheck. So the answer is 4."
    a = segment_text(txt); b = segment_text(txt)
    assert [s["sentence"] for s in a] == [s["sentence"] for s in b]
    assert all(0.0 <= s["norm_pos"] <= 1.0 for s in a)
