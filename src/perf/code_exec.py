"""Compatibility API for the canonical LiveCodeBench execution grader.

Production CLI: ``python -m src.perf.grade_code_exec``.  This module keeps the
older ``grade_code`` function but delegates extraction, test decoding, target-method
resolution, resource limits, and result semantics to that single implementation.
Generated code is untrusted; use only in a disposable, network-isolated worker.
"""
from __future__ import annotations

from typing import Optional

from .grade_code_exec import (decode_tests, derive_fn_name, extract_submission,
                              grade_submission)


def extract_code(answer_text: str) -> Optional[str]:
    return extract_submission(answer_text)[0]


def grade_code(answer_text: str, metadata: dict, *,
               timeout_s: float = 6.0, mem_mb: int = 2048) -> dict:
    code, language = extract_submission(answer_text)
    tests = decode_tests(metadata)
    if code and language != "python":
        return {"pass_rate": None, "passed_all": None, "n_tests": len(tests),
                "n_passed": 0, "status": f"unsupported_language:{language}",
                "gradeable": False}
    result = grade_submission(code, tests, derive_fn_name(metadata),
                              max(1, int(timeout_s)), mem_mb, max(1, int(timeout_s)))
    total, passed = result["total"], result["passed"]
    return {"pass_rate": (passed / total) if result["gradeable"] and total else None,
            "passed_all": (bool(result["success"]) if result["success"] is not None else None),
            "n_tests": total, "n_passed": passed, "status": result["err"] or "ok",
            "gradeable": result["gradeable"], "language": language}
