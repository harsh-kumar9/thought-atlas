"""src/perf/code_exec.py — Sandboxed execution of model-generated code vs LiveCodeBench tests.

Runs each candidate solution in a separate subprocess with wall-clock timeout, capped
memory (resource.RLIMIT_AS), and no network (the SLURM job itself runs without egress on
the compute node; we additionally avoid importing anything that opens sockets). Returns
pass@1 = fraction of test cases passed (1.0 only if ALL pass, plus a finer pass_rate).

SECURITY: this executes untrusted model output. Run ONLY inside the SLURM job sandbox,
never on the login node. Each solution runs in its own `python -c` subprocess; we do not
exec in-process. For stronger isolation use firejail/nsjail if available (hook below).

LiveCodeBench has two harness styles:
  * stdin/stdout: feed `input`, compare stdout to `output`.
  * functional ("fn_name"): call the named function with parsed args, compare return.
We support both; `fn_name` in instance metadata selects functional mode.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional


def extract_code(answer_text: str) -> Optional[str]:
    """Pull the last ```python ...``` block (model's final solution); fallback to last ``` block."""
    if not answer_text:
        return None
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", answer_text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return None


_HARNESS_STDIN = r"""
import sys
_SOLUTION_
"""

_HARNESS_FUNC = r"""
import json, sys
_SOLUTION_
_args = json.loads(sys.stdin.read())
_res = {fn}(*_args)
print(json.dumps(_res))
"""


def _run_one(code: str, stdin_data: str, timeout_s: float, mem_mb: int) -> tuple[bool, str]:
    """Run `code` feeding stdin_data; return (ok, stdout). ok=False on timeout/crash."""
    preexec = None
    if sys.platform != "win32":
        def _limit():
            import resource
            soft = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        preexec = _limit
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            input=stdin_data, capture_output=True, text=True,
            timeout=timeout_s, preexec_fn=preexec,
        )
        if proc.returncode != 0:
            return False, proc.stderr[-500:]
        return True, proc.stdout
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:  # noqa
        return False, f"RUNNER_ERROR: {e}"


def _norm(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def grade_code(answer_text: str, metadata: dict, *,
               timeout_s: float = 6.0, mem_mb: int = 2048) -> dict:
    """Return {pass_rate, passed_all, n_tests, n_passed, status}."""
    code = extract_code(answer_text)
    if not code:
        return {"pass_rate": 0.0, "passed_all": False, "n_tests": 0, "n_passed": 0,
                "status": "no_code_block"}

    # Gather test cases (public + private). LiveCodeBench stores them as JSON strings/lists.
    tests = []
    for key in ("public_test_cases", "private_test_cases"):
        raw = metadata.get(key)
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                continue
        if isinstance(raw, list):
            tests.extend(raw)
    if not tests:
        return {"pass_rate": None, "passed_all": None, "n_tests": 0, "n_passed": 0,
                "status": "no_tests"}

    fn = metadata.get("fn_name")
    n_pass = 0
    for t in tests:
        tin = t.get("input", "") if isinstance(t, dict) else ""
        tout = t.get("output", "") if isinstance(t, dict) else ""
        if fn:
            harness = _HARNESS_FUNC.format(fn=fn).replace("_SOLUTION_", textwrap.dedent(code))
            ok, out = _run_one(harness, tin if isinstance(tin, str) else json.dumps(tin),
                               timeout_s, mem_mb)
            if ok:
                try:
                    if json.loads(out.strip()) == json.loads(tout) if isinstance(tout, str) else tout:
                        n_pass += 1
                except Exception:
                    if _norm(out) == _norm(str(tout)):
                        n_pass += 1
        else:
            harness = _HARNESS_STDIN.replace("_SOLUTION_", textwrap.dedent(code))
            ok, out = _run_one(harness, tin, timeout_s, mem_mb)
            if ok and _norm(out) == _norm(str(tout)):
                n_pass += 1

    n = len(tests)
    return {"pass_rate": n_pass / n if n else 0.0, "passed_all": n_pass == n,
            "n_tests": n, "n_passed": n_pass, "status": "ok"}
