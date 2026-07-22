"""src/perf/grade_code_exec.py — execution-based pass@1 grading for the `code` domain (LiveCodeBench).

SUCCESS = passes ALL test cases (public + private). LiveCodeBench pass@1 with one sample/problem.

SAFETY (read before running):
  - This EXECUTES model-generated code. Run ONLY in a sandboxed compute job, never a login node.
  - Each submission runs in a separate process with: wall-clock timeout, address-space (memory) cap,
    CPU-time cap, and stdout/stderr captured. We also blank out a few obvious escape hatches.
  - This is best-effort isolation (rlimits + subprocess), NOT a security sandbox. Run inside SLURM
    on a compute node, ideally with no outbound network. Do not run untrusted code anywhere you care about.

Two test types in LiveCodeBench:
  functional : call Solution().<method>(*args) ; compare return value
  stdin      : run the program as a script, feed `input` on stdin, compare stdout (whitespace-normalized)

Test cases live in trace instance_metadata: public_test_cases (JSON str),
private_test_cases (base64 -> zlib -> pickle -> JSON str).

Usage (cluster compute job, NOT login node):
  python -m src.perf.grade_code_exec --traces-glob "data/traces/traces_*.parquet" \
      --out data/perf/code_grades.parquet --timeout 8
"""
from __future__ import annotations
import argparse, ast, base64, hashlib, json, multiprocessing as mp, pickle, re, resource, sys, zlib
from pathlib import Path
import polars as pl

from src.utils.io import resolve_trace_paths


CODE_GRADE_VERSION = "code-exec-v2"


# ---------- test-case decoding ----------
def decode_tests(meta: dict):
    """Return (public+private) list of {input, output, testtype}."""
    tests = []
    try:
        tests += json.loads(meta.get("public_test_cases") or "[]")
    except Exception:
        pass
    priv = meta.get("private_test_cases")
    if priv:
        try:
            raw = base64.b64decode(priv)
            dec = pickle.loads(zlib.decompress(raw))      # -> usually a JSON string
            if isinstance(dec, (bytes, str)):
                dec = json.loads(dec)
            tests += dec
        except Exception:
            try:                                          # fallback: zlib->json directly
                tests += json.loads(zlib.decompress(base64.b64decode(priv)).decode())
            except Exception:
                pass
    return tests


# ---------- code extraction from a free-text answer ----------
def extract_submission(answer: str) -> tuple[str | None, str | None]:
    """Return ``(code, language)`` without pretending non-Python code is Python.

    Older generations were not constrained to a language, so C++/Java fences are
    common.  This runner intentionally supports Python only; other languages are
    recorded as unsupported rather than scored as wrong answers.
    """
    if not answer:
        return None, None
    blocks = re.findall(r"```\s*([\w+#.-]*)\s*\n(.*?)```", answer, flags=re.S | re.I)
    if blocks:
        for lang, b in reversed(blocks):
            if "def " in b or "class " in b or len(b.strip()) > 40:
                raw = (lang or "").lower()
                language = ({"py": "python", "python3": "python", "c++": "cpp",
                             "cc": "cpp", "cxx": "cpp"}.get(raw, raw) or _infer_language(b))
                return b.strip(), language
        lang, b = blocks[-1]
        return b.strip(), (lang.lower() or _infer_language(b))
    # no fence: grab from the first class/def to the end of the answer
    m = re.search(
        r"(?:^|\n)(?:\s*)(#include\s*<|using\s+namespace\s+std|public\s+class\b|"
        r"import\s+java\.|class\s+Solution\b|def\s+\w+\s*\(|int\s+main\s*\()", answer)
    if m:
        code = answer[m.start():].strip()
        return code, _infer_language(code)
    return None, None


def _infer_language(code: str) -> str:
    if re.search(r"#include\s*<|using\s+namespace\s+std", code):
        return "cpp"
    if re.search(r"import\s+java\.|public\s+class|public\s+static\s+void\s+main", code):
        return "java"
    if re.search(r"\bpublic:|\bvector\s*<|\bint\s+main\s*\(", code):
        return "cpp"
    return "python"


def extract_code(answer: str, starter: str = "") -> str | None:
    """Backward-compatible code-only extraction helper."""
    return extract_submission(answer)[0]


def derive_fn_name(meta: dict) -> str | None:
    if meta.get("fn_name"):
        return str(meta["fn_name"])
    starter = str(meta.get("starter_code") or "")
    methods = re.findall(r"^\s{1,8}def\s+([A-Za-z_]\w*)\s*\(", starter, flags=re.M)
    public = [m for m in methods if not m.startswith("_")]
    if public:
        return public[0]
    funcs = re.findall(r"^def\s+([A-Za-z_]\w*)\s*\(", starter, flags=re.M)
    return funcs[0] if funcs else None


# ---------- the sandboxed worker (runs in a child process) ----------
def _limit_resources(mem_mb: int, cpu_s: int):
    soft = mem_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except Exception:
        pass


def _parse_literal(value):
    if not isinstance(value, str):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except Exception:
            pass
    return value.strip()


def _value_equal(got, expected) -> bool:
    exp = _parse_literal(expected)
    if isinstance(got, float) and isinstance(exp, (int, float)):
        return abs(got - float(exp)) <= 1e-6 * max(1.0, abs(float(exp)))
    if isinstance(got, tuple) and isinstance(exp, list):
        got = list(got)
    return got == exp


def _run_one(code: str, tests: list, fn_name, mem_mb, cpu_s, q):
    """Queue variant (kept for compatibility)."""
    class _P:
        def send(self, x): q.put(x)
    _run_one_pipe(code, tests, fn_name, mem_mb, cpu_s, _P())


def _run_one_pipe(code: str, tests: list, fn_name, mem_mb, cpu_s, conn):
    """Executed in child process. Sends (passed:int, total:int, err:str) over the connection.
    Mode decided ONCE per problem: functional (Solution class, call method) vs stdin (script + stdin)."""
    import io, contextlib
    passed = 0; total = len(tests); err = ""
    # RLIMIT_CPU applies to the whole child, not one test.  Budget the complete
    # suite while retaining the parent hard-wall cap.
    _limit_resources(mem_mb, min(max(cpu_s, cpu_s * max(1, total)), 180))

    # decide mode from the test cases (LiveCodeBench tags each) + code shape
    ttypes = {tc.get("testtype") for tc in tests}
    is_functional = ("functional" in ttypes) and ("class Solution" in code or "def " in code) and "stdin" not in ttypes
    # if mixed/ambiguous, prefer functional only if a Solution class exists
    if "functional" in ttypes and "stdin" in ttypes:
        is_functional = "class Solution" in code

    if is_functional:
        # LiveCodeBench starter annotations commonly use these names without imports.
        import bisect, collections, functools, heapq, itertools, math, typing
        ns: dict = {name: getattr(typing, name) for name in dir(typing)}
        ns.update({"collections": collections, "functools": functools, "heapq": heapq,
                   "itertools": itertools, "math": math, "bisect": bisect,
                   "deque": collections.deque, "Counter": collections.Counter,
                   "defaultdict": collections.defaultdict, "heappush": heapq.heappush,
                   "heappop": heapq.heappop, "bisect_left": bisect.bisect_left,
                   "bisect_right": bisect.bisect_right})
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(code, ns)
        except Exception as e:
            try: conn.send((0, total, f"compile/exec: {type(e).__name__}: {str(e)[:80]}"))
            except Exception: pass
            return
        sol_cls = ns.get("Solution")
        method = fn_name
        if sol_cls is not None and method is None:
            try: conn.send((0, total, "missing_fn_name"))
            except Exception: pass
            return
        import signal
        def _alarm(signum, frame):
            raise TimeoutError("per-test timeout")
        for tc in tests:
            inp = tc.get("input", ""); exp = str(tc.get("output", "")).strip()
            try:
                signal.signal(signal.SIGALRM, _alarm); signal.alarm(max(1, int(cpu_s)))
                args = [_parse_literal(x) for x in str(inp).split("\n") if x.strip()]
                with contextlib.redirect_stdout(io.StringIO()):
                    if sol_cls is not None and method:
                        got = getattr(sol_cls(), method)(*args)
                    else:
                        got = ns.get(fn_name or "solve", lambda *a: None)(*args)
                signal.alarm(0)
                ok = _value_equal(got, exp)
                passed += int(ok)
            except Exception as e:
                signal.alarm(0)
                err = f"{type(e).__name__}: {str(e)[:60]}"; continue
    else:
        # stdin mode: re-exec code as __main__ with stdin piped, per test. No top-level pre-exec.
        compiled = None
        try:
            compiled = compile(code, "<sub>", "exec")
        except Exception as e:
            try: conn.send((0, total, f"compile: {type(e).__name__}: {str(e)[:80]}"))
            except Exception: pass
            return
        import signal
        def _alarm(signum, frame):
            raise TimeoutError("per-test timeout")
        per_test = max(2, int(cpu_s))   # per-test wall budget inside the child
        for tc in tests:
            inp = tc.get("input", ""); exp = str(tc.get("output", "")).strip()
            out_buf = io.StringIO()
            try:
                signal.signal(signal.SIGALRM, _alarm); signal.alarm(per_test)
                with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(io.StringIO()):
                    sys.stdin = io.StringIO(str(inp))
                    exec(compiled, {"__name__": "__main__"})
                signal.alarm(0)
                ok = " ".join(out_buf.getvalue().split()) == " ".join(exp.split())
                passed += int(ok)
            except Exception as e:
                signal.alarm(0)
                err = f"{type(e).__name__}: {str(e)[:60]}"; continue
        try: signal.alarm(0)
        except Exception: pass
    try: conn.send((passed, total, err))
    except Exception: pass


def grade_submission(code, tests, fn_name, timeout, mem_mb, cpu_s):
    """Run one submission in a child process with a hard wall-clock timeout. Returns dict.
    Uses fork (pure-CPU grading, no GPU) for speed + no re-import. Pipe for result (no Queue deadlock).
    Parent wall-clock scales with test count (child enforces per-test budget) so many-test
    submissions aren't killed just for having many tests."""
    if not code:
        return {"success": None, "passed": 0, "total": len(tests or []),
                "err": "no_code", "gradeable": False}
    if not tests:
        return {"success": None, "passed": 0, "total": 0,
                "err": "no_tests", "gradeable": False}
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    def _target():
        _run_one_pipe(code, tests, fn_name, mem_mb, cpu_s, child_conn)

    p = ctx.Process(target=_target)
    p.start(); child_conn.close()
    # outer budget: per-test child budget (cpu_s) * n + slack, capped so nothing runs forever
    wall = min(timeout + cpu_s * len(tests), 180)
    p.join(wall)
    if p.is_alive():
        p.terminate(); p.join(0.5)
        if p.is_alive():
            p.kill(); p.join(0.5)
        return {"success": 0, "passed": 0, "total": len(tests),
                "err": "timeout", "gradeable": True}
    if parent_conn.poll():
        try:
            passed, total, err = parent_conn.recv()
            return {"success": int(passed == total and total > 0), "passed": passed,
                    "total": total, "err": err, "gradeable": True}
        except Exception:
            pass
    return {"success": None, "passed": 0, "total": len(tests),
            "err": f"worker_no_result_exit_{p.exitcode}", "gradeable": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out", default="data/perf/code_grades.parquet")
    ap.add_argument("--timeout", type=int, default=8, help="wall-clock seconds per submission")
    ap.add_argument("--mem-mb", type=int, default=2048)
    ap.add_argument("--cpu-s", type=int, default=10)
    ap.add_argument("--max-tests", type=int, default=0,
                    help="optional test cap for debugging only; 0 grades the full official suite")
    args = ap.parse_args()

    paths = resolve_trace_paths(args.traces_glob)
    if not paths:
        raise SystemExit(f"no traces matched {args.traces_glob}")
    tr = pl.concat([pl.read_parquet(p) for p in paths],
                   how="diagonal_relaxed").filter(pl.col("task_type") == "code")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = set(pl.read_parquet(out)["trace_id"].to_list())

    rows = []
    n = tr.height
    for k, r in enumerate(tr.iter_rows(named=True)):
        tid = r["trace_id"]
        if tid in done:
            continue
        try:
            meta = json.loads(r["instance_metadata"])
        except Exception:
            meta = {}
        tests = decode_tests(meta)
        if args.max_tests and len(tests) > args.max_tests:
            tests = tests[:args.max_tests]
        answer = r.get("answer_text") or ""
        code, language = extract_submission(answer)
        fn_name = derive_fn_name(meta)
        if code and language != "python":
            res = {"success": None, "passed": 0, "total": len(tests),
                   "err": f"unsupported_language:{language}", "gradeable": False}
        else:
            res = grade_submission(code, tests, fn_name,
                                   args.timeout, args.mem_mb, args.cpu_s)
        rows.append({"trace_id": tid, "task_type": "code", "success": res["success"],
                     "parsed": code is not None, "completed": (r.get("finish_reason") == "stop"),
                     "difficulty_raw": r.get("difficulty_raw"),
                     "tests_passed": res["passed"], "tests_total": res["total"],
                     "exec_err": res["err"], "gradeable": res["gradeable"],
                     "language": language, "target_fn": fn_name,
                     "grader_version": CODE_GRADE_VERSION,
                     "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                     "grade_method": "python_execution_full_suite"})
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{n} graded")
    res_df = pl.DataFrame(rows)
    if out.exists() and res_df.height:
        res_df = pl.concat([pl.read_parquet(out), res_df], how="diagonal_relaxed").unique("trace_id", keep="last")
    if res_df.height:
        tmp = out.with_suffix(out.suffix + ".tmp")
        res_df.write_parquet(tmp); tmp.replace(out)
    # report
    g = pl.read_parquet(out) if out.exists() else res_df
    j = g.join(tr.select(["trace_id", "gen_model"]), on="trace_id", how="left")
    print(f"\ncode grades -> {out} ({g.height} traces)")
    for m in sorted(j["gen_model"].drop_nulls().unique().to_list()):
        s = j.filter(pl.col("gen_model") == m)
        if s.height:
            gs = s.filter(pl.col("gradeable"))
            pass1 = gs["success"].mean() if gs.height else float("nan")
            print(f"  {m:9s} pass@1_gradeable={pass1:.3f} | gradeable={gs.height/s.height:.0%} "
                  f"| code-extracted={s['parsed'].mean():.0%} "
                  f"| timeouts={100*(s['exec_err']=='timeout').mean():.0f}% (n={s.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
