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
      --out data/perf/code_grades.parquet --workers 16 --timeout 8
"""
from __future__ import annotations
import argparse, ast, base64, json, multiprocessing as mp, pickle, re, resource, sys, zlib
from pathlib import Path
import polars as pl


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
def extract_code(answer: str, starter: str = "") -> str | None:
    """Pull the submission. Prefer the LAST ```python fenced block; else a class Solution / def region."""
    if not answer:
        return None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, flags=re.S | re.I)
    if blocks:
        # last substantive block (skip tiny ones that are just imports/examples)
        for b in reversed(blocks):
            if "def " in b or "class " in b or len(b.strip()) > 40:
                return b.strip()
        return blocks[-1].strip()
    # no fence: grab from the first class/def to the end of the answer
    m = re.search(r"(?:^|\n)(class\s+Solution\b|def\s+\w+\s*\()", answer)
    if m:
        return answer[m.start():].strip()
    return None


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


def _run_one(code: str, tests: list, fn_name, mem_mb, cpu_s, q):
    """Queue variant (kept for compatibility)."""
    class _P:
        def send(self, x): q.put(x)
    _run_one_pipe(code, tests, fn_name, mem_mb, cpu_s, _P())


def _run_one_pipe(code: str, tests: list, fn_name, mem_mb, cpu_s, conn):
    """Executed in child process. Sends (passed:int, total:int, err:str) over the connection.
    Mode decided ONCE per problem: functional (Solution class, call method) vs stdin (script + stdin)."""
    _limit_resources(mem_mb, cpu_s)
    import io, contextlib
    passed = 0; total = len(tests); err = ""

    # decide mode from the test cases (LiveCodeBench tags each) + code shape
    ttypes = {tc.get("testtype") for tc in tests}
    is_functional = ("functional" in ttypes) and ("class Solution" in code or "def " in code) and "stdin" not in ttypes
    # if mixed/ambiguous, prefer functional only if a Solution class exists
    if "functional" in ttypes and "stdin" in ttypes:
        is_functional = "class Solution" in code

    if is_functional:
        ns: dict = {}
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
            cand = [m for m in dir(sol_cls) if not m.startswith("_")]
            method = cand[0] if cand else None
        for tc in tests:
            inp = tc.get("input", ""); exp = str(tc.get("output", "")).strip()
            try:
                args = [ast.literal_eval(x) for x in str(inp).split("\n") if x.strip()]
                with contextlib.redirect_stdout(io.StringIO()):
                    if sol_cls is not None and method:
                        got = getattr(sol_cls(), method)(*args)
                    else:
                        got = ns.get(fn_name or "solve", lambda *a: None)(*args)
                ok = str(got).strip() == exp
                # some functional outputs are lists/bools; try literal compare too
                if not ok:
                    try: ok = ast.literal_eval(str(got)) == ast.literal_eval(exp)
                    except Exception: pass
                passed += int(ok)
            except Exception as e:
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
    if not code or not tests:
        return {"success": 0, "passed": 0, "total": len(tests or []), "err": "no_code_or_tests"}
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
        return {"success": 0, "passed": 0, "total": len(tests), "err": "timeout"}
    if parent_conn.poll():
        try:
            passed, total, err = parent_conn.recv()
            return {"success": int(passed == total and total > 0), "passed": passed,
                    "total": total, "err": err}
        except Exception:
            pass
    return {"success": 0, "passed": 0, "total": len(tests), "err": "no_result"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out", default="data/perf/code_grades.parquet")
    ap.add_argument("--timeout", type=int, default=8, help="wall-clock seconds per submission")
    ap.add_argument("--mem-mb", type=int, default=2048)
    ap.add_argument("--cpu-s", type=int, default=10)
    ap.add_argument("--max-private", type=int, default=60, help="cap private tests/problem for speed")
    args = ap.parse_args()

    tr = pl.concat([pl.read_parquet(p) for p in sorted(Path().glob(args.traces_glob))],
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
        if args.max_private and len(tests) > args.max_private:
            tests = tests[:args.max_private]                       # cap for runtime; pass@1 still strict
        code = extract_code(r.get("answer_text") or "", meta.get("starter_code") or "")
        res = grade_submission(code, tests, meta.get("fn_name"),
                               args.timeout, args.mem_mb, args.cpu_s)
        rows.append({"trace_id": tid, "task_type": "code", "success": res["success"],
                     "parsed": code is not None, "completed": (r.get("finish_reason") == "stop"),
                     "difficulty_raw": r.get("difficulty_raw"),
                     "tests_passed": res["passed"], "tests_total": res["total"],
                     "exec_err": res["err"], "gradeable": True, "grade_method": "execution"})
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{n} graded")
    res_df = pl.DataFrame(rows)
    if out.exists() and res_df.height:
        res_df = pl.concat([pl.read_parquet(out), res_df], how="diagonal_relaxed").unique("trace_id", keep="last")
    if res_df.height:
        res_df.write_parquet(out)
    # report
    g = pl.read_parquet(out) if out.exists() else res_df
    j = g.join(tr.select(["trace_id", "gen_model"]), on="trace_id", how="left")
    print(f"\ncode grades -> {out} ({g.height} traces)")
    for m in ["reasoner", "anchor"]:
        s = j.filter(pl.col("gen_model") == m)
        if s.height:
            print(f"  {m:9s} pass@1={s['success'].mean():.3f} | code-extracted={s['parsed'].mean():.0%} "
                  f"| timeouts={100*(s['exec_err']=='timeout').mean():.0f}% (n={s.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())