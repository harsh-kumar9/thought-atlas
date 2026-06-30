"""grade.py — deterministic success grading for ground-truth domains.

math/gpqa/planning graded here (string/symbolic match). code requires execution
(separate sandboxed job — see grade_code_exec stub). moral/idea are judge-scored
(read from judge output, not graded here).
"""
from __future__ import annotations
import re, json
import polars as pl

# ---------- answer extraction ----------
def _last_boxed(s: str):
    """Extract the content of the last \\boxed{...}, brace-balanced."""
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{": depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i+1:j]
    return None


def extract_math(answer_text: str):
    b = _last_boxed(answer_text or "")
    if b is not None:
        return b.strip()
    # fallback: "final answer ... X" or last number
    m = re.findall(r"(?:final answer|answer)\s*[:=]?\s*\$?([-\d./\\]+)", (answer_text or "").lower())
    return m[-1].strip() if m else None


def extract_choice(answer_text: str):
    """MCQ letter for gpqa/planning. Returns (letter|None). Ordered most->least reliable."""
    t = answer_text or ""
    for pat in [r"\*\*answer:\*\*\s*\\?boxed\{?([A-E])", r"answer\s*[:=]?\s*\*?\*?\s*([A-E])\b",
                r"\\boxed\{([A-E])\}", r"\bthe (?:correct |final )?answer is\s*:?\s*\*?\*?\s*([A-E])\b",
                r"\boption\s+([A-E])\b\s*(?:is correct|is the)", r"\*\*([A-E])[.\.]\s"]:
        m = re.findall(pat, t, flags=re.I)
        if m:
            return m[-1].upper()
    # last resort: a lone capital letter in the final line(s)
    tail = "\n".join([ln for ln in t.strip().splitlines()[-3:]])
    m = re.findall(r"\b([A-E])\b", tail)
    return m[-1].upper() if m else None


# ---------- normalization / equivalence ----------
def _norm_math(x: str):
    if x is None:
        return None
    x = x.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    x = x.replace("$", "").replace("\\!", "").replace(" ", "").replace("\\,", "")
    x = x.replace("\\%", "").rstrip(".")
    return x


def math_equal(pred: str, ref: str) -> bool:
    if pred is None or ref is None:
        return False
    p, r = _norm_math(pred), _norm_math(ref)
    if p == r:
        return True
    # symbolic equivalence via sympy (handles 1/4 == 0.25, etc.)
    try:
        from sympy import simplify, sympify, Rational, nsimplify
        from sympy.parsing.latex import parse_latex
        def _to(e):
            try: return parse_latex(e)
            except Exception:
                try: return sympify(e.replace("\\frac", ""))
                except Exception: return None
        pe, re_ = _to(p), _to(r)
        if pe is not None and re_ is not None:
            return bool(simplify(pe - re_) == 0)
    except Exception:
        pass
    # numeric fallback
    try:
        return abs(float(p) - float(r)) < 1e-6
    except Exception:
        return False


# ---------- per-domain grading ----------
def grade_traces(tr: pl.DataFrame) -> pl.DataFrame:
    """Return per-trace success + grading metadata.
    success: 1/0 for graded domains, None for judge/exec domains.
    parsed: did we extract an answer at all (False -> success is unreliable, not necessarily wrong).
    completed: did the generation finish (length-truncated traces have no real answer).
    """
    rows = []
    for r in tr.iter_rows(named=True):
        d = r["task_type"]; tid = r["trace_id"]; ans = r.get("answer_text") or ""
        ref = r.get("reference_answer"); diff = r.get("difficulty_raw")
        fin = r.get("finish_reason"); comp = (fin == "stop") if fin is not None else r.get("completed")
        succ, gradeable, parsed, method = None, True, None, ""
        if d == "math":
            pred = extract_math(ans); parsed = pred is not None
            succ = int(math_equal(pred, ref)); method = "symbolic_match"
        elif d in ("gpqa", "planning"):
            pred = extract_choice(ans); parsed = pred is not None
            succ = int(pred == (ref or "").strip().upper()); method = "mcq_match"
        elif d == "code":
            gradeable = False; method = "needs_execution"
        elif d in ("moral", "idea"):
            gradeable = False; method = "judge_scored"
        rows.append({"trace_id": tid, "task_type": d, "success": succ, "parsed": parsed,
                     "completed": bool(comp) if comp is not None else None,
                     "difficulty_raw": diff, "gradeable": gradeable, "grade_method": method})
    return pl.DataFrame(rows)


if __name__ == "__main__":
    import argparse, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out", default="data/perf/success_grades.parquet")
    args = ap.parse_args()
    paths = glob.glob(args.traces_glob)
    if not paths:
        raise SystemExit(f"no traces matched {args.traces_glob}")
    tr = pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")
    g = grade_traces(tr)
    det = g.filter(pl.col("gradeable"))
    print("Graded (deterministic) by domain:")
    j = g.join(tr.select(["trace_id", "gen_model"]), on="trace_id")
    for d in ["math", "gpqa", "planning"]:
        sub = j.filter((pl.col("task_type") == d))
        for m in ["reasoner", "anchor"]:
            s = sub.filter(pl.col("gen_model") == m)
            if s.height:
                sp = s.filter(pl.col("parsed"))
                acc_p = sp["success"].mean() if sp.height else float("nan")
                print(f"  {d:9s} {m:9s} acc_all={s['success'].mean():.3f} acc_parsed={acc_p:.3f} "
                      f"parse={100*s['parsed'].mean():.0f}% (n={s.height})")
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    g.write_parquet(args.out)
    print(f"-> {args.out}")