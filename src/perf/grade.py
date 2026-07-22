"""grade.py — auditable deterministic grading for ground-truth domains.

math/gpqa/planning graded here (string/symbolic match). code requires execution
(separate isolated compute job — see grade_code_exec). moral/idea are judge-scored
(read from judge output, not graded here).
"""
from __future__ import annotations
import hashlib
import polars as pl
from .answer_grading import (GRADE_VERSION, extract_math_candidates,
                             extract_choice as _extract_choice,
                             grade_math_answer, grade_mcq_answer)
from src.utils.io import resolve_trace_paths


# Compatibility helpers used by older notebooks. Production uses the structured
# grade functions below.
def extract_math(answer_text: str):
    candidates = extract_math_candidates(answer_text)
    return candidates[0][0] if candidates else None


def extract_choice(answer_text: str):
    return _extract_choice(answer_text)[0]


def math_equal(pred: str, ref: str) -> bool:
    return bool(pred is not None and grade_math_answer(pred, ref).get("success") == 1)


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
        comp = r.get("completed")
        if comp is None:
            comp = r.get("finish_reason") == "stop"
        succ, gradeable, parsed, method = None, True, None, ""
        prediction, status = None, None
        if d == "math":
            result = grade_math_answer(ans, ref)
            prediction, parsed, succ = result["prediction"], result["parsed"], result["success"]
            method, status = result["parse_method"], result["status"]
        elif d in ("gpqa", "planning"):
            result = grade_mcq_answer(ans, ref, r.get("prompt") or "")
            prediction, parsed, succ = result["prediction"], result["parsed"], result["success"]
            method, status = result["parse_method"], result["status"]
        elif d == "code":
            gradeable = False; method = "needs_execution"
        elif d in ("moral", "idea"):
            gradeable = False; method = "judge_scored"
        rows.append({"trace_id": tid, "task_type": d, "success": succ, "parsed": parsed,
                     "completed": bool(comp) if comp is not None else None,
                     "difficulty_raw": diff, "gradeable": gradeable, "grade_method": method,
                     "predicted_answer": prediction, "reference_answer": ref,
                     "grade_status": status, "grader_version": GRADE_VERSION,
                     "answer_sha256": hashlib.sha256(ans.encode("utf-8")).hexdigest()})
    return pl.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out", default="data/perf/success_grades.parquet")
    args = ap.parse_args()
    paths = resolve_trace_paths(args.traces_glob)
    if not paths:
        raise SystemExit(f"no traces matched {args.traces_glob}")
    tr = pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")
    g = grade_traces(tr)
    det = g.filter(pl.col("gradeable"))
    print("Graded (deterministic) by domain:")
    j = g.join(tr.select(["trace_id", "gen_model"]), on="trace_id")
    for d in ["math", "gpqa", "planning"]:
        sub = j.filter((pl.col("task_type") == d))
        for m in sorted(sub["gen_model"].drop_nulls().unique().to_list()):
            s = sub.filter(pl.col("gen_model") == m)
            if s.height:
                sp = s.filter(pl.col("parsed"))
                acc_p = sp["success"].mean() if sp.height else float("nan")
                correct = sp["success"].sum() or 0
                lower = correct / s.height
                upper = (correct + s.height - sp.height) / s.height
                print(f"  {d:9s} {m:9s} acc_parsed={acc_p:.3f} "
                      f"parse={100*sp.height/s.height:.0f}% bounds_all=[{lower:.3f},{upper:.3f}] "
                      f"(n={s.height})")
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    g.write_parquet(tmp); tmp.replace(out)
    print(f"-> {args.out}")
