"""Accuracy as a function of the token budget, by truncating long-budget traces (D-40).

A trace generated with budget Bmax, cut at B tokens, is what a run with budget B would have produced with the same
sampling stream: if it stopped before B it is unchanged; otherwise the first B tokens are decoded and parsed as a
length-truncated generation. Outcome per budget: answered (non-empty final answer) and correct (unanswered = wrong).
Math is graded like mc.analyze (repo grader, bold stripped); gpqa/planning with the repo MCQ grader.

Output under data/runs/<run>/analysis/: budget_outcomes.parquet (trace x budget), budget_summary.csv (condition x
budget levels), budget_paired.csv (condition minus reference, paired over tasks after averaging seeds, bootstrap CI).
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from .common import DATA, load_tokenizer
from src.perf.answer_grading import grade_math_answer, grade_mcq_answer
from src.utils.parse import parse_generation_detailed


def grade(answer: str, ref, task: str, prompt: str) -> int:
    if not answer:
        return 0
    if task == "math":
        return int(bool(grade_math_answer(answer.replace("**", ""), ref).get("success")))
    return int(bool(grade_mcq_answer(answer, ref, prompt).get("success")))


GRADES = DATA / "extract_free" / "grades.parquet"


def load_grades() -> dict:
    """Per-trace grades from the free-text answer extraction (D-47); empty if not built yet."""
    if not GRADES.exists():
        return {}
    g = pl.read_parquet(GRADES, columns=["trace_id", "success"])
    return dict(zip(g["trace_id"].to_list(), g["success"].to_list()))


def outcomes(tr: pl.DataFrame, tok, budgets: list[int], grades: dict | None = None) -> pl.DataFrame:
    """With `grades`, a trace that finished within the budget takes its extraction grade; a trace cut by the budget
    is graded directly on whatever answer the cut text contains (usually none)."""
    rows = []
    for r in tr.iter_rows(named=True):
        ids = r["gen_token_ids"]; n = len(ids)
        for B in budgets:
            if n <= B:
                ans = r["answer_text"] or ""
                c = grades.get(r["trace_id"]) if grades else None
                c = grade(ans, r["reference_answer"], r["task_type"], r["prompt"]) if c is None else int(c)
            else:
                ans = parse_generation_detailed(tok.decode(ids[:B], skip_special_tokens=False), "reasoning", finish_reason="length")["answer_text"] or ""
                c = grade(ans, r["reference_answer"], r["task_type"], r["prompt"])
            rows.append({"condition": r["condition"], "instance_id": r["instance_id"], "task_type": r["task_type"], "seed": r["seed"],
                         "budget": B, "tokens": min(n, B), "answered": int(bool(ans.strip())), "correct": c})
    return pl.DataFrame(rows)


def paired(df: pl.DataFrame, ref: str, reps: int = 2000) -> pl.DataFrame:
    """Condition minus ref per (task_type, budget): mean over seeds within task, then paired over tasks."""
    task = df.group_by(["condition", "task_type", "budget", "instance_id"]).agg(pl.col("correct").mean(), pl.col("answered").mean(), pl.col("tokens").mean())
    base = task.filter(pl.col("condition") == ref).drop("condition")
    rng = np.random.default_rng(0); out = []
    for (c, t, b), g in task.filter(pl.col("condition") != ref).group_by(["condition", "task_type", "budget"]):
        j = g.join(base, on=["task_type", "budget", "instance_id"], suffix="_ref")
        rec = {"condition": c, "task_type": t, "budget": b, "n_tasks": j.height, "reference": ref}
        for m in ("correct", "answered", "tokens"):
            d = (j[m] - j[f"{m}_ref"]).to_numpy()
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)])
            rec.update({f"{m}_ref": float(j[f"{m}_ref"].mean()), m: float(j[m].mean()), f"d_{m}": float(d.mean()),
                        f"d_{m}_lo": float(np.quantile(bs, 0.025)), f"d_{m}_hi": float(np.quantile(bs, 0.975))})
        out.append(rec)
    return pl.DataFrame(out).sort(["task_type", "condition", "budget"]) if out else pl.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--budgets", default="4096,8192,12288,16384,24576,32768")
    ap.add_argument("--refs", default="baseline", help="comma list of reference conditions for paired contrasts")
    ap.add_argument("--direct", action="store_true", help="grade with the direct grader instead of the extraction grades")
    a = ap.parse_args()
    rd = DATA / "runs" / a.run; out = rd / "analysis"; out.mkdir(parents=True, exist_ok=True)
    tr = pl.concat([pl.read_parquet(p) for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed")
    budgets = [int(x) for x in a.budgets.split(",")]
    budgets = [b for b in budgets if b <= int(tr["n_new_tokens"].max()) or b == budgets[0]]
    grades = {} if a.direct else load_grades()
    print(f"[budget] grades: {'direct grader' if not grades else f'extraction grades for {len(grades)} traces'}")
    df = outcomes(tr, load_tokenizer(), budgets, grades)
    df.write_parquet(out / "budget_outcomes.parquet")
    summ = df.group_by(["condition", "task_type", "budget"]).agg(pl.len().alias("n"), pl.col("correct").mean(), pl.col("answered").mean(),
                                                               pl.col("tokens").mean()).sort(["task_type", "condition", "budget"])
    summ = summ.with_columns((pl.col("correct") / pl.col("answered")).alias("correct_given_answered"))
    summ.write_csv(out / "budget_summary.csv")
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(220)
    print(summ)
    pd = pl.concat([x for x in (paired(df, r) for r in a.refs.split(",")) if x.height])  if df["condition"].n_unique() > 1 else pl.DataFrame()
    if not pd.height:
        return
    pd.write_csv(out / "budget_paired.csv")
    print(pd.select(["reference", "condition", "task_type", "budget", "n_tasks", "d_correct", "d_correct_lo", "d_correct_hi", "d_answered", "d_tokens"]))


if __name__ == "__main__":
    main()
