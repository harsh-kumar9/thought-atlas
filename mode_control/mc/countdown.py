"""Countdown tasks and grader for the Society of Thought crossover (D-51).

Tasks come from Jiayi-Pan/Countdown-Tasks-3to4 (the TinyZero set; rows {target, nums}). We keep four-number rows, sample
200 with a fixed seed, and use a TinyZero-style instruction: reach the target with + - * / using each number exactly
once, final equation in <answer> tags. reference_answer stores {"nums", "target"} as JSON so the generic graders can
read it.

  tasks  -> data/runs/cd_countdown/tasks.parquet
  test   grader self-test
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from fractions import Fraction

import polars as pl

from .common import DATA

HF_ID = "Jiayi-Pan/Countdown-Tasks-3to4"
PROMPT = ("Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations "
          "(+, -, *, /) and each number must be used exactly once. Think it through, then give the final equation in "
          "<answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.")


def _eval(node):
    """Exact rational evaluation of + - * / over integer literals; anything else raises."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if b == 0: raise ZeroDivisionError
        return a / b
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return Fraction(node.value)
    raise ValueError("unsupported expression")


def _numbers(node) -> list[int]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, int)]


def extract_equation(answer: str) -> str | None:
    """Last <answer>...</answer> in the final answer; else the last line that looks like an arithmetic expression.
    A trailing '= target' is dropped."""
    m = re.findall(r"<answer>(.*?)</answer>", answer or "", flags=re.S)
    cand = m[-1] if m else None
    if cand is None:
        lines = [l.strip().strip("$").strip() for l in (answer or "").splitlines() if re.search(r"\d\s*[-+*/x×÷]\s*\(?\d", l)]
        cand = lines[-1] if lines else None
    if cand is None:
        return None
    cand = cand.replace("×", "*").replace("÷", "/").replace("\\times", "*").replace("\\div", "/").replace("−", "-")
    cand = re.sub(r"\\left|\\right|\\!|\\,|\$", "", cand).replace("**", "")
    cand = cand.split("=")[0] if cand.count("=") == 1 and re.search(r"=\s*-?\d+\s*$", cand) else cand
    cand = cand.strip().rstrip(".")
    if cand and not re.fullmatch(r"[\d\s+\-*/().]+", cand):          # prose around the equation: keep the longest arithmetic run
        runs = [x.strip() for x in re.findall(r"[\d\s+\-*/().]+", cand) if re.search(r"\d", x) and re.search(r"[-+*/]", x)]
        cand = max(runs, key=len) if runs else None
    return cand or None


def grade_countdown(answer: str, ref) -> int:
    """1 if the extracted equation uses exactly the given numbers (as a multiset) and evaluates to the target."""
    r = json.loads(ref) if isinstance(ref, str) else ref
    eq = extract_equation(answer)
    if not eq or not re.fullmatch(r"[\d\s+\-*/().]+", eq):
        return 0
    try:
        tree = ast.parse(eq, mode="eval")
        val = _eval(tree)
    except Exception:
        return 0
    return int(Counter(_numbers(tree)) == Counter(int(x) for x in r["nums"]) and val == Fraction(int(r["target"])))


def cmd_tasks(n: int, seed: int):
    from datasets import load_dataset
    d = load_dataset(HF_ID, split="train")
    d = d.filter(lambda x: len(x["nums"]) == 4)
    d = d.shuffle(seed=seed).select(range(n))
    rows = []
    for i, x in enumerate(d):
        nums = [int(v) for v in x["nums"]]; target = int(x["target"])
        rows.append({"instance_id": f"countdown:{seed}:{i:03d}", "task_type": "countdown",
                     "prompt": PROMPT.format(nums=nums, target=target),
                     "reference_answer": json.dumps({"nums": nums, "target": target}), "difficulty_raw": None,
                     "metadata": json.dumps({"source_dataset": HF_ID, "nums": nums, "target": target})})
    out = DATA / "runs" / "cd_countdown"; out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out / "tasks.parquet")
    print(f"[countdown] {len(rows)} tasks -> {out / 'tasks.parquet'}; e.g. {rows[0]['prompt']}")


def cmd_test():
    ref = json.dumps({"nums": [3, 7, 25, 50], "target": 25 * 3 + 50 - 7})
    cases = [("<answer> 25 * 3 + 50 - 7 </answer>", 1), ("<answer>(25*3)+50-7 = 118</answer>", 1), ("<answer> 25 * 3 + 50 </answer>", 0),
             ("<answer> 25 * 3 + 50 - 7 + 3 - 3 </answer>", 0), ("The equation is $25 \\times 3 + 50 - 7$", 1), ("", 0),
             ("<answer> 25 * 3 + 50 - 8 </answer>", 0), ("<answer>import os</answer>", 0)]
    for a, want in cases:
        got = grade_countdown(a, ref)
        assert got == want, (a, got, want)
    assert grade_countdown("<answer> (50 - 25) / 5 * 4 </answer>", json.dumps({"nums": [50, 25, 5, 4], "target": 20})) == 1
    print("[countdown] grader self-test passed")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks"); t.add_argument("--n", type=int, default=200); t.add_argument("--seed", type=int, default=20261006)
    sub.add_parser("test")
    a = ap.parse_args()
    cmd_tasks(a.n, a.seed) if a.cmd == "tasks" else cmd_test()


if __name__ == "__main__":
    main()
