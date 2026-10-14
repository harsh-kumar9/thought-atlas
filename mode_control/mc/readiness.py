"""Forced-answer readiness probe (D-62): was the answer already there when the model kept thinking?

For each trace of the chosen arms, cut the generated reasoning at k tokens, close the think block, force an answer with a
short greedy continuation (no steering, one reader for every arm) and grade it. R(k) = share of tasks whose answer at k is
correct. A trace that closed its think block before k keeps its own final answer and its extraction grade (D-47,
budget.load_grades), as mc.budget does for a finished trace.

Arms are run:condition pairs; traces come from data/runs/<run>/traces_<condition>.parquet. gen_token_ids holds the generated
tokens and the think block ends at the first </think> token.

Usage:
  python -m mode_control.mc.readiness --arms p1_budget:baseline,p1_budget:ol_math_A2,p1_budget:ol_random,p4_eval3:rl_zeroT_s60,p4_eval2:rl_a2initT_s60
  python -m mode_control.mc.readiness --arms cd_countdown:baseline,cd_countdown:ol_math_A2 --cuts 1024,2048,4096,8192
  python -m mode_control.mc.readiness --arms p1_budget:baseline --smoke 8      # 8 traces, quick engine check
Outputs: data/readiness/<run>__<cond>.parquet (trace x cut) and data/readiness/summary.csv (arm x cut).
Written 2026-10-14; not yet run.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from .budget import load_grades
from .common import DATA, MODEL_ID, load_tokenizer, render_prompt
from .extract_free import math_correct

RD = DATA / "readiness"
FORCE = {
    "math": "\n</think>\n\nThe final answer is $\\boxed{",
    "countdown": "\n</think>\n\n<answer>",
    "gpqa": "\n</think>\n\nThe correct answer is (",
    "planning": "\n</think>\n\nThe correct answer is (",
}


def boxed_content(text: str) -> str:
    """Text up to the brace that closes the forced \\boxed{ (depth starts at 1)."""
    depth, out = 1, []
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def grade_forced(task: str, completion: str, ref, prompt: str) -> tuple[str, int]:
    if task == "math":
        fa = boxed_content(completion)
        return fa, int(math_correct(fa, ref)) if fa else 0
    if task == "countdown":
        from .countdown import grade_countdown
        fa = completion.split("</answer>")[0].strip()
        return fa, int(grade_countdown(fa, ref)) if fa else 0
    from src.perf.answer_grading import grade_mcq_answer
    fa = "(" + completion.strip()[:3]
    return fa, int(bool(grade_mcq_answer(fa, ref, prompt).get("success"))) if fa else 0


def build_engine(max_model_len: int):
    from vllm import LLM
    return LLM(model=MODEL_ID, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=0.88, max_num_seqs=128,
               max_num_batched_tokens=8192, enable_prefix_caching=True, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="comma list of run:condition")
    ap.add_argument("--cuts", default="512,1024,2048,4096,8192,12288")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--smoke", type=int, default=0, help="use only the first N traces per arm")
    a = ap.parse_args()
    cuts = [int(c) for c in a.cuts.split(",")]
    RD.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    end_id = tok.convert_tokens_to_ids("</think>")
    assert isinstance(end_id, int) and end_id >= 0, "</think> is not a single token"
    grades = load_grades()

    jobs, meta = [], []          # prompts to run, and (arm index, trace row, cut)
    arms = [s.split(":") for s in a.arms.split(",")]
    own = {}                     # (arm, trace_id, cut) -> dict for traces finished before the cut
    rows_by_arm = {}
    for run, cond in arms:
        tr = pl.read_parquet(DATA / "runs" / run / f"traces_{cond}.parquet")
        if a.smoke:
            tr = tr.head(a.smoke)
        rows_by_arm[(run, cond)] = tr
        for r in tr.iter_rows(named=True):
            ids = list(r["gen_token_ids"] or [])
            pos = ids.index(end_id) if end_id in ids else len(ids)
            task = r["task_type"]
            for k in cuts:
                if pos <= k:          # closed its think block by k: own answer and grade
                    g = grades.get(r["trace_id"])
                    if g is None:
                        _, g = grade_forced(task, r.get("answer_text") or "", r["reference_answer"], r["prompt"]) if task != "math" else ("", 0)
                    own[(run, cond, r["trace_id"], k)] = {"finished": True, "forced_answer": None, "correct": int(bool(g)), "think_tokens": pos}
                else:
                    text = render_prompt(tok, r["prompt"]) + tok.decode(ids[:k]) + FORCE.get(task, FORCE["math"])
                    jobs.append(text)
                    meta.append((run, cond, r, k, pos))
    print(f"[readiness] {len(jobs)} forced continuations, {len(own)} finished-before-cut cells", flush=True)

    from vllm import SamplingParams
    longest = max(cuts) + 1024
    llm = build_engine(max_model_len=longest + a.max_tokens + 64)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=0)
    t0 = time.time()
    outs = llm.generate(jobs, sp) if jobs else []
    print(f"[readiness] generated in {time.time() - t0:.0f}s", flush=True)

    recs = []
    for (run, cond, r, k, pos), o in zip(meta, outs):
        comp = o.outputs[0].text
        fa, ok = grade_forced(r["task_type"], comp, r["reference_answer"], r["prompt"])
        recs.append({"run": run, "condition": cond, "trace_id": r["trace_id"], "instance_id": r["instance_id"], "seed": r["seed"],
                     "task_type": r["task_type"], "cut": k, "finished": False, "think_tokens": pos, "forced_answer": fa,
                     "completion": comp, "correct": ok})
    for (run, cond, tid, k), d in own.items():
        r = rows_by_arm[(run, cond)].filter(pl.col("trace_id") == tid).row(0, named=True)
        recs.append({"run": run, "condition": cond, "trace_id": tid, "instance_id": r["instance_id"], "seed": r["seed"],
                     "task_type": r["task_type"], "cut": k, "completion": None, **d})
    df = pl.DataFrame(recs)
    for (run, cond), g in df.group_by(["run", "condition"]):
        g.write_parquet(RD / f"{run}__{cond}.parquet")

    # summary: R(k) over all tasks (mean over seeds, then tasks); forced-only R among traces not finished by k
    summ = (df.group_by(["run", "condition", "cut"])
              .agg([pl.col("correct").mean().alias("R"), pl.col("finished").mean().alias("finished_by_cut"),
                    pl.col("correct").filter(~pl.col("finished")).mean().alias("R_forced_only"), pl.len().alias("n")])
              .sort(["run", "condition", "cut"]))
    # subset: tasks the unsteered arm had not answered at the largest cut (per seed)
    base = df.filter((pl.col("condition") == "baseline") & (pl.col("cut") == max(cuts)) & (~pl.col("finished")))
    late = base.select(["instance_id", "seed"]).unique()
    sub = (df.join(late, on=["instance_id", "seed"]).group_by(["run", "condition", "cut"])
             .agg([pl.col("correct").mean().alias("R_unanswered_at_max_cut"), pl.len().alias("n_sub")]))
    summ = summ.join(sub, on=["run", "condition", "cut"], how="left")
    summ.write_csv(RD / ("summary_smoke.csv" if a.smoke else "summary.csv"))
    print(summ)
    # paired contrast of every arm against baseline at each cut (tasks averaged over seeds; task bootstrap)
    rng = np.random.default_rng(0)
    out = []
    per = df.group_by(["run", "condition", "instance_id", "cut"]).agg(pl.col("correct").mean())
    b = per.filter(pl.col("condition") == "baseline").select(["instance_id", "cut", pl.col("correct").alias("c0")])
    for (run, cond), g in per.filter(pl.col("condition") != "baseline").group_by(["run", "condition"]):
        j = g.join(b, on=["instance_id", "cut"])
        for k in cuts:
            d = (j.filter(pl.col("cut") == k)["correct"] - j.filter(pl.col("cut") == k)["c0"]).to_numpy()
            if len(d) == 0:
                continue
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            out.append({"run": run, "condition": cond, "cut": k, "delta_vs_baseline": float(d.mean()),
                        "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "n_tasks": int(len(d))})
    if out:
        pl.DataFrame(out).write_csv(RD / ("paired_smoke.csv" if a.smoke else "paired.csv"))
        print(pl.DataFrame(out))
    (RD / "config.json").write_text(json.dumps({"arms": a.arms, "cuts": cuts, "max_tokens": a.max_tokens, "force": FORCE, "model": MODEL_ID}, indent=1))


if __name__ == "__main__":
    main()
