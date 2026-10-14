"""Forced-answer readiness probe (D-62): was the answer already there when the model kept thinking?

For each trace of the chosen arms, cut the generated reasoning at k tokens, close the think block, force an answer with a
short greedy continuation (no steering, one reader for every arm) and grade it. R(k) = share of tasks whose answer at k is
correct. A trace that closed its think block before k keeps its own final answer and its grade (the D-47 free-text grade
when one exists, else the repo grader on its answer text), as mc.budget does for a finished trace.

Arms are run:condition pairs; traces come from data/runs/<run>/traces_<condition>.parquet. gen_token_ids holds the generated
tokens and the think block ends at the first </think> token.

Gate: before the full batch the first 8 traces of the first arm are run and printed; the job stops (exit 2) if fewer than
60% of their forced continuations contain a parseable answer, so a malformed forcing string cannot waste the run.
The raw completions are written to raw_completions.parquet right after generation; --from-raw redoes grading and the
summaries from that file without the engine.

Usage:
  python -m mode_control.mc.readiness --arms p1_budget:baseline,p1_budget:ol_math_A2,p1_budget:ol_random,p4_eval3:rl_zeroT_s60,p4_eval2:rl_a2initT_s60
  python -m mode_control.mc.readiness --arms cd_countdown:baseline,cd_countdown:ol_math_A2 --cuts 1024,2048,4096,8192,12288 --tag countdown
  python -m mode_control.mc.readiness --arms p1_budget:baseline --smoke 8      # 8 traces, quick engine check
  python -m mode_control.mc.readiness --arms ... --dry-run                    # build prompts and own grades, no engine
  python -m mode_control.mc.readiness --arms ... --tag math --from-raw        # regrade + summarise from raw_completions.parquet
Outputs under data/readiness/<tag>/: raw_completions.parquet, <run>__<cond>.parquet (trace x cut), summary.csv (arm x cut),
paired.csv (arm minus baseline per cut, task bootstrap), samples.txt (the gate's forced continuations), config.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import polars as pl

from .budget import grade as grade_own, load_grades
from .common import DATA, MODEL_ID, load_tokenizer, render_prompt
from .extract_free import math_correct

FORCE = {
    "math": "\n</think>\n\nThe final answer is $\\boxed{",
    "countdown": "\n</think>\n\n<answer>",
    "gpqa": "\n</think>\n\nThe correct answer is (",
    "planning": "\n</think>\n\nThe correct answer is (",
}
SCHEMA = {"run": pl.Utf8, "condition": pl.Utf8, "trace_id": pl.Utf8, "instance_id": pl.Utf8, "seed": pl.Int64, "task_type": pl.Utf8,
          "think_tokens": pl.Int64, "cut": pl.Int64, "finished": pl.Boolean, "forced_answer": pl.Utf8, "completion": pl.Utf8,
          "correct": pl.Int64}


def boxed_content(text: str) -> str:
    """Text up to the brace that closes the forced \\boxed{ (depth starts at 1); empty if no closing brace."""
    depth, out = 1, []
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
    return ""


def grade_forced(task: str, completion: str, ref, prompt: str) -> tuple[str, int]:
    """Returns (parsed answer or empty, correct)."""
    if task == "math":
        fa = boxed_content(completion)
        return fa, (int(bool(math_correct(fa, ref))) if fa else 0)
    if task == "countdown":
        from .countdown import grade_countdown
        fa = completion.split("</answer>")[0].strip() if "</answer>" in completion else ""
        return fa, (int(bool(grade_countdown(fa, ref))) if fa else 0)
    from src.perf.answer_grading import grade_mcq_answer
    m = completion.strip()[:1]
    fa = f"({m})" if m.isalpha() else ""
    return fa, (int(bool(grade_mcq_answer(fa, ref, prompt).get("success"))) if fa else 0)


def build_engine(max_model_len: int):
    from vllm import LLM
    return LLM(model=MODEL_ID, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=0.88, max_num_seqs=128,
               max_num_batched_tokens=8192, enable_prefix_caching=True, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})


def frame(recs: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(recs, schema_overrides=SCHEMA, infer_schema_length=None)


def summarise(df: pl.DataFrame, rd, cuts: list[int], base_cond: str, suffix: str):
    for (run, cond), g in df.group_by(["run", "condition"]):
        g.write_parquet(rd / f"{run}__{cond}.parquet")
    # R(k) over all traces; R among traces not finished by k; R on the tasks the baseline had not answered at the largest cut
    summ = (df.group_by(["run", "condition", "cut"])
              .agg([pl.col("correct").mean().alias("R"), pl.col("finished").mean().alias("finished_by_cut"),
                    pl.col("correct").filter(~pl.col("finished")).mean().alias("R_forced_only"), pl.len().alias("n")])
              .sort(["run", "condition", "cut"]))
    late = (df.filter((pl.col("condition") == base_cond) & (pl.col("cut") == max(cuts)) & (~pl.col("finished")))
              .select(["instance_id", "seed"]).unique())
    sub = (df.join(late, on=["instance_id", "seed"]).group_by(["run", "condition", "cut"])
             .agg([pl.col("correct").mean().alias("R_on_baseline_unfinished_at_max_cut"), pl.len().alias("n_sub")]))
    summ = summ.join(sub, on=["run", "condition", "cut"], how="left")
    summ.write_csv(rd / f"summary{suffix}.csv")
    print(summ)
    # paired contrast of every arm against the baseline arm at each cut (tasks averaged over seeds; task bootstrap)
    rng = np.random.default_rng(0)
    per = df.group_by(["run", "condition", "instance_id", "cut"]).agg(pl.col("correct").mean())
    b = per.filter(pl.col("condition") == base_cond).select(["instance_id", "cut", pl.col("correct").alias("c0")])
    out = []
    for (run, cond), g in per.filter(pl.col("condition") != base_cond).group_by(["run", "condition"]):
        j = g.join(b, on=["instance_id", "cut"])
        for k in cuts:
            jk = j.filter(pl.col("cut") == k)
            d = (jk["correct"] - jk["c0"]).to_numpy()
            if len(d) == 0:
                continue
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            out.append({"run": run, "condition": cond, "cut": k, "delta_vs_baseline": float(d.mean()),
                        "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "n_tasks": int(len(d))})
    if out:
        pl.DataFrame(out).write_csv(rd / f"paired{suffix}.csv")
        print(pl.DataFrame(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="comma list of run:condition; the first arm's condition is the baseline for paired contrasts")
    ap.add_argument("--cuts", default="512,1024,2048,4096,8192,12288")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--tag", default="math")
    ap.add_argument("--smoke", type=int, default=0, help="use only the first N traces per arm")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and own grades, write nothing to the engine")
    ap.add_argument("--from-raw", action="store_true", help="regrade and summarise from raw_completions.parquet, no engine")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    cuts = [int(c) for c in a.cuts.split(",")]
    rd = DATA / "readiness" / a.tag
    rd.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if a.smoke else ""
    arms = [tuple(s.split(":")) for s in a.arms.split(",")]
    base_cond = arms[0][1]

    if a.from_raw:
        raw = pl.read_parquet(rd / "raw_completions.parquet")
        recs = []
        for r in raw.iter_rows(named=True):
            if r["finished"]:
                recs.append({k: r[k] for k in SCHEMA})
            else:
                fa, ok = grade_forced(r["task_type"], r["completion"] or "", r["reference_answer"], r["prompt"])
                recs.append({**{k: r[k] for k in SCHEMA if k not in ("forced_answer", "correct")}, "forced_answer": fa, "correct": ok})
        summarise(frame(recs), rd, cuts, base_cond, suffix)
        return

    tok = load_tokenizer()
    end_id = tok.convert_tokens_to_ids("</think>")
    assert isinstance(end_id, int) and end_id >= 0, "</think> is not a single token"
    grades = load_grades()

    jobs, meta, recs = [], [], []
    n_own_free, n_own_fallback = 0, 0
    for run, cond in arms:
        tr = pl.read_parquet(DATA / "runs" / run / f"traces_{cond}.parquet")
        if a.smoke:
            tr = tr.head(a.smoke)
        for r in tr.iter_rows(named=True):
            ids = list(r["gen_token_ids"] or [])
            pos = ids.index(end_id) if end_id in ids else len(ids)
            task = r["task_type"]
            base = {"run": run, "condition": cond, "trace_id": r["trace_id"], "instance_id": r["instance_id"], "seed": int(r["seed"]),
                    "task_type": task, "think_tokens": int(pos)}
            for k in cuts:
                if pos <= k:                                   # closed its think block by k: own answer and grade
                    g = grades.get(r["trace_id"])
                    if g is None:
                        g = grade_own(r.get("answer_text") or "", r["reference_answer"], task, r["prompt"]); n_own_fallback += 1
                    else:
                        n_own_free += 1
                    recs.append({**base, "cut": k, "finished": True, "forced_answer": None, "completion": None, "correct": int(bool(g)),
                                 "reference_answer": str(r["reference_answer"]), "prompt": r["prompt"]})
                else:
                    jobs.append(render_prompt(tok, r["prompt"]) + tok.decode(ids[:k]) + FORCE.get(task, FORCE["math"]))
                    meta.append((base, k, r["reference_answer"], r["prompt"]))
    print(f"[readiness] {len(jobs)} forced continuations; {n_own_free + n_own_fallback} finished-before-cut cells "
          f"({n_own_free} with D-47 grades, {n_own_fallback} repo-grader fallback)", flush=True)
    if a.dry_run:
        print(jobs[0][-600:] if jobs else "(no forced prompts)")
        return

    from vllm import SamplingParams
    llm = build_engine(max_model_len=max(cuts) + 1024 + a.max_tokens + 64)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=0)

    if not a.no_gate and jobs:
        gate_idx = [i for i, m in enumerate(meta) if m[0]["run"] == arms[0][0] and m[0]["condition"] == base_cond][:8 * len(cuts)]
        outs = llm.generate([jobs[i] for i in gate_idx], sp)
        lines, parsed = [], 0
        for i, o in zip(gate_idx, outs):
            b, k, ref, prompt = meta[i]
            comp = o.outputs[0].text
            fa, ok = grade_forced(b["task_type"], comp, ref, prompt)
            parsed += bool(fa)
            lines.append(f"{b['instance_id']} seed {b['seed']} cut {k} think {b['think_tokens']} | ref {str(ref)[:40]} | forced {fa[:60]!r} ok={ok} | raw {comp[:80]!r}")
        (rd / "samples.txt").write_text("\n".join(lines))
        print("\n".join(lines), flush=True)
        if parsed < 0.6 * len(gate_idx):
            print(f"[readiness] GATE FAILED: {parsed}/{len(gate_idx)} forced continuations parsed; stopping", flush=True)
            sys.exit(2)
        print(f"[readiness] gate ok: {parsed}/{len(gate_idx)} parsed", flush=True)

    t0 = time.time()
    outs = llm.generate(jobs, sp) if jobs else []
    print(f"[readiness] generated {len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    for (b, k, ref, prompt), o in zip(meta, outs):
        recs.append({**b, "cut": k, "finished": False, "forced_answer": None, "completion": o.outputs[0].text, "correct": 0,
                     "reference_answer": str(ref), "prompt": prompt})
    raw = pl.DataFrame(recs, schema_overrides={**SCHEMA, "reference_answer": pl.Utf8, "prompt": pl.Utf8}, infer_schema_length=None)
    raw.write_parquet(rd / "raw_completions.parquet")
    print(f"[readiness] raw completions saved ({raw.height} rows)", flush=True)

    graded = []
    for r in recs:
        if r["finished"]:
            graded.append({k: r[k] for k in SCHEMA})
        else:
            fa, ok = grade_forced(r["task_type"], r["completion"] or "", r["reference_answer"], r["prompt"])
            graded.append({**{k: r[k] for k in SCHEMA if k not in ("forced_answer", "correct")}, "forced_answer": fa, "correct": ok})
    summarise(frame(graded), rd, cuts, base_cond, suffix)
    (rd / "config.json").write_text(json.dumps({"arms": a.arms, "cuts": cuts, "max_tokens": a.max_tokens, "force": FORCE,
                                                "model": MODEL_ID, "smoke": a.smoke}, indent=1))


if __name__ == "__main__":
    main()
