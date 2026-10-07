"""Label the solution method of each Math trace (strategy-level diversity).

The judge (same model as the atlas judge) reads the problem, the reasoning (clipped) and the final answer
and returns a method category from a fixed taxonomy plus a one-line method summary. Distinct categories
per task across seeds, and embedding diversity of the summaries, measure strategy diversity.
Output: data/runs/<run>/judge/methods__<judge>.parquet (resumable by trace_id).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .common import DATA, JUDGE_MODEL
from src.judge.vllm_engine import EngineConfig, build_llm, chat_batch, make_sampling, safe_json  # noqa: E402

CATEGORIES = [
    "direct_computation", "algebraic_manipulation", "substitution_or_change_of_variable", "casework_or_enumeration",
    "geometric_or_coordinate_argument", "number_theoretic_argument", "counting_principle_or_combinatorial_identity",
    "inequality_or_bounding", "pattern_finding_or_guess_and_check", "recursion_or_induction", "probability_or_expectation_argument",
    "other",
]

PROMPT = """You are classifying HOW a solver approached a competition math problem. Read the problem, the solver's reasoning and its final answer. Ignore whether the answer is correct.

Return JSON with:
- "method_category": exactly one of {cats}
- "method_summary": one sentence (max 25 words) naming the key idea or route the solver actually used
- "n_approaches_tried": integer, how many distinct solution routes the solver tried before settling (1 if it went straight through)
- "switched_approach": true if the solver abandoned a route and switched to a different one

PROBLEM:
{problem}

SOLVER'S REASONING (may be clipped):
{reasoning}

SOLVER'S FINAL ANSWER:
{answer}
"""

SCHEMA = {"type": "object", "properties": {
    "method_category": {"type": "string", "enum": CATEGORIES},
    "method_summary": {"type": "string", "maxLength": 300},
    "n_approaches_tried": {"type": "integer", "minimum": 1, "maximum": 10},
    "switched_approach": {"type": "boolean"}},
    "required": ["method_category", "method_summary", "n_approaches_tried", "switched_approach"]}


def clip(tok, text: str, limit: int) -> str:
    ids = tok(text or "", add_special_tokens=False)["input_ids"]
    if len(ids) <= limit:
        return text or ""
    head, tail = ids[: limit * 2 // 3], ids[-limit // 3:]
    return tok.decode(head) + "\n...[clipped]...\n" + tok.decode(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--reasoning-tokens", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    rd = DATA / "runs" / a.run
    tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "condition", "instance_id", "task_type", "seed", "prompt", "think_text", "answer_text"])
                    for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed").filter(pl.col("task_type") == "math")
    out = rd / "judge" / f"methods__{a.judge_model.replace('/', '_')}.parquet"; out.parent.mkdir(parents=True, exist_ok=True)
    prev = pl.read_parquet(out) if out.exists() else None
    done = set(prev["trace_id"].to_list()) if prev is not None else set()
    todo = tr.filter(~pl.col("trace_id").is_in(list(done)))
    print(f"[methods] {todo.height} to label ({len(done)} done)", flush=True)
    if todo.height == 0:
        return
    eng = EngineConfig(model=a.judge_model, dtype="bfloat16", tensor_parallel_size=1, max_model_len=16384)
    llm = build_llm(eng)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.judge_model, trust_remote_code=True)
    samp = make_sampling(temperature=0.0, max_tokens=200, json_schema=SCHEMA)
    rows = []
    for i in range(0, todo.height, a.batch):
        chunk = todo[i:i + a.batch]
        prompts = [PROMPT.format(cats=json.dumps(CATEGORIES), problem=r["prompt"], reasoning=clip(tok, r["think_text"], a.reasoning_tokens),
                                 answer=(r["answer_text"] or "")[:1500]) for r in chunk.iter_rows(named=True)]
        outs = chat_batch(llm, tok, prompts, samp, max_input_tokens=16384 - 300)
        for r, o in zip(chunk.iter_rows(named=True), outs):
            j = safe_json(o) or {}
            rows.append({"trace_id": r["trace_id"], "condition": r["condition"], "instance_id": r["instance_id"], "seed": r["seed"],
                         "method_category": j.get("method_category"), "method_summary": j.get("method_summary"),
                         "n_approaches_tried": j.get("n_approaches_tried"), "switched_approach": j.get("switched_approach"),
                         "parsed": bool(j), "judge_model": a.judge_model})
        res = pl.DataFrame(rows)
        allres = pl.concat([prev, res], how="diagonal_relaxed") if prev is not None else res
        tmp = out.with_suffix(".parquet.tmp"); allres.write_parquet(tmp); tmp.replace(out)
        print(f"[methods] {min(i + a.batch, todo.height)}/{todo.height}", flush=True)


if __name__ == "__main__":
    main()
