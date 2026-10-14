"""F1 minimal pairs (D-50): does deliberation follow the task's demand, its topic, or its answer format?

80 MATH-500 problems and 80 GPQA questions, four prompts each:
  solve_free  math as in the atlas; GPQA stem without options + "Reason step by step, then end with your final answer."
  solve_mcq   GPQA as in the atlas; math stem + four options (reference + three wrong final answers) + the GPQA instruction
  generate    "Do not solve this problem. ... list several genuinely different ways ..." (no options)
  evaluate    the problem (no options) and a proposed final answer, correct for half of the items; verdict Correct/Incorrect
task_type stays "math" / "gpqa" so the atlas generator and judge treat the rows as usual; the design factors are in
metadata and in the instance_id suffix ("<atlas id>::<prompt type>").

  tasks     -> data/f1/tasks/{math,gpqa}.parquet (CPU)
  judge_in  merged traces -> data/f1/judge_in/traces_<model>.parquet, reasoning text cut to the 8,192-token window (CPU)
  acts      residual stream at the last prompt token, every layer (GPU; MC_MODEL = qwen35_9b or reasoner)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

import numpy as np
import polars as pl

from .common import MC, TASKS_DIR

F1 = MC / "data" / "f1"
SEED = 20261006
N_PER_TOPIC = 80
WINDOW = 8192
MCQ_INSTR = "Reason step by step, then end with your final answer as a single letter."
FREE_INSTR = "Reason step by step, then end with your final answer."
GEN = ("Do not solve this problem. Instead, list several genuinely different ways someone could approach it, and say briefly "
       "what each approach would involve.\n\nProblem:\n{stem}")
EVAL = ("Here is a problem and a proposed final answer. Decide whether the proposed answer is correct. Reason it through, then "
        "end with your verdict: Correct or Incorrect.\n\nProblem:\n{stem}\n\nProposed answer: {ans}")
GEN_MODELS = {"qwen35_9b": "Qwen/Qwen3.5-9B", "gemma4_12b": "google/gemma-4-12B-it", "reasoner": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}


def _h(s: str) -> int:
    return int(hashlib.sha256(f"{SEED}|{s}".encode()).hexdigest()[:8], 16)


def parse_gpqa(prompt: str):
    """stem, [4 option texts] from an atlas GPQA prompt ('<stem>\\n\\nA. ..\\nB. ..\\nC. ..\\nD. ..\\n\\n<instruction>')."""
    body = prompt.rsplit("\n\n", 1)[0] if prompt.rstrip().endswith(MCQ_INSTR) else prompt
    m = re.search(r"\n\nA\. (.*)\nB\. (.*)\nC\. (.*)\nD\. (.*)$", body, flags=re.S)
    if not m:
        return None, None
    opts = [x.strip() for x in m.groups()]
    if any("\n" in o or not o for o in opts):
        return None, None
    return body[:m.start()].strip(), opts


def math_distractors(iid: str, ref: str, wrong: list[str]) -> list[str] | None:
    """Three wrong integer answers for an integer-answer problem: other atlas models' wrong integers first, then nearby
    integers. Non-integer references are skipped so every option is a plain, unambiguous value."""
    if not re.fullmatch(r"-?\d+", ref.strip()):
        return None
    wrong = [w for w in ((x or "").strip().strip("$").replace(",", "") for x in wrong) if re.fullmatch(r"-?\d+", w)]
    return _math_distractors(iid, ref.strip(), wrong)


def _math_distractors(iid: str, ref: str, wrong: list[str]) -> list[str] | None:
    from .extract_free import math_correct, norm_math

    def as_set(x):                                      # unordered answer lists ("-2,1" == "1, -2"); not tuples or intervals
        x = x.strip()
        if "," not in x or x[:1] in "([{" or x.startswith("\\"):
            return None
        return frozenset(norm_math(e).replace(" ", "") for e in x.split(","))
    def clean(w):                                       # a bare final value, or None for parser debris ("**", "x", prose)
        w = re.sub(r"^\s*[A-Za-z]{1,3}\s*=\s*", "", (w or "").strip().strip("$").strip())
        if not w or "**" in w or len(w) > 25 or re.search(r"\\(right|left)?arrow|\\to\b|\\text", w):
            return None
        if re.search(r"[A-Za-z]{2,}", re.sub(r"\\[A-Za-z]+", "", w)) or re.fullmatch(r"[A-Za-z]", w) or "=" in w:
            return None
        return w
    out, seen = [], {norm_math(ref).replace(" ", "")}
    ref_set = as_set(ref)
    for w in (clean(x) for x in wrong):
        if w is None:
            continue
        k = norm_math(w).replace(" ", "")
        if not k or len(w) > 40 or k in seen or math_correct(w, ref) or (ref_set is not None and as_set(w) == ref_set):
            continue
        seen.add(k); out.append(w.strip().strip("$"))
        if len(out) == 3:
            return out
    if re.fullmatch(r"-?\d+", ref.strip()):                       # integer reference: fill with nearby integers
        n = int(ref); rng = np.random.default_rng(_h(iid))
        for d in rng.permutation([1, -1, 2, -2, 3, 10, -10, n if n else 5]):
            c = str(n + int(d))
            if c.replace(" ", "") not in seen:
                seen.add(c); out.append(c)
            if len(out) == 3:
                return out
    return None


def cmd_tasks():
    F1.mkdir(parents=True, exist_ok=True); (F1 / "tasks").mkdir(exist_ok=True)
    # wrong final answers other atlas models gave, per math problem (canonical grades; re-checked with the D-47 matcher)
    g = pl.read_parquet(MC.parent / "data/v2/perf/success_grades.parquet", columns=["trace_id", "task_type", "success", "predicted_answer"])
    tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "instance_id"]) for p in sorted((MC.parent / "data/v2/traces").glob("traces_*.parquet")) if ".shard" not in p.name])
    g = g.filter(pl.col("task_type") == "math").join(tr, on="trace_id")
    wrong = {i: [w for w in grp.filter((pl.col("success") == 0) & pl.col("predicted_answer").is_not_null())["predicted_answer"].to_list()]
             for (i,), grp in g.group_by(["instance_id"])}
    math = pl.read_parquet(TASKS_DIR / "math.parquet")
    elig = []
    for r in math.iter_rows(named=True):
        d = math_distractors(r["instance_id"], str(r["reference_answer"]), wrong.get(r["instance_id"], []))
        if d:
            elig.append({**r, "distractors": d})
    rng = np.random.default_rng(SEED); picks = []
    for lv in sorted({e["difficulty_raw"] for e in elig}):
        pool = [e for e in elig if e["difficulty_raw"] == lv]
        picks += [pool[i] for i in sorted(rng.choice(len(pool), min(N_PER_TOPIC // 5, len(pool)), replace=False))]
    gp = pl.read_parquet(TASKS_DIR / "gpqa.parquet")
    gel = []
    for r in gp.iter_rows(named=True):
        stem, opts = parse_gpqa(r["prompt"])
        if stem and sum(o in stem for o in opts) < 2:      # skip stems that already list the options (no real free format)
            gel.append({**r, "stem": stem, "opts": opts})
    gpicks = [gel[i] for i in sorted(rng.choice(len(gel), N_PER_TOPIC, replace=False))]
    rows = {"math": [], "gpqa": []}

    def add(topic, base, ptype, prompt, ref, extra):
        rows[topic].append({"instance_id": f"{base['instance_id']}::{ptype}", "task_type": topic, "prompt": prompt, "reference_answer": ref,
                            "difficulty_raw": base.get("difficulty_raw"),
                            "metadata": json.dumps({"f1_item": base["instance_id"], "topic": topic, "ptype": ptype,
                                                    "demand": "solve" if ptype.startswith("solve") else ptype,
                                                    "format": "mcq" if ptype == "solve_mcq" else "free", **extra})})
    for r in picks:
        iid, stem, ref = r["instance_id"], r["prompt"], str(r["reference_answer"])
        opts = [ref] + r["distractors"]; order = np.random.default_rng(_h(iid + "opts")).permutation(4)
        letters = "ABCD"; shown = [opts[i] for i in order]; key = letters[list(order).index(0)]
        add("math", r, "solve_free", stem, ref, {})
        add("math", r, "solve_mcq", stem + "\n\n" + "\n".join(f"{letters[k]}. {o}" for k, o in enumerate(shown)) + "\n\n" + MCQ_INSTR, key, {"options": shown})
        add("math", r, "generate", GEN.format(stem=stem), None, {})
        ok = _h(iid + "eval") % 2 == 0; prop = ref if ok else r["distractors"][0]
        add("math", r, "evaluate", EVAL.format(stem=stem, ans=prop), "Correct" if ok else "Incorrect", {"proposed": prop, "proposed_correct": ok})
    for r in gpicks:
        iid, stem, opts, key = r["instance_id"], r["stem"], r["opts"], str(r["reference_answer"]).strip().upper()
        right = opts["ABCD".index(key)]
        add("gpqa", r, "solve_free", stem + "\n\n" + FREE_INSTR, right, {})
        add("gpqa", r, "solve_mcq", r["prompt"], key, {})
        add("gpqa", r, "generate", GEN.format(stem=stem), None, {})
        ok = _h(iid + "eval") % 2 == 0
        wrong_opts = [o for o in opts if o != right]; prop = right if ok else wrong_opts[_h(iid + "w") % len(wrong_opts)]
        add("gpqa", r, "evaluate", EVAL.format(stem=stem, ans=prop), "Correct" if ok else "Incorrect", {"proposed": prop, "proposed_correct": ok})
    for topic, rr in rows.items():
        pl.DataFrame(rr).write_parquet(F1 / "tasks" / f"{topic}.parquet")
        print(f"[f1 tasks] {topic}: {len(rr)} prompts from {len(rr) // 4} items ({len(elig) if topic == 'math' else len(gel)} eligible)")
    ex = rows["math"][1]["prompt"]; print("[f1 tasks] example math solve_mcq:\n" + ex[-400:])


def cmd_judge_in(suffix: str = ""):
    """Merged traces -> judge input with reasoning_text_for_analysis cut to the first WINDOW tokens of each model's tokenizer."""
    from transformers import AutoTokenizer
    (F1 / f"judge_in{suffix}").mkdir(parents=True, exist_ok=True)
    for key, hf in GEN_MODELS.items():
        p = F1 / f"traces{suffix}" / f"traces_{key}.parquet"
        if not p.exists():
            print(f"[f1 judge_in] {key}: no merged traces yet"); continue
        tok = AutoTokenizer.from_pretrained(hf)
        t = pl.read_parquet(p)
        cut = []
        # the atlas generator leaves reasoning_text_for_analysis empty when the budget ends inside <think>; the first 8k tokens of
        # such a trace are still its opening reasoning, so fall back to think_text (D-50 note; finished-only is a robustness check)
        src = [r if r else th for r, th in zip(t["reasoning_text_for_analysis"].to_list(), t["think_text"].to_list())]
        for x in src:
            if not x:
                cut.append(x); continue
            ids = tok(x, add_special_tokens=False)["input_ids"]
            cut.append(x if len(ids) <= WINDOW else tok.decode(ids[:WINDOW]))
        t = t.with_columns(pl.Series("reasoning_text_for_analysis", cut), pl.lit("baseline").alias("condition"))
        t.write_parquet(F1 / f"judge_in{suffix}" / f"traces_{key}.parquet")
        print(f"[f1 judge_in] {key}: {t.height} traces, {sum(1 for x in cut if x)} with reasoning text")


def cmd_acts():
    import torch
    from .common import GEN_MODEL_KEY, decoder_layers, load_model, load_tokenizer, render_prompt, _layer_output
    tok = load_tokenizer(); model = load_model(); dl = decoder_layers(model)
    tasks = pl.concat([pl.read_parquet(F1 / "tasks" / f"{t}.parquet").with_columns(pl.col("difficulty_raw").cast(pl.Utf8)) for t in ("math", "gpqa")])
    got = {}
    hs = [l.register_forward_hook(lambda m, a, o, i=i: got.__setitem__(i, _layer_output(o)[0, -1].float().cpu().numpy())) for i, l in enumerate(dl)]
    H = np.zeros((tasks.height, len(dl), int(getattr(model.config, "text_config", model.config).hidden_size)), np.float16)
    try:
        for n, r in enumerate(tasks.iter_rows(named=True)):
            ids = tok(render_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=model.device), logits_to_keep=1)
            H[n] = np.stack([got[i] for i in range(len(dl))]).astype(np.float16)
            if n % 100 == 0:
                print(f"[f1 acts] {n}/{tasks.height}", flush=True)
    finally:
        for h in hs: h.remove()
    meta = [json.loads(m) for m in tasks["metadata"].to_list()]
    np.savez(F1 / f"acts_{GEN_MODEL_KEY}.npz", H=H, instance_id=np.array(tasks["instance_id"].to_list()),
             topic=np.array([m["topic"] for m in meta]), ptype=np.array([m["ptype"] for m in meta]), item=np.array([m["f1_item"] for m in meta]))
    print(f"[f1 acts] {H.shape} -> {F1 / f'acts_{GEN_MODEL_KEY}.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tasks", "judge_in", "acts"])
    ap.add_argument("--suffix", default="", help='judge_in: "" for the main samples, "_s2" for the noise-floor samples')
    a = ap.parse_args()
    if a.cmd == "judge_in":
        cmd_judge_in(a.suffix)
    else:
        {"tasks": cmd_tasks, "acts": cmd_acts}[a.cmd]()


if __name__ == "__main__":
    main()
