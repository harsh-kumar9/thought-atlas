"""Candidate qualitative examples for the story page (run on ada, CPU)."""
import json, glob
import numpy as np
import polars as pl
from pathlib import Path
from mode_control.mc.common import BEH, DATA, JUDGE_MODEL, load_tokenizer
from mode_control.mc.phenotype import _segments_with_offsets
from mode_control.mc.budget import grade

R = DATA / "runs"; tok = load_tokenizer(); TAG = JUDGE_MODEL.replace("/", "_")
SHORT = {"Question_and_Answering": "Q&A", "Perspective_Shift": "shift", "Conflict_of_Perspectives": "conflict", "Reconciliation": "reconcile",
         "verification": "verify", "backtracking": "backtrack", "subgoal": "subgoal", "backward_chaining": "backward"}

def load_tb(run):
    fs = sorted((R / run / "judge").rglob(f"trackB_full__{TAG}*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed").unique(["trace_id", "seg_idx"], keep="last")

def segments(row, tb):
    g = tb.filter(pl.col("trace_id") == row["trace_id"]).sort("seg_idx")
    if g.height == 0 or not row.get("reasoning_text_for_analysis"): return None
    n = int(g["n_segments"][0]); text = row["reasoning_text_for_analysis"]
    starts = _segments_with_offsets(tok, text, n)
    if starts is None: return None
    ids = tok(text, add_special_tokens=False)["input_ids"]; ends = starts[1:] + [len(ids)]
    lab = {int(r["seg_idx"]): [SHORT[b] for b in BEH if r[b]] for r in g.iter_rows(named=True)}
    return [{"i": k, "text": tok.decode(ids[a:b]).strip(), "labels": lab.get(k, [])} for k, (a, b) in enumerate(zip(starts, ends))]

out = {}
# 1. math vs idea signature examples (corpus runs, seed 1)
for dom in ("math", "idea"):
    run = f"corpus1_{dom}_a"; tb = load_tb(run)
    tr = pl.concat([pl.read_parquet(p) for p in sorted((R / run).glob("traces_*.parquet"))], how="diagonal_relaxed").filter(pl.col("seed") == 1)
    tr = tr.filter(pl.col("completed")).sort("n_new_tokens")
    picks = tr[[tr.height // 3, tr.height // 2]]
    out[f"sig_{dom}"] = []
    for r in picks.iter_rows(named=True):
        segs = segments(r, tb)
        if segs: out[f"sig_{dom}"].append({"instance_id": r["instance_id"], "prompt": r["prompt"][:900], "tokens": r["n_new_tokens"], "segs": segs[:40]})

# 2. same math problem: unsteered vs A2 vs RL from zero (seed 1, 32k budget)
tb1 = load_tb("p1_budget"); tb3 = load_tb("p4_eval3")
cols = ["trace_id", "instance_id", "seed", "prompt", "reference_answer", "answer_text", "n_new_tokens", "finish_reason", "hard_loop", "reasoning_text_for_analysis", "full_text"]
b = pl.read_parquet(R / "p1_budget/traces_baseline.parquet", columns=cols).filter(pl.col("seed") == 1)
a = pl.read_parquet(R / "p1_budget/traces_ol_math_A2.parquet", columns=cols).filter(pl.col("seed") == 1)
z = pl.read_parquet(R / "p4_eval3/traces_rl_zeroT_s60.parquet", columns=cols).filter(pl.col("seed") == 1)
j = b.join(a, on="instance_id", suffix="_a").join(z, on="instance_id", suffix="_z")
cands = []
for r in j.iter_rows(named=True):
    cb = grade(r["answer_text"] or "", r["reference_answer"], "math", r["prompt"]); ca = grade(r["answer_text_a"] or "", r["reference_answer"], "math", r["prompt"])
    cz = grade(r["answer_text_z"] or "", r["reference_answer"], "math", r["prompt"])
    if cb == 0 and ca == 1 and cz == 1 and (r["hard_loop"] or r["finish_reason"] == "length") and len(r["prompt"]) < 500:
        cands.append((r["n_new_tokens"] - r["n_new_tokens_a"], r))
cands.sort(key=lambda x: -x[0])
out["same_problem"] = []
for _, r in cands[:4]:
    ex = {"instance_id": r["instance_id"], "prompt": r["prompt"], "reference": r["reference_answer"]}
    for sfx, name, tb in (("", "unsteered", tb1), ("_a", "A2", tb1), ("_z", "rl_zero", tb3)):
        row = {k: r[k + sfx] if (k + sfx) in r else r[k] for k in ["trace_id", "reasoning_text_for_analysis", "n_new_tokens", "finish_reason", "hard_loop", "answer_text"]}
        segs = segments(row, tb)
        ex[name] = {"tokens": row["n_new_tokens"], "finish": row["finish_reason"], "loop": row["hard_loop"], "answer": (row["answer_text"] or "")[-300:],
                    "segs_head": segs[:14] if segs else None, "n_segs": len(segs) if segs else None,
                    "tail": (r["full_text" + sfx] or "")[-700:] if name == "unsteered" else None}
    out["same_problem"].append(ex)

# 3. idea task under explore vs check (sched1, full dose)
q = pl.read_parquet(R / "sched1/judge/quality__google_gemma-4-31B-it.parquet").select(["trace_id", "axis_originality", "axis_feasibility"])
ti = pl.concat([pl.read_parquet(p, columns=["trace_id", "instance_id", "condition", "prompt", "answer_text", "task_type", "n_new_tokens"]) for p in sorted((R / "sched1").glob("traces_*.parquet"))], how="diagonal_relaxed").filter(pl.col("task_type") == "idea").join(q, on="trace_id")
piv = {}
for r in ti.iter_rows(named=True): piv.setdefault(r["instance_id"], {})[r["condition"]] = r
ideas = []
for iid, d in piv.items():
    if all(c in d for c in ("baseline", "sus_idea_L19", "sus_math_L19")):
        e, c, bl = d["sus_idea_L19"], d["sus_math_L19"], d["baseline"]
        if e["answer_text"] and c["answer_text"] and e["axis_originality"] > c["axis_originality"] and c["axis_feasibility"] > e["axis_feasibility"] and len(e["answer_text"]) < 2500 and len(c["answer_text"]) < 2500:
            ideas.append({"instance_id": iid, "prompt": bl["prompt"][:1200], "explore": {k: e[k] for k in ("answer_text", "axis_originality", "axis_feasibility", "n_new_tokens")},
                          "check": {k: c[k] for k in ("answer_text", "axis_originality", "axis_feasibility", "n_new_tokens")},
                          "baseline": {k: bl[k] for k in ("answer_text", "axis_originality", "axis_feasibility", "n_new_tokens")}})
out["idea"] = ideas[:6]

# 4. P3: draw-a-picture changes the method and solves what unprompted misses
m = pl.read_parquet(R / "p3_strat/judge/methods__google_gemma-4-31B-it.parquet").filter(pl.col("parsed"))
t3 = pl.concat([pl.read_parquet(p, columns=["trace_id", "instance_id", "seed", "condition", "prompt", "reference_answer", "answer_text", "n_new_tokens", "think_text"]) for p in sorted((R / "p3_strat").glob("traces_*.parquet"))], how="diagonal_relaxed")
t3 = t3.join(m.select(["trace_id", "method_category", "method_summary"]), on="trace_id")
t3 = t3.with_columns(pl.Series("correct", [grade(x or "", rr, "math", p) for x, rr, p in zip(t3["answer_text"].to_list(), t3["reference_answer"].to_list(), t3["prompt"].to_list())]))
p3 = []
for iid, g in t3.group_by("instance_id"):
    iid = iid[0] if isinstance(iid, tuple) else iid
    bl = g.filter(pl.col("condition") == "baseline"); fg = g.filter(pl.col("condition") == "prompt_figure")
    if bl.height and fg.height and bl["correct"].sum() == 0 and fg["correct"].sum() >= 2 and (fg["method_category"] == "geometric_or_coordinate_argument").any() and not (bl["method_category"] == "geometric_or_coordinate_argument").any() and len(bl["prompt"][0]) < 450:
        f1 = fg.filter(pl.col("correct") == 1).row(0, named=True); b1 = bl.row(0, named=True)
        p3.append({"instance_id": iid, "prompt": b1["prompt"], "reference": b1["reference_answer"],
                   "baseline_methods": bl.select(["method_category", "method_summary", "correct"]).to_dicts(),
                   "figure_methods": fg.select(["method_category", "method_summary", "correct"]).to_dicts(),
                   "figure_think_head": (f1["think_text"] or "")[:1500], "baseline_think_head": (b1["think_text"] or "")[:1000]})
out["p3"] = p3[:5]
Path("/ada1/u/harsh/tmp/story_examples.json").write_text(json.dumps(out))
print({k: len(v) for k, v in out.items()})
