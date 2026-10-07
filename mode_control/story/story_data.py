"""Collect the numbers for the story page into one JSON (run on ada, CPU)."""
import json
import numpy as np
import polars as pl
from pathlib import Path
from mode_control.mc.common import BEH, DATA

R = DATA / "runs"; V = DATA / "vsteer"
out = {}
RATES = [f"{b}__rate" for b in BEH]

# 1. signatures per domain (unsteered corpus, 800 traces each)
sig = {}
for dom in ["math", "gpqa", "planning", "idea"]:
    s = pl.concat([pl.read_csv(R / f"corpus1_{dom}_{x}/analysis/summary.csv") for x in "ab"]).filter(pl.col("condition") == "baseline")
    sig[dom] = {b: float(s[f"{b}__rate"].mean()) for b in BEH}
    sig[dom]["tokens"] = float(s["n_new_tokens"].mean())
out["signatures"] = sig

# 2. rank-one knob (powered run): Idea-likeness score by condition and domain
sc = pl.read_parquet(R / "power1/analysis/scored.parquet")
out["power1"] = sc.group_by(["condition", "task_type"]).agg(pl.col("s_all").mean(), pl.len()).sort(["task_type", "condition"]).to_dicts()

# 3. budget curves
def summ(run, conds=None):
    s = pl.read_csv(R / f"{run}/analysis/budget_summary.csv")
    if conds: s = s.filter(pl.col("condition").is_in(conds))
    return s.select(["condition", "task_type", "budget", "correct", "answered", "tokens", "n"]).to_dicts()
out["budget_math"] = summ("p1_budget") + summ("p4_eval2") + summ("p4_eval3", ["rl_zeroT_s60"])
out["budget_gpqa"] = summ("p2_gpqa"); out["budget_planning"] = summ("p2_planning")
def paired(run):
    return pl.read_csv(R / f"{run}/analysis/budget_paired.csv").select(["reference", "condition", "task_type", "budget", "d_correct", "d_correct_lo", "d_correct_hi"]).to_dicts()
out["paired_math"] = paired("p1_budget"); out["paired_gpqa"] = paired("p2_gpqa"); out["paired_planning"] = paired("p2_planning")
loops = {}
for run in ["p1_budget", "p2_gpqa", "p2_planning", "p4_eval2", "p4_eval3"]:
    for p in sorted((R / run).glob("traces_*.parquet")):
        t = pl.read_parquet(p, columns=["condition", "hard_loop"])
        loops[f"{run}/{t['condition'][0]}"] = float(t["hard_loop"].mean())
out["loops"] = loops

# 4. behaviour shifts (seed 1, paired vs unsteered on same task and seed), with task-bootstrap CI
rng = np.random.default_rng(0)
def shifts(scored_run, cond, base_run=None):
    df = pl.read_parquet(R / f"{scored_run}/analysis/scored.parquet")
    bdf = pl.read_parquet(R / f"{base_run or scored_run}/analysis/scored.parquet")
    base = bdf.filter(pl.col("condition") == "baseline").select(["instance_id", "seed"] + RATES)
    j = df.filter(pl.col("condition") == cond).select(["instance_id", "seed"] + RATES).join(base, on=["instance_id", "seed"], suffix="_b")
    res = {}
    for b, r in zip(BEH, RATES):
        d = (j[r] - j[r + "_b"]).drop_nulls().to_numpy()
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
        res[b] = [float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    return {"n": j.height, "shift": res}
out["shifts"] = {
    "math/A2": shifts("p1_budget", "ol_math_A2"), "math/sealdir": shifts("p1_budget", "sealdir_matched"),
    "math/seal": shifts("p1_budget", "seal"), "math/concise": shifts("p1_budget", "concise"), "math/random": shifts("p1_budget", "ol_random"),
    "math/rl_zero": shifts("p4_eval3", "rl_zeroT_s60", "p1_budget"), "math/rl_a2": shifts("p4_eval2", "rl_a2initT_s60", "p1_budget"),
    "gpqa/A2": shifts("p2_gpqa", "ol_math_A2"), "gpqa/own": shifts("p2_gpqa", "ol_gpqa_A4"),
    "planning/A2": shifts("p2_planning", "ol_math_A2"), "planning/own": shifts("p2_planning", "ol_planning_A2"),
}
base_rates = {}
for run in ["p1_budget", "p2_gpqa", "p2_planning"]:
    b = pl.read_parquet(R / f"{run}/analysis/scored.parquet").filter(pl.col("condition") == "baseline")
    base_rates[run] = {x: float(b[r].mean()) for x, r in zip(BEH, RATES)}
out["base_rates"] = base_rates

# 5. cosines: residual vectors vs behaviour-shift vectors
vec = {"A2": "ol_math_A2", "SEAL direction": "sealdir_L17_n6p75", "RL from zero": "rl_zero8k_T_s60", "A2 + RL": "rl_a2init8k_T_s60", "random": "ol_random"}
sh = {"A2": "math/A2", "SEAL direction": "math/sealdir", "RL from zero": "math/rl_zero", "A2 + RL": "math/rl_a2", "random": "math/random"}
vv = {k: np.load(V / f"vec_{f}.npz")["v"] for k, f in vec.items()}
ss = {k: np.array([out["shifts"][f]["shift"][b][0] for b in BEH]) for k, f in sh.items()}
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
names = list(vec)
out["cos"] = {"names": names, "residual": [[cos(vv[a], vv[b]) for b in names] for a in names],
              "behaviour": [[cos(ss[a], ss[b]) for b in names] for a in names], "norms": {k: float(np.linalg.norm(v)) for k, v in vv.items()}}

# 6. RL training curves (fixed-gradient runs)
out["rl"] = {r: [json.loads(l) for l in open(DATA / f"rlbias/{r}/metrics.jsonl")] for r in ["zero8k_T", "a2init8k_T"]}
for r in out["rl"]:
    d = {x["step"]: x for x in out["rl"][r]}; out["rl"][r] = [{k: d[s][k] for k in ("step", "reward", "answered", "tokens", "b_norm", "cos_ol_math_A2")} for s in sorted(d)]

# 7. P3 method mix and pools
m = pl.read_parquet(R / "p3_strat/judge/methods__google_gemma-4-31B-it.parquet").filter(pl.col("parsed"))
mix = m.group_by(["condition", "method_category"]).len().with_columns((pl.col("len") / pl.col("len").sum().over("condition")).alias("share"))
out["p3_mix"] = mix.select(["condition", "method_category", "share"]).to_dicts()
def offmode(m):
    """Share of samples whose method is not the usual one for their task. Usual = most common unprompted method; with ties,
    every tied method counts as usual (a plain group_by().first() after sort picks an arbitrary mode)."""
    cnt = m.filter(pl.col("condition") == "baseline").group_by(["instance_id", "method_category"]).len()
    modes = cnt.filter(pl.col("len") == pl.col("len").max().over("instance_id")).select(["instance_id", "method_category"]).with_columns(pl.lit(True).alias("is_mode"))
    j = m.join(modes, on=["instance_id", "method_category"], how="left").with_columns(pl.col("is_mode").fill_null(False))
    return j.group_by("condition").agg((~pl.col("is_mode")).mean().alias("off")).to_dicts()
out["p3_offmode"] = offmode(m)
ms = pl.read_parquet(R / "p3_steer/judge/methods__google_gemma-4-31B-it.parquet").filter(pl.col("parsed"))
out["p3s_offmode"] = offmode(ms)
out["p3_budget"] = summ("p3_strat"); out["p3s_budget"] = summ("p3_steer")
loops3 = {}
for run in ["p3_strat", "p3_steer"]:
    for p in sorted((R / run).glob("traces_*.parquet")):
        t = pl.read_parquet(p, columns=["condition", "hard_loop"]); loops3[f"{run}/{t['condition'][0]}"] = float(t["hard_loop"].mean())
out["loops3"] = loops3

Path("/ada1/u/harsh/tmp/story_data.json").write_text(json.dumps(out))
print("ok", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items()})
print(json.dumps(out["power1"], indent=0)[:1500])
