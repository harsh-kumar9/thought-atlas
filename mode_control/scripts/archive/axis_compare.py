import polars as pl, numpy as np, json, sys
RUN, DOM, REFRUN = sys.argv[1], sys.argv[2], "sched1_div"
from mode_control.mc.common import BEH
pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_rows(40); pl.Config.set_fmt_float("mixed")
rates = [f"{b}__rate" for b in BEH]
def shift_vec(run, cond, dom):
    sc = pl.read_parquet(f"mode_control/data/runs/{run}/analysis/scored.parquet").filter(pl.col("task_type") == dom)
    b = sc.filter(pl.col("condition") == "baseline").select(["instance_id"] + rates)
    g = sc.filter(pl.col("condition") == cond).select(["instance_id"] + rates).join(b, on="instance_id", suffix="_b")
    S = g.select(rates).to_numpy().astype(float); B = g.select([r + "_b" for r in rates]).to_numpy().astype(float)
    return S - B                                   # [n, 8] paired shifts
ref = {"rank1 explore (sus_idea, sched1_div)": shift_vec(REFRUN, "sus_idea_L19", DOM), "rank1 check (sus_math, sched1_div)": shift_vec(REFRUN, "sus_math_L19", DOM)}
arch = json.load(open("mode_control/data/sigctrl/archetypes.json"))
spec = json.load(open(f"mode_control/data/runs/{RUN}/run.json"))
base_mean = pl.read_parquet(f"mode_control/data/runs/{RUN}/analysis/scored.parquet").filter(pl.col("condition") == "baseline").select(rates).to_numpy().astype(float).mean(0)
rng = np.random.default_rng(0)
rows = []
for c in spec["conditions"]:
    if c["mode"] != "target": continue
    D = shift_vec(RUN, c["name"], DOM); m = D.mean(0)
    tgt = np.asarray(arch[c["phi_star"]]["phi"][:8]) - base_mean          # target gap from this run's own baseline
    u = tgt / np.linalg.norm(tgt)
    covered = D @ u / np.linalg.norm(tgt)                                     # fraction of the gap covered along the target direction (>1 = overshoot)
    bs = [covered[rng.integers(0, len(covered), len(covered))].mean() for _ in range(2000)]
    r = {"condition": c["name"], "|shift|": float(np.linalg.norm(m)), "gap_covered": float(covered.mean()), "gc_lo": float(np.percentile(bs, 2.5)), "gc_hi": float(np.percentile(bs, 97.5))}
    for k, R in ref.items():
        rm = R.mean(0); r[f"cos vs {k.split(' (')[0]}"] = float(np.dot(m, rm) / (np.linalg.norm(m) * np.linalg.norm(rm) + 1e-9))
    # component orthogonal to the rank-1 explore/check axis
    axis = ref["rank1 explore (sus_idea, sched1_div)"].mean(0); axis = axis / np.linalg.norm(axis)
    orth = m - np.dot(m, axis) * axis; r["|orth to rank1 axis|"] = float(np.linalg.norm(orth)); r["orth verification"] = float(orth[BEH.index("verification")]); r["orth backtracking"] = float(orth[BEH.index("backtracking")])
    rows.append(r)
print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(3)))
print(f"rank-1 explore shift vector ({DOM}, 40 tasks x 5 seeds):", dict(zip(BEH, ref["rank1 explore (sus_idea, sched1_div)"].mean(0).round(3))))
for c in spec["conditions"]:
    if c["mode"] == "target" and not c.get("random_subspace") and c.get("gain") == spec["conditions"][1].get("gain"):
        print("archetype gap from baseline:", c["phi_star"], dict(zip(BEH, (np.asarray(arch[c["phi_star"]]["phi"][:8]) - base_mean).round(3))))
