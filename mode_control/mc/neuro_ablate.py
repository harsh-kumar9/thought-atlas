"""D-69: do the gap-carrying heads cause the behaviour? Builds run.json for a head-ablation run on held-out Math and Idea
tasks; generation and judging use the standard stages (mode.sbatch steer <run>, judge <run>, mc.analyze).

Conditions: baseline; abl_idea<k> = the top-k Idea-ward carriers of the layer-19 axis (map_components.csv, conjunction
passed, heads only); abl_math<k> = the top-k Math-ward carriers; abl_rand<k> = k random heads, one per layer of the Idea
set, not in either set. Heads are silenced on every position (common.HeadAblator).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from .common import DATA

NEURO = DATA / "neuro"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="abl1"); ap.add_argument("--k", type=int, default=8); ap.add_argument("--n-per-domain", type=int, default=40)
    ap.add_argument("--task-seed", type=int, default=20261014); ap.add_argument("--window", type=int, default=4096); ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--seeds", default="1"); ap.add_argument("--rand-seed", type=int, default=7)
    ap.add_argument("--exclude-runs", default="corpus1_math_a,corpus1_math_b,corpus1_idea_a,corpus1_idea_b")
    ap.add_argument("--directional", action="store_true", help="D-69b: remove only the heads' writes along the layer-19 axis; adds 2k sets and the neutralize ceiling")
    a = ap.parse_args()
    df = pl.read_csv(NEURO / "analysis" / "map_components.csv").filter(pl.col("kind").is_in(["attn", "lin"]) & (pl.col("layer") <= 19) & pl.col("dom19_conj"))
    idea = df.sort("dom19_diff", descending=True).head(a.k); math = df.sort("dom19_diff").head(a.k)
    used = set(zip(idea["layer"], idea["head"])) | set(zip(math["layer"], math["head"]))
    allh = pl.read_csv(NEURO / "analysis" / "map_components.csv").filter(pl.col("kind").is_in(["attn", "lin"]))
    rng = np.random.default_rng(a.rand_seed); rand = []
    for l in idea["layer"].to_list():
        pool = [(int(r["layer"]), r["kind"], int(r["head"])) for r in allh.filter(pl.col("layer") == l).iter_rows(named=True) if (int(r["layer"]), int(r["head"])) not in used]
        pick = pool[int(rng.integers(len(pool)))]; used.add((pick[0], pick[2])); rand.append(pick)
    sets = {"idea": [[int(r["layer"]), r["kind"], int(r["head"])] for r in idea.iter_rows(named=True)],
            "math": [[int(r["layer"]), r["kind"], int(r["head"])] for r in math.iter_rows(named=True)],
            "rand": [list(x) for x in rand]}
    if a.directional:
        k2 = 2 * a.k
        idea2 = df.sort("dom19_diff", descending=True).head(k2); math2 = df.sort("dom19_diff").head(k2)
        used2 = set(used) | set(zip(idea2["layer"], idea2["head"])) | set(zip(math2["layer"], math2["head"])); rand2 = []
        for l in idea2["layer"].to_list():
            pool = [(int(r["layer"]), r["kind"], int(r["head"])) for r in allh.filter(pl.col("layer") == l).iter_rows(named=True) if (int(r["layer"]), int(r["head"])) not in used2]
            pick = pool[int(rng.integers(len(pool)))]; used2.add((pick[0], pick[2])); rand2.append(pick)
        big = {f"idea{k2}": [[int(r["layer"]), r["kind"], int(r["head"])] for r in idea2.iter_rows(named=True)],
               f"math{k2}": [[int(r["layer"]), r["kind"], int(r["head"])] for r in math2.iter_rows(named=True)], f"rand{k2}": [list(x) for x in rand2]}
        ax = {"direction": "prompt_r1", "layer": 19, "phase": "mean32"}
        conds = [{"name": "baseline", "mode": "baseline"}]
        conds += [{"name": f"dabl_{nm}{a.k}", "mode": "ablate", "heads": hs, "sustained": True, **ax} for nm, hs in sets.items()]
        conds += [{"name": f"dabl_{nm}", "mode": "ablate", "heads": hs, "sustained": True, **ax} for nm, hs in big.items()]
        conds.append({"name": "neutralize_sus_L19", "mode": "gamma", "gamma": 0.0, "sustained": True, "pulse_k": 64, **ax})
        sets.update(big)
    else:
        conds = [{"name": "baseline", "mode": "baseline"}] + [{"name": f"abl_{nm}{a.k}", "mode": "ablate", "heads": hs, "sustained": True} for nm, hs in sets.items()]
    excl = sorted({i for r in a.exclude_runs.split(",") for i in pl.read_parquet(DATA / "runs" / r / "tasks.parquet")["instance_id"].to_list()}) if a.exclude_runs else None
    spec = {"run": a.run, "window": a.window, "max_new_tokens": a.max_new_tokens, "n_per_domain": a.n_per_domain, "task_seed": a.task_seed, "domains": ["math", "idea"],
            "exclude_instances": excl, "seeds": [int(x) for x in a.seeds.split(",")], "directions": "directions_w8192_v2w8192.npz", "layer_selection": None,
            "derived": {"source": "data/neuro/analysis/map_components.csv", "k": a.k, "decision": "D-69b" if a.directional else "D-69"}, "conditions": conds}
    rd = DATA / "runs" / a.run; rd.mkdir(parents=True, exist_ok=True); (rd / "run.json").write_text(json.dumps(spec, indent=2))
    for nm, hs in sets.items():
        print(f"[ablate] {nm}: " + ", ".join(f"L{l} {k} h{h}" for l, k, h in hs))
    print(f"[ablate] {len(conds)} conditions, {a.n_per_domain} tasks per domain (excluding {len(excl or [])} corpus tasks) -> {rd / 'run.json'}")


if __name__ == "__main__":
    main()
