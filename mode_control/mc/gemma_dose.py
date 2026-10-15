"""D-64 stage 4a: random-norm dose sweep for Gemma-4-12B (which norms keep generation coherent at the readout layer).

Writes Gaussian random vectors at the given norms for the chosen layer into data/gemma4_12b/vsteer/ and a run
data/gemma4_12b/runs/gdose_math/ (the first N held-out math tasks, 2 seeds, 8k budget) with one arm per norm plus
baseline. Outcomes read afterwards with mc.budget (accuracy by budget) and the loop flags in the traces: the coherent
range is the largest norm whose hard-loop rate stays within 10 points of baseline and whose traces still parse.
Run with MC_MODEL=gemma4_12b: python -m mode_control.mc.gemma_dose --layer 24 --norms 1,2,4,8,16 --n-tasks 20
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from .common import DATA, GEN_MODEL_KEY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--norms", default="1,2,4,8,16")
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--vector", default=None, help="also sweep this calibrated vector at --scales")
    ap.add_argument("--scales", default="0.5,1,2,4")
    a = ap.parse_args()
    assert GEN_MODEL_KEY == "gemma4_12b"
    VS = DATA / "vsteer"; VS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    u = rng.standard_normal(3840); u /= np.linalg.norm(u)
    conds = [{"name": "baseline", "mode": "baseline"}]
    for n in [float(x) for x in a.norms.split(",")]:
        name = f"rand_n{n:g}"
        np.savez(VS / f"vec_{name}.npz", v=(u * n).astype(np.float32), layer=a.layer, meta=json.dumps({"source": "random_gaussian", "norm": n, "layer": a.layer, "decision": "D-64 dose sweep"}))
        conds.append({"name": name, "mode": "vector", "vector": name, "positions": "all"})
    if a.vector:
        z = np.load(VS / f"vec_{a.vector}.npz", allow_pickle=True)
        v0, meta0 = z["v"].astype(np.float32), json.loads(str(z["meta"]))
        for sc in [float(x) for x in a.scales.split(",")]:
            name = f"{a.vector}_x{sc:g}"
            np.savez(VS / f"vec_{name}.npz", v=(v0 * sc).astype(np.float32), layer=int(z["layer"]), meta=json.dumps({**meta0, "scale": sc, "norm": float(np.linalg.norm(v0) * sc)}))
            conds.append({"name": name, "mode": "vector", "vector": name, "positions": "all"})
    rd = DATA / "runs" / "gdose_math"; rd.mkdir(parents=True, exist_ok=True)
    tasks = pl.read_parquet(DATA / "tasks_heldout_math.parquet").head(a.n_tasks)
    tasks.write_parquet(rd / "tasks.parquet")
    spec = {"run": "gdose_math", "engine": "vllm", "seeds": [1, 2], "max_new_tokens": a.budget, "window": 8192, "tasks_from": "gdose_math",
            "decision": "D-64 dose sweep", "layer": a.layer, "conditions": conds}
    (rd / "run.json").write_text(json.dumps(spec, indent=2))
    print(f"[gdose] {tasks.height} tasks, layer {a.layer}, arms {[c['name'] for c in conds]} -> {rd}")


if __name__ == "__main__":
    main()
