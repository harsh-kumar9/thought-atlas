"""D-64 stage 5: run specs for the Gemma-4-12B core cells (vsteer format), written only; submission follows the dose sweep.

Runs under data/gemma4_12b/runs/ (MC_MODEL=gemma4_12b), 2 seeds, 32k budget, 8k judge window, fixed vectors at layer 28:
  g_math      held-out math (200):     baseline, own_math_A1, rand_own_math_A1, sig_idea, rand_sig_idea
  g_gpqa      held-out gpqa (200):     baseline, own_gpqa_A1, own_math_A1 (the Gemma Math vector on GPQA), rand_own_gpqa_A1
  g_planning  held-out planning (200): baseline, own_planning_A3, rand_own_planning_A3
  g_idea      held-out idea (200):     baseline, sig_math, rand_sig_math
  g_countdown the 200 cd_countdown tasks (read-only copy): baseline, conv_up, conv_down, rand_conv_up
Optional --scale rescales every calibrated vector by one factor (after the dose sweep), writing vec_<name>_s<scale> copies.
Usage: MC_MODEL=gemma4_12b python -m mode_control.mc.gemma_runs [--scale 1]
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import polars as pl

from .common import DATA, GEN_MODEL_KEY, MC

# rand_n2 = Gaussian at norm 2, the largest norm a random direction survives in the dose sweep (D-64 stage 4b); the norm-matched
# rand_<name> controls are kept as pre-registered but sit above that range. conv_*_x0.5 are the half-dose insurance arms.
RUNS = {
    "g_math": ("math", ["own_math_A1", "rand_own_math_A1", "sig_idea", "rand_sig_idea", "rand_n2"]),
    "g_gpqa": ("gpqa", ["own_gpqa_A1", "own_math_A1", "rand_own_gpqa_A1", "rand_n2"]),
    "g_planning": ("planning", ["own_planning_A3", "rand_own_planning_A3", "rand_n2"]),
    "g_idea": ("idea", ["sig_math", "rand_sig_math", "rand_n2"]),
    "g_countdown": ("countdown", ["conv_up", "conv_down", "rand_conv_up", "conv_up_x0.5", "conv_down_x0.5", "rand_n2"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--budget", type=int, default=32768)
    a = ap.parse_args()
    assert GEN_MODEL_KEY == "gemma4_12b"
    VS = DATA / "vsteer"
    for base in ("conv_up", "conv_down"):                                   # half-dose copies for the Countdown insurance arms
        z = np.load(VS / f"vec_{base}.npz", allow_pickle=True)
        np.savez(VS / f"vec_{base}_x0.5.npz", v=(z["v"].astype(np.float32) * 0.5), layer=int(z["layer"]),
                 meta=json.dumps({**json.loads(str(z["meta"])), "scale": 0.5, "norm": float(np.linalg.norm(z["v"]) * 0.5)}))
    for run, (dom, vecs) in RUNS.items():
        rd = DATA / "runs" / run; rd.mkdir(parents=True, exist_ok=True)
        if dom == "countdown":
            shutil.copy(MC / "data" / "runs" / "cd_countdown" / "tasks.parquet", rd / "tasks.parquet")
        else:
            pl.read_parquet(DATA / f"tasks_heldout_{dom}.parquet").write_parquet(rd / "tasks.parquet")
        conds = [{"name": "baseline", "mode": "baseline"}]
        for v in vecs:
            name = v
            if a.scale != 1.0:
                z = np.load(VS / f"vec_{v}.npz", allow_pickle=True)
                name = f"{v}_s{a.scale:g}"
                np.savez(VS / f"vec_{name}.npz", v=(z["v"].astype(np.float32) * a.scale), layer=int(z["layer"]),
                         meta=json.dumps({**json.loads(str(z["meta"])), "scale": a.scale, "norm": float(np.linalg.norm(z["v"]) * a.scale)}))
            conds.append({"name": name, "mode": "vector", "vector": name, "positions": "all"})
        spec = {"run": run, "engine": "vllm", "seeds": [1, 2], "max_new_tokens": a.budget, "window": 8192, "tasks_from": run,
                "decision": "D-64 stage 5", "model": "google/gemma-4-12B-it", "layer": 28, "scale": a.scale, "conditions": conds}
        (rd / "run.json").write_text(json.dumps(spec, indent=2))
        t = pl.read_parquet(rd / "tasks.parquet")
        print(f"[gruns] {run}: {t.height} tasks, arms {[c['name'] for c in conds]}")


if __name__ == "__main__":
    main()
