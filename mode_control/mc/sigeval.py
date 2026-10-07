"""Signature-controller evaluation (D-33 stage 5): gap closure, selectivity, naturality per condition.

gap closure(task) = 1 - d(phi_steered, phi*) / d(phi_unsteered, phi*), d = Euclidean over the 8 overall behavior
rates (from the judge on the steered trace), paired by task; selectivity = movement along the intended
direction (phi* - phi_unsteered) divided by movement orthogonal to it. Naturality: loops and completion from
summary.csv. Utility comes from analyze.py (accuracy, has_answer) and the quality judge.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from .common import BEH, DATA


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
    rd = DATA / "runs" / a.run; spec = json.loads((rd / "run.json").read_text())
    arch = json.load(open(DATA / "sigctrl" / "archetypes.json"))
    sc = pl.read_parquet(rd / "analysis" / "scored.parquet")
    rates = [f"{b}__rate" for b in BEH]
    base = sc.filter(pl.col("condition") == "baseline").select(["instance_id", "task_type"] + rates)
    rng = np.random.default_rng(0); rows = []
    for c in spec["conditions"]:
        if c["mode"] != "target": continue
        phi = np.asarray(arch[c["phi_star"]]["phi"][:8]) if isinstance(c["phi_star"], str) else np.asarray(c["phi_star"][:8])
        g = sc.filter(pl.col("condition") == c["name"]).select(["instance_id", "task_type"] + rates)
        j = g.join(base, on=["instance_id", "task_type"], suffix="_b")
        for dom in sorted(j["task_type"].unique().to_list()):
            jj = j.filter(pl.col("task_type") == dom)
            S = jj.select(rates).to_numpy().astype(float); B = jj.select([r + "_b" for r in rates]).to_numpy().astype(float)
            d_s = np.linalg.norm(S - phi, axis=1); d_b = np.linalg.norm(B - phi, axis=1)
            gc = 1 - d_s / np.maximum(d_b, 1e-6)
            v = phi - B; vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
            move = S - B; along = (move * vn).sum(1); orth = np.linalg.norm(move - along[:, None] * vn, axis=1)
            sel = along / np.maximum(orth, 1e-6)
            bs = [gc[rng.integers(0, len(gc), len(gc))].mean() for _ in range(2000)]
            rows.append({"condition": c["name"], "target": c["phi_star"], "gain": c.get("gain"), "random": bool(c.get("random_subspace")), "task_type": dom, "n": len(gc),
                         "gap_closure": float(gc.mean()), "gc_lo": float(np.percentile(bs, 2.5)), "gc_hi": float(np.percentile(bs, 97.5)),
                         "along": float(along.mean()), "orth": float(orth.mean()), "selectivity": float(np.median(sel)),
                         "d_unsteered": float(d_b.mean()), "d_steered": float(d_s.mean())})
    res = pl.DataFrame(rows).sort(["task_type", "gain", "condition"]); res.write_csv(rd / "analysis" / "sigctrl_eval.csv")
    su = pl.read_csv(rd / "analysis" / "summary.csv").select(["condition", "task_type", "hard_loop", "has_answer", "n_new_tokens", "math_success_all"])
    pl.Config.set_tbl_width_chars(240); pl.Config.set_tbl_rows(80); pl.Config.set_fmt_float("mixed")
    print(res.join(su, on=["condition", "task_type"], how="left").with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
