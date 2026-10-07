"""Power for the primary causal contrast from baseline within-domain variability.

Primary estimand per condition γ: Δγ = E[s | idea, γ] - E[s | math, γ]. Attenuation test compares
Δ_0 to Δ_1 on the same tasks. Conservative model: score SD under intervention equals the baseline
within-domain SD (σ), no pairing benefit. Var(Δ̂_γ - Δ̂_1) ≈ 4σ²/N (N tasks per domain).
Detectable attenuation fraction f at 80% power: f·Δ_1 = (z_.975 + z_.8)·2σ/√N.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from .common import DATA


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--window", type=int, default=4096); a = ap.parse_args()
    ph = pl.read_parquet(DATA / f"phenotype/baseline_w{a.window}.parquet")
    z = 1.96 + 0.84
    rows = []
    for fam in ["cog", "conv", "all"]:
        s = ph[f"s_{fam}"].to_numpy(); dom = (ph["task_type"] == "idea").to_numpy()
        delta = s[dom].mean() - s[~dom].mean()
        sigma = np.sqrt(0.5 * (s[dom].var() + s[~dom].var()))
        for N in [20, 40, 80, 150, 340]:
            f = z * 2 * sigma / np.sqrt(N) / delta
            rows.append({"family": fam, "N_per_domain": N, "delta_baseline": round(float(delta), 2),
                         "sigma_within": round(float(sigma), 2), "min_detectable_attenuation": round(float(f), 3)})
        for f in [0.1, 0.25, 0.5]:
            N = (z * 2 * sigma / (f * delta)) ** 2
            rows.append({"family": fam, "N_per_domain": int(np.ceil(N)), "delta_baseline": round(float(delta), 2),
                         "sigma_within": round(float(sigma), 2), "min_detectable_attenuation": f})
    df = pl.DataFrame(rows).sort(["family", "N_per_domain"])
    out = DATA / "phenotype" / f"power_w{a.window}.csv"; df.write_csv(out)
    print(df); print("->", out)


if __name__ == "__main__":
    main()
