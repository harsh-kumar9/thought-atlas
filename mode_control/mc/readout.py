"""Within-domain linear readout of the frozen scores by layer and phase (64-PC ridge, out of fold).

Answers RQ1 without committing to one direction: is the later signature linearly readable from
early activations within each domain? Writes data/directions/readout_by_phase{tag}.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from .common import DATA, FAMILIES, stable_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageA", default=str(DATA / "acts/stageA_k192.npz"))
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--tag", default="_k192")
    ap.add_argument("--layers", default="4,8,12,16,19,22,24,28")
    ap.add_argument("--n-pcs", type=int, default=64)
    a = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr
    A = np.load(a.stageA, allow_pickle=True); phases = A["phases"].tolist()
    ph = pl.read_parquet(DATA / f"phenotype/baseline_w{a.window}.parquet")
    idx = {t: i for i, t in enumerate(A["trace_id"].tolist())}; ph = ph.filter(pl.col("trace_id").is_in(list(idx)))
    order = np.array([idx[t] for t in ph["trace_id"].to_list()]); dom = (ph["task_type"] == "idea").to_numpy()
    folds = np.array([stable_fold(s) for s in ph["instance_id"].to_list()])
    rows = []
    for phase in phases:
        pi = phases.index(phase)
        for l in [int(x) for x in a.layers.split(",")]:
            for dname, m in (("math", ~dom), ("idea", dom)):
                X = A["acts"][order][m, l, pi, :].astype(np.float32); X = X - X.mean(0); f = folds[m]
                Z = PCA(n_components=min(a.n_pcs, X.shape[0] - 1)).fit_transform(X)
                r = {"phase": phase, "layer": l, "domain": dname, "n": int(m.sum())}
                for fam in FAMILIES:
                    s = ph[f"s_{fam}"].to_numpy()[m]; p = np.zeros(len(s))
                    for k in np.unique(f):
                        tr, te = f != k, f == k; p[te] = Ridge(alpha=10.0).fit(Z[tr], s[tr]).predict(Z[te])
                    r[f"rho_{fam}"] = float(spearmanr(p, s).correlation)
                rows.append(r)
            print(f"{phase:11s} L{l:2d} " + " ".join(f"{r['domain']}: cog={r['rho_cog']:+.2f} conv={r['rho_conv']:+.2f} all={r['rho_all']:+.2f}" for r in rows[-2:]), flush=True)
    out = DATA / "directions" / f"readout_by_phase{a.tag}.csv"; pl.DataFrame(rows).write_csv(out); print("->", out)


if __name__ == "__main__":
    main()
