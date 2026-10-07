"""Representation-space and layer-map figures.

  layer_phase_heatmaps : Gate 1 (spontaneous domain gap, AUROC), Gate 2 (within-domain rho) per layer x phase
  rep_space            : at chosen layers, PCA of baseline activations colored by domain / score, and the
                         2-D plane spanned by out-of-fold ridge readout directions for s_conv and s_cog
                         (conversational vs cognitive), colored by domain.
Outputs under data/directions/figures/.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from .common import DATA, stable_fold


def heatmaps(map_csv, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    df = pl.read_csv(map_csv).filter(pl.col("direction") == "prompt_r1")
    phases = df["phase"].unique().sort().to_list(); layers = sorted(df["layer"].unique().to_list())
    metrics = [("delta_sd_units", "Gate 1: Idea−Math gap on prompt direction (SD units)"),
               ("proj_domain_auroc", "Gate 1: domain AUROC of projection"),
               ("paired_sign_frac", "paired condition decodability (frac diffs > 0)"),
               ("rawdom_oof_auroc", "raw domain direction: OOF domain AUROC"),
               ("g2_cog_rho_math", "Gate 2: rho(proj, s_cog) within Math"), ("g2_cog_rho_idea", "Gate 2: rho(proj, s_cog) within Idea"),
               ("g2_conv_rho_math", "Gate 2: rho(proj, s_conv) within Math"), ("g2_conv_rho_idea", "Gate 2: rho(proj, s_conv) within Idea"),
               ("g2_all_oof_r2_gain", "Gate 2: OOF R² gain for s_all beyond domain")]
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    for ax, (m, title) in zip(axes.ravel(), metrics):
        M = np.full((len(phases), len(layers)), np.nan)
        for r in df.iter_rows(named=True):
            M[phases.index(r["phase"]), layers.index(r["layer"])] = r[m]
        im = ax.imshow(M, aspect="auto", cmap="coolwarm" if "rho" in m or "delta" in m or "cos" in m else "viridis")
        ax.set_yticks(range(len(phases))); ax.set_yticklabels(phases); ax.set_xticks(layers[::2]); ax.set_xlabel("layer")
        ax.set_title(title, fontsize=9); plt.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out / "layer_phase_map.png", dpi=140); plt.close(fig)


def rep_space(stageA, pheno, layers, phase, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    A = np.load(stageA, allow_pickle=True)
    idx = {t: i for i, t in enumerate(A["trace_id"].tolist())}
    ph = pheno.filter(pl.col("trace_id").is_in(list(idx)))
    order = np.array([idx[t] for t in ph["trace_id"].to_list()])
    pi = A["phases"].tolist().index(phase)
    dom = (ph["task_type"] == "idea").to_numpy(); folds = np.array([stable_fold(s) for s in ph["instance_id"].to_list()])
    s_cog, s_conv = ph["s_cog"].to_numpy(), ph["s_conv"].to_numpy()
    fig, axes = plt.subplots(len(layers), 4, figsize=(20, 4.5 * len(layers)), squeeze=False)
    for row, l in enumerate(layers):
        X = A["acts"][order, l, pi, :].astype(np.float32); X = X - X.mean(0)
        P = PCA(n_components=2).fit_transform(X)
        for c, (vals, name, cmap) in enumerate([(dom.astype(float), "domain (blue Math, red Idea)", "coolwarm"),
                                                 (s_cog, "s_cog", "viridis"), (s_conv, "s_conv", "plasma")]):
            ax = axes[row, c]; ax.scatter(P[:, 0], P[:, 1], c=vals, cmap=cmap, s=8, alpha=0.7)
            ax.set_title(f"layer {l} {phase}: PCA, color = {name}", fontsize=9)
        # OOF ridge readouts of s_cog and s_conv -> 2-D "conversational vs cognitive" plane
        pc = PCA(n_components=64).fit(X); Z = pc.transform(X)
        pred = {}
        for name, s in (("cog", s_cog), ("conv", s_conv)):
            p = np.zeros(len(s))
            for k in np.unique(folds):
                tr, te = folds != k, folds == k
                # within-domain target: remove domain means so the readout is not just the domain axis
                s_c = s - np.where(dom, s[dom].mean(), s[~dom].mean())
                m = Ridge(alpha=10.0).fit(Z[tr], s_c[tr]); p[te] = m.predict(Z[te])
            pred[name] = p
        ax = axes[row, 3]
        ax.scatter(pred["cog"], pred["conv"], c=dom.astype(float), cmap="coolwarm", s=8, alpha=0.7)
        from scipy.stats import spearmanr
        rc = [spearmanr(pred["cog"][m], (s_cog - np.where(dom, s_cog[dom].mean(), s_cog[~dom].mean()))[m]).correlation for m in (~dom, dom)]
        rv = [spearmanr(pred["conv"][m], (s_conv - np.where(dom, s_conv[dom].mean(), s_conv[~dom].mean()))[m]).correlation for m in (~dom, dom)]
        ax.set_xlabel(f"OOF readout of within-domain s_cog (rho Math {rc[0]:.2f}, Idea {rc[1]:.2f})")
        ax.set_ylabel(f"OOF readout of within-domain s_conv (rho Math {rv[0]:.2f}, Idea {rv[1]:.2f})")
        ax.set_title(f"layer {l}: cognitive vs conversational readouts", fontsize=9)
    fig.tight_layout(); fig.savefig(out / f"rep_space_{phase}.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=4096); ap.add_argument("--layers", default="8,16,24")
    ap.add_argument("--phase", default="mean32"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = DATA / "directions/figures"; out.mkdir(parents=True, exist_ok=True)
    heatmaps(DATA / f"directions/layer_phase_map{a.tag}.csv", out)
    pheno = pl.read_parquet(DATA / f"phenotype/baseline_w{a.window}.parquet")
    rep_space(DATA / "acts/stageA_k64.npz", pheno, [int(x) for x in a.layers.split(",")], a.phase, out)
    print("->", out)


if __name__ == "__main__":
    main()
