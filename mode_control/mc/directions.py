"""Fit candidate control subspaces and run Gates 1-2 plus the layer x phase representation map.

Inputs: data/acts/stageA_k64.npz (baseline acts), data/acts/stageB_k32.npz (paired prompt diffs),
data/phenotype/baseline_w{W}.parquet (OOF frozen scores).

Directions per layer l and phase (all unit-norm, sign fixed so that +u = exploratory / Idea-like):
  prompt_r1   : mean paired difference E_i[d_il] (exploratory - checking), unwhitened.
  prompt_pw_r1: the same mean difference whitened inside the top-k PCA subspace of the baseline
                activations (k=64). Full-covariance whitening is not identifiable here (n=783 < d=4096;
                Ledoit-Wolf collapses the direction into the null space), so this is the robustness variant.
  rawdom_r1   : mu_idea - mu_math on baseline acts (topic-confounded control); its domain AUROC is
                reported out-of-fold on the prompt folds.
  prompt_r2/4 : mean difference plus the leading principal directions of the centered paired diffs.

Gate 1: delta_l = E[u^T h | idea] - E[u^T h | math] on baseline acts, AUROC of the projection.
Gate 2: out-of-fold within-domain prediction of the frozen scores from the projection (domain intercepts).
Outputs: data/directions/layer_phase_map{tag}.csv, directions_w{W}{tag}.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from .common import DATA, FAMILIES, orthonormal, stable_fold


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def fit_directions(A: np.ndarray, dom: np.ndarray, folds: np.ndarray, D: np.ndarray, *, k_pca: int = 64,
                   ranks=(2, 4)):
    A = A.astype(np.float32); D = D.astype(np.float32)
    center = A.mean(0)
    m = D.mean(0)
    out = {"center": center}
    out["prompt_r1"] = _unit(m)[:, None].astype(np.float32)
    # PCA-subspace whitening (top-k PCs of the baseline acts)
    Ac = A - center
    U_, S_, Vt = np.linalg.svd(Ac, full_matrices=False)
    V = Vt[:k_pca].T; lam = (S_[:k_pca] ** 2) / (len(A) - 1)
    m_pc = V.T @ m
    out["prompt_pw_r1"] = _unit(V @ (m_pc / lam))[:, None].astype(np.float32)
    out["rawdom_r1"] = _unit(A[dom].mean(0) - A[~dom].mean(0))[:, None].astype(np.float32)
    # rank-r: mean diff + leading principal directions of centered diffs
    _, _, VtD = np.linalg.svd(D - m, full_matrices=False)
    for r in ranks:
        out[f"prompt_r{r}"] = orthonormal(np.concatenate([m[:, None], VtD[:r - 1].T], axis=1))
    out["cos_prompt_rawdom"] = float(out["prompt_r1"][:, 0] @ out["rawdom_r1"][:, 0])
    out["cos_prompt_pw"] = float(out["prompt_r1"][:, 0] @ out["prompt_pw_r1"][:, 0])
    out["paired_sign_frac"] = float(((D @ out["prompt_r1"][:, 0]) > 0).mean())
    out["diff_norm_mean"] = float(np.linalg.norm(D, axis=1).mean()); out["mean_diff_norm"] = float(np.linalg.norm(m))
    # OOF domain AUROC of the raw domain direction (fit on train folds)
    from sklearn.metrics import roc_auc_score
    z = np.zeros(len(A))
    for kf in np.unique(folds):
        tr, te = folds != kf, folds == kf
        u = _unit(A[tr][dom[tr]].mean(0) - A[tr][~dom[tr]].mean(0)); z[te] = A[te] @ u
    out["rawdom_oof_auroc"] = float(roc_auc_score(dom.astype(int), z))
    return out


def readout_direction(A: np.ndarray, dom: np.ndarray, s: np.ndarray, *, k_pca: int = 64, alpha: float = 10.0):
    """Supervised readout: ridge from the top-k PCs of centered acts to the within-domain-centered score,
    mapped back to raw space (unit norm, sign: +u = Idea-like)."""
    from sklearn.linear_model import Ridge
    A = A.astype(np.float32); c = A.mean(0); Ac = A - c
    _, S_, Vt = np.linalg.svd(Ac, full_matrices=False); V = Vt[:k_pca].T
    Z = Ac @ V
    sc = s - np.where(dom, s[dom].mean(), s[~dom].mean())
    m = Ridge(alpha=alpha).fit(Z, sc)
    u = V @ m.coef_
    u = _unit(u)
    if (A[dom] @ u).mean() < (A[~dom] @ u).mean():
        u = -u
    return u[:, None].astype(np.float32)


def readout_oof_projection(A: np.ndarray, dom: np.ndarray, folds: np.ndarray, s: np.ndarray, **kw):
    z = np.zeros(len(A), np.float32)
    for kf in np.unique(folds):
        tr, te = folds != kf, folds == kf
        u = readout_direction(A[tr], dom[tr], s[tr], **kw)
        z[te] = A[te].astype(np.float32) @ u[:, 0]
    return z


def gate_stats(A: np.ndarray, dom: np.ndarray, folds: np.ndarray, scores: dict[str, np.ndarray], U: np.ndarray,
               z_override: np.ndarray | None = None):
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr
    z = A.astype(np.float32) @ U
    if z_override is not None:
        z = z.copy(); z[:, 0] = z_override
    z1 = z[:, 0]
    st = {"delta_idea_minus_math": float(z1[dom].mean() - z1[~dom].mean()),
          "delta_sd_units": float((z1[dom].mean() - z1[~dom].mean()) / (z1.std() + 1e-9)),
          "proj_domain_auroc": float(roc_auc_score(dom.astype(int), z1)),
          "proj_sd_math": float(z1[~dom].std()), "proj_sd_idea": float(z1[dom].std()),
          "mu_math": z[~dom].mean(0).tolist(), "mu_idea": z[dom].mean(0).tolist()}
    for name, s in scores.items():
        X = np.concatenate([dom[:, None].astype(np.float32), z], axis=1)
        pred = np.zeros(len(s)); base = np.zeros(len(s))
        for kf in np.unique(folds):
            tr, te = folds != kf, folds == kf
            pred[te] = Ridge(alpha=1.0).fit(X[tr], s[tr]).predict(X[te])
            base[te] = Ridge(alpha=1.0).fit(X[tr, :1], s[tr]).predict(X[te, :1])
        st[f"g2_{name}_oof_r2_gain"] = float(1 - ((s - pred) ** 2).sum() / ((s - base) ** 2).sum())
        for d_name, mask in (("math", ~dom), ("idea", dom)):
            st[f"g2_{name}_rho_{d_name}"] = float(spearmanr(z1[mask], s[mask]).correlation)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--stageA", default=str(DATA / "acts/stageA_k64.npz"))
    ap.add_argument("--stageB", default=str(DATA / "acts/stageB_k32.npz"))
    ap.add_argument("--layers", default="all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--k-pca", type=int, default=64)
    a = ap.parse_args()
    A = np.load(a.stageA, allow_pickle=True); B = np.load(a.stageB, allow_pickle=True)
    pheno = pl.read_parquet(DATA / f"phenotype/baseline_w{a.window}.parquet")
    tidA = A["trace_id"].tolist(); idx = {t: i for i, t in enumerate(tidA)}
    ph = pheno.filter(pl.col("trace_id").is_in(tidA))
    order = np.array([idx[t] for t in ph["trace_id"].to_list()])
    acts = A["acts"][order]
    dom = (ph["task_type"] == "idea").to_numpy()
    folds = np.array([stable_fold(s) for s in ph["instance_id"].to_list()])
    scores = {f: ph[f"s_{f}"].to_numpy() for f in FAMILIES}
    phasesA = A["phases"].tolist(); phasesB = B["phases"].tolist()
    diffs = B["diffs"]
    L = acts.shape[1]
    layers = range(L) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    rows = []; store = {}
    keys = ["prompt_r1", "prompt_pw_r1", "rawdom_r1", "prompt_r2", "prompt_r4", "readout_r1"]
    for l in layers:
        for pi, phase in enumerate(phasesA):
            if phase not in phasesB:
                continue
            pb = phasesB.index(phase)
            Al = acts[:, l, pi, :].astype(np.float32)
            Dl = diffs[:, :, l, pb, :].reshape(-1, diffs.shape[-1]).astype(np.float32)
            fit = fit_directions(Al, dom, folds, Dl, k_pca=a.k_pca)
            fit["readout_r1"] = readout_direction(Al, dom, scores["all"])
            z_ro = readout_oof_projection(Al, dom, folds, scores["all"])
            for key in keys:
                st = gate_stats(Al, dom, folds, scores, fit[key], z_override=(z_ro if key == "readout_r1" else None))
                rows.append({"layer": l, "phase": phase, "direction": key, "rank": fit[key].shape[1],
                             **{k: fit[k] for k in ["cos_prompt_rawdom", "cos_prompt_pw", "paired_sign_frac",
                                                    "diff_norm_mean", "mean_diff_norm", "rawdom_oof_auroc"]},
                             **{k: v for k, v in st.items() if not isinstance(v, list)}})
                store[f"{key}|{l}|{phase}"] = fit[key]
                store[f"mu|{key}|{l}|{phase}"] = np.array([st["mu_math"], st["mu_idea"]], np.float32)
                store[f"sd|{key}|{l}|{phase}"] = np.float32(np.sqrt(0.5 * (st["proj_sd_math"] ** 2 + st["proj_sd_idea"] ** 2)))
            store[f"center|{l}|{phase}"] = fit["center"]
            r0 = rows[-len(keys)]
            print(f"layer {l:2d} {phase:11s} prompt_r1: δ={r0['delta_sd_units']:+.2f}sd auroc={r0['proj_domain_auroc']:.3f} "
                  f"g2 ρ(cog) M={r0['g2_cog_rho_math']:+.2f} I={r0['g2_cog_rho_idea']:+.2f} ρ(conv) M={r0['g2_conv_rho_math']:+.2f} "
                  f"I={r0['g2_conv_rho_idea']:+.2f} r2gain(all)={r0['g2_all_oof_r2_gain']:+.3f} cos(raw)={fit['cos_prompt_rawdom']:.2f} "
                  f"rawdom_oof={fit['rawdom_oof_auroc']:.3f}", flush=True)
    out = DATA / "directions"; out.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows); df.write_csv(out / f"layer_phase_map{a.tag}.csv")
    np.savez(out / f"directions_w{a.window}{a.tag}.npz", **store)
    print(f"wrote {df.height} rows -> layer_phase_map{a.tag}.csv")


if __name__ == "__main__":
    main()
