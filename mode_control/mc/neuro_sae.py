"""E6 (D-60): the Math-Idea axis and path in the Qwen-Scope SAE basis (Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50).

The release has one TopK SAE (k = 50, 65,536 features, d = 4096) per decoder layer, hooked on the layer output
(model.model.layers[L] forward hook, 0-based: the same convention as our layer index). It was trained on the Base
model; we apply it to the instruct model's reasoning activations, so the first step is a gate on reconstruction.

fetch     download layer<L>.sae.pt into the HF cache (run on ada, which has network; the sbatch sets HF_HUB_OFFLINE=1)
gate      fraction of variance unexplained (FVU), L0 and dead-feature share on corpus segment states at layer L
          (sigctrl acts), per domain                                     -> data/neuro/sae_gate_L<L>.json
contrast  mean feature activation per domain, the Idea-minus-Math feature difference, its concentration, and each
          feature's share of the difference along the domain axis and in the readout subspace
                                                                         -> data/neuro/sae_contrast_L<L>.csv / .json
path      SAE codes along the straight line and the data curve (geometry_L<L>.npz): active-set turnover between
          consecutive points (Jaccard), distinct features over the path, reconstruction FVU of each point
                                                                         -> data/neuro/sae_path_L<L>.json
vectors   steering vectors from the top-k features of the Idea-minus-Math feature difference:
          v = sum_{f in top-k} (mean_idea_f - mean_math_f) d_f, scaled to the norm of nm_<dir>_line_s0p5, k in {8, 64};
          the reverse sign for toward_math                              -> data/vsteer/vec_nm_<dir>_sae_k<k>.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .common import DATA

NEURO = DATA / "neuro"; SIG = DATA / "sigctrl"; VS = DATA / "vsteer"
REPO_ID = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"
TOPK = 50
DOMS = ["math", "idea", "gpqa", "planning"]


def sae_path(layer: int) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, f"layer{layer}.sae.pt")


def load_sae(layer: int, device: str = "cpu"):
    import torch
    sd = torch.load(sae_path(layer), map_location=device)
    sd = sd.get("state_dict", sd)
    W_enc, b_enc, W_dec, b_dec = (sd[k].float() for k in ("W_enc", "b_enc", "W_dec", "b_dec"))
    assert W_enc.shape[1] == 4096 and W_dec.shape[0] == 4096, (W_enc.shape, W_dec.shape)
    return {"W_enc": W_enc, "b_enc": b_enc, "W_dec": W_dec, "b_dec": b_dec, "device": device}


def encode(sae, X, chunk: int = 2048):
    """TopK codes [n, F] (dense float32 on CPU is too large for 64K features: returns indices and values [n, k])."""
    import torch
    idx, val = [], []
    for i in range(0, len(X), chunk):
        x = torch.as_tensor(np.asarray(X[i:i + chunk], np.float32), device=sae["device"])
        pre = x @ sae["W_enc"].T + sae["b_enc"]
        v, j = pre.topk(TOPK, dim=-1)
        idx.append(j.cpu().numpy()); val.append(v.cpu().numpy())
    return np.concatenate(idx), np.concatenate(val)


def decode(sae, idx, val):
    import torch
    out = []
    for i in range(0, len(idx), 2048):
        j = torch.as_tensor(idx[i:i + 2048], device=sae["device"]); v = torch.as_tensor(val[i:i + 2048], device=sae["device"])
        out.append((torch.einsum("nk,dnk->nd", v, sae["W_dec"][:, j]) + sae["b_dec"]).cpu().numpy())
    return np.concatenate(out)


def _sample_states(dom: str, layer: int, n: int, rng):
    Hs = []
    for ab in ("a", "b"):
        z = np.load(SIG / f"corpus1_{dom}_{ab}__acts.npz", mmap_mode="r"); li = z["layers"].tolist().index(layer)
        H = z["H"][:, li, :]; idx = np.sort(rng.choice(H.shape[0], min(n // 2, H.shape[0]), replace=False))
        Hs.append(np.asarray(H[idx]).astype(np.float32))
    return np.concatenate(Hs)


def _device():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def cmd_gate(layer: int, n: int):
    rng = np.random.default_rng(0); sae = load_sae(layer, _device()); out = {"layer": layer, "repo": REPO_ID, "k": TOPK, "n_per_domain": n}
    F = sae["W_enc"].shape[0]; alive = np.zeros(F, bool)
    for d in DOMS:
        X = _sample_states(d, layer, n, rng)
        idx, val = encode(sae, X); R = decode(sae, idx, val)
        res = ((X - R) ** 2).sum(1); tot = ((X - X.mean(0)) ** 2).sum(1)
        fvu = float(res.sum() / tot.sum()); alive[np.unique(idx)] = True
        cos = float(np.mean((X * R).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(R, axis=1) + 1e-9)))
        out[d] = {"fvu": fvu, "cos_recon": cos, "mean_norm": float(np.linalg.norm(X, axis=1).mean()), "n_active_features": int(len(np.unique(idx)))}
        print(f"[sae gate] L{layer} {d}: FVU {fvu:.3f}, cosine {cos:.3f}, {len(np.unique(idx))} features used", flush=True)
    out["features_alive_any_domain"] = int(alive.sum()); out["pass"] = bool(all(out[d]["fvu"] < 0.35 for d in DOMS))
    NEURO.mkdir(parents=True, exist_ok=True); (NEURO / f"sae_gate_L{layer}.json").write_text(json.dumps(out, indent=1))
    print(f"[sae gate] pass = {out['pass']} (FVU < 0.35 in every domain)")


def cmd_contrast(layer: int, n: int):
    import polars as pl
    rng = np.random.default_rng(1); sae = load_sae(layer, _device()); F = sae["W_enc"].shape[0]
    means, freq = {}, {}
    for d in DOMS:
        X = _sample_states(d, layer, n, rng); idx, val = encode(sae, X)
        m = np.zeros(F); fr = np.zeros(F)
        np.add.at(m, idx.ravel(), val.ravel()); np.add.at(fr, idx.ravel(), 1.0)
        means[d] = m / len(X); freq[d] = fr / len(X)
    delta = means["idea"] - means["math"]
    dirs = np.load(DATA / "directions" / "directions_w8192_v2w8192.npz", allow_pickle=True)
    axis = dirs[f"prompt_r1|{layer}|mean32"][:, 0].astype(np.float32)
    dm = json.loads((NEURO / "directions.json").read_text()) if (NEURO / "directions.json").exists() else None
    Wd = sae["W_dec"].cpu().numpy()                                                 # [d, F]
    dec_norm = np.linalg.norm(Wd, axis=0); dec_axis = (axis @ Wd)                    # d_f . axis
    share_axis = delta * dec_axis                                                    # feature f's contribution to the gap on the axis
    total_axis = float(share_axis.sum()); gap_axis = float((Wd @ delta) @ axis)
    order = np.argsort(-np.abs(share_axis))
    cum = np.cumsum(share_axis[order]) / (total_axis if total_axis else 1.0)
    rows = []
    for r, f in enumerate(order[:400]):
        rows.append({"rank": r, "feature": int(f), "delta_mean": float(delta[f]), "mean_math": float(means["math"][f]), "mean_idea": float(means["idea"][f]),
                     "mean_gpqa": float(means["gpqa"][f]), "mean_planning": float(means["planning"][f]), "freq_math": float(freq["math"][f]), "freq_idea": float(freq["idea"][f]),
                     "dec_norm": float(dec_norm[f]), "dec_dot_axis": float(dec_axis[f]), "share_axis": float(share_axis[f] / (total_axis or 1.0))})
    pl.DataFrame(rows).write_csv(NEURO / f"sae_contrast_L{layer}.csv")
    n_half = int(np.searchsorted(np.cumsum(np.abs(share_axis[order])) / np.abs(share_axis).sum(), 0.5) + 1)
    out = {"layer": layer, "gap_on_axis_via_sae": gap_axis, "sum_feature_shares": total_axis, "n_features_half_abs_share": n_half,
           "n_features_nonzero_delta": int((delta != 0).sum()), "top_signed_cum_at_8": float(cum[7]), "top_signed_cum_at_64": float(cum[63]),
           "top_features": [int(f) for f in order[:64]]}
    (NEURO / f"sae_contrast_L{layer}.json").write_text(json.dumps(out, indent=1))
    np.savez(NEURO / f"sae_means_L{layer}.npz", **{f"mean_{d}": means[d] for d in DOMS}, **{f"freq_{d}": freq[d] for d in DOMS}, delta=delta, order=order)
    print(f"[sae contrast] L{layer}: gap on axis through the SAE {gap_axis:+.3f}; {n_half} features carry half of |share|; signed cumulative share top-8 {cum[7]:.2f}, top-64 {cum[63]:.2f}")


def cmd_path(layer: int):
    g = np.load(NEURO / f"geometry_L{layer}.npz", allow_pickle=True); sae = load_sae(layer, _device())
    out = {"layer": layer}
    for name, pts, s in (("line", g["line_pts"], g["s_line"]), ("curve", g["curve_pts"], g["s_curve"])):
        idx, val = encode(sae, pts); R = decode(sae, idx, val)
        fvu = ((pts - R) ** 2).sum(1) / (((pts - pts.mean(0)) ** 2).sum(1) + 1e-9)
        sets = [set(i.tolist()) for i in idx]
        jac = [len(a & b) / len(a | b) for a, b in zip(sets, sets[1:])]
        out[name] = {"s": s.tolist(), "fvu_vs_pathmean": fvu.tolist(), "recon_rel_err": (np.linalg.norm(pts - R, axis=1) / np.linalg.norm(pts, axis=1)).tolist(),
                     "jaccard_consecutive": jac, "n_distinct_features": len(set.union(*sets)), "active_sets": [sorted(x) for x in sets]}
        print(f"[sae path] {name}: distinct features over the path {len(set.union(*sets))}, mean consecutive Jaccard {np.mean(jac):.2f}, "
              f"relative recon error (min/median/max) {np.min(out[name]['recon_rel_err']):.3f}/{np.median(out[name]['recon_rel_err']):.3f}/{np.max(out[name]['recon_rel_err']):.3f}")
    (NEURO / f"sae_path_L{layer}.json").write_text(json.dumps(out, indent=1))


def cmd_vectors(layer: int, ks=(8, 64)):
    m = np.load(NEURO / f"sae_means_L{layer}.npz"); sae = load_sae(layer, "cpu"); Wd = sae["W_dec"].numpy()
    delta, order = m["delta"], m["order"]
    for direction, sign in (("toward_idea", 1.0), ("toward_math", -1.0)):
        ref = np.load(VS / f"vec_nm_{direction}_line_s0p5.npz", allow_pickle=True); target = float(np.linalg.norm(ref["v"]))
        for k in ks:
            f = order[:k]; v = sign * (Wd[:, f] @ delta[f]); v = v / (np.linalg.norm(v) + 1e-9) * target
            axis_proj = float(v @ np.load(NEURO / f"geometry_L{layer}.npz")["axis"])
            meta = {"source": "sae_topk_features", "repo": REPO_ID, "layer": layer, "k_features": int(k), "features": [int(x) for x in f], "norm": target,
                    "direction": direction, "axis_proj": axis_proj, "norm_matched_to": "line_s0p5"}
            np.savez(VS / f"vec_nm_{direction}_sae_k{k}.npz", v=v.astype(np.float32), layer=layer, meta=json.dumps(meta))
            print(f"[sae vectors] nm_{direction}_sae_k{k}: |v| {target:.3f}, axis proj {axis_proj:+.3f}, cos with line {float(v @ ref['v'] / (target * np.linalg.norm(ref['v']) + 1e-9)):+.3f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fetch", "gate", "contrast", "path", "vectors"])
    ap.add_argument("--layer", type=int, default=16); ap.add_argument("--n", type=int, default=12000)
    a = ap.parse_args()
    if a.cmd == "fetch": print(sae_path(a.layer))
    elif a.cmd == "gate": cmd_gate(a.layer, a.n)
    elif a.cmd == "contrast": cmd_contrast(a.layer, a.n)
    elif a.cmd == "path": cmd_path(a.layer)
    else: cmd_vectors(a.layer)


if __name__ == "__main__":
    main()
