"""D-64 stage 4: Gemma steering vectors with a readout-calibrated dose (replaces the norm-fraction rule, see D-64 amendment).

The rank-8 map predicts demeaned rates from the standardised residual: phi_hat = ((h - mu) / sd) @ W + b. A residual change v
therefore implies the rate change (v / sd) @ W. The open-loop direction for a target gap g (16 readout targets) is
u = unit(diag(sd) W (W^T W + lam I)^-1 g), the controller's own open-loop direction (common.Steerer); the implied coverage of the gap at norm n is
c(n) = n * ((u / sd) @ W) . g / |g|^2.

Reference dose: Qwen's A2 vector (map_corpus1_dm8_L16_r8, archetype math_A2 minus the math mean) at its norm 6.75 implies a
coverage c_ref; every Gemma vector gets the norm that implies the same coverage under the Gemma map. The construction is
checked by rebuilding Qwen's A2 and comparing it with the stored vec_ol_math_A2.npz (cosine should be ~1).

Gemma vectors written to data/gemma4_12b/vsteer/vec_<name>.npz at the map's layer:
  own_math_A1, own_gpqa_A1, own_planning_A3  archetype minus its domain mean (the own efficient archetypes)
  sig_math, sig_idea                          domain mean profile minus the mean of the four domain profiles (D-52 recipe)
  conv_up, conv_down                          +/- one target SD on the four conversational rates (D-51 recipe)
  rand_<name>                                 Gaussian direction norm-matched to each vector above
Run from the repo root (any MC_MODEL): python -m mode_control.mc.gemma_calib
"""
from __future__ import annotations

import json

import numpy as np

from .common import BEH, MC

QSIG = MC / "data" / "sigctrl"
GSIG = MC / "data" / "gemma4_12b" / "sigctrl"
GVS = MC / "data" / "gemma4_12b" / "vsteer"
REF_NORM = 6.75


def load_map(path):
    z = np.load(path, allow_pickle=True)
    return {"W": z["W"].astype(np.float64), "mu": z["mu"].astype(np.float64), "sd": z["sd"].astype(np.float64), "layer": int(z["layer"]),
            "domain_means": {str(d): m.astype(np.float64) for d, m in zip(z["domain_names"].tolist(), z["domain_means"])},
            "target_sd": z["target_sd"].astype(np.float64), "target_names": [str(t) for t in z["target_names"].tolist()]}


REL_LAM = 1e-3   # the controller's ridge, relative to the mean eigenvalue of W^T W (D-35 recipe)


def direction(m, g):
    """The controller's open-loop direction: G g with G = diag(sd) W (W^T W + lam I)^-1, i.e. the ridge pseudo-inverse step in
    standardised coordinates mapped back to raw units (common.Steerer, mode target)."""
    W = m["W"]; k = W.shape[1]
    lam = REL_LAM * float(np.trace(W.T @ W) / k)
    u = m["sd"] * (W @ np.linalg.solve(W.T @ W + lam * np.eye(k), g))
    return u / np.linalg.norm(u)


def coverage_per_unit_norm(m, u, g):
    return float(((u / m["sd"]) @ m["W"]) @ g / (g @ g))


def main():
    # ---- reference: Qwen A2
    qm = load_map(QSIG / "map_corpus1_dm8_L16_r8.npz")
    qa = json.loads((QSIG / "archetypes.json").read_text())
    g_a2 = np.asarray(qa["math_A2"]["phi"], np.float64)[:16] - qm["domain_means"]["math"]
    u_a2 = direction(qm, g_a2)
    stored = np.load(MC / "data" / "vsteer" / "vec_ol_math_A2.npz", allow_pickle=True)["v"].astype(np.float64)
    cos = float(u_a2 @ stored / (np.linalg.norm(stored) + 1e-9))
    c_ref = REF_NORM * coverage_per_unit_norm(qm, u_a2, g_a2)
    print(f"[calib] Qwen A2 rebuilt: cosine with stored vector {cos:.4f} (stored norm {np.linalg.norm(stored):.3f}); implied coverage at {REF_NORM}: {c_ref:.3f}")

    # ---- Gemma
    gm = load_map(GSIG / "map_corpusG_dm8_L28_r8.npz")
    ga = json.loads((GSIG / "archetypes.json").read_text())
    dm = gm["domain_means"]; grand = np.mean(np.stack(list(dm.values())), axis=0)
    conv_idx = [gm["target_names"].index(n) for n in gm["target_names"] if any(n.startswith(b) for b in BEH[:4]) and not n.endswith("_next4")] or [0, 1, 2, 3]
    conv = np.zeros(16); conv[conv_idx] = gm["target_sd"][conv_idx]
    targets = {
        "own_math_A1": np.asarray(ga["math_A1"]["phi"], np.float64)[:16] - dm["math"],
        "own_gpqa_A1": np.asarray(ga["gpqa_A1"]["phi"], np.float64)[:16] - dm["gpqa"],
        "own_planning_A3": np.asarray(ga["planning_A3"]["phi"], np.float64)[:16] - dm["planning"],
        "sig_math": dm["math"] - grand, "sig_idea": dm["idea"] - grand,
        "conv_up": conv, "conv_down": -conv,
    }
    GVS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    out = {"reference": {"model": "qwen35_9b", "vector": "ol_math_A2", "norm": REF_NORM, "implied_coverage": c_ref, "rebuild_cosine": cos},
           "gemma_layer": gm["layer"], "conv_targets": [gm["target_names"][i] for i in conv_idx], "vectors": {}}
    for name, g in targets.items():
        u = direction(gm, g); cpn = coverage_per_unit_norm(gm, u, g); n = c_ref / cpn
        v = (u * n).astype(np.float32)
        meta = {"source": "gemma_calibrated", "target": name, "layer": gm["layer"], "norm": float(n), "implied_coverage": c_ref,
                "coverage_per_unit_norm": cpn, "gap": dict(zip(gm["target_names"], np.round(g, 4).tolist())), "decision": "D-64"}
        np.savez(GVS / f"vec_{name}.npz", v=v, layer=gm["layer"], meta=json.dumps(meta))
        r = rng.standard_normal(len(u)); r = r / np.linalg.norm(r) * n
        np.savez(GVS / f"vec_rand_{name}.npz", v=r.astype(np.float32), layer=gm["layer"], meta=json.dumps({"source": "random_gaussian", "matched_to": name, "norm": float(n), "layer": gm["layer"]}))
        out["vectors"][name] = {"norm": float(n), "coverage_per_unit_norm": cpn, "gap_norm": float(np.linalg.norm(g)),
                                "gap8": dict(zip([b[:10] for b in BEH], np.round(g[:8], 3).tolist()))}
        print(f"[calib] {name:16s} norm {n:7.2f}  coverage/unit {cpn:.4f}  gap8 {out['vectors'][name]['gap8']}")
    cosines = {}
    names = list(targets)
    V = {k: np.load(GVS / f"vec_{k}.npz")["v"] for k in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cosines[f"{a}|{b}"] = float(V[a] @ V[b] / (np.linalg.norm(V[a]) * np.linalg.norm(V[b])))
    out["cosines"] = cosines
    print("[calib] cosines:", {k: round(v, 2) for k, v in cosines.items()})
    (GSIG / "calibration.json").write_text(json.dumps(out, indent=1))
    print("->", GSIG / "calibration.json")


if __name__ == "__main__":
    main()
