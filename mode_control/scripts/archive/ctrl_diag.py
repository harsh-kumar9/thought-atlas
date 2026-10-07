import numpy as np, polars as pl, json
pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_float("mixed")
e = pl.read_csv("mode_control/data/runs/sigpilot_idea/analysis/sigctrl_eval.csv")
print(e.select(["condition","d_unsteered","d_steered","gap_closure","gc_lo","gc_hi"]).with_columns(pl.col(pl.Float64).round(3)).sort("condition"))
# controller magnitude diagnostic on corpus states (layer 16), Idea domain, target idea_A3 and idea_A4
z = np.load("mode_control/data/sigctrl/map_corpus1_dm8_L16_r8.npz", allow_pickle=True)
W, b, mu, sd = z["W"].astype(np.float64), z["b"].astype(np.float64), z["mu"].astype(np.float64), z["sd"].astype(np.float64)
dm = dict(zip(z["domain_names"].tolist(), z["domain_means"]))
arch = json.load(open("mode_control/data/sigctrl/archetypes.json"))
acts = np.load("mode_control/data/sigctrl/corpus1_idea_a__acts.npz"); li = acts["layers"].tolist().index(16)
idx = np.random.default_rng(0).choice(acts["H"].shape[0], 3000, replace=False)
H = acts["H"][idx, li, :].astype(np.float64)
print("W frobenius", np.linalg.norm(W).round(4), "W col norms", np.linalg.norm(W, axis=0).round(4).tolist()[:8])
for lam in [1e-3, 1e-2, 1e-1]:
    G = (sd[:, None] * W) @ np.linalg.inv(W.T @ W + lam * np.eye(16))
    for name in ["idea_A3", "idea_A4"]:
        phi_star = np.asarray(arch[name]["phi"])[:16] - dm["idea"][:16]
        phi_hat = ((H - mu) / sd) @ W + b
        err = phi_star - phi_hat
        delta = err @ G.T
        hn = np.linalg.norm(H, axis=1); dn = np.linalg.norm(delta, axis=1)
        # what the readout says after the correction (should hit the target when lam is small)
        phi_after = (((H + delta) - mu) / sd) @ W + b
        closed = 1 - np.linalg.norm(phi_star - phi_after, axis=1).mean() / np.linalg.norm(err, axis=1).mean()
        print(f"lam={lam:g} {name}: |delta|/|h| median {np.median(dn/hn):.3f} (p90 {np.percentile(dn/hn,90):.3f}); |h| median {np.median(hn):.1f}; readout gap closed by the correction {closed:.2f}; |err| median {np.median(np.linalg.norm(err,axis=1)):.3f}")
