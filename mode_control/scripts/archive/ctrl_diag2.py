import numpy as np, json
z = np.load("mode_control/data/sigctrl/map_corpus1_dm8_L16_r8.npz", allow_pickle=True)
W, b, mu, sd = z["W"].astype(np.float64), z["b"].astype(np.float64), z["mu"].astype(np.float64), z["sd"].astype(np.float64)
dm = dict(zip(z["domain_names"].tolist(), z["domain_means"])); arch = json.load(open("mode_control/data/sigctrl/archetypes.json"))
for dom, run, names in [("idea", "corpus1_idea_a", ["idea_A3", "idea_A4"]), ("math", "corpus1_math_a", ["math_A4", "math_A2"])]:
    acts = np.load(f"mode_control/data/sigctrl/{run}__acts.npz"); li = acts["layers"].tolist().index(16)
    idx = np.random.default_rng(0).choice(acts["H"].shape[0], 3000, replace=False); H = acts["H"][idx, li, :].astype(np.float64)
    hn = np.median(np.linalg.norm(H, axis=1))
    tr = np.trace(W.T @ W) / 16
    for rel in [1e-2, 1e-3, 1e-4]:
        lam = rel * tr; G = (sd[:, None] * W) @ np.linalg.inv(W.T @ W + lam * np.eye(16))
        for name in names:
            phi_star = np.asarray(arch[name]["phi"])[:16] - dm[dom][:16]
            err = phi_star - (((H - mu) / sd) @ W + b); delta = err @ G.T
            phi_after = (((H + delta) - mu) / sd) @ W + b
            closed = 1 - np.linalg.norm(phi_star - phi_after, axis=1).mean() / np.linalg.norm(err, axis=1).mean()
            print(f"{dom} closed-loop rel_lam={rel:g} {name}: |delta|/|h| median {np.median(np.linalg.norm(delta,axis=1))/hn:.3f}, readout closure {closed:.2f}")
    # open-loop constant direction: v = G (phi_star - 0) i.e. move the readout from the domain mean to the archetype; report its norm
    G = (sd[:, None] * W) @ np.linalg.inv(W.T @ W + 1e-3 * tr * np.eye(16))
    for name in names:
        v = G @ (np.asarray(arch[name]["phi"])[:16] - dm[dom][:16])
        print(f"{dom} open-loop direction {name}: |v| = {np.linalg.norm(v):.2f} (|h| median {hn:.1f}); cosine between the two archetype directions:", end=" ")
    v1 = G @ (np.asarray(arch[names[0]]["phi"])[:16] - dm[dom][:16]); v2 = G @ (np.asarray(arch[names[1]]["phi"])[:16] - dm[dom][:16])
    print(f"{np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2):.2f}")
# compare with the rank-1 prompt direction dose used before (|alpha| = 2 gap at layer 19)
d = np.load("mode_control/data/directions/directions_w8192_v2w8192.npz"); mu19 = d["mu|prompt_r1|19|mean32"]; print("rank-1 dose |alpha| = 2 gap =", 2 * abs(float(mu19[1][0] - mu19[0][0])))
