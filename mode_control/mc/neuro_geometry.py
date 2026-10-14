"""E5 (D-59): is the Math-to-Idea path curved, and does steering along the data path avoid the loops?

geometry  PCA-50 on segment states (sigctrl acts at one layer) sampled from corpus1 math, idea, gpqa, planning (a+b).
          Straight line = mu_math + s (mu_idea - mu_math). Curve = shortest path through the k-NN graph of the sample
          from the state nearest mu_math to the state nearest mu_idea, each node replaced by the mean of its 50
          neighbours (full space), smoothed along the path, parameterised by arc length s in [0, 1] with the two
          centroids as the end points. Void ratio of a point = its 10-NN distance / the median 10-NN distance of the
          data (in PCA-50 space). Also the domain mix of the neighbours along the curve and 2-D coordinates for the
          atlas figure. Writes data/neuro/geometry_L<l>.npz and .json.
vectors   vsteer vectors (layer l) for the steering arms, both directions:
            toward_idea (for Math tasks): line_s0p5, line_pm, line_s1p0, curve_s0p5, curve_n0p5, rand_n0p5, rand_n1p0
            toward_math (for Idea tasks): the same from the Idea end (line_s1p0 is the Math centroid)
          line_pm = the line point whose projection on the domain axis equals curve_n0p5's; curve_n0p5 = the curve
          point whose displacement norm equals line_s0p5's. Random vectors are norm-matched Gaussian directions.
runs      run.json for nm_math (tasks_from sigpilot3_math; p1_budget baseline/ol_random shared) and nm_idea
          (tasks_from tx_idea; its baseline shared), 2 seeds, 32k budget, 8k window, vLLM fixed vectors.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from .common import DATA

NEURO = DATA / "neuro"
SIG = DATA / "sigctrl"
VS = DATA / "vsteer"
DOMS = ["math", "idea", "gpqa", "planning"]


def _states(dom: str, layer: int, n_sample: int, rng):
    Hs, means, n_all = [], [], 0
    for ab in ("a", "b"):
        z = np.load(SIG / f"corpus1_{dom}_{ab}__acts.npz", mmap_mode="r")
        li = z["layers"].tolist().index(layer)
        H = z["H"][:, li, :]
        n = H.shape[0]; n_all += n
        s = np.zeros(H.shape[1], np.float64)
        for i in range(0, n, 4096):
            s += H[i:i + 4096].astype(np.float64).sum(0)
        means.append((s, n))
        idx = np.sort(rng.choice(n, min(n_sample // 2, n), replace=False))
        Hs.append(np.asarray(H[idx]).astype(np.float32))
    mu = sum(s for s, _ in means) / n_all
    return np.concatenate(Hs), mu.astype(np.float32), n_all


def cmd_geometry(layer: int, n_sample: int, k_graph: int, k_smooth: int, seed: int):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    X, dom, mus = [], [], {}
    for d in DOMS:
        H, mu, n = _states(d, layer, n_sample, rng); X.append(H); dom += [d] * len(H); mus[d] = mu
        print(f"[geometry] {d}: {n} states, sampled {len(H)}", flush=True)
    X = np.concatenate(X); dom = np.array(dom)
    pca = PCA(n_components=50, random_state=0).fit(X)
    Z = pca.transform(X).astype(np.float32)
    ev = pca.explained_variance_ratio_
    zmu = {d: pca.transform(mus[d][None])[0] for d in DOMS}
    nn = NearestNeighbors(n_neighbors=k_graph + 1).fit(Z)
    dist, nbr = nn.kneighbors(Z)
    d10_data = dist[:, 10]; d10_med = float(np.median(d10_data))
    # k-NN graph (symmetric), shortest path between the states nearest the two centroids
    rows = np.repeat(np.arange(len(Z)), k_graph); cols = nbr[:, 1:].ravel(); w = dist[:, 1:].ravel()
    G = coo_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(len(Z), len(Z))).tocsr()
    src = int(np.argmin(np.linalg.norm(Z - zmu["math"], axis=1))); dst = int(np.argmin(np.linalg.norm(Z - zmu["idea"], axis=1)))
    D, pred = dijkstra(G, directed=False, indices=src, return_predecessors=True)
    assert np.isfinite(D[dst]), "centroid states are not connected in the k-NN graph; raise --k-graph"
    path = [dst]
    while path[-1] != src:
        path.append(int(pred[path[-1]]))
    path = path[::-1]
    print(f"[geometry] geodesic: {len(path)} nodes, graph length {D[dst]:.2f}; straight distance {np.linalg.norm(zmu['idea'] - zmu['math']):.2f} (PCA-50)")
    # smoothed curve in full space: node -> mean of its k_smooth neighbours; then a moving average along the path
    nn_s = NearestNeighbors(n_neighbors=k_smooth).fit(Z)
    _, nb_s = nn_s.kneighbors(Z[path])
    C_full = np.stack([X[nb_s[i]].mean(0) for i in range(len(path))])
    mix = np.stack([[(dom[nb_s[i]] == d).mean() for d in DOMS] for i in range(len(path))])
    win = 5
    C_sm = np.stack([C_full[max(0, i - win // 2): i + win // 2 + 1].mean(0) for i in range(len(path))])
    C_sm = np.concatenate([mus["math"][None], C_sm, mus["idea"][None]])
    mix = np.concatenate([[[1, 0, 0, 0]], mix, [[0, 1, 0, 0]]])
    seg = np.linalg.norm(np.diff(C_sm, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)]); s = s / s[-1]
    # void ratio along the line and the curve (PCA-50 space), plus the raw path nodes as the in-data reference
    def d10(points_full):
        Zp = pca.transform(points_full); dd, _ = nn.kneighbors(Zp, n_neighbors=10); return dd[:, 9] / d10_med
    s_line = np.round(np.arange(0, 1.501, 0.05), 3)
    line_pts = mus["math"][None] + s_line[:, None] * (mus["idea"] - mus["math"])[None]
    s_curve = np.round(np.arange(0, 1.001, 0.05), 3)
    curve_pts = np.stack([np.array([np.interp(t, s, C_sm[:, j]) for j in range(C_sm.shape[1])]) for t in s_curve])
    void_line, void_curve = d10(line_pts), d10(curve_pts)
    void_nodes = d10(X[path])
    # neighbour domain mix along the line too
    def mix_of(points_full):
        Zp = pca.transform(points_full); _, nb = nn.kneighbors(Zp, n_neighbors=k_smooth)
        return np.stack([[(dom[nb[i]] == d).mean() for d in DOMS] for i in range(len(points_full))])
    mix_line = mix_of(line_pts); mix_curve = mix_of(curve_pts)
    # projection on the domain axis (layer 16 or 19, phase mean32) for the vectors
    dirs = np.load(DATA / "directions" / "directions_w8192_v2w8192.npz", allow_pickle=True)
    axis = dirs[f"prompt_r1|{layer}|mean32"][:, 0].astype(np.float32)
    delta = mus["idea"] - mus["math"]
    gap_axis = float(delta @ axis)
    # 2-D coordinates for the atlas figure: PC1/PC2 of the sample, and the (axis, first orthogonal PC) plane
    sub = rng.choice(len(Z), min(8000, len(Z)), replace=False)
    out = {"layer": layer, "n_sample_per_domain": n_sample, "k_graph": k_graph, "k_smooth": k_smooth, "explained_var_50": float(ev.sum()),
           "explained_var_first5": ev[:5].tolist(), "d10_median": d10_med, "path_len_nodes": len(path), "graph_length": float(D[dst]),
           "straight_pca50": float(np.linalg.norm(zmu["idea"] - zmu["math"])), "straight_full": float(np.linalg.norm(delta)),
           "curve_length_full": float(np.sum(seg)), "gap_on_axis": gap_axis, "cos_delta_axis": float(delta @ axis / np.linalg.norm(delta)),
           "s_line": s_line.tolist(), "void_line": void_line.tolist(), "s_curve": s_curve.tolist(), "void_curve": void_curve.tolist(),
           "void_nodes_mean": float(void_nodes.mean()), "void_nodes_max": float(void_nodes.max()),
           "mix_curve": {d: mix_curve[:, i].tolist() for i, d in enumerate(DOMS)}, "mix_line": {d: mix_line[:, i].tolist() for i, d in enumerate(DOMS)},
           "curve_axis_proj": ((curve_pts - mus["math"]) @ axis).tolist(), "curve_disp_norm": np.linalg.norm(curve_pts - mus["math"], axis=1).tolist(),
           "line_axis_proj": ((line_pts - mus["math"]) @ axis).tolist(), "line_disp_norm": np.linalg.norm(line_pts - mus["math"], axis=1).tolist(),
           "centroid_dist": {f"{a}-{b}": float(np.linalg.norm(mus[a] - mus[b])) for a in DOMS for b in DOMS if a < b}}
    NEURO.mkdir(parents=True, exist_ok=True)
    (NEURO / f"geometry_L{layer}.json").write_text(json.dumps(out, indent=1))
    np.savez(NEURO / f"geometry_L{layer}.npz", mu_math=mus["math"], mu_idea=mus["idea"], mu_gpqa=mus["gpqa"], mu_planning=mus["planning"],
             curve=C_sm, curve_s=s, curve_mix=mix, pcs=pca.components_[:2], pca_mean=pca.mean_, Z2=Z[sub, :2], Z2_dom=dom[sub],
             line_pts=line_pts, s_line=s_line, curve_pts=curve_pts, s_curve=s_curve, path_Z2=Z[path, :2], axis=axis,
             axis_plane=np.stack([(X[sub] - pca.mean_) @ axis, Z[sub, 0]], axis=1))
    print("[geometry] void ratio along the line (s=0..1.5):", np.round(void_line[::2], 2).tolist())
    print("[geometry] void ratio along the curve (s=0..1):", np.round(void_curve[::2], 2).tolist())
    print(f"[geometry] raw path nodes: mean {void_nodes.mean():.2f}, max {void_nodes.max():.2f}; gap on axis {gap_axis:.3f}, |delta| {np.linalg.norm(delta):.3f}, "
          f"cos(delta, axis) {out['cos_delta_axis']:.3f}; curve length {np.sum(seg):.2f} vs straight {np.linalg.norm(delta):.2f}")
    print("[geometry] neighbour mix along the curve (math, idea, gpqa, planning) at s=0,.25,.5,.75,1:", [np.round(mix_curve[i], 2).tolist() for i in (0, 5, 10, 15, 20)])


def _curve_point(g, t: float):
    s, C = g["curve_s"], g["curve"]
    return np.array([np.interp(t, s, C[:, j]) for j in range(C.shape[1])])


def _curve_at_norm(g, origin, target_norm: float, reverse: bool):
    """The first curve point (from `origin`'s end) whose displacement norm reaches target_norm."""
    ts = np.linspace(0, 1, 2001)
    for t in ts:
        tt = 1 - t if reverse else t
        p = _curve_point(g, tt)
        if np.linalg.norm(p - origin) >= target_norm:
            return p, float(t)
    return _curve_point(g, 0.0 if reverse else 1.0), 1.0


def cmd_vectors(layer: int, seed: int):
    VS.mkdir(parents=True, exist_ok=True)
    g = np.load(NEURO / f"geometry_L{layer}.npz", allow_pickle=True)
    mu_m, mu_i, axis = g["mu_math"].astype(np.float32), g["mu_idea"].astype(np.float32), g["axis"].astype(np.float32)
    delta = mu_i - mu_m
    rng = np.random.default_rng(seed)
    spec = {}
    for direction, origin, end, sign in (("toward_idea", mu_m, mu_i, 1.0), ("toward_math", mu_i, mu_m, -1.0)):
        rev = direction == "toward_math"
        d = end - origin
        vecs = {}
        vecs["line_s0p5"] = (0.5 * d, {"s": 0.5})
        vecs["line_s1p0"] = (1.0 * d, {"s": 1.0})
        c05 = _curve_point(g, 0.5) - origin
        vecs["curve_s0p5"] = (c05, {"s": 0.5})
        cn, t_n = _curve_at_norm(g, origin, 0.5 * np.linalg.norm(d), rev)
        vecs["curve_n0p5"] = (cn - origin, {"s": t_n, "norm_matched_to": "line_s0p5"})
        # line point with the same projection on the domain axis as curve_n0p5
        pm = float(((cn - origin) @ axis) / (d @ axis))
        vecs["line_pm"] = (pm * d, {"s": pm, "projection_matched_to": "curve_n0p5"})
        for nm, ref in (("rand_n0p5", "line_s0p5"), ("rand_n1p0", "line_s1p0")):
            u = rng.standard_normal(len(d)).astype(np.float32); u /= np.linalg.norm(u)
            vecs[nm] = (u * np.linalg.norm(vecs[ref][0]), {"norm_matched_to": ref})
        for nm, (v, meta) in vecs.items():
            name = f"nm_{direction}_{nm}"
            full = {"source": "neuro_geometry", "direction": direction, "layer": layer, "norm": float(np.linalg.norm(v)),
                    "axis_proj": float(v @ axis), "axis_proj_rel_gap": float(v @ axis / (delta @ axis)), **meta}
            np.savez(VS / f"vec_{name}.npz", v=v.astype(np.float32), layer=layer, meta=json.dumps(full))
            spec[name] = full
            print(f"[vectors] {name:28s} |v| {full['norm']:6.3f}  axis {full['axis_proj']:+6.3f} ({full['axis_proj_rel_gap']:+.2f} gap)  {meta}")
    (NEURO / f"vectors_L{layer}.json").write_text(json.dumps(spec, indent=1))


def cmd_runs(layer: int):
    arms = ["line_s0p5", "line_pm", "line_s1p0", "curve_s0p5", "curve_n0p5", "rand_n0p5", "rand_n1p0"]
    for run, direction, tasks_from, borrow in (("nm_math", "toward_idea", "sigpilot3_math", "p1_budget:baseline,ol_random"),
                                               ("nm_idea", "toward_math", "tx_idea", "tx_idea:baseline,ol_random")):
        conds = [] if borrow else [{"name": "baseline", "mode": "baseline"}]
        conds += [{"name": a, "mode": "vector", "vector": f"nm_{direction}_{a}", "positions": "all"} for a in arms]
        spec = {"run": run, "engine": "vllm", "tasks_from": tasks_from, "borrow": borrow, "seeds": [1, 2], "max_new_tokens": 32768, "window": 8192,
                "decision": "D-59", "geometry": f"geometry_L{layer}.json", "conditions": conds}
        rd = DATA / "runs" / run; rd.mkdir(parents=True, exist_ok=True)
        (rd / "run.json").write_text(json.dumps(spec, indent=2))
        print(f"[runs] {run}: {len(conds)} conditions -> {rd / 'run.json'}")


def cmd_figure(layer: int):
    """Atlas map (paper style): states by domain in PC1/PC2 with the line and the curve, void ratio along both, neighbour
    domain mix along the curve, and the (domain axis, PC1) plane."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42})
    g = np.load(NEURO / f"geometry_L{layer}.npz", allow_pickle=True); J = json.loads((NEURO / f"geometry_L{layer}.json").read_text())
    col = {"math": "#1b6ca8", "idea": "#c7521d", "gpqa": "#5e8c3a", "planning": "#8a5fb8"}
    Z2, dom = g["Z2"], g["Z2_dom"]; pcs, pm = g["pcs"], g["pca_mean"]
    proj = lambda X: (np.asarray(X) - pm) @ pcs.T
    fig, axs = plt.subplots(2, 2, figsize=(7, 5.2))
    ax = axs[0, 0]
    for d in DOMS:
        m = dom == d; ax.scatter(Z2[m, 0], Z2[m, 1], s=1.2, color=col[d], alpha=0.25, lw=0, label=d, rasterized=True)
    L2, C2 = proj(g["line_pts"]), proj(g["curve_pts"])
    ax.plot(L2[:, 0], L2[:, 1], color="k", lw=1.2, label="straight line (s = 0 to 1.5)"); ax.plot(C2[:, 0], C2[:, 1], color="k", lw=1.2, ls="--", label="data path (s = 0 to 1)")
    for d in ("math", "idea"):
        c = proj(g[f"mu_{d}"][None])[0]; ax.scatter([c[0]], [c[1]], s=28, color=col[d], edgecolor="k", lw=0.6, zorder=5)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.legend(frameon=False, markerscale=6, loc="best"); ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax = axs[0, 1]
    ax.plot(J["s_line"], J["void_line"], color="k", label="straight line"); ax.plot(J["s_curve"], J["void_curve"], color="k", ls="--", label="data path (smoothed points)")
    ax.axhline(J["void_nodes_mean"], color="grey", lw=0.8, ls=":", label="raw path nodes (mean)"); ax.axhline(1.0, color="grey", lw=0.5)
    ax.set_xlabel("s (fraction of the Math-to-Idea path)"); ax.set_ylabel("10-NN distance / data median"); ax.legend(frameon=False); ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax = axs[1, 0]
    mix = np.stack([J["mix_curve"][d] for d in DOMS]); ax.stackplot(J["s_curve"], mix, colors=[col[d] for d in DOMS], labels=DOMS, alpha=0.85)
    ax.set_xlabel("s along the data path"); ax.set_ylabel("domain of the 50 nearest states"); ax.set_ylim(0, 1); ax.legend(frameon=False, loc="center left", fontsize=6); ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax = axs[1, 1]
    A = g["axis_plane"]
    for d in DOMS:
        m = dom == d; ax.scatter(A[m, 0], A[m, 1], s=1.2, color=col[d], alpha=0.25, lw=0, rasterized=True)
    ax.plot((g["line_pts"] - pm) @ g["axis"], L2[:, 0], color="k", lw=1.2); ax.plot((g["curve_pts"] - pm) @ g["axis"], C2[:, 0], color="k", lw=1.2, ls="--")
    ax.set_xlabel(f"projection on the domain axis (layer {layer})"); ax.set_ylabel("PC 1"); ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    out = NEURO / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"fig_neuro_atlas_L{layer}.pdf", bbox_inches="tight"); fig.savefig(out / f"fig_neuro_atlas_L{layer}.png", bbox_inches="tight")
    print("[figure]", out / f"fig_neuro_atlas_L{layer}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["geometry", "vectors", "runs", "figure"])
    ap.add_argument("--layer", type=int, default=16); ap.add_argument("--n-sample", type=int, default=10000)
    ap.add_argument("--k-graph", type=int, default=15); ap.add_argument("--k-smooth", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "geometry": cmd_geometry(a.layer, a.n_sample, a.k_graph, a.k_smooth, a.seed)
    elif a.cmd == "vectors": cmd_vectors(a.layer, a.seed + 11)
    elif a.cmd == "figure": cmd_figure(a.layer)
    else: cmd_runs(a.layer)


if __name__ == "__main__":
    main()
