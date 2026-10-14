"""CPU analysis for the neuro-style readouts (D-55 to D-58). Inputs: data/neuro/{corpus,prompts,taskset}/*.npz from
mc.neuro_extract, components.json, directions.json, the Track B labels of the corpus runs.

  map       E1: per-component domain contrast (signed dom19 write, activity in the readout subspace), conjunction over
            seeds, gap decomposition, three-class map.           -> analysis/map_components.csv, figures/fig_neuro_map
  taskset   E2: time courses in the instruction delay, demand decodability per bin, magnitude vs coupling codes.
                                                                 -> analysis/taskset_*.csv, figures/fig_neuro_taskset_*
  coupling  E3: hub-specific coupling during reasoning by domain and strategy prompt (power1 arms).
                                                                 -> analysis/coupling_*.csv, figures/fig_neuro_coupling
  events    E4: event-locked averages around labelled sentences.  -> analysis/events_*.csv, figures/fig_neuro_events
  all       every analysis whose inputs exist
Figures are paper style: panel letters, axis labels, no claim titles; PDF + PNG, 7 in wide.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from .common import BEH, DATA, JUDGE_MODEL, load_tokenizer

NEURO = DATA / "neuro"; OUT = NEURO / "analysis"; FIG = NEURO / "figures"
DIRS = ["dom16", "dom19"] + [f"sig16_{j}" for j in range(8)] + ["a2_16"]
J16, J19, JA2, SIG = 0, 1, 10, slice(2, 10)
SEEDS = (1, 2, 3, 4)
TARGETS_SHOWN = ["future_verification", "future_backtracking", "future_subgoal", "future_Perspective_Shift"]
DOM_COL = {"math": "#1b6ca8", "idea": "#c7521d", "gpqa": "#5e8c3a", "planning": "#8a5fb8"}


# ---------------------------------------------------------------- loading
def comps() -> list[dict]:
    return json.loads((NEURO / "components.json").read_text())


def dmeta() -> dict:
    return json.loads((NEURO / "directions.json").read_text())


def load_shards(mode: str, prefix: str = "", need_series: bool = False):
    files = sorted((NEURO / mode).glob(f"{prefix}*__s*.npz"))
    if not files:
        return None, [], []
    cols, resid, series = {}, [], []
    for f in files:
        z = np.load(f, allow_pickle=True); n = len(z["trace_id"])
        for k in z.files:
            if k in ("resid", "series", "gen_ids") or k.endswith("_off"):
                continue
            cols.setdefault(k, []).append(z[k])
        off, R = z["resid_off"], z["resid"]
        resid += [R[off[i]:off[i + 1]].astype(np.float32) for i in range(n)]
        if need_series:
            off, S = z["series_off"], z["series"]
            series += [S[off[i]:off[i + 1]] for i in range(n)]
    meta = {k: np.concatenate(v) for k, v in cols.items()}
    return meta, resid, series


def pred_targets(sig_proj: np.ndarray, dm: dict) -> np.ndarray:
    """Readout-predicted behaviour targets [T, 16] from the 8-dim projections of the layer-16 residual."""
    mu = np.asarray(dm["readout_mu_proj"])[SIG]
    return (sig_proj - mu) @ np.asarray(dm["U8_to_targets"]) + np.asarray(dm["readout_b"])


# ---------------------------------------------------------------- statistics
def welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b) + 1e-12))


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)) + 1e-12
    return float((a.mean() - b.mean()) / sp)


D_MIN = 0.5          # conjunction: |Cohen's d| > 0.5 with one sign in every seed (|t| > 2 passes almost everything at n = 200 + 200)
REL_MIN = 0.10       # activity classes also need the contrast to be at least 10% of the mean activity


def paired_t(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)) if len(d) > 2 else 0.0


def boot_ci(x, reps=2000, seed=0, axis=0):
    """Mean and 95% CI over the first axis by bootstrap."""
    x = np.asarray(x, float); rng = np.random.default_rng(seed); n = x.shape[0]
    if n == 0:
        return np.nan, np.nan, np.nan
    bs = np.stack([x[rng.integers(0, n, n)].mean(0) for _ in range(reps)])
    return x.mean(0), np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 and np.isfinite(s).all() else np.nan


def fisher_z(r):
    r = np.clip(np.asarray(r, float), -0.999, 0.999); return np.arctanh(r)


def corr_rows(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson r of x [T] with every column of Y [T, m]."""
    x = x - x.mean(); Y = Y - Y.mean(0)
    return (x @ Y) / (np.linalg.norm(x) * np.linalg.norm(Y, axis=0) + 1e-9)


# ---------------------------------------------------------------- figure helpers
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300,
                         "pdf.fonttype": 42, "lines.linewidth": 1.0})
    return plt


def _save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    print(f"[fig] {FIG / name}.pdf")


def _panel(ax, letter):
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def comp_grid(values: np.ndarray, C: list[dict], max_layer: int = 19, fill=np.nan):
    """[rows, layers] grid: rows 0-31 heads, 32 = MLP, 33 = embedding (layer column 0)."""
    G = np.full((34, max_layer + 1), fill, float)
    for v, c in zip(values, C):
        if c["kind"] == "embed":
            G[33, 0] = v
        elif c["layer"] <= max_layer:
            G[32 if c["kind"] == "mlp" else c["head"], c["layer"]] = v
    return G


# ================================================================ E1: map
def cmd_map():
    meta, resid, _ = load_shards("corpus")
    if meta is None:
        print("[map] no corpus shards"); return
    C = comps(); dm = dmeta(); OUT.mkdir(parents=True, exist_ok=True)
    M, A = meta["comp_mean"], meta["comp_act"]; dom, seed = meta["task_type"], meta["seed"]
    im, ii = dom == "math", dom == "idea"
    print(f"[map] {M.shape[0]} traces ({im.sum()} math, {ii.sum()} idea), {M.shape[1]} components")
    rows = []
    for c in range(len(C)):
        r = {"idx": c, **C[c]}
        for j, nm in ((J19, "dom19"), (J16, "dom16"), (JA2, "a2")):
            x = M[:, c, j]
            r[f"{nm}_math"], r[f"{nm}_idea"] = float(x[im].mean()), float(x[ii].mean()); r[f"{nm}_diff"] = r[f"{nm}_idea"] - r[f"{nm}_math"]
            ds = [cohen_d(x[ii & (seed == s)], x[im & (seed == s)]) for s in SEEDS]
            r[f"{nm}_conj"] = bool(all(abs(d) > D_MIN for d in ds) and len({np.sign(d) for d in ds}) == 1)
            r[f"{nm}_d_min"] = float(min(ds, key=abs)); r[f"{nm}_t_pooled"] = welch_t(x[ii], x[im])
        a = A[:, c]
        r["act_math"], r["act_idea"] = float(a[im].mean()), float(a[ii].mean()); r["act_diff"] = r["act_idea"] - r["act_math"]
        ds = [cohen_d(a[ii & (seed == s)], a[im & (seed == s)]) for s in SEEDS]
        rel = abs(r["act_diff"]) / (0.5 * (r["act_math"] + r["act_idea"]) + 1e-9)
        r["act_conj"] = bool(all(abs(d) > D_MIN for d in ds) and len({np.sign(d) for d in ds}) == 1 and rel > REL_MIN)
        r["act_d_min"] = float(min(ds, key=abs)); r["act_rel_diff"] = float(rel)
        rows.append(r)
    df = pl.DataFrame(rows)
    # gap decomposition on dom19 (components in layers <= 19 only; the embedding counts)
    gap_resid = float(np.mean([r[:, 1, J19].mean() for r, d in zip(resid, dom) if d == "idea"]) - np.mean([r[:, 1, J19].mean() for r, d in zip(resid, dom) if d == "math"]))
    valid19 = (df["layer"] <= 19)
    gap_sum = float(df.filter(valid19)["dom19_diff"].sum())
    df = df.with_columns((pl.col("dom19_diff") / gap_sum).alias("gap_share"))
    order = np.argsort(-np.abs(df["dom19_diff"].to_numpy()))
    cum = np.cumsum(df["dom19_diff"].to_numpy()[order]) / gap_sum
    k10 = max(1, int(0.1 * len(C)))
    # classes on activity: Idea > Math, Math > Idea (conjunction), active in both (no conjunction, both activities in the top quartile)
    q = np.quantile(np.concatenate([df["act_math"].to_numpy(), df["act_idea"].to_numpy()]), 0.75)
    cls = np.where(df["act_conj"].to_numpy() & (df["act_diff"].to_numpy() > 0), "idea>math",
                   np.where(df["act_conj"].to_numpy() & (df["act_diff"].to_numpy() < 0), "math>idea",
                            np.where((~df["act_conj"].to_numpy()) & (df["act_math"].to_numpy() >= q) & (df["act_idea"].to_numpy() >= q), "both", "none")))
    df = df.with_columns(pl.Series("act_class", cls)).sort("idx")
    df.write_csv(OUT / "map_components.csv")
    summary = {"n_traces": int(M.shape[0]), "gap_resid_L19": gap_resid, "gap_sum_components": gap_sum, "conjunction_rule": f"|d| > {D_MIN} every seed; activity also rel diff > {REL_MIN}",
               "share_by_kind": df.filter(valid19).group_by("kind").agg((pl.col("dom19_diff").sum() / gap_sum).alias("share"), pl.len().alias("n")).to_dicts(),
               "share_top10pct_abs": float(np.abs(df["dom19_diff"].to_numpy()[order][:k10]).sum() / np.abs(df["dom19_diff"].to_numpy()).sum()),
               "signed_share_top10pct": float(cum[k10 - 1]), "n_conj_dom19": int(df["dom19_conj"].sum()), "n_conj_act": int(df["act_conj"].sum()),
               "classes": {k: int((cls == k).sum()) for k in ("idea>math", "math>idea", "both", "none")},
               "layer_profile": df.filter(valid19).group_by("layer").agg(pl.col("dom19_diff").sum()).sort("layer").to_dict(as_series=False),
               "top_carriers": df.sort("dom19_diff", descending=True).head(12).select(["layer", "kind", "head", "dom19_diff", "dom19_conj", "gap_share"]).to_dicts(),
               "top_counter": df.sort("dom19_diff").head(8).select(["layer", "kind", "head", "dom19_diff", "dom19_conj", "gap_share"]).to_dicts(),
               "hub_candidates": df.filter(pl.col("act_class") == "both").sort(pl.col("act_math") + pl.col("act_idea"), descending=True).head(12)
               .select(["layer", "kind", "head", "act_math", "act_idea", "dom19_diff"]).to_dicts()}
    (OUT / "map_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[map] gap on the L19 axis: residual {gap_resid:+.3f}, sum of components {gap_sum:+.3f}; conjunction (dom19) {summary['n_conj_dom19']}; "
          f"top 10% of components carry {summary['share_top10pct_abs']:.2f} of |contrast|, signed {summary['signed_share_top10pct']:.2f}; classes {summary['classes']}")
    for t in summary["top_carriers"][:8]:
        print(f"[map]   L{t['layer']:2d} {t['kind']:5s} {t['head']:3d}  diff {t['dom19_diff']:+.3f}  share {t['gap_share']:.3f}  conj {t['dom19_conj']}")
    # figure
    plt = _mpl()
    fig, axs = plt.subplots(2, 2, figsize=(7, 5.2), gridspec_kw={"height_ratios": [1.4, 1]})
    cmap_cls = {"idea>math": "#2a9d4b", "math>idea": "#c4287a", "both": "#e6b422", "none": "#e8e8e8"}
    G = np.zeros((34, 20, 3)); code = np.full((34, 20), -1)
    for i, c in enumerate(C):
        if c["layer"] > 19 and c["kind"] != "embed": continue
        r = 33 if c["kind"] == "embed" else (32 if c["kind"] == "mlp" else c["head"]); l = 0 if c["kind"] == "embed" else c["layer"]
        code[r, l] = ["none", "math>idea", "idea>math", "both"].index(cls[i])
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    ax = axs[0, 0]; ax.imshow(np.ma.masked_less(code, 0), cmap=ListedColormap([cmap_cls["none"], cmap_cls["math>idea"], cmap_cls["idea>math"], cmap_cls["both"]]),
                             vmin=-0.5, vmax=3.5, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("layer"); ax.set_ylabel("head (32 = MLP, 33 = embedding)"); ax.set_title("activity in the readout subspace"); _panel(ax, "a")
    for lab, col in (("Math > Idea", cmap_cls["math>idea"]), ("Idea > Math", cmap_cls["idea>math"]), ("both", cmap_cls["both"])):
        ax.plot([], [], "s", color=col, label=lab)
    ax.legend(loc="upper left", frameon=False, ncol=3, bbox_to_anchor=(0, -0.18))
    Gd = comp_grid(df["dom19_diff"].to_numpy(), C); mask = comp_grid(df["dom19_conj"].to_numpy().astype(float), C, fill=0.0) > 0
    Gshow = np.where(mask, Gd, np.nan); v = np.nanmax(np.abs(Gd)) or 1.0
    ax = axs[0, 1]; im_ = ax.imshow(np.ma.masked_invalid(Gshow), cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), aspect="auto", origin="lower", interpolation="nearest")
    ax.imshow(np.ma.masked_invalid(np.where(~mask, Gd, np.nan)), cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), alpha=0.25, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("layer"); ax.set_ylabel("head"); ax.set_title("Idea minus Math write on the domain axis (layer 19)"); _panel(ax, "b")
    fig.colorbar(im_, ax=ax, fraction=0.04, pad=0.02, label="contrast (axis units)")
    ax = axs[1, 0]; ax.plot(np.arange(1, len(cum) + 1), cum, color="k"); ax.axhline(1, color="grey", lw=0.5, ls=":"); ax.axvline(k10, color="grey", lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("components, ranked by |contrast|"); ax.set_ylabel("cumulative share of the gap"); _panel(ax, "c")
    lp = df.filter(valid19).group_by(["layer", "kind"]).agg(pl.col("dom19_diff").sum()).sort("layer")
    ax = axs[1, 1]
    for kind, col in (("lin", "#1b6ca8"), ("attn", "#5e8c3a"), ("mlp", "#c7521d"), ("embed", "grey")):
        g = lp.filter(pl.col("kind") == kind)
        ax.bar(g["layer"].to_numpy() + {"lin": -0.2, "attn": -0.2, "mlp": 0.2, "embed": 0}[kind], g["dom19_diff"].to_numpy(), width=0.4, color=col, label={"lin": "linear-attention heads", "attn": "full-attention heads", "mlp": "MLP", "embed": "embedding"}[kind])
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("layer"); ax.set_ylabel("summed contrast"); ax.legend(frameon=False); _panel(ax, "d")
    fig.tight_layout(); _save(fig, "fig_neuro_map"); plt.close(fig)


# ================================================================ E2: task set
def _bins(sig: np.ndarray, start: int, stop: int, width: int, n_bins: int):
    out = np.full((n_bins, sig.shape[1]), np.nan)
    for b in range(n_bins):
        a, z = start + b * width, min(start + (b + 1) * width, stop)
        if a >= stop: break
        out[b] = sig[a:z].mean(0)
    return out


def cmd_taskset(tag: str = ""):
    """tag '' = the fixed-filler run (data/neuro/taskset: deterministic delay, no item variance in the pre-task span);
    tag 'varied' = one filler per item (data/neuro/taskset_varied), the run used for the pre-task statistics."""
    global OUT, FIG
    OUT = NEURO / ("analysis" if not tag else f"analysis_taskset_{tag}"); FIG = NEURO / ("figures" if not tag else f"figures_taskset_{tag}")
    meta, resid, series = load_shards("taskset" + (f"_{tag}" if tag else ""), need_series=True)
    if meta is None:
        print("[taskset] no shards"); return
    C = comps(); dm = dmeta(); OUT.mkdir(parents=True, exist_ok=True)
    n = len(meta["trace_id"]); tn = dm["target_names"]
    sig_names = ["dom19", "dom16", "a2"] + TARGETS_SHOWN
    sys0 = meta["span_system"][:, 0]; task0 = meta["span_task"][:, 0]; task1 = meta["span_task"][:, 1]; np_ = meta["n_prompt"]; ng = meta["n_gen"]
    # per-prompt signals [T, 7]
    S = []
    for r in resid:
        pt = pred_targets(r[:, 0, SIG], dm)
        S.append(np.stack([r[:, 1, J19], r[:, 0, J16], r[:, 0, JA2]] + [pt[:, tn.index(t)] for t in TARGETS_SHOWN], axis=1))
    W = 8; NB = {"instruction": 44, "task": 32, "generation": 24}
    B = {k: np.stack([_bins(S[i], {"instruction": sys0[i], "task": task0[i], "generation": np_[i]}[k],
                             {"instruction": task0[i], "task": task1[i], "generation": np_[i] + ng[i]}[k], W, nb) for i in range(n)]) for k, nb in NB.items()}
    keys = ["topic", "demand", "op", "delay"]
    groups = pl.DataFrame({k: meta[k] for k in keys}).with_row_index("i")
    rows, arows = [], []
    for g, idx in groups.group_by(keys, maintain_order=True):
        ii = idx["i"].to_numpy(); cond = dict(zip(keys, g))
        for align, nb in NB.items():
            X = B[align][ii]                                                     # [m, nb, 7]
            for b in range(nb):
                xb = X[:, b, :]; ok = np.isfinite(xb[:, 0])
                if ok.sum() < 5: continue
                m, lo, hi = boot_ci(xb[ok], reps=500)
                for s, nm in enumerate(sig_names):
                    rows.append({**cond, "align": align, "bin": b, "rel_token": b * W, "signal": nm, "n": int(ok.sum()), "mean": float(m[s]), "lo": float(lo[s]), "hi": float(hi[s])})
    pl.DataFrame(rows).write_csv(OUT / "taskset_timecourse.csv")
    # demand decodability per bin (within topic x op x delay): AUROC on dom19; cross-validated logistic on the 8-dim readout projection
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    Bsig = {k: np.stack([_bins(resid[i][:, 0, SIG], {"instruction": sys0[i], "task": task0[i], "generation": np_[i]}[k],
                               {"instruction": task0[i], "task": task1[i], "generation": np_[i] + ng[i]}[k], W, nb) for i in range(n)]) for k, nb in NB.items()}
    for g, idx in groups.group_by(["topic", "op", "delay"], maintain_order=True):
        ii = idx["i"].to_numpy(); y = (meta["demand"][ii] == "generate").astype(int)
        for align, nb in NB.items():
            for b in range(nb):
                x = B[align][ii, b, 0]; ok = np.isfinite(x)
                if ok.sum() < 20 or y[ok].mean() in (0, 1): continue
                a1 = auroc(y[ok], x[ok])
                Z = Bsig[align][ii, b, :][ok]
                try:
                    p = cross_val_predict(LogisticRegression(C=1.0, max_iter=500), (Z - Z.mean(0)) / (Z.std(0) + 1e-6), y[ok], cv=5, method="predict_proba")[:, 1]
                    a8 = auroc(y[ok], p)
                except Exception:
                    a8 = np.nan
                arows.append({"topic": g[0], "op": g[1], "delay": int(g[2]), "align": align, "bin": b, "rel_token": b * W, "n": int(ok.sum()), "auroc_dom19": a1, "auroc_readout8": a8})
    pl.DataFrame(arows).write_csv(OUT / "taskset_auroc.csv")
    # component codes in the pre-task span: demand contrast and operation contrast, paired over items
    Mp, Ap = meta["comp_mean_pre"][:, :, J19], meta["comp_act_pre"]
    item = meta["item"]; dem = meta["demand"]; op = meta["op"]; dl = meta["delay"]; topic = meta["topic"]
    crow = []
    for delay in sorted(set(dl.tolist())):
        sel = dl == delay
        # demand: per item, mean over ops of (generate - solve); operation: per item x demand, exploratory - checking
        items = sorted(set(item[sel].tolist()))
        d_dem = np.full((len(items), Mp.shape[1]), np.nan); d_op = np.full((len(items) * 2, Mp.shape[1]), np.nan)
        for k, it in enumerate(items):
            g_ = sel & (item == it)
            a_, b_ = Mp[g_ & (dem == "generate")], Mp[g_ & (dem == "solve")]
            if len(a_) and len(b_): d_dem[k] = a_.mean(0) - b_.mean(0)
            for q, d_ in enumerate(("solve", "generate")):
                e_, c_ = Mp[g_ & (dem == d_) & (op == "exploratory")], Mp[g_ & (dem == d_) & (op == "checking")]
                if len(e_) and len(c_): d_op[2 * k + q] = e_.mean(0) - c_.mean(0)
        for c in range(Mp.shape[1]):
            crow.append({"delay": int(delay), "idx": c, **C[c], "act_pre": float(Ap[sel, c].mean()), "demand_diff": float(np.nanmean(d_dem[:, c])), "t_demand": paired_t(d_dem[:, c]),
                         "op_diff": float(np.nanmean(d_op[:, c])), "t_op": paired_t(d_op[:, c])})
    cdf = pl.DataFrame(crow); cdf.write_csv(OUT / "taskset_components.csv")
    # hub and specific sets from the longest delay
    # Sets by effect size with a consistency floor (the model is deterministic, so t alone favours tiny but hyper-consistent
    # shifts): specific = largest |demand shift| each way among |t| > 4; hub = most active among |demand shift| below the median.
    c256 = cdf.filter(pl.col("delay") == max(DELAYS_PRESENT := sorted(set(dl.tolist())))).filter(pl.col("kind") != "embed")
    med = float(c256["demand_diff"].abs().median())
    hub = c256.filter(pl.col("demand_diff").abs() < med).sort("act_pre", descending=True).head(8)["idx"].to_list()
    cons = c256.filter((pl.col("t_demand").abs() > 4) & ~pl.col("idx").is_in(hub))
    spec_idea = cons.sort("demand_diff", descending=True).head(8)["idx"].to_list(); spec_math = cons.sort("demand_diff").head(8)["idx"].to_list()
    sets = {"hub": hub, "spec_idea": spec_idea, "spec_math": spec_math, "delay_used": int(max(DELAYS_PRESENT)),
            "hub_desc": [C[i] for i in hub], "spec_idea_desc": [C[i] for i in spec_idea], "spec_math_desc": [C[i] for i in spec_math],
            "n_demand_coded": int((c256["t_demand"].abs() > 2).sum()), "n_op_coded": int((c256["t_op"].abs() > 2).sum()),
            "n_both_coded": int(((c256["t_demand"].abs() > 2) & (c256["t_op"].abs() > 2)).sum())}
    (OUT / "taskset_sets.json").write_text(json.dumps(sets, indent=1))
    print(f"[taskset] pre-task span, delay {sets['delay_used']}: {sets['n_demand_coded']} components code demand (|t|>2), {sets['n_op_coded']} code operation, {sets['n_both_coded']} both")
    # coupling in the pre-task span (longest delay), per prompt: hub signal = mean over the hub set
    crows, wrows = [], []
    long = np.where(dl == sets["delay_used"])[0]
    def _coup(ser, a, b, hub_set, spec_set):
        X = ser[a:b].astype(np.float32); h = X[:, hub_set].mean(1); Y = X[:, spec_set]
        r = corr_rows(h, Y); slope = ((Y - Y.mean(0)) * (h - h.mean())[:, None]).sum(0) / (((h - h.mean()) ** 2).sum() + 1e-9)
        return float(fisher_z(r).mean()), float(slope.mean())
    # coupling over the filler tokens only (the system message is identical across items within a condition and would add
    # a shared time course); instruction-aligned windows below still cover the whole pre-task span
    fil0 = meta["span_filler"][:, 0]; fil1 = meta["span_filler"][:, 1]
    by_cond = {}
    for i in long:
        a0, a1 = (fil0[i], fil1[i]) if fil0[i] >= 0 else (sys0[i], task0[i])
        z_i, s_i = _coup(series[i], a0, a1, hub, spec_idea); z_m, s_m = _coup(series[i], a0, a1, hub, spec_math)
        by_cond.setdefault((topic[i], dem[i], op[i]), []).append(i)
        crows.append({"i": int(i), "item": item[i], "topic": topic[i], "demand": dem[i], "op": op[i], "z_idea": z_i, "slope_idea": s_i, "z_math": z_m, "slope_math": s_m})
    # item-shuffled null: hub series of prompt i with the specific series of the next prompt in the same condition
    for key, ii in by_cond.items():
        for a_, b_ in zip(ii, ii[1:] + ii[:1]):
            if a_ == b_: continue
            sa, sb = (fil0[a_] if fil0[a_] >= 0 else sys0[a_]), (fil0[b_] if fil0[b_] >= 0 else sys0[b_])
            ea, eb = (fil1[a_] if fil0[a_] >= 0 else task0[a_]), (fil1[b_] if fil0[b_] >= 0 else task0[b_])
            T = min(ea - sa, eb - sb)
            h = series[a_][sa:sa + T][:, hub].astype(np.float32).mean(1)
            for nm, ss in (("idea", spec_idea), ("math", spec_math)):
                Y = series[b_][sb:sb + T][:, ss].astype(np.float32)
                crows.append({"i": int(a_), "item": item[a_], "topic": key[0], "demand": key[1], "op": "null_shuffle", f"z_{nm}": float(fisher_z(corr_rows(h, Y)).mean())})
    cdf2 = pl.DataFrame(crows, infer_schema_length=None); cdf2.write_csv(OUT / "taskset_coupling.csv")
    summ = cdf2.group_by(["topic", "demand", "op"]).agg(pl.col("z_idea").mean(), pl.col("z_math").mean(), pl.len().alias("n")).sort(["topic", "demand", "op"])
    print("[taskset] coupling (Fisher z, hub x specific) by condition:"); print(summ)
    # operation contrast on coupling, paired over items, with a permutation of the op label within item
    prow = []
    rng = np.random.default_rng(0)
    for (tp, de), g in cdf2.filter(pl.col("op").is_in(["checking", "exploratory"])).group_by(["topic", "demand"]):
        piv = g.pivot(on="op", index="item", values=["z_idea", "z_math"]).drop_nulls()
        for nm in ("z_idea", "z_math"):
            d = (piv[f"{nm}_exploratory"] - piv[f"{nm}_checking"]).to_numpy()
            perm = np.array([(d * rng.choice([-1, 1], len(d))).mean() for _ in range(4000)])
            prow.append({"topic": tp, "demand": de, "measure": nm, "n_items": len(d), "explore_minus_check": float(d.mean()), "t": paired_t(d), "p_perm": float((np.abs(perm) >= abs(d.mean())).mean())})
    pl.DataFrame(prow).write_csv(OUT / "taskset_coupling_op.csv"); print(pl.DataFrame(prow))
    # sliding-window coupling along the sequence (window 64, step 16): instruction-aligned, then generation-aligned
    for i in long:
        for align, a0, a1 in (("instruction", sys0[i], task0[i]), ("generation", np_[i], np_[i] + ng[i])):
            for w0 in range(a0, a1 - 64 + 1, 16):
                z_i, _ = _coup(series[i], w0, w0 + 64, hub, spec_idea); z_m, _ = _coup(series[i], w0, w0 + 64, hub, spec_math)
                wrows.append({"topic": topic[i], "demand": dem[i], "op": op[i], "align": align, "rel_token": int(w0 - a0), "z_idea": z_i, "z_math": z_m})
    wdf = pl.DataFrame(wrows); wdf.write_csv(OUT / "taskset_coupling_windows.csv")
    _fig_taskset(pl.DataFrame(rows), pl.DataFrame(arows), cdf, cdf2, wdf, sets)


def _fig_taskset(tc: pl.DataFrame, au: pl.DataFrame, cdf: pl.DataFrame, cp: pl.DataFrame, wdf: pl.DataFrame, sets: dict):
    plt = _mpl()
    delays = sorted(tc["delay"].unique().to_list())
    fig, axs = plt.subplots(2, len(delays), figsize=(7, 4.2), sharey="row")
    ls = {"none": "-", "checking": "--", "exploratory": ":"}
    for k, d in enumerate(delays):
        ax = axs[0, k]
        for dem_, col in (("solve", DOM_COL["math"]), ("generate", DOM_COL["idea"])):
            for op_ in ("none", "checking", "exploratory"):
                g = tc.filter((pl.col("delay") == d) & (pl.col("demand") == dem_) & (pl.col("op") == op_) & (pl.col("signal") == "dom19") & (pl.col("topic") == "math"))
                gi = g.filter(pl.col("align") == "instruction").sort("bin"); gg = g.filter(pl.col("align") == "generation").sort("bin")
                if gi.height:
                    ax.plot(gi["rel_token"], gi["mean"], color=col, ls=ls[op_], label=f"{dem_}, {op_}" if k == 0 else None)
                    ax.fill_between(gi["rel_token"], gi["lo"], gi["hi"], color=col, alpha=0.12, lw=0)
                if gg.height:
                    x0 = gi["rel_token"].max() + 40 if gi.height else 0
                    ax.plot(x0 + gg["rel_token"], gg["mean"], color=col, ls=ls[op_]); ax.fill_between(x0 + gg["rel_token"], gg["lo"], gg["hi"], color=col, alpha=0.12, lw=0)
        ax.set_title(f"delay {d} tokens")
        if k == 0: ax.set_ylabel("domain-axis projection (layer 19)"); ax.legend(frameon=False, ncol=2, fontsize=5.5, loc="lower left"); _panel(ax, "a")
        ax2 = axs[1, k]
        for op_ in ("none", "checking", "exploratory"):
            g = au.filter((pl.col("delay") == d) & (pl.col("op") == op_) & (pl.col("topic") == "math")).sort(["align", "bin"])
            gi = g.filter(pl.col("align") == "instruction"); gg = g.filter(pl.col("align") == "generation")
            if gi.height: ax2.plot(gi["rel_token"], gi["auroc_dom19"], color="k", ls=ls[op_], label=op_ if k == 0 else None)
            if gg.height:
                x0 = gi["rel_token"].max() + 40 if gi.height else 0
                ax2.plot(x0 + gg["rel_token"], gg["auroc_dom19"], color="k", ls=ls[op_]); ax2.plot(x0 + gg["rel_token"], gg["auroc_readout8"], color="grey", ls=ls[op_])
            if gi.height: ax2.plot(gi["rel_token"], gi["auroc_readout8"], color="grey", ls=ls[op_])
        ax2.axhline(0.5, color="grey", lw=0.5, ls=":"); ax2.set_ylim(0.3, 1.02)
        if k == 0: ax2.set_ylabel("demand AUROC (black: axis, grey: 8-dim readout)"); ax2.legend(frameon=False, fontsize=5.5, loc="lower left"); _panel(ax2, "b")
    fig.supxlabel("tokens from instruction onset (pre-task span), then from generation onset (task epoch)", fontsize=6.5)
    fig.tight_layout(); _save(fig, "fig_neuro_taskset_timecourse"); plt.close(fig)
    # magnitude vs coupling
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.4))
    c = cdf.filter(pl.col("delay") == sets["delay_used"])
    ax = axs[0]; ax.scatter(c["t_demand"], c["t_op"], s=4, color="grey", alpha=0.6)
    for nm, col in (("hub", "#e6b422"), ("spec_idea", DOM_COL["idea"]), ("spec_math", DOM_COL["math"])):
        s = c.filter(pl.col("idx").is_in(sets[nm])); ax.scatter(s["t_demand"], s["t_op"], s=10, color=col, label=nm.replace("spec_", "specific: "))
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4); ax.set_xlabel("t, demand (generate - solve)"); ax.set_ylabel("t, operation (explore - check)"); ax.legend(frameon=False); _panel(ax, "a")
    ax = axs[1]
    summ = cp.group_by(["demand", "op"]).agg(pl.col("z_idea").mean(), (pl.col("z_idea").std() / pl.len().sqrt()).alias("z_idea_std"),
                                             pl.col("z_math").mean(), (pl.col("z_math").std() / pl.len().sqrt()).alias("z_math_std")).sort(["demand", "op"])
    xs = np.arange(summ.height)
    ax.bar(xs - 0.2, summ["z_idea"], 0.4, yerr=summ["z_idea_std"], color=DOM_COL["idea"], label="hub x Idea-specific"); ax.bar(xs + 0.2, summ["z_math"], 0.4, yerr=summ["z_math_std"], color=DOM_COL["math"], label="hub x Math-specific")
    ax.set_xticks(xs); ax.set_xticklabels([a + "\n" + b for a, b in zip(summ["demand"], summ["op"])], fontsize=5.5); ax.set_ylabel("coupling (Fisher z)"); ax.legend(frameon=False); _panel(ax, "b")
    ax = axs[2]
    for (dem_, op_), g in wdf.filter(pl.col("topic") == "math").group_by(["demand", "op"], maintain_order=True):
        if op_ == "none": continue
        gi = g.filter(pl.col("align") == "instruction").group_by("rel_token").agg(pl.col("z_idea").mean()).sort("rel_token")
        ax.plot(gi["rel_token"], gi["z_idea"], color=DOM_COL["idea" if dem_ == "generate" else "math"], ls=ls[op_], label=f"{dem_}, {op_}")
    ax.set_xlabel("window start, tokens from instruction onset"); ax.set_ylabel("hub x Idea-specific z (64-token windows)"); ax.legend(frameon=False, fontsize=5.5); _panel(ax, "c")
    fig.tight_layout(); _save(fig, "fig_neuro_taskset_codes"); plt.close(fig)


# ================================================================ E3: coupling during reasoning
def _sets_from_map() -> dict:
    """Disjoint sets. Hub = the most active components (readout subspace, both domains) that carry no signed domain
    contrast (not dom19_conj); specific = the strongest signed carriers each way, excluding the hub."""
    df = pl.read_csv(OUT / "map_components.csv").filter(pl.col("kind") != "embed")
    nohub = df.filter(~pl.col("dom19_conj"))
    hub = nohub.filter(pl.col("act_class") == "both").sort(pl.col("act_math") + pl.col("act_idea"), descending=True).head(8)["idx"].to_list()
    if len(hub) < 4:
        hub = nohub.sort(pl.col("act_math") + pl.col("act_idea"), descending=True).head(8)["idx"].to_list()
    rest = df.filter(pl.col("dom19_conj") & ~pl.col("idx").is_in(hub))
    spec_idea = rest.sort("dom19_diff", descending=True).head(8)["idx"].to_list()
    spec_math = rest.sort("dom19_diff").head(8)["idx"].to_list()
    assert not (set(hub) & set(spec_idea)) and not (set(hub) & set(spec_math)) and not (set(spec_idea) & set(spec_math))
    return {"hub": hub, "spec_idea": spec_idea, "spec_math": spec_math}


def cmd_coupling():
    if not (OUT / "map_components.csv").exists():
        cmd_map()
    sets = _sets_from_map(); C = comps()
    (OUT / "coupling_sets.json").write_text(json.dumps({**sets, **{k + "_desc": [C[i] for i in sets[k]] for k in sets}}, indent=1))
    rows, wrows = [], []
    for mode, prefix in (("prompts", ""), ("corpus", "")):
        meta, _, series = load_shards(mode, prefix, need_series=True)
        if meta is None: continue
        has = np.array([len(s) > 0 for s in series])
        for i in np.where(has)[0]:
            X = series[i][:1536].astype(np.float32); h = X[:, sets["hub"]].mean(1)
            rec = {"mode": mode, "trace_id": meta["trace_id"][i], "instance_id": meta["instance_id"][i], "task_type": meta["task_type"][i], "condition": meta["condition"][i], "seed": int(meta["seed"][i])}
            for nm in ("idea", "math"):
                Y = X[:, sets[f"spec_{nm}"]]; r = corr_rows(h, Y)
                rec[f"z_{nm}"] = float(fisher_z(r).mean()); rec[f"slope_{nm}"] = float((((Y - Y.mean(0)) * (h - h.mean())[:, None]).sum(0) / (((h - h.mean()) ** 2).sum() + 1e-9)).mean())
            rows.append(rec)
            for w0 in range(0, len(X) - 64 + 1, 32):
                hw = h[w0:w0 + 64]
                wrows.append({"mode": mode, "task_type": meta["task_type"][i], "condition": meta["condition"][i], "rel_token": w0,
                              "z_idea": float(fisher_z(corr_rows(hw, X[w0:w0 + 64][:, sets["spec_idea"]])).mean()), "z_math": float(fisher_z(corr_rows(hw, X[w0:w0 + 64][:, sets["spec_math"]])).mean())})
        # shuffled-pair null within condition x domain
        idx = np.where(has)[0]
        for key in sorted({(meta["task_type"][i], meta["condition"][i]) for i in idx}):
            ii = [i for i in idx if (meta["task_type"][i], meta["condition"][i]) == key]
            for a_, b_ in zip(ii, ii[1:] + ii[:1]):
                if a_ == b_: continue
                T = min(len(series[a_]), len(series[b_]), 1536)
                h = series[a_][:T][:, sets["hub"]].astype(np.float32).mean(1)
                rec = {"mode": mode, "trace_id": meta["trace_id"][a_], "instance_id": meta["instance_id"][a_], "task_type": key[0], "condition": key[1] + "__null", "seed": int(meta["seed"][a_])}
                for nm in ("idea", "math"):
                    rec[f"z_{nm}"] = float(fisher_z(corr_rows(h, series[b_][:T][:, sets[f"spec_{nm}"]].astype(np.float32))).mean())
                rows.append(rec)
    if not rows:
        print("[coupling] no series found"); return
    df = pl.DataFrame(rows, infer_schema_length=None); df.write_csv(OUT / "coupling_traces.csv")
    summ = df.group_by(["mode", "task_type", "condition"]).agg(pl.len().alias("n"), pl.col("z_idea").mean(), pl.col("z_math").mean(), pl.col("slope_idea").mean(), pl.col("slope_math").mean()).sort(["mode", "task_type", "condition"])
    summ.write_csv(OUT / "coupling_summary.csv"); pl.Config.set_tbl_rows(60); print(summ)
    # paired contrasts vs baseline within power1 (same tasks, seed 1)
    prow = []
    p1 = df.filter((pl.col("mode") == "prompts") & ~pl.col("condition").str.ends_with("__null"))
    base = p1.filter(pl.col("condition") == "baseline")
    for (dom, cond), g in p1.filter(pl.col("condition") != "baseline").group_by(["task_type", "condition"]):
        j = g.join(base.filter(pl.col("task_type") == dom).select(["instance_id", "z_idea", "z_math"]), on="instance_id", suffix="_b")
        for nm in ("z_idea", "z_math"):
            d = (j[nm] - j[f"{nm}_b"]).to_numpy(); m, lo, hi = boot_ci(d)
            prow.append({"task_type": dom, "condition": cond, "measure": nm, "n": len(d), "shift_vs_baseline": float(m), "lo": float(lo), "hi": float(hi), "t": paired_t(d)})
    if prow:
        pl.DataFrame(prow).write_csv(OUT / "coupling_paired.csv"); print(pl.DataFrame(prow))
    wdf = pl.DataFrame(wrows); wdf.write_csv(OUT / "coupling_windows.csv")
    # magnitude code for the same sets in the same traces: paired shift of each component's mean dom19 write vs baseline
    # (Cohen's d over tasks), next to the coupling shift, so the two codes can be compared on one scale
    metaP, _, _ = load_shards("prompts", "")
    if metaP is not None:
        mrows = []
        Mp = metaP["comp_mean"][:, :, J19]; cond = metaP["condition"]; inst = metaP["instance_id"]; dom = metaP["task_type"]
        for dname in ("math", "idea"):
            for c in ("prompt_checking", "prompt_exploratory"):
                ib = {i: k for k, i in enumerate(inst) if cond[k] == "baseline" and dom[k] == dname}
                pairs = [(k, ib[i]) for k, i in enumerate(inst) if cond[k] == c and dom[k] == dname and i in ib]
                if len(pairs) < 10: continue
                A_ = np.stack([Mp[a] for a, _ in pairs]); B_ = np.stack([Mp[b] for _, b in pairs]); D_ = A_ - B_
                d_all = D_.mean(0) / (D_.std(0, ddof=1) + 1e-9)
                for nm in ("hub", "spec_idea", "spec_math"):
                    idx = sets[nm]
                    mrows.append({"task_type": dname, "condition": c, "set": nm, "n_pairs": len(pairs), "mean_shift": float(D_[:, idx].mean()),
                                  "mean_abs_d": float(np.abs(d_all[idx]).mean()), "max_abs_d": float(np.abs(d_all[idx]).max())})
                mrows.append({"task_type": dname, "condition": c, "set": "all_components", "n_pairs": len(pairs), "mean_shift": float(D_.mean()),
                              "mean_abs_d": float(np.abs(d_all).mean()), "max_abs_d": float(np.abs(d_all).max())})
        pl.DataFrame(mrows).write_csv(OUT / "coupling_magnitude.csv"); print("[coupling] magnitude code of the same sets (paired d vs baseline):"); print(pl.DataFrame(mrows))
    plt = _mpl(); fig, axs = plt.subplots(1, 2, figsize=(7, 2.4))
    ax = axs[0]; s = summ.filter(pl.col("mode") == "prompts"); xs = np.arange(s.height)
    ax.bar(xs - 0.2, s["z_idea"], 0.4, color=DOM_COL["idea"], label="hub x Idea-specific"); ax.bar(xs + 0.2, s["z_math"], 0.4, color=DOM_COL["math"], label="hub x Math-specific")
    labels = [a + "\n" + b.replace("prompt_", "").replace("__null", "\nnull") for a, b in zip(s["task_type"], s["condition"])]
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=4.5, rotation=90); ax.set_ylabel("coupling (Fisher z)"); ax.legend(frameon=False, fontsize=5.5); _panel(ax, "a")
    ax = axs[1]
    for (dom, cond), g in wdf.filter(pl.col("mode") == "prompts").group_by(["task_type", "condition"], maintain_order=True):
        gg = g.group_by("rel_token").agg(pl.col("z_idea").mean()).sort("rel_token")
        ax.plot(gg["rel_token"], gg["z_idea"], color=DOM_COL[dom], ls={"baseline": "-", "prompt_checking": "--", "prompt_exploratory": ":"}.get(cond, "-"), label=f"{dom}, {cond.replace('prompt_', '')}")
    ax.set_xlabel("window start (reasoning tokens)"); ax.set_ylabel("hub x Idea-specific z"); ax.legend(frameon=False, fontsize=5.5); _panel(ax, "b")
    fig.tight_layout(); _save(fig, "fig_neuro_coupling"); plt.close(fig)


# ================================================================ E4: event-locked averages
def cmd_events(runs=("corpus1_math_a", "corpus1_math_b", "corpus1_idea_a", "corpus1_idea_b"), W: int = 64, n_null: int = 10):
    from .phenotype import _segments_with_offsets
    meta, resid, _ = load_shards("corpus")
    if meta is None:
        print("[events] no corpus shards"); return
    dm = dmeta(); tn = dm["target_names"]; tok = load_tokenizer(); OUT.mkdir(parents=True, exist_ok=True)
    tag = JUDGE_MODEL.replace("/", "_")
    tb = pl.concat([pl.read_parquet(p) for r in runs for p in sorted((DATA / "runs" / r / "judge").rglob(f"trackB_full__{tag}*.parquet"))], how="diagonal_relaxed").unique(["trace_id", "seg_idx"], keep="last")
    tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "reasoning_text_for_analysis"]) for r in runs for p in sorted((DATA / "runs" / r).glob("traces_baseline.parquet"))])
    rtext = dict(zip(tr["trace_id"].to_list(), tr["reasoning_text_for_analysis"].to_list()))
    nseg = dict(zip(tb["trace_id"].to_list(), tb["n_segments"].to_list()))
    lab_by = {(k[0] if isinstance(k, tuple) else k): g.sort("seg_idx") for k, g in tb.group_by("trace_id")}
    sig_names = ["dom19", "dom16", "a2"] + TARGETS_SHOWN
    rng = np.random.default_rng(0)
    acc = {}                                                    # (beh, dom) -> list of per-trace [2W, 7] (event minus null)
    acc_ev, acc_null = {}, {}
    n_events = {}
    for i, tid in enumerate(meta["trace_id"]):
        if tid not in nseg or tid not in rtext or not rtext[tid]: continue
        starts = _segments_with_offsets(tok, rtext[tid], nseg[tid])
        if starts is None: continue
        g = lab_by[tid]; valid = (g["kim_parsed"].fill_null(False) & g["gandhi_parsed"].fill_null(False)).to_numpy()
        labels = g.select(BEH).fill_null(0).to_numpy().astype(int)
        r = resid[i]; T = len(r); dom = meta["task_type"][i]
        pt = pred_targets(r[:, 0, SIG], dm)
        S = np.stack([r[:, 1, J19], r[:, 0, J16], r[:, 0, JA2]] + [pt[:, tn.index(t)] for t in TARGETS_SHOWN], axis=1)
        starts = np.asarray(starts)
        for b, beh in enumerate(BEH):
            on = starts[(valid) & (labels[:, b] == 1)]; on = on[(on >= W) & (on < T - W)]
            if len(on) == 0: continue
            E = np.stack([S[o - W:o + W] for o in on]); E = (E - E[:, :W // 2].mean(1, keepdims=True)).mean(0)
            pool = np.arange(W, T - W)
            Nn = np.stack([(lambda o_: (np.stack([S[o - W:o + W] for o in o_]) - np.stack([S[o - W:o + W] for o in o_])[:, :W // 2].mean(1, keepdims=True)).mean(0))(rng.choice(pool, len(on), replace=False)) for _ in range(n_null)]).mean(0)
            acc_ev.setdefault((beh, dom), []).append(E); acc_null.setdefault((beh, dom), []).append(Nn); n_events[(beh, dom)] = n_events.get((beh, dom), 0) + len(on)
    rows, lead = [], []
    rel = np.arange(-W, W)
    for key in acc_ev:
        beh, dom = key; E = np.stack(acc_ev[key]); Nn = np.stack(acc_null[key]); D = E - Nn
        m, lo, hi = boot_ci(D, reps=500)
        for s, nm in enumerate(sig_names):
            for t in range(2 * W):
                rows.append({"behaviour": beh, "task_type": dom, "signal": nm, "rel_token": int(rel[t]), "n_traces": len(E), "n_events": n_events[key],
                             "event_mean": float(E[:, t, s].mean()), "null_mean": float(Nn[:, t, s].mean()), "diff": float(m[t, s]), "lo": float(lo[t, s]), "hi": float(hi[t, s])})
            # lead time: bins of 8 before onset; first bin of the longest run of significant bins that ends at onset
            sig_bins = [(lo[t0:t0 + 8, s] > 0).all() or (hi[t0:t0 + 8, s] < 0).all() for t0 in range(0, W, 8)]
            peak = float(np.abs(m[W:W + 48, s]).max()) + 1e-9
            big_bins = [abs(m[t0:t0 + 8, s].mean()) >= 0.25 * peak and sig_bins[k] for k, t0 in enumerate(range(0, W, 8))]
            def _run(bins):
                n = 0
                for sb in reversed(bins):
                    if sb: n += 1
                    else: break
                return n
            run, run_big = _run(sig_bins), _run(big_bins)
            lead.append({"behaviour": beh, "task_type": dom, "signal": nm, "lead_tokens_ci": 8 * run, "lead_tokens": 8 * run_big,
                         "sign_pre": float(np.sign(m[W - 8:W, s].mean())) if run_big else 0.0, "pre_mean_-16_0": float(m[W - 16:W, s].mean()),
                         "post_mean_0_32": float(m[W:W + 32, s].mean()), "post_peak_abs": peak, "n_traces": len(E), "n_events": n_events[key]})
    pl.DataFrame(rows).write_csv(OUT / "events_locked.csv"); ld = pl.DataFrame(lead); ld.write_csv(OUT / "events_lead.csv")
    pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200); print(ld.filter(pl.col("signal").is_in(["dom19", "a2"])).sort(["signal", "behaviour", "task_type"]))
    plt = _mpl(); df = pl.DataFrame(rows)
    short = {"Question_and_Answering": "question / answer", "Perspective_Shift": "perspective shift", "Conflict_of_Perspectives": "conflict", "Reconciliation": "reconciliation",
             "verification": "verification", "backtracking": "backtracking", "subgoal": "subgoal", "backward_chaining": "backward chaining"}
    fig, axs = plt.subplots(3, 8, figsize=(7.2, 3.4), sharex=True)
    for s, nm in enumerate(["dom19", "dom16", "a2"]):
        for b, beh in enumerate(BEH):
            ax = axs[s, b]
            for dom in ("math", "idea"):
                g = df.filter((pl.col("behaviour") == beh) & (pl.col("task_type") == dom) & (pl.col("signal") == nm)).sort("rel_token")
                if not g.height: continue
                ax.plot(g["rel_token"], g["diff"], color=DOM_COL[dom], lw=0.8, label=dom if (s == 0 and b == 0) else None); ax.fill_between(g["rel_token"], g["lo"], g["hi"], color=DOM_COL[dom], alpha=0.15, lw=0)
            ax.axvline(0, color="k", lw=0.4); ax.axhline(0, color="grey", lw=0.4, ls=":"); ax.set_xlim(-W, W); ax.set_xticks([-W, 0, W]); ax.tick_params(labelsize=5, length=2, pad=1)
            if s == 0: ax.set_title(short[beh], fontsize=6, pad=2)
            if b == 0: ax.set_ylabel({"dom19": "domain axis, layer 19", "dom16": "domain axis, layer 16", "a2": "A2 direction, layer 16"}[nm], fontsize=5.5); _panel(ax, "abc"[s])
    axs[0, 0].legend(frameon=False, fontsize=5.5, loc="upper left")
    fig.supxlabel("tokens from sentence onset (event minus matched random onsets, baseline -64..-33)", fontsize=6)
    fig.tight_layout(h_pad=0.4, w_pad=0.3); _save(fig, "fig_neuro_events"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["map", "taskset", "coupling", "events", "all"])
    ap.add_argument("--tag", default="", help="taskset: '' (fixed filler) or 'varied' (one filler per item)")
    a = ap.parse_args()
    if a.cmd in ("map", "all"): cmd_map()
    if a.cmd in ("events", "all"): cmd_events()
    if a.cmd in ("coupling", "all"): cmd_coupling()
    if a.cmd in ("taskset", "all"): cmd_taskset(a.tag)


if __name__ == "__main__":
    main()
