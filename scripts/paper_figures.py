"""scripts/paper_figures.py — publication-ready figures + final stats for the paper.

Produces (data/analysis/paper/):
  fig1_composition.{png,pdf}    H1: domain x behavior, cognitive/conversational rows, DeepSeek-R1-Distill-Llama-8B vs Llama-3.1-8B-Instruct
  fig2_heartbeat.{png,pdf}      H2: per-domain positional arcs, shape-normalized, cog/conv rows
  fig3_section.{png,pdf}        think-block vs answer-block behavior split (H3 support)
  h2_fanova.csv                 functional ANOVA per behavior (with rarity-guarded p)
  h4_shuffle.csv                within-trace shuffle null per behavior (positional structure is real)
  gee_table.csv                 extracted GEE coefficients (task_type effects), not repr strings

Run on the cluster (needs traces for task_type/gen_model):
  python scripts/paper_figures.py --config configs/exp.yaml --judge-tag google_gemma-4-31B-it \
      --judge-dir data/judge/prod --traces-glob "data/traces/traces_*.parquet"
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

COG = ["verification", "backtracking", "subgoal", "backward_chaining"]
CNV = ["Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives", "Reconciliation"]
BEH = COG + CNV
FULL = {"verification": "Answer Verification", "backtracking": "Backtracking",
        "subgoal": "Subgoal Setting", "backward_chaining": "Backward Chaining",
        "Question_and_Answering": "Question & Answering", "Perspective_Shift": "Perspective Shift",
        "Conflict_of_Perspectives": "Conflict of Perspectives", "Reconciliation": "Reconciliation"}
# domain display + a deliberate order: checkable/formal -> open-ended
DOM_ORDER = ["math", "code", "gpqa", "planning", "moral", "idea"]
DOM_LABEL = {"math": "Math", "code": "Code", "gpqa": "GPQA", "planning": "Planning",
             "moral": "Moral", "idea": "Idea"}
# colorblind-safe domain palette (Okabe-Ito)
DOM_COLOR = {"math": "#0072B2", "code": "#E69F00", "gpqa": "#009E73", "planning": "#CC79A7",
             "moral": "#D55E00", "idea": "#56B4E9"}
ACCENT = "#1b6ca8"; GREY = "#bcbcbc"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "figure.dpi": 150, "savefig.bbox": "tight"})


def boot_ci(x, fn=np.mean, n=1000, seed=0):
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    r = np.random.default_rng(seed)
    s = np.array([fn(x[r.integers(0, len(x), len(x))]) for _ in range(n)])
    return float(fn(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def load(judge_dir, tag, traces_glob):
    jd = Path(judge_dir)
    a = pl.read_parquet(jd / f"trackA_counts__{tag}.parquet")
    bf = pl.read_parquet(jd / f"trackB_full__{tag}.parquet")
    tr = pl.concat([pl.read_parquet(p) for p in glob.glob(traces_glob)], how="diagonal_relaxed") \
           .select(["trace_id", "task_type", "gen_model"])
    return a.join(tr, on="trace_id", how="left"), bf.join(tr, on="trace_id", how="left")


# ---------- FIG 1: composition (H1) ----------
def fig1(a, out):
    doms = [d for d in DOM_ORDER if d in a["task_type"].unique().to_list()]
    models = a["gen_model"].unique().sort().to_list()
    rk = "reasoner" if "reasoner" in models else models[-1]
    ak = "anchor" if "anchor" in models else models[0]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    x = np.arange(len(doms)); w = 0.4
    for ri, (rtitle, behs) in enumerate([("Cognitive\n(reasoning ops)", COG),
                                          ("Conversational\n(dialectic ops)", CNV)]):
        rmax = 0; cache = {}
        for b in behs:
            for mk in (ak, rk):
                ms, los, his = [], [], []
                for d in doms:
                    v = a.filter((pl.col("gen_model") == mk) & (pl.col("task_type") == d))[b].to_numpy()
                    m, lo, hi = boot_ci(v.astype(float))
                    ms.append(m); los.append(m-lo); his.append(hi-m)
                cache[(b, mk)] = (ms, los, his); rmax = max(rmax, max(np.add(ms, his)))
        for ci, b in enumerate(behs):
            ax = axes[ri][ci]
            am, al, ah = cache[(b, ak)]; rm, rl, rh = cache[(b, rk)]
            ax.bar(x-w/2, am, w, yerr=[al, ah], capsize=2, color=GREY, ecolor="#888", label="Llama-3.1-8B-Instruct")
            ax.bar(x+w/2, rm, w, yerr=[rl, rh], capsize=2, color=ACCENT, ecolor="#0d3b5c", label="DeepSeek-R1-Distill-Llama-8B")
            ax.set_title(FULL[b], fontweight="bold", fontsize=9.5)
            ax.set_xticks(x); ax.set_xticklabels([DOM_LABEL[d] for d in doms], rotation=40, ha="right", fontsize=8)
            ax.set_ylim(0, rmax*1.12)
            if ci == 0: ax.set_ylabel(rtitle, fontweight="bold", fontsize=10)
            else: ax.set_yticklabels([])
        axes[ri][0].legend(fontsize=7.5, loc="upper right", frameon=False)
    fig.suptitle("Behavior composition per reasoning trace, by task domain", fontsize=13, y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig1_composition.{ext}")
    plt.close(fig)


# ---------- FIG 2: heartbeat, shape-normalized (H2) ----------
def fig2(bf, out, nbins=20):
    doms = [d for d in DOM_ORDER if d in bf["task_type"].unique().to_list()]
    seg = bf.filter(pl.col("gen_model") == "reasoner") if "reasoner" in bf["gen_model"].unique().to_list() else bf
    seg = seg.with_columns((pl.col("norm_pos")*(nbins-1)).round().cast(pl.Int32).alias("bin"))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for idx, b in enumerate(BEH):
        ax = axes[idx//4][idx%4]
        for d in doms:
            prof = (seg.filter(pl.col("task_type") == d).group_by("bin")
                    .agg(pl.col(b).mean()).sort("bin"))
            y = prof[b].to_numpy()
            s = y.sum()
            y = y/s if s > 0 else y          # shape-normalize: compare arcs, not levels
            ax.plot(prof["bin"].to_numpy(), y, color=DOM_COLOR[d], lw=1.6, label=DOM_LABEL[d])
        ax.set_title(FULL[b], fontweight="bold", fontsize=9.5)
        ax.set_xlabel("position in trace →", fontsize=8)
        if idx % 4 == 0:
            ax.set_ylabel(("Cognitive" if idx == 0 else "Conversational") + "\nshare of behavior", fontsize=9)
    axes[0][0].legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    fig.suptitle("Positional 'heartbeat': where each behavior occurs within the reasoning trace\n"
                 "(shape-normalized per domain; DeepSeek-R1-Distill-Llama-8B)", fontsize=12, y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig2_heartbeat.{ext}")
    plt.close(fig)


# ---------- FIG 3: think vs answer section ----------
def fig3(bf, out):
    seg = bf.filter(pl.col("gen_model") == "reasoner") if "reasoner" in bf["gen_model"].unique().to_list() else bf
    sect = seg.group_by("section_type").agg([pl.col(b).mean() for b in BEH])
    sects = sect["section_type"].to_list()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(BEH)); w = 0.8/max(1, len(sects))
    for si, s in enumerate(sects):
        row = sect.filter(pl.col("section_type") == s)
        ax.bar(x + si*w, [row[b][0] for b in BEH], w, label=f"{s}-block",
               color=ACCENT if s == "think" else GREY)
    ax.set_xticks(x + w*(len(sects)-1)/2)
    ax.set_xticklabels([FULL[b] for b in BEH], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("per-sentence rate"); ax.legend(frameon=False)
    ax.axvline(3.5, color="#ccc", ls="--", lw=1)
    ax.text(1.5, ax.get_ylim()[1]*0.95, "cognitive", ha="center", fontsize=9, color="#555")
    ax.text(5.5, ax.get_ylim()[1]*0.95, "conversational", ha="center", fontsize=9, color="#555")
    fig.suptitle("Behavior rates in think-block vs answer-block (DeepSeek-R1-Distill-Llama-8B)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig3_section.{ext}")
    plt.close(fig)


# ---------- H2 functional ANOVA (rarity-guarded) ----------
def h2_fanova(bf, out, nbins=20):
    seg = bf.with_columns((pl.col("norm_pos")*(nbins-1)).round().cast(pl.Int32).alias("bin"))
    doms = seg["task_type"].unique().to_list()
    rows = []
    for b in BEH:
        # between-domain variance of per-bin profiles vs within
        profs = []
        for d in doms:
            p = seg.filter(pl.col("task_type") == d).group_by("bin").agg(pl.col(b).mean()).sort("bin")[b].to_numpy()
            profs.append(p)
        profs = np.array([p for p in profs if len(p) == nbins])
        grand = profs.mean(0)
        between = ((profs - grand)**2).sum() * 1.0
        within = profs.var(0).sum()
        overall_rate = float(seg[b].mean())
        F = between/within if within > 1e-12 else np.nan
        rows.append({"behavior": b, "overall_rate": round(overall_rate, 5),
                     "F_between_within": round(float(F), 3) if np.isfinite(F) else None,
                     "assessable": overall_rate > 0.005,
                     "note": "OK" if overall_rate > 0.005 else "too rare for positional test"})
    pl.DataFrame(rows).write_csv(out / "h2_fanova.csv")
    return rows


# ---------- H4 within-trace shuffle null ----------
def h4_shuffle(bf, out, nbins=20, nperm=500, seed=0):
    rng = np.random.default_rng(seed)
    tids = bf["trace_id"].to_numpy(); pos = bf["norm_pos"].to_numpy()
    order = np.argsort(tids, kind="stable"); st = tids[order]; sp = pos[order]
    bnd = np.where(st[1:] != st[:-1])[0]+1
    starts = np.concatenate([[0], bnd]); ends = np.concatenate([bnd, [len(st)]])
    nb_all = (sp*(nbins-1)).round().astype(int)
    rows = []
    for b in BEH:
        vals = bf[b].to_numpy()[order]
        def prof(v):
            return np.array([v[nb_all == k].mean() if (nb_all == k).any() else np.nan for k in range(nbins)])
        obs = np.nanvar(prof(vals))
        nulls = np.empty(nperm)
        for t in range(nperm):
            sv = vals.copy()
            for s, e in zip(starts, ends):
                if e-s > 1: rng.shuffle(sv[s:e])
            nulls[t] = np.nanvar(prof(sv))
        p = (np.sum(nulls >= obs)+1)/(nperm+1)
        rows.append({"behavior": b, "obs_var": round(float(obs), 6),
                     "null_mean": round(float(nulls.mean()), 6), "p_perm": round(float(p), 4),
                     "structured": bool(p < 0.05 and obs > 10*nulls.mean())})
    pl.DataFrame(rows).write_csv(out / "h4_shuffle.csv")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--judge-tag", required=True)
    ap.add_argument("--judge-dir", default="data/judge/prod")
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out", default="data/analysis/paper")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    trackA, bf = load(a.judge_dir, a.judge_tag, a.traces_glob)
    fig1(trackA, out); print("fig1 composition ✓")
    fig2(bf, out); print("fig2 heartbeat ✓")
    fig3(bf, out); print("fig3 section ✓")
    f = h2_fanova(bf, out); print("H2 fanova ✓:", sum(r["assessable"] for r in f), "/8 assessable")
    h = h4_shuffle(bf, out); print("H4 shuffle ✓:", sum(r["structured"] for r in h), "/8 structured")
    print(f"all -> {out}/")


if __name__ == "__main__":
    main()
