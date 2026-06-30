"""scripts/analyze_trackA.py — H1 (domain × behavior interaction) plots from Track A counts.

Track A tests H1 only (domain x behavior; rate ratios). Temporal H2/H4 need Track B.
Produces four panels that disambiguate domain / mechanism / model / presence-vs-frequency,
all with bootstrap 95% CIs (per-domain N varies: code=175 vs 500, so CIs must be visible).

Outputs (data/analysis/):
  trackA_rate_heatmap.png        domain x behavior, per model — the headline interaction
  trackA_presence_vs_rate.png    presence (frac>0) and frequency (rate) side by side
  trackA_model_contrast.png      reasoner vs anchor per behavior, CIs
  trackA_rate_ratios.csv         H1 test: max/min across-domain rate ratio per behavior

Run:  python scripts/analyze_trackA.py --config configs/exp.yaml --judge-tag google_gemma-4-31B-it
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KIM = ["Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives", "Reconciliation"]
GAN = ["verification", "backtracking", "subgoal", "backward_chaining"]
BEH = GAN + KIM
# full human-readable names for panel titles
FULL = {"verification": "Answer Verification", "backtracking": "Backtracking",
        "subgoal": "Subgoal Setting", "backward_chaining": "Backward Chaining",
        "Question_and_Answering": "Question & Answering", "Perspective_Shift": "Perspective Shift",
        "Conflict_of_Perspectives": "Conflict of Perspectives", "Reconciliation": "Reconciliation"}
SHORT = FULL  # titles use full names; alias kept so existing references resolve


def boot_ci(x: np.ndarray, fn, n=2000, seed=0):
    """Bootstrap 95% CI for statistic fn over 1-D array x."""
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    stats = np.array([fn(x[i]) for i in idx])
    return float(fn(x)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--judge-tag", required=True)
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--judge-glob", default=None)
    ap.add_argument("--out-dir", default="data/analysis")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    jg = args.judge_glob or f"data/judge/prod/trackA_counts__{args.judge_tag}*.parquet"
    a = pl.concat([pl.read_parquet(s) for s in glob.glob(jg)], how="diagonal_relaxed").unique("trace_id")
    tr = pl.concat([pl.read_parquet(f) for f in glob.glob(args.traces_glob)], how="diagonal_relaxed")

    sel = ["trace_id", "task_type", "gen_model"]
    df = a.join(tr.select(sel), on="trace_id", how="left")

    models = df["gen_model"].unique().sort().to_list()
    domains = df["task_type"].unique().sort().to_list()
    # exact display names from config; fall back to the key if a model isn't listed
    def _disp(k):
        try:
            return cfg.gen_models[k].hf_id.split("/")[-1]
        except Exception:
            return k
    name = {k: _disp(k) for k in models}

    def rate_and_presence(sub: pl.DataFrame, beh: str):
        c = sub[beh].to_numpy().astype(float)
        return c, (c > 0).astype(float)

    # ===== Main figure: 2 rows (cognitive / conversational) x 4 behavior columns =====
    # vertical bars, domain on x; reasoner accent color, anchor faint grey; shared y within a row.
    ACCENT = "#1b6ca8"   # reasoner
    GREY = "#bdbdbd"     # anchor (de-emphasized)
    rows_spec = [("Cognitive behaviors (Gandhi)", GAN), ("Conversational behaviors (Kim)", KIM)]
    reasoner_key = "reasoner" if "reasoner" in models else models[-1]
    anchor_key = "anchor" if "anchor" in models else models[0]

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    x = np.arange(len(domains)); w = 0.4
    for ri, (rtitle, behs) in enumerate(rows_spec):
        row_max, cache = 0.0, {}
        for b in behs:
            for mk in (anchor_key, reasoner_key):
                means, los, his = [], [], []
                for d in domains:
                    sub = df.filter((pl.col("gen_model") == mk) & (pl.col("task_type") == d))
                    r, _ = rate_and_presence(sub, b)
                    mm, lo, hi = boot_ci(r, np.mean)
                    means.append(mm); los.append(mm-lo); his.append(hi-mm)
                cache[(b, mk)] = (means, los, his)
                row_max = max(row_max, max(np.add(means, his)))
        for ci, b in enumerate(behs):
            ax = axes[ri][ci]
            am, alo, ahi = cache[(b, anchor_key)]
            rm, rlo, rhi = cache[(b, reasoner_key)]
            ax.bar(x - w/2, am, w, yerr=[alo, ahi], capsize=2, color=GREY,
                   error_kw=dict(ecolor="#7a7a7a", lw=1), label=name[anchor_key])
            ax.bar(x + w/2, rm, w, yerr=[rlo, rhi], capsize=2, color=ACCENT,
                   error_kw=dict(ecolor="#0d3b5c", lw=1), label=name[reasoner_key])
            ax.set_title(SHORT[b], fontsize=9.5, fontweight="bold")
            ax.set_xticks(x); ax.set_xticklabels(domains, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, row_max * 1.12)
            ax.spines[["top", "right"]].set_visible(False)
            if ci == 0:
                ax.set_ylabel(rtitle, fontsize=11, fontweight="bold")
            else:
                ax.set_yticklabels([])
        axes[ri][0].legend(fontsize=8, loc="upper right", frameon=False)
    fig.suptitle("Behavior counts per reasoning trace, by task domain\n"
                 f"{name[reasoner_key]} (reasoning) vs {name[anchor_key]} (non-reasoning) · "
                 "mean ± 95% bootstrap CI", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "trackA_behavior_by_domain.png", dpi=150); plt.close(fig)

    # ===== H1 test artifact: across-domain rate ratio per behavior, per model =====
    rows = []
    for m in models:
        for b in BEH:
            dmeans = {}
            for d in domains:
                sub = df.filter((pl.col("gen_model") == m) & (pl.col("task_type") == d))
                r, _ = rate_and_presence(sub, b)
                dmeans[d] = float(np.mean(r)) if len(r) else 0.0
            vals = np.array(list(dmeans.values()))
            mn, mx = float(vals.min()), float(vals.max())
            # H1: does the behavior's per-domain rate vary by >1.5x? min=0 & max>0 (present in
            # some domains, absent in others) is the STRONGEST domain effect -> passes.
            h1_pass = mx >= 1.5 * mn if mn > 0 else (mx > 0)
            ratio = (mx / mn) if mn > 0 else float("inf")
            rows.append({"model": m, "behavior": b, "min_domain": min(dmeans, key=dmeans.get),
                         "max_domain": max(dmeans, key=dmeans.get),
                         "min_rate": round(mn, 3), "max_rate": round(mx, 3),
                         "rate_ratio": (round(ratio, 2) if np.isfinite(ratio) else 999.0),
                         "H1_pass_gt1p5": bool(h1_pass)})
    rr = pl.DataFrame(rows).with_columns(pl.col("H1_pass_gt1p5").cast(pl.Boolean))
    rr.write_csv(out / "trackA_rate_ratios.csv")
    npass = rr.filter((pl.col("model") == reasoner_key) & pl.col("H1_pass_gt1p5")).height
    print(f"[H1] {reasoner_key}: {npass}/{len(BEH)} behaviors show >1.5x across-domain rate ratio "
          f"(prereg threshold: >=4).  ->  {'SUPPORTED' if npass >= 4 else 'NOT met'}")
    print(rr.filter(pl.col("model") == reasoner_key).sort("rate_ratio", descending=True))
    print(f"\nfigures + trackA_rate_ratios.csv -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())