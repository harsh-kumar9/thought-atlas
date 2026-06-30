"""scripts/run_analysis.py — orchestrate aggregate + heartbeat + HMM + figures.
Usage: python scripts/run_analysis.py --config configs/exp.yaml --judge-tag <tag>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl
from omegaconf import OmegaConf
from src.analysis import aggregate as agg, heartbeat as hb, hmm_regimes as hmr


def plot_heatmap(ct: pl.DataFrame, path: Path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import numpy as np
    models = ct["gen_model"].unique().to_list(); tasks = ct["task_type"].unique().to_list()
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 4), squeeze=False)
    for mi, m in enumerate(models):
        sub = ct.filter(pl.col("gen_model") == m)
        M = np.array([[sub.filter((pl.col("task_type")==t)&(pl.col("behavior")==b))["mean"].to_list()[0]
                       for t in tasks] for b in agg.BEHAVIORS])
        ax = axes[0][mi]; im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.set_yticks(range(len(agg.BEHAVIORS))); ax.set_yticklabels(agg.BEHAVIORS, fontsize=7)
        ax.set_title(f"model={m} (mean count/trace)"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_heartbeat(segments: pl.DataFrame, path: Path, n_bins=20):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, beh in zip(axes.flat, agg.BEHAVIORS):
        c = hb.curve(segments, beh, n_bins=n_bins)
        for task in c["task_type"].unique().to_list():
            agg_t = (c.filter(pl.col("task_type")==task).group_by("bin")
                       .agg(pl.col("freq").mean()).sort("bin"))
            ax.plot(agg_t["bin"].to_list(), agg_t["freq"].to_list(), label=task, marker=".")
        ax.set_title(beh, fontsize=9); ax.set_xlabel("position bin"); ax.set_ylabel("norm freq")
    axes.flat[0].legend(fontsize=7); fig.suptitle("Cross-domain heartbeat (ThinkARM-style)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--judge-tag", required=True)
    ap.add_argument("--judge-dir", default="data/judge")
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out", default="data/analysis")
    a = ap.parse_args()
    cfg = OmegaConf.load(a.config); out = Path(a.out); (out/"figures").mkdir(parents=True, exist_ok=True)
    traces = pl.concat([pl.read_parquet(p) for p in sorted(Path().glob(a.traces_glob))], how="diagonal_relaxed")

    # Track A aggregate
    trackA = pl.read_parquet(Path(a.judge_dir)/f"trackA_counts__{a.judge_tag}.parquet")
    joined = agg.join_counts(trackA, traces)
    ct = agg.cell_table(joined); ct.write_parquet(out/"cell_table.parquet")
    plot_heatmap(ct, out/"figures"/"domain_behavior_heatmap.png")
    gee = {b: str(agg.gee_cluster_robust(joined, b)) for b in agg.BEHAVIORS}
    (out/"glmm_formulas.json").write_text(json.dumps(
        {"lme4_canonical": {b: agg.lme4_formula(b) for b in agg.BEHAVIORS}, "gee_python": gee}, indent=2))

    # Track B temporal
    bf = pl.read_parquet(Path(a.judge_dir)/f"trackB_full__{a.judge_tag}.parquet")
    segs = bf.join(traces.select(["trace_id","task_type","gen_model"]), on="trace_id", how="inner")
    plot_heartbeat(segs, out/"figures"/"heartbeat.png", n_bins=int(cfg.analysis.heartbeat.n_position_bins))
    fanova = {b: hb.functional_anova(segs, b, n_bins=int(cfg.analysis.heartbeat.n_position_bins))
              for b in agg.BEHAVIORS}
    (out/"fanova.json").write_text(json.dumps(fanova, indent=2, default=str))
    hmm = {t: hmr.select_regimes(segs, t, k_range=tuple(cfg.analysis.hmm.regime_ranges[t]))
           for t in segs["task_type"].unique().to_list() if t in cfg.analysis.hmm.regime_ranges}
    (out/"hmm_regimes.json").write_text(json.dumps(hmm, indent=2, default=str))
    print(f"[analysis] wrote figures + tables -> {out}")


if __name__ == "__main__":
    main()
