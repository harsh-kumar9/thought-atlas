"""src/perf/trajectory_viz.py — the mechanism figure: deliberation trajectories, success vs failure.

For each (domain, behavior): the per-position behavior rate, split into traces that SUCCEEDED
vs FAILED, each with a bootstrap 95% CI band. Where the bands separate, the *timing* of that
behavior distinguishes outcome — a positional mechanism the regression coefficients can't show.

Cognitive (Gandhi) and conversational (Kim) families are drawn as SEPARATE figures (different
taxonomies, different scales). Binary-outcome domains (math/gpqa/planning/code) use success/fail;
continuous-quality domains (moral/idea) split at the median quality (top half vs bottom half).

Usage:
  python -m src.perf.trajectory_viz --trackB data/judge/prod/trackB_full__<tag>.parquet \
      --grades data/perf/success_grades.parquet data/perf/code_grades.parquet \
               data/judge/prod/quality__<tag>.parquet \
      --traces-glob "data/traces/traces_*.parquet" --out-dir data/perf/figures
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KIM = ["Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives", "Reconciliation"]
GAN = ["verification", "backtracking", "subgoal"]    # drop backward_chaining (too rare)
FULL = {"verification": "Answer Verification", "backtracking": "Backtracking", "subgoal": "Subgoal Setting",
        "Question_and_Answering": "Question & Answering", "Perspective_Shift": "Perspective Shift",
        "Conflict_of_Perspectives": "Conflict of Perspectives", "Reconciliation": "Reconciliation"}
DOM_ORDER = ["math", "code", "gpqa", "planning", "moral", "idea"]
DOM_LABEL = {"math": "Math", "code": "Code", "gpqa": "GPQA", "planning": "Planning", "moral": "Moral", "idea": "Idea"}
SUCC_C = "#1b7837"; FAIL_C = "#c0392b"     # green=success, red=failure (colorblind-distinguishable by linestyle too)


def _profile_ci(sub_traces, behavior, nbins=12, nboot=400, seed=0):
    """Mean per-bin rate + bootstrap 95% CI across traces. sub_traces: list of (pos[], val[])."""
    rng = np.random.default_rng(seed)
    # bin each trace's events, then bootstrap over TRACES (cluster unit = trace)
    binned = []
    for pos, val in sub_traces:
        bn = np.clip((pos * nbins).astype(int), 0, nbins - 1)
        prof = np.array([val[bn == k].mean() if (bn == k).any() else np.nan for k in range(nbins)])
        binned.append(prof)
    if not binned:
        return None
    B = np.array(binned)                       # n_traces x nbins
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(B, 0)
    boot = np.empty((nboot, nbins))
    n = len(B)
    for i in range(nboot):
        idx = rng.integers(0, n, n)
        with np.errstate(invalid="ignore"):
            boot[i] = np.nanmean(B[idx], 0)
    lo = np.nanpercentile(boot, 2.5, axis=0); hi = np.nanpercentile(boot, 97.5, axis=0)
    return mean, lo, hi


def _outcome_split(df):
    """Return a function tid->'success'|'fail'|None for binary, or median-split for continuous."""
    if "success" in df.columns and df["success"].drop_nulls().len() >= 30:
        m = dict(zip(df["trace_id"].to_list(), df["success"].to_list()))
        return lambda t: (None if m.get(t) is None else ("success" if m[t] == 1 else "fail")), "outcome (solved vs failed)"
    if "quality_score" in df.columns and df["quality_score"].drop_nulls().len() >= 30:
        med = df["quality_score"].median()
        m = dict(zip(df["trace_id"].to_list(), df["quality_score"].to_list()))
        return (lambda t: (None if m.get(t) is None else ("success" if m[t] >= med else "fail"))), f"quality (≥ vs < median {med:.2f})"
    return None, None


def _domain_boundary(bd_traces):
    """Median + IQR (as %) of the think->answer boundary across a domain's traces (outcome-invariant)."""
    vals = []
    for sub in bd_traces:
        secs = sub["section_type"].to_list(); pos = sub["norm_pos"].to_list()
        ans = [i for i, s in enumerate(secs) if s == "answer"]
        if ans and ans[0] > 0:
            vals.append(pos[ans[0]])
    if not vals:
        return None
    v = np.array(vals) * 100
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def make_family_figure(trackB, grades, traces_meta, behaviors, family_name, out_path, nbins=12):
    # join domain + outcome
    g = grades
    tr = traces_meta.select(["trace_id", "task_type", "gen_model"])
    b = trackB.join(tr, on="trace_id", how="left").filter(pl.col("gen_model") == "reasoner")
    doms = [d for d in DOM_ORDER if d in b["task_type"].unique().to_list()]
    nrows, ncols = len(behaviors), len(doms)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows), squeeze=False)
    xpct = np.linspace(0, 100, nbins)

    for di, dom in enumerate(doms):
        gd = g.filter(pl.col("task_type") == dom) if "task_type" in g.columns else g
        split_fn, _ = _outcome_split(gd)
        bd = b.filter(pl.col("task_type") == dom)
        # group sentences by trace once
        per_trace = {}
        for tid, sub in bd.group_by("trace_id"):
            tid0 = tid[0] if isinstance(tid, tuple) else tid
            sub = sub.sort("seg_idx")
            per_trace[tid0] = (sub["norm_pos"].to_numpy(), sub)
        # per-domain think->answer boundary band (outcome-invariant; verified empirically)
        bnd = _domain_boundary([sub for _, sub in per_trace.values()])
        # outcome label per trace
        for bi, beh in enumerate(behaviors):
            ax = axes[bi][di]
            if bnd is not None:
                ax.axvspan(bnd[1], bnd[2], color="#000000", alpha=0.06, lw=0)
                ax.axvline(bnd[0], color="#999", lw=0.6, ls=(0, (2, 2)))
            if split_fn is not None and beh in bd.columns:
                succ_tr, fail_tr = [], []
                for tid, (pos, sub) in per_trace.items():
                    lab = split_fn(tid)
                    if lab is None:
                        continue
                    val = sub[beh].to_numpy().astype(float)
                    (succ_tr if lab == "success" else fail_tr).append((pos, val))
                for grp, color, ls, lab in [(succ_tr, SUCC_C, "-", "solved/high"),
                                            (fail_tr, FAIL_C, "--", "failed/low")]:
                    r = _profile_ci(grp, beh, nbins=nbins)
                    if r is None:
                        continue
                    mean, lo, hi = r
                    ax.fill_between(xpct, lo, hi, color=color, alpha=0.15, lw=0)
                    ax.plot(xpct, mean, color=color, ls=ls, lw=1.6, label=lab)
            if bi == 0:
                ax.set_title(DOM_LABEL[dom], fontweight="bold", fontsize=11)
            if di == 0:
                ax.set_ylabel(FULL[beh], fontsize=8.5, fontweight="bold")
            ax.tick_params(labelsize=6, length=2)
            ax.set_xlim(0, 100); ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            if bi == nrows - 1:
                ax.set_xlabel("% through trace", fontsize=7)
    # one legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=SUCC_C, lw=2, label="solved / high-quality"),
               Line2D([0], [0], color=FAIL_C, lw=2, ls="--", label="failed / low-quality"),
               Line2D([0], [0], color="#999", lw=5, alpha=0.3, label="think→answer boundary (IQR)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Deliberation trajectories by outcome — {family_name} behaviors\n"
                 "per-position rate, success vs failure, 95% bootstrap CI (DeepSeek-R1-Distill-Llama-8B)",
                 fontsize=12, y=1.04 if nrows > 2 else 1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}.png/pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackB", required=True)
    ap.add_argument("--grades", nargs="+", required=True)
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out-dir", default="data/perf/figures")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    trackB = pl.read_parquet(a.trackB)
    grades = pl.concat([pl.read_parquet(g) for g in a.grades], how="diagonal_relaxed")
    # coalesce duplicate trace_ids (success + quality files overlap)
    for c in ("success", "quality_score"):
        if c not in grades.columns:
            grades = grades.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
    grades = grades.group_by("trace_id").agg([
        pl.col("success").drop_nulls().first().alias("success"),
        pl.col("quality_score").drop_nulls().first().alias("quality_score"),
        pl.col("task_type").drop_nulls().first().alias("task_type")])
    tr = pl.concat([pl.read_parquet(p) for p in glob.glob(a.traces_glob)], how="diagonal_relaxed")
    make_family_figure(trackB, grades, tr, GAN, "Cognitive", f"{a.out_dir}/traj_success_cognitive")
    make_family_figure(trackB, grades, tr, KIM, "Conversational", f"{a.out_dir}/traj_success_conversational")


if __name__ == "__main__":
    main()
