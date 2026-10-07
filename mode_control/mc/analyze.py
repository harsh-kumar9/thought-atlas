"""Post-judge analysis of a steering run: Δγ curves, attenuation, reversal, raw rates, task preservation.

Inputs: data/runs/<run>/traces_*.parquet + data/runs/<run>/judge/trackB_full__*.parquet (+ trackA)
Outputs: data/runs/<run>/analysis/{scored.parquet, summary.csv, contrasts.csv, figures/*.png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from .common import BEH, DATA, JUDGE_MODEL, load_tokenizer
from .phenotype import phase_amounts, score_with_frozen, windowed_amounts
from src.perf.answer_grading import grade_math_answer  # noqa: E402


def bootstrap_mean(x: np.ndarray, reps: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed); x = x[np.isfinite(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)])
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--window", type=int, default=None)
    a = ap.parse_args()
    rd = DATA / "runs" / a.run; spec = json.loads((rd / "run.json").read_text())
    window = a.window or spec["window"]
    frozen = json.loads((DATA / "phenotype" / f"frozen_w{window}.json").read_text())
    tag = JUDGE_MODEL.replace("/", "_")
    tr = pl.concat([pl.read_parquet(p) for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed")
    tr = tr.with_columns(pl.col("condition").replace({"rawdom_gm1": "nullspace_gm1"}))   # D-11: discarded whitened direction
    tb_paths = sorted((rd / "judge").rglob(f"trackB_full__{tag}*.parquet"))
    tb = pl.concat([pl.read_parquet(p) for p in tb_paths], how="diagonal_relaxed").unique(["trace_id", "seg_idx"], keep="last")
    tok = load_tokenizer()
    amt = windowed_amounts(tr, tb, window=window, tok=tok)
    sc = score_with_frozen(amt, frozen)
    meta_cols = ["trace_id", "condition", "instance_id", "task_type", "seed", "n_new_tokens", "completed", "finish_reason",
                 "parse_status", "hard_loop", "soft_loop", "answer_text", "reference_answer", "think_text", "system_prompt"]
    df = sc.join(tr.select(meta_cols), on="trace_id", how="inner")
    # task-preservation metrics. Math is graded from each trace's raw text (not after joins) with markdown
    # bold stripped: the repo grader's "final answer line" rule otherwise reads "**Answer:**" as the answer.
    def _grade(answer, ref, task):
        if task != "math" or not answer:
            return None
        return grade_math_answer(answer.replace("**", ""), ref).get("success")
    grades = pl.DataFrame({
        "trace_id": tr["trace_id"].to_list(),
        "math_success": [_grade(a, r, t) for a, r, t in zip(tr["answer_text"].to_list(), tr["reference_answer"].to_list(), tr["task_type"].to_list())],
    }).with_columns(pl.col("math_success").cast(pl.Int64))
    df = df.join(grades, on="trace_id", how="left")
    df = df.with_columns([
        pl.when(pl.col("task_type") == "math").then(pl.col("math_success").fill_null(0)).otherwise(None).alias("math_success_all"),
        pl.col("answer_text").str.len_chars().alias("answer_chars"),
        pl.col("think_text").str.len_chars().alias("think_chars"),
        (pl.col("answer_text").str.len_chars() > 0).alias("has_answer"),
    ])
    out = rd / "analysis"; (out / "figures").mkdir(parents=True, exist_ok=True)
    df.drop(["answer_text", "think_text"]).write_parquet(out / "scored.parquet")
    # per condition x domain summaries with bootstrap CIs
    rows = []
    metrics = ["s_cog", "s_conv", "s_all"] + [f"{b}__rate" for b in BEH] + \
              ["n_new_tokens", "completed", "has_answer", "math_success", "math_success_all", "hard_loop", "soft_loop", "n_valid_segments"]
    for (cond, dom), g in df.group_by(["condition", "task_type"]):
        r = {"condition": cond, "task_type": dom, "n": g.height}
        for m in metrics:
            if m not in g.columns: continue
            x = g[m].cast(pl.Float64, strict=False).to_numpy().astype(float)
            mean, lo, hi = bootstrap_mean(x)
            r[m] = mean; r[f"{m}_lo"] = lo; r[f"{m}_hi"] = hi
        rows.append(r)
    summ = pl.DataFrame(rows).sort(["condition", "task_type"]); summ.write_csv(out / "summary.csv")
    # Δγ = E[s|idea] - E[s|math] per condition, with paired-by-instance bootstrap vs baseline
    conds = sorted(df["condition"].unique().to_list())
    crows = []
    rng = np.random.default_rng(0)
    for fam in ["s_cog", "s_conv", "s_all"]:
        base = df.filter(pl.col("condition") == "baseline")
        for c in conds:
            g = df.filter(pl.col("condition") == c)
            mi = g.filter(pl.col("task_type") == "idea")[fam].to_numpy(); mm = g.filter(pl.col("task_type") == "math")[fam].to_numpy()
            delta = mi.mean() - mm.mean()
            bi = base.filter(pl.col("task_type") == "idea")[fam].to_numpy(); bm = base.filter(pl.col("task_type") == "math")[fam].to_numpy()
            d1 = bi.mean() - bm.mean()
            boots = []
            for _ in range(2000):
                di = mi[rng.integers(0, len(mi), len(mi))].mean() - mm[rng.integers(0, len(mm), len(mm))].mean()
                boots.append(di)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            crows.append({"score": fam, "condition": c, "delta": float(delta), "delta_lo": float(lo), "delta_hi": float(hi),
                          "delta_baseline": float(d1), "attenuation": float(1 - delta / d1) if d1 else np.nan,
                          "idea_mean": float(mi.mean()), "math_mean": float(mm.mean())})
    con = pl.DataFrame(crows); con.write_csv(out / "contrasts.csv")
    print(con)
    # paired shift vs in-run baseline, per condition x domain x score (matched on instance_id + seed)
    srows = []
    base = df.filter(pl.col("condition") == "baseline")
    if base.height:
        for fam in ["s_cog", "s_conv", "s_all", "backtracking__rate", "verification__rate", "Perspective_Shift__rate",
                    "Conflict_of_Perspectives__rate", "n_new_tokens", "has_answer", "math_success_all"]:
            for c in conds:
                if c == "baseline":
                    continue
                g = df.filter(pl.col("condition") == c).select(["instance_id", "seed", "task_type", pl.col(fam).cast(pl.Float64, strict=False)])
                j = g.join(base.select(["instance_id", "seed", pl.col(fam).cast(pl.Float64, strict=False).alias("b")]), on=["instance_id", "seed"], how="inner")
                for dom in ["math", "idea"]:
                    dvals = (j.filter(pl.col("task_type") == dom)[fam] - j.filter(pl.col("task_type") == dom)["b"]).to_numpy().astype(float)
                    m, lo, hi = bootstrap_mean(dvals, seed=1)
                    srows.append({"metric": fam, "condition": c, "task_type": dom, "n_pairs": int(np.isfinite(dvals).sum()),
                                  "paired_shift": m, "shift_lo": lo, "shift_hi": hi,
                                  "baseline_mean": float(np.nanmean(j.filter(pl.col("task_type") == dom)["b"].to_numpy().astype(float)))})
        sh = pl.DataFrame(srows); sh.write_csv(out / "paired_shifts.csv")
        print(sh.filter(pl.col("metric").is_in(["s_cog", "s_conv", "s_all"])))
        # robustness: the same paired shifts restricted to pairs where neither trace carries a loop flag
        clean = df.filter(~(pl.col("hard_loop").fill_null(False) | pl.col("soft_loop").fill_null(False)))
        cb = clean.filter(pl.col("condition") == "baseline")
        crows = []
        for fam in ["s_cog", "s_conv", "s_all", "backtracking__rate", "verification__rate", "Perspective_Shift__rate", "Conflict_of_Perspectives__rate"]:
            for c in conds:
                if c == "baseline":
                    continue
                g = clean.filter(pl.col("condition") == c).select(["instance_id", "seed", "task_type", pl.col(fam).cast(pl.Float64, strict=False)])
                j = g.join(cb.select(["instance_id", "seed", pl.col(fam).cast(pl.Float64, strict=False).alias("b")]), on=["instance_id", "seed"], how="inner")
                for dom in ["math", "idea"]:
                    dvals = (j.filter(pl.col("task_type") == dom)[fam] - j.filter(pl.col("task_type") == dom)["b"]).to_numpy().astype(float)
                    m, lo, hi = bootstrap_mean(dvals, seed=1)
                    crows.append({"metric": fam, "condition": c, "task_type": dom, "n_pairs": int(np.isfinite(dvals).sum()),
                                  "paired_shift": m, "shift_lo": lo, "shift_hi": hi})
        pl.DataFrame(crows).write_csv(out / "paired_shifts_noloop.csv")
    # early/late phase profile (schedule runs): split every trace at the primary per-task switch point,
    # half of the paired unsteered generation length clipped to [512, 6144], so all arms share the split
    if "switch_at" in tr.columns and tr["switch_at"].drop_nulls().len() > 0:
        try:
            bl = tr.filter(pl.col("condition") == "baseline").select(["instance_id", "seed", "n_new_tokens"])
            bmap = {(i, int(sd)): int(min(max(round(0.5 * n), 512), 6144)) for i, sd, n in zip(bl["instance_id"].to_list(), bl["seed"].to_list(), bl["n_new_tokens"].to_list())}
            split = {t: bmap[(i, int(sd))] for t, i, sd in zip(tr["trace_id"].to_list(), tr["instance_id"].to_list(), tr["seed"].to_list()) if (i, int(sd)) in bmap}
            pa = phase_amounts(tr, tb, split, window=window, tok=tok)
            if pa.height:
                ps = score_with_frozen(pa, frozen).join(tr.select(["trace_id", "condition", "instance_id", "task_type", "seed", "hard_loop", "soft_loop"]), on="trace_id", how="inner")
                ps.write_parquet(out / "phase_scored.parquet")
                prow = []
                for (cond, dom, ph), g in ps.group_by(["condition", "task_type", "phase"]):
                    r = {"condition": cond, "task_type": dom, "phase": ph, "n": g.height}
                    for m in ["s_cog", "s_conv", "s_all"] + [f"{b}__rate" for b in BEH]:
                        mean, lo, hi = bootstrap_mean(g[m].to_numpy().astype(float)); r[m] = mean; r[f"{m}_lo"] = lo; r[f"{m}_hi"] = hi
                    prow.append(r)
                pl.DataFrame(prow).sort(["condition", "task_type", "phase"]).write_csv(out / "phase_summary.csv")
                # paired vs baseline within phase
                pb = ps.filter(pl.col("condition") == "baseline")
                sr = []
                for fam in ["s_all", "backtracking__rate", "verification__rate", "Perspective_Shift__rate", "Conflict_of_Perspectives__rate"]:
                    for c in sorted(ps["condition"].unique().to_list()):
                        if c == "baseline": continue
                        for ph in ["early", "late"]:
                            g = ps.filter((pl.col("condition") == c) & (pl.col("phase") == ph)).select(["instance_id", "seed", "task_type", fam])
                            j = g.join(pb.filter(pl.col("phase") == ph).select(["instance_id", "seed", pl.col(fam).alias("b")]), on=["instance_id", "seed"], how="inner")
                            for dom in ["math", "idea"]:
                                dv = (j.filter(pl.col("task_type") == dom)[fam] - j.filter(pl.col("task_type") == dom)["b"]).to_numpy().astype(float)
                                m, lo, hi = bootstrap_mean(dv, seed=1)
                                sr.append({"metric": fam, "condition": c, "task_type": dom, "phase": ph, "n_pairs": int(np.isfinite(dv).sum()), "paired_shift": m, "shift_lo": lo, "shift_hi": hi})
                pl.DataFrame(sr).write_csv(out / "phase_paired_shifts.csv")
                print("phase analysis ->", out / "phase_paired_shifts.csv")
        except Exception as e:
            print("phase analysis failed:", e)
    # central figure: math / idea score by condition
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, fam in zip(axes, ["s_cog", "s_conv", "s_all"]):
            s = summ.sort("condition")
            for dom, col in (("math", "#1f77b4"), ("idea", "#d62728")):
                ss = s.filter(pl.col("task_type") == dom)
                x = np.arange(ss.height)
                ax.errorbar(x, ss[fam].to_numpy(), yerr=[ss[fam].to_numpy() - ss[f"{fam}_lo"].to_numpy(), ss[f"{fam}_hi"].to_numpy() - ss[fam].to_numpy()],
                            fmt="o-", color=col, label=dom, capsize=3)
                ax.set_xticks(x); ax.set_xticklabels(ss["condition"].to_list(), rotation=45, ha="right")
            ax.set_title(f"{fam} (frozen, + = Idea-like)"); ax.axhline(0, color="k", lw=0.5); ax.legend()
        fig.tight_layout(); fig.savefig(out / "figures/mode_score_by_condition.png", dpi=150)
        fig, axes = plt.subplots(2, 4, figsize=(18, 7))
        for ax, b in zip(axes.ravel(), BEH):
            s = summ.sort("condition")
            for dom, col in (("math", "#1f77b4"), ("idea", "#d62728")):
                ss = s.filter(pl.col("task_type") == dom); x = np.arange(ss.height)
                ax.errorbar(x, ss[f"{b}__rate"].to_numpy(), yerr=[ss[f"{b}__rate"].to_numpy() - ss[f"{b}__rate_lo"].to_numpy(), ss[f"{b}__rate_hi"].to_numpy() - ss[f"{b}__rate"].to_numpy()],
                            fmt="o-", color=col, label=dom, capsize=3)
                ax.set_xticks(x); ax.set_xticklabels(ss["condition"].to_list(), rotation=45, ha="right", fontsize=7)
            ax.set_title(b)
        axes[0, 0].legend(); fig.tight_layout(); fig.savefig(out / "figures/behavior_rates_by_condition.png", dpi=150)
    except Exception as e:
        print("plotting failed:", e)
    print("->", out)


if __name__ == "__main__":
    main()
