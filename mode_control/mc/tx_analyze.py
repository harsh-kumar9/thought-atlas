"""Analysis for the transplant matrix (D-52), the Countdown crossover (D-51) and F3 beyond Math (D-53).

Targets and where their arms live (the p1/p2 runs share task sets and sampling seeds with the tx runs, so their
unsteered, A2, own-archetype and random arms are reused; vsteer's run fingerprint check guarantees the pairing):
  math -> tx_math + p1_budget    gpqa -> tx_gpqa + p2_gpqa    planning -> tx_planning + p2_planning
  idea -> tx_idea                moral -> tx_moral            countdown -> cd_countdown, cd_math + p1_budget

Subcommands
  stage_judge <run>   seed-1 copies of every arm -> runs/<run>/judge_in_s1/traces_<cond>.parquet (Track B input)
  stage_extract       math traces of the tx/cd runs -> data/extract_in_tx (free-text extraction input, D-47)
  shifts              per-trace behaviour rates over the 8k window (Track B, seed 1) and paired shifts vs unsteered,
                      plus answered / loops / tokens over both seeds -> data/transplant/shifts.csv
  prereg_observed     H2a prediction (ii): each target's slopes (prereg.json) times the measured shift, written with its
                      sha256 before any transplant outcome is read -> data/transplant/prereg_observed.json
  outcomes            outcome change per arm vs unsteered: accuracy at 12k and 32k (budget.py outputs) for math/gpqa/
                      planning; quality for idea/moral (answered traces, and unanswered as 0) -> data/transplant/outcomes.csv
  h2a                 the pre-registered tests -> data/transplant/h2a.json
  crossover           D-51 contrast (conv_up - A2) on Countdown minus on Math, at each budget -> data/transplant/crossover.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import polars as pl

from .common import BEH, DATA, JUDGE_MODEL

TX = DATA / "transplant"
TARGETS = {"math": ("tx_math", "p1_budget", "p4_eval3"), "gpqa": ("tx_gpqa", "p2_gpqa"), "planning": ("tx_planning", "p2_planning"),
           "idea": ("tx_idea",), "moral": ("tx_moral",)}
DOMS = list(TARGETS)
TAG = JUDGE_MODEL.replace("/", "_")
WINDOW = 8192


def _boot(x: np.ndarray, reps: int = 2000, seed: int = 0):
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)])
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def run_traces(run: str) -> pl.DataFrame:
    rd = DATA / "runs" / run
    return pl.concat([pl.read_parquet(p) for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed").with_columns(pl.lit(run).alias("run"))


SHIFT_TARGETS = TARGETS | {"countdown": ("cd_countdown",), "math_conv": ("cd_math", "p1_budget")}


def target_traces(dom: str) -> pl.DataFrame:
    """The target's own run plus every borrowed run (same run fingerprint, so arms are paired by sampling seed); a condition
    present in an earlier run is not taken again from a later one."""
    own, *others = SHIFT_TARGETS[dom]
    tr = run_traces(own) if any((DATA / "runs" / own).glob("traces_*.parquet")) else None
    for other in others:
        ob = run_traces(other)
        if tr is not None:
            assert set(tr["run_fingerprint"].unique()) == set(ob["run_fingerprint"].unique()), f"{dom}: {own} and {other} are not paired"
            ob = ob.filter(~pl.col("condition").is_in(tr["condition"].unique().to_list()))
        tr = ob if tr is None else pl.concat([tr, ob], how="diagonal_relaxed")
    return tr


def trackb(runs: list[str]) -> pl.DataFrame:
    paths = [p for r in runs for p in sorted((DATA / "runs" / r / "judge").rglob(f"trackB_full__{TAG}*.parquet"))]
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed").unique(["trace_id", "seg_idx"], keep="last")


# ---------------------------------------------------------------- staging
def cmd_stage_judge(run: str):
    rd = DATA / "runs" / run; out = rd / "judge_in_s1"; out.mkdir(exist_ok=True)
    done = set()
    tb_paths = sorted((rd / "judge").rglob(f"trackB_full__{TAG}*.parquet"))
    if tb_paths:
        done = set(pl.concat([pl.read_parquet(p, columns=["trace_id"]) for p in tb_paths])["trace_id"].unique().to_list())
    for p in sorted(rd.glob("traces_*.parquet")):
        t = pl.read_parquet(p).filter(pl.col("seed") == 1)
        todo = t.filter(~pl.col("trace_id").is_in(list(done)))
        if todo.height:
            todo.write_parquet(out / p.name)                 # only what still needs labels
        elif (out / p.name).exists():
            (out / p.name).unlink()
        print(f"[stage_judge] {run}/{p.stem}: {t.height} seed-1 traces, {todo.height} not yet labelled")


def cmd_stage_extract():
    out = DATA / "extract_in_tx"; out.mkdir(exist_ok=True)
    for run in ("tx_math", "cd_math"):
        for p in sorted((DATA / "runs" / run).glob("traces_*.parquet")):
            link = out / f"traces_{run}__{p.stem.removeprefix('traces_')}.parquet"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(p)
            print("[stage_extract]", link.name)


# ---------------------------------------------------------------- behaviour shifts
def rates(tr: pl.DataFrame, tb: pl.DataFrame) -> pl.DataFrame:
    from .common import load_tokenizer
    from .phenotype import windowed_amounts
    amt = windowed_amounts(tr.filter(pl.col("trace_id").is_in(tb["trace_id"].unique().to_list())), tb, window=WINDOW, tok=load_tokenizer())
    return amt.select(["trace_id", "n_valid_segments"] + [f"{b}__rate" for b in BEH])


def cmd_shifts():
    rows = []
    for dom in SHIFT_TARGETS:
        try:
            tr = target_traces(dom)
        except ValueError:                                   # no trace files yet
            continue
        if tr is None:
            continue
        tb = trackb(list(SHIFT_TARGETS[dom]))
        rt = rates(tr, tb)
        tr = tr.with_columns((pl.col("answer_text").fill_null("").str.len_chars() > 0).alias("answered"),
                             (pl.col("hard_loop").fill_null(False) | pl.col("soft_loop").fill_null(False)).alias("loop"))
        d = tr.select(["trace_id", "condition", "instance_id", "seed", "answered", "loop", "n_new_tokens"]).join(rt, on="trace_id", how="left")
        base = d.filter(pl.col("condition") == "baseline")
        for c in sorted(set(d["condition"].unique()) - {"baseline"}):
            g = d.filter(pl.col("condition") == c).join(base, on=["instance_id", "seed"], suffix="_b")
            rec = {"target": dom, "arm": c, "n_pairs_all": g.height}
            for m in ("answered", "loop", "n_new_tokens"):
                rec[f"{m}"] = float(g[m].cast(pl.Float64).mean()); rec[f"{m}_base"] = float(g[f"{m}_b"].cast(pl.Float64).mean())
            s1 = g.filter((pl.col("seed") == 1) & pl.col("n_valid_segments").is_not_null() & pl.col("n_valid_segments_b").is_not_null())
            rec["n_pairs_trackB"] = s1.height
            shift = []
            for b in BEH:
                x = (s1[f"{b}__rate"] - s1[f"{b}__rate_b"]).to_numpy().astype(float) * 100
                m, lo, hi = _boot(x)
                rec[f"d_{b}"], rec[f"d_{b}_lo"], rec[f"d_{b}_hi"] = m, lo, hi; shift.append(m)
                rec[f"base_{b}"] = float(100 * s1[f"{b}__rate_b"].mean()) if s1.height else np.nan
            rows.append(rec)
        print(f"[shifts] {dom}: {len([r for r in rows if r['target'] == dom])} arms")
    df = pl.DataFrame(rows); TX.mkdir(exist_ok=True); df.write_csv(TX / "shifts.csv")
    # cosine of each arm's shift with A2's shift on the same target (and with the same vector's math shift)
    sv = {(r["target"], r["arm"]): np.array([r[f"d_{b}"] for b in BEH]) for r in rows}
    cos = []
    for (t, a), v in sv.items():
        ref = sv.get((t, "ol_math_A2"))
        cos.append({"target": t, "arm": a, "size": float(np.linalg.norm(v)),
                    "cos_with_A2_same_target": float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-9)) if ref is not None else None,
                    "cos_with_same_arm_on_math": float(v @ sv[("math", a)] / (np.linalg.norm(v) * np.linalg.norm(sv[("math", a)]) + 1e-9)) if ("math", a) in sv else None})
    pl.DataFrame(cos).write_csv(TX / "shift_cosines.csv")
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(240)
    print(df.select(["target", "arm", "n_pairs_trackB", "loop", "loop_base", "answered", "answered_base"] + [f"d_{b}" for b in BEH]))


def cmd_prereg_observed():
    pre = json.loads((TX / "prereg.json").read_text())
    sh = pl.read_csv(TX / "shifts.csv")
    from .transplant import OUTCOMES
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "decision": "D-52 H2a prediction (ii)",
           "prereg_sha256": (TX / "prereg.sha256").read_text().strip(), "cells": []}
    for r in sh.filter(pl.col("target").is_in(DOMS)).iter_rows(named=True):
        d = np.array([r[f"d_{b}"] for b in BEH]) / 100
        for y in OUTCOMES[r["target"]]:
            beta = np.array([pre["domains"][r["target"]][y]["beta"][b] for b in BEH])
            sd = pre["domains"][r["target"]][y]["outcome_sd"]
            out["cells"].append({"target": r["target"], "outcome": y, "arm": r["arm"], "pred_observed": float(beta @ d),
                                 "pred_observed_sd_units": float(beta @ d / sd) if sd else None, "n_pairs_trackB": r["n_pairs_trackB"]})
    blob = json.dumps(out, indent=1, sort_keys=True); (TX / "prereg_observed.json").write_text(blob)
    h = hashlib.sha256(blob.encode()).hexdigest(); (TX / "prereg_observed.sha256").write_text(h + "\n")
    print(f"[prereg_observed] {len(out['cells'])} cells -> {TX / 'prereg_observed.json'} sha256 {h}")


# ---------------------------------------------------------------- outcomes
def quality_table(dom: str) -> pl.DataFrame:
    run = TARGETS[dom][0]
    q = pl.concat([pl.read_parquet(p) for p in sorted((DATA / "runs" / run / "judge").rglob(f"quality__{TAG}*.parquet"))], how="diagonal_relaxed").unique("trace_id", keep="last")
    cols = {"quality_score": "quality"} | ({"axis_originality": "originality", "axis_feasibility": "feasibility"} if dom == "idea" else {})
    tr = target_traces(dom).with_columns((pl.col("answer_text").fill_null("").str.len_chars() > 0).alias("answered"))
    return tr.select(["trace_id", "condition", "instance_id", "seed", "answered"]).join(q.select(["trace_id"] + list(cols)).rename(cols), on="trace_id", how="left")


def cmd_outcomes():
    from .transplant import OUTCOMES
    rows = []
    for dom in ("math", "gpqa", "planning"):
        p = DATA / "runs" / TARGETS[dom][0] / "analysis" / "budget_paired.csv"
        if not p.exists():
            print(f"[outcomes] {dom}: run budget.py on {TARGETS[dom][0]} with --borrow first"); continue
        b = pl.read_csv(p).filter((pl.col("reference") == "baseline") & pl.col("budget").is_in([12288, 32768]))
        for r in b.iter_rows(named=True):
            rows.append({"target": dom, "outcome": "correct", "budget": r["budget"], "arm": r["condition"], "n_tasks": r["n_tasks"],
                         "base": r["correct_ref"], "arm_mean": r["correct"], "delta": r["d_correct"], "lo": r["d_correct_lo"], "hi": r["d_correct_hi"],
                         "d_answered": r["d_answered"], "d_tokens": r["d_tokens"]})
    for dom in ("idea", "moral"):
        try:
            q = quality_table(dom)
        except ValueError:
            print(f"[outcomes] {dom}: no quality judge output yet"); continue
        for y in OUTCOMES[dom]:
            for mode in ("answered", "unanswered_as_zero"):
                qq = q.with_columns(pl.when(pl.col("answered")).then(pl.col(y)).otherwise(0.0 if mode == "unanswered_as_zero" else None).alias("v"))
                task = qq.group_by(["condition", "instance_id"]).agg(pl.col("v").mean())
                base = task.filter(pl.col("condition") == "baseline").drop("condition")
                for c in sorted(set(task["condition"].unique()) - {"baseline"}):
                    j = task.filter(pl.col("condition") == c).join(base, on="instance_id", suffix="_b").drop_nulls()
                    m, lo, hi = _boot((j["v"] - j["v_b"]).to_numpy().astype(float))
                    rows.append({"target": dom, "outcome": y if mode == "answered" else f"{y}_with_unanswered_0", "budget": 32768, "arm": c,
                                 "n_tasks": j.height, "base": float(j["v_b"].mean()), "arm_mean": float(j["v"].mean()), "delta": m, "lo": lo, "hi": hi})
    df = pl.DataFrame(rows); df.write_csv(TX / "outcomes.csv")
    pl.Config.set_tbl_rows(300); pl.Config.set_tbl_width_chars(220); print(df)


def cmd_h2a():
    """Pre-registered tests (D-52): sign agreement over the off-diagonal signature cells, within-target Spearman of predicted
    vs observed outcome change averaged over targets, pooled correlation in outcome-SD units. Primary outcome per target:
    accuracy at 12k (math, gpqa, planning), quality among answered (idea: the originality/feasibility mean; moral: rubric)."""
    from scipy.stats import binomtest, spearmanr
    pre = json.loads((TX / "prereg.json").read_text()); obs = json.loads((TX / "prereg_observed.json").read_text())
    oc = pl.read_csv(TX / "outcomes.csv")
    prim = {"math": ("correct", 12288), "gpqa": ("correct", 12288), "planning": ("correct", 12288), "idea": ("quality", 32768), "moral": ("quality", 32768)}
    pobs = {(c["target"], c["outcome"], c["arm"]): c["pred_observed"] for c in obs["cells"]}
    cells = []
    for dom, (y, bud) in prim.items():
        o = oc.filter((pl.col("target") == dom) & (pl.col("outcome") == y) & (pl.col("budget") == bud))
        sd = pre["domains"][dom][y]["outcome_sd"]
        for r in o.iter_rows(named=True):
            a = r["arm"]
            pi = pre["domains"][dom][y]["pred_implied"].get(a.replace("sealdir_matched", "sealdir_L17_n6p75").replace("rl_zeroT_s60", "rl_zero8k_T_s60"), {}).get("pred")
            cells.append({"target": dom, "arm": a, "source": a[4:] if a.startswith("sig_") and "_x" not in a else None, "observed": r["delta"],
                          "observed_sd": r["delta"] / sd if sd else None, "pred_implied": pi, "pred_observed": pobs.get((dom, y, a)),
                          "pred_implied_sd": pi / sd if (pi is not None and sd) else None})
    df = pl.DataFrame(cells); df.write_csv(TX / "h2a_cells.csv")
    res = {}
    off = df.filter(pl.col("source").is_not_null() & (pl.col("source") != pl.col("target")))
    for k in ("pred_implied", "pred_observed"):
        o = off.drop_nulls([k, "observed"])
        agree = int((np.sign(o[k].to_numpy()) == np.sign(o["observed"].to_numpy())).sum())
        rho = [spearmanr(g[k], g["observed"]).statistic for _, g in o.group_by("target") if g.height >= 3]
        x = o.drop_nulls([f"{k}_sd" if k == "pred_implied" else k]).to_pandas()
        res[k] = {"n_cells": o.height, "sign_agree": agree, "sign_p": float(binomtest(agree, o.height, 0.5, alternative="greater").pvalue) if o.height else None,
                  "mean_within_target_spearman": float(np.nanmean(rho)) if rho else None, "per_target_spearman": [float(r) for r in rho]}
    df_all = df.drop_nulls(["observed_sd", "pred_implied_sd"])
    res["pooled_corr_sd_units_implied"] = float(np.corrcoef(df_all["pred_implied_sd"], df_all["observed_sd"])[0, 1]) if df_all.height > 2 else None
    (TX / "h2a.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    pl.Config.set_tbl_rows(200); print(df.sort(["target", "arm"]))


def cmd_crossover(reps: int = 4000):
    def task_means(run, borrow=None):
        p = DATA / "runs" / run / "analysis" / "budget_outcomes.parquet"
        return pl.read_parquet(p).group_by(["condition", "budget", "instance_id"]).agg(pl.col("correct").mean())
    cd, cm = task_means("cd_countdown"), task_means("cd_math")
    rng = np.random.default_rng(0); rows = []
    for bud in sorted(cd["budget"].unique()):
        diff = {}
        for name, t in (("countdown", cd), ("math", cm)):
            w = t.filter(pl.col("budget") == bud).pivot(on="condition", index="instance_id", values="correct")
            need = {"conv_up", "ol_math_A2", "baseline", "conv_down"}
            if not need <= set(w.columns):
                diff = None; break
            diff[name] = {k: w[k].to_numpy() - w["baseline"].to_numpy() for k in ("conv_up", "conv_down", "ol_math_A2")}
        if diff is None:
            continue
        c = diff["countdown"]["conv_up"] - diff["countdown"]["ol_math_A2"]; m = diff["math"]["conv_up"] - diff["math"]["ol_math_A2"]
        bs = [c[rng.integers(0, len(c), len(c))].mean() - m[rng.integers(0, len(m), len(m))].mean() for _ in range(reps)]
        rows.append({"budget": bud, **{f"{dn}_{k}": float(v.mean()) for dn, dd in diff.items() for k, v in dd.items()},
                     "crossover": float(c.mean() - m.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))})
    df = pl.DataFrame(rows); df.write_csv(TX / "crossover.csv"); pl.Config.set_tbl_width_chars(240); print(df)


VEC_OF = {"ol_math_A2": "ol_math_A2", "ol_gpqa_A4": "ol_gpqa_A4", "ol_planning_A2": "ol_planning_A2", "ol_random": "ol_random",
          "sealdir_matched": "sealdir_L17_n6p75", "rl_zeroT_s60": "rl_zero8k_T_s60", "idea_commit": "idea_commit",
          **{f"sig_{d}": f"sig_{d}" for d in DOMS}}


def cmd_f3():
    """D-53: over every vector arm on math, gpqa and planning, does the accuracy change at 12k follow the behaviour shift or
    the residual direction? x_beh = the target's pre-registered slopes times the measured shift (prediction ii);
    x_res = residual-space cosine of the vector with A2; x_proj = the shift projected on A2's shift on the same target."""
    from .transplant import load_vec, unit
    obs = {(c["target"], c["arm"]): c["pred_observed"] for c in json.loads((TX / "prereg_observed.json").read_text())["cells"] if c["outcome"] == "correct"}
    sh = pl.read_csv(TX / "shifts.csv"); oc = pl.read_csv(TX / "outcomes.csv").filter((pl.col("outcome") == "correct") & (pl.col("budget") == 12288))
    a2 = unit(load_vec("ol_math_A2")); rows = []
    for r in oc.iter_rows(named=True):
        if r["arm"] not in VEC_OF:
            continue
        s = sh.filter((pl.col("target") == r["target"]) & (pl.col("arm") == r["arm"]))
        sa = sh.filter((pl.col("target") == r["target"]) & (pl.col("arm") == "ol_math_A2"))
        if not s.height or not sa.height:
            continue
        v, va = np.array([s[f"d_{b}"][0] for b in BEH]), np.array([sa[f"d_{b}"][0] for b in BEH])
        rows.append({"target": r["target"], "arm": r["arm"], "d_acc_12k": r["delta"], "x_beh": obs.get((r["target"], r["arm"])),
                     "x_proj": float(v @ va / (va @ va)), "x_res": float(unit(load_vec(VEC_OF[r["arm"]])) @ a2)})
    df = pl.DataFrame(rows).drop_nulls(); df.write_csv(TX / "f3_cells.csv")
    y = df["d_acc_12k"].to_numpy(); res = {"n_cells": df.height}
    for name, cols in {"behaviour (slopes x shift)": ["x_beh"], "shift projected on A2's shift": ["x_proj"], "residual cosine with A2": ["x_res"],
                       "projection + residual cosine": ["x_proj", "x_res"]}.items():
        X = np.column_stack([np.ones(len(y))] + [df[c].to_numpy() for c in cols])
        b = np.linalg.lstsq(X, y, rcond=None)[0]; r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        res[name] = {"coef": dict(zip(["intercept"] + cols, np.round(b, 4).tolist())), "r2": float(r2)}
    (TX / "f3.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); print(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage_judge", "stage_extract", "shifts", "prereg_observed", "outcomes", "h2a", "crossover", "f3"])
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    if a.cmd == "stage_judge":
        cmd_stage_judge(a.run)
    else:
        {"stage_extract": cmd_stage_extract, "shifts": cmd_shifts, "prereg_observed": cmd_prereg_observed, "outcomes": cmd_outcomes,
         "h2a": cmd_h2a, "crossover": cmd_crossover, "f3": cmd_f3}[a.cmd]()


if __name__ == "__main__":
    main()
