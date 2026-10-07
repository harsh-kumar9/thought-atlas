"""P2 step 1 (D-42): which parts of the reasoning signature go with success, within task, in each domain.

Data: the unsteered corpus runs (corpus1_<domain>_{a,b}: 200 tasks x 4 seeds, 12k budget, Track B labels over the
8k window). Per trace: the 8 whole-trace behavior rates (as in archetypes.py), answered, correct (math: repo grader,
bold stripped; gpqa/planning: repo MCQ grader). Within-task analysis: rates and outcomes are demeaned within task
(task fixed effects), so a coefficient says what distinguishes a successful sample from an unsuccessful sample of the
same problem. Only tasks whose outcome varies across seeds contribute.

Per domain and outcome (correct, answered, correct among answered):
  - univariate within-task correlation of each behavior with the outcome (task-bootstrap CI)
  - multivariate within-task OLS on the 8 rates -> beta (outcome per unit rate)
  - prediction for a steering shift d (8 rates): beta . d, with task-bootstrap CI. Shifts used: the Math A2 open-loop
    shift and the rank-one sustained check shift measured on the 199 held-out Math tasks (sigpilot3_math)
  - the archetype of that domain whose (phi - centroid) has the highest predicted gain (the P2 own-domain arm)
Output: data/sigctrl/success_dir.csv, success_pred.csv, success_dir.json.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from .common import BEH, DATA, JUDGE_MODEL
from .archetypes import whole_signature
from .budget import grade

SIG = DATA / "sigctrl"
DOMAINS = ["math", "gpqa", "planning"]
SHIFT_RUN, SHIFTS = "sigpilot3_math", {"math_A2_open": "ol_math_A2_g3p0", "rank1_check": "sus_math_L19"}


def load_domain(dom: str) -> pl.DataFrame:
    trs, tbs = [], []
    tag = JUDGE_MODEL.replace("/", "_")
    for s in "ab":
        rd = DATA / "runs" / f"corpus1_{dom}_{s}"
        trs.append(pl.concat([pl.read_parquet(p) for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed"))
        tbs.append(pl.concat([pl.read_parquet(p) for p in sorted((rd / "judge").rglob(f"trackB_full__{tag}*.parquet"))], how="diagonal_relaxed"))
    tr = pl.concat(trs, how="diagonal_relaxed"); tb = pl.concat(tbs, how="diagonal_relaxed").unique(["trace_id", "seg_idx"], keep="last")
    sig = whole_signature(tb).filter(pl.col("n_valid") >= 4)
    tr = tr.with_columns(pl.Series("answered", [int(bool((a or "").strip())) for a in tr["answer_text"].to_list()]),
                         pl.Series("correct", [grade(a or "", r, t, p) for a, r, t, p in zip(tr["answer_text"].to_list(), tr["reference_answer"].to_list(),
                                                                                         tr["task_type"].to_list(), tr["prompt"].to_list())]))
    return tr.select(["trace_id", "instance_id", "task_type", "seed", "answered", "correct", "n_new_tokens"]).join(sig, on="trace_id", how="inner")


def demean(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.with_columns([(pl.col(c) - pl.col(c).mean().over("instance_id")).alias(c) for c in cols])


def fit(df: pl.DataFrame, y: str, X: list[str]):
    g = df.filter(pl.col(y).is_not_null())
    g = g.filter(pl.col(y).std().over("instance_id") > 0)                     # tasks whose outcome varies
    d = demean(g, X + [y])
    Xm, ym = d.select(X).to_numpy(), d[y].to_numpy().astype(float)
    uni = np.array([np.corrcoef(Xm[:, k], ym)[0, 1] if Xm[:, k].std() > 0 else np.nan for k in range(Xm.shape[1])])
    beta = np.linalg.lstsq(Xm, ym, rcond=None)[0]
    return g, uni, beta


def main():
    sc = pl.read_parquet(DATA / "runs" / SHIFT_RUN / "analysis" / "scored.parquet")
    R = [f"{b}__rate" for b in BEH]
    base = sc.filter(pl.col("condition") == "baseline").select(["instance_id", "seed"] + R)
    shifts = {}
    for k, c in SHIFTS.items():                                          # paired mean shift of the 8 rates vs unsteered
        j = sc.filter(pl.col("condition") == c).select(["instance_id", "seed"] + R).join(base, on=["instance_id", "seed"], suffix="_b")
        shifts[k] = np.array([float(np.nanmean((j[r] - j[f"{r}_b"]).to_numpy())) for r in R])
    arch = json.loads((SIG / "archetypes.json").read_text())
    X = [f"all_{b}" for b in BEH]
    rows, preds, summary = [], [], {"shifts": {k: dict(zip(BEH, v.round(4).tolist())) for k, v in shifts.items()}}
    rng = np.random.default_rng(0)
    for dom in DOMAINS:
        df = load_domain(dom)
        df = df.with_columns(pl.when(pl.col("answered") == 1).then(pl.col("correct")).otherwise(None).alias("correct_if_answered"))
        summary[dom] = {"n_traces": df.height, "n_tasks": df["instance_id"].n_unique(), "accuracy": float(df["correct"].mean()),
                        "answered": float(df["answered"].mean()), "mean_rates": dict(zip(BEH, df.select(X).mean().row(0)))}
        cand = {k: np.array(arch[k]["phi"][:8]) - np.array(arch[f"{dom}_centroid"]["phi"][:8]) for k in arch if k.startswith(f"{dom}_A")}
        for y in ("correct", "answered", "correct_if_answered"):
            g, uni, beta = fit(df, y, X)
            tasks = g["instance_id"].unique().to_list()
            bs_uni, bs_pred = [], {k: [] for k in list(shifts) + list(cand)}
            for _ in range(500):
                pick = pl.DataFrame({"instance_id": [tasks[i] for i in rng.integers(0, len(tasks), len(tasks))]}).with_row_index("rep")
                gb = pick.join(g, on="instance_id").with_columns((pl.col("instance_id") + "#" + pl.col("rep").cast(pl.Utf8)).alias("instance_id"))
                _, u, b = fit(gb, y, X)
                bs_uni.append(u)
                for k, d in {**shifts, **cand}.items():
                    bs_pred[k].append(float(b @ d))
            bs_uni = np.array(bs_uni)
            for k, b in enumerate(BEH):
                rows.append({"domain": dom, "outcome": y, "behavior": b, "n_tasks_varying": len(tasks), "within_task_corr": float(uni[k]),
                             "corr_lo": float(np.nanquantile(bs_uni[:, k], 0.025)), "corr_hi": float(np.nanquantile(bs_uni[:, k], 0.975)),
                             "beta": float(beta[k])})
            for k, d in {**shifts, **cand}.items():
                v = np.array(bs_pred[k])
                preds.append({"domain": dom, "outcome": y, "shift": k, "pred": float(beta @ d), "pred_lo": float(np.quantile(v, 0.025)),
                              "pred_hi": float(np.quantile(v, 0.975))})
        best = max(cand, key=lambda k: [p for p in preds if p["domain"] == dom and p["outcome"] == "correct" and p["shift"] == k][0]["pred"])
        summary[dom]["own_archetype"] = best
    pl.DataFrame(rows).write_csv(SIG / "success_dir.csv"); pl.DataFrame(preds).write_csv(SIG / "success_pred.csv")
    (SIG / "success_dir.json").write_text(json.dumps(summary, indent=1))
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(200)
    print(json.dumps({d: {k: v for k, v in summary[d].items() if k != "mean_rates"} for d in DOMAINS}, indent=1))
    print(pl.DataFrame(rows).filter(pl.col("outcome") == "correct").select(["domain", "behavior", "n_tasks_varying", "within_task_corr", "corr_lo", "corr_hi"]))
    print(pl.DataFrame(preds))


if __name__ == "__main__":
    main()
