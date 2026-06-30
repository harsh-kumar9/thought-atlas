"""src/perf/mechanism.py — mechanisms of success: which deliberation predicts correctness, per domain.

CLAIM CEILING: associative, difficulty-controlled. NOT causal (deliberation is endogenous to
difficulty; we control for it but cannot rule out residual confounding or reverse causation).

Per domain (never pooled — the reasoning effect REVERSES across domains, pooling cancels it):
  success ~ behavior_features + difficulty
3 nested feature sets: counts -> +positional -> +trajectory (does timing/shape add predictive power?).

Outcome: binary (math/gpqa/planning/code) -> logistic; continuous quality (moral/idea) -> OLS.
Conditioned on parsed & completed (extraction/truncation failures are not reasoning failures).

Outputs:
  mechanism_coefs.csv   per-domain, per-feature standardized coef + p (difficulty-adjusted)
  mechanism_model_fit.csv   nested-model fit comparison (does timing add over counts?)
"""
from __future__ import annotations
import warnings
import numpy as np
import polars as pl

warnings.simplefilter("ignore")
KIM = ["Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives", "Reconciliation"]
GAN = ["verification", "backtracking", "subgoal", "backward_chaining"]
BEH = GAN + KIM
COUNT_FEATS = [f"count_{b}" for b in BEH] + ["cog_total", "cnv_total"]
POS_FEATS = [f"{b}_{s}" for b in BEH for s in ("late", "com")]   # early is collinear w/ late; keep late+com
TRAJ_FEATS = ["subgoal_frontload", "qa_frontload", "verify_backload", "reconcile_terminal", "arc_score"]


def _difficulty_numeric(df: pl.DataFrame, domain: str) -> pl.DataFrame:
    """Map difficulty_raw to a numeric control where graded; else a categorical code."""
    if domain == "math":      # levels "Level 1".."Level 5" or 1..5
        d = df["difficulty_raw"].cast(pl.Utf8).str.extract(r"(\d+)").cast(pl.Float64, strict=False)
    elif domain == "code":    # easy/medium/hard
        m = {"easy": 0.0, "medium": 1.0, "hard": 2.0}
        d = df["difficulty_raw"].cast(pl.Utf8).str.to_lowercase().replace(m, default=None).cast(pl.Float64, strict=False)
    else:
        # gpqa/planning/moral/idea: difficulty_raw is categorical (subject/competency/theory).
        # Use a per-category mean-success baseline as the difficulty proxy (computed by caller).
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias("difficulty_num"))
    return df.with_columns(d.alias("difficulty_num"))


def fit_domain(df: pl.DataFrame, domain: str, outcome: str):
    """Fit nested models for one domain. Returns (coef_rows, fit_rows).
    Regularized (L2) logistic to handle correlated behavior features at modest n; OLS for continuous."""
    import statsmodels.api as sm

    df = _difficulty_numeric(df, domain)
    if df["difficulty_num"].null_count() == df.height:
        # fall back to a categorical stratum: difficulty_raw, else meta_stratum (moral/idea)
        strat_col = None
        if "difficulty_raw" in df.columns and df["difficulty_raw"].drop_nulls().len() > 0:
            strat_col = "difficulty_raw"
        elif "meta_stratum" in df.columns and df["meta_stratum"].drop_nulls().len() > 0:
            strat_col = "meta_stratum"
        if strat_col:
            base = df.group_by(strat_col).agg(pl.col(outcome).mean().alias("_b"))
            df = df.join(base, on=strat_col, how="left").with_columns(pl.col("_b").alias("difficulty_num"))
    # if difficulty is still entirely missing (e.g. moral/idea have no graded difficulty),
    # drop the control rather than nan-mask every row out. Note this in the fit row.
    has_difficulty = df["difficulty_num"].drop_nulls().len() >= 30 if "difficulty_num" in df.columns else False

    # FOCUSED feature sets (avoid dumping ~30 correlated cols on n~130 -> non-convergence/overfit).
    # counts: the 8 behavior counts. positional: center-of-mass of the 4 behaviors H2 flagged temporal.
    # trajectory: the 4 canonical arc components + arc_score.
    count_feats = [f"count_{b}" for b in BEH]
    pos_feats = [f"{b}_com" for b in ["verification", "subgoal", "Reconciliation", "Question_and_Answering"]]
    traj_feats = ["subgoal_frontload", "verify_backload", "reconcile_terminal", "arc_score"]
    feature_sets = {"counts": count_feats,
                    "counts+positional": count_feats + pos_feats,
                    "counts+positional+trajectory": count_feats + pos_feats + traj_feats}

    coef_rows, fit_rows = [], []
    y = df[outcome].to_numpy().astype(float)
    binary = set(np.unique(y[~np.isnan(y)])) <= {0.0, 1.0}

    for set_name, feats in feature_sets.items():
        feats = [f for f in feats if f in df.columns]
        cols = feats + (["difficulty_num"] if has_difficulty else [])
        X = df.select(cols).to_numpy().astype(float)
        mask = ~np.isnan(X).any(1) & ~np.isnan(y)
        Xm, ym = X[mask], y[mask]
        if Xm.shape[0] < 30 or np.std(ym) == 0:
            fit_rows.append({"domain": domain, "feature_set": set_name, "n": int(mask.sum()),
                             "note": "insufficient_n_or_no_variance"}); continue
        mu, sd = Xm.mean(0), Xm.std(0); sd[sd == 0] = 1
        Xs = sm.add_constant((Xm - mu) / sd, has_constant="add")
        names = ["const"] + cols
        try:
            if binary:
                # L2-regularized logistic (alpha tuned to n); regularized fit converges where MLE won't
                alpha = max(1.0, Xs.shape[1] / 5.0)
                model = sm.Logit(ym, Xs).fit_regularized(alpha=alpha, disp=0, maxiter=500)
                # refit unpenalized only if well-conditioned (for clean p-values); else keep regularized
                try:
                    m2 = sm.Logit(ym, Xs).fit(disp=0, maxiter=300)
                    if np.all(np.isfinite(m2.bse)) and m2.mle_retvals.get("converged", False):
                        model = m2
                except Exception:
                    pass
                fitm = ("pseudo_R2", getattr(model, "prsquared", float("nan")), "AIC", getattr(model, "aic", float("nan")))
            else:
                model = sm.OLS(ym, Xs).fit()
                fitm = ("R2", model.rsquared, "AIC", model.aic)
            pvals = getattr(model, "pvalues", [np.nan]*len(names))
        except Exception as e:
            fit_rows.append({"domain": domain, "feature_set": set_name, "n": int(mask.sum()),
                             "note": f"fit_error: {type(e).__name__}: {str(e)[:60]}"}); continue
        for nm, b, p in zip(names, model.params, pvals):
            if nm == "const":
                continue
            coef_rows.append({"domain": domain, "feature_set": set_name, "feature": nm,
                              "coef": round(float(b), 4),
                              "p_value": round(float(p), 4) if np.isfinite(p) else None,
                              "sig": "*" if (np.isfinite(p) and p < 0.05) else ""})
        fit_rows.append({"domain": domain, "feature_set": set_name, "n": int(mask.sum()),
                         fitm[0]: round(float(fitm[1]), 4) if np.isfinite(fitm[1]) else None,
                         fitm[2]: round(float(fitm[3]), 1) if np.isfinite(fitm[3]) else None,
                         "outcome": "binary" if binary else "continuous",
                         "difficulty_controlled": bool(has_difficulty)})
    return coef_rows, fit_rows


def run(features: pl.DataFrame, grades: pl.DataFrame, traces_meta: pl.DataFrame, out_dir="data/perf"):
    from pathlib import Path
    import json
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Extract a categorical STRATUM for difficulty control where difficulty_raw is absent:
    #   moral -> dilemma_type ; idea -> keyword's coarse field (first 2 words). These adjust for
    #   "some dilemma types / topics score higher" — the moral/idea analogue of difficulty.
    if "instance_metadata" in traces_meta.columns:
        strata = []
        for r in traces_meta.iter_rows(named=True):
            s = None
            try:
                md = json.loads(r["instance_metadata"])
                if r["task_type"] == "moral":
                    s = md.get("dilemma_type") or md.get("theory")
                elif r["task_type"] == "idea":
                    kw = md.get("keyword", "")
                    s = " ".join(kw.split()[:2]) if kw else None
            except Exception:
                pass
            strata.append(s)
        traces_meta = traces_meta.with_columns(pl.Series("meta_stratum", strata))
    # Grade files may overlap (e.g. success_grades lists moral/idea with null success, quality file
    # supplies their score) -> duplicate trace_ids. Coalesce to ONE row/trace, merging outcomes:
    # prefer a non-null `success`; else use `quality_score`. Collapse metadata by first-non-null.
    if "quality_score" not in grades.columns:
        grades = grades.with_columns(pl.lit(None).cast(pl.Float64).alias("quality_score"))
    if "success" not in grades.columns:
        grades = grades.with_columns(pl.lit(None).cast(pl.Float64).alias("success"))
    agg_exprs = [pl.col("success").drop_nulls().first().alias("success"),
                 pl.col("quality_score").drop_nulls().first().alias("quality_score")]
    for c in ("parsed", "completed", "difficulty_raw", "task_type"):
        if c in grades.columns:
            agg_exprs.append(pl.col(c).drop_nulls().first().alias(c))
    grades = grades.group_by("trace_id").agg(agg_exprs)

    df = (features.join(grades, on="trace_id", how="inner")
                  .join(traces_meta.select([c for c in ["trace_id", "task_type", "gen_model", "meta_stratum"]
                                            if c in traces_meta.columns]), on="trace_id", how="left"))
    # task_type may arrive from both grades and traces; prefer traces' (authoritative)
    if "task_type_right" in df.columns:
        df = df.with_columns(pl.coalesce(["task_type_right", "task_type"]).alias("task_type")).drop("task_type_right")
    df = df.filter(pl.col("gen_model") == "reasoner")
    if "parsed" in df.columns:
        df = df.filter(pl.col("parsed").fill_null(True))
    if "completed" in df.columns:
        df = df.filter(pl.col("completed").fill_null(True))

    all_coef, all_fit = [], []
    for domain in ["math", "gpqa", "planning", "code", "moral", "idea"]:
        sub = df.filter(pl.col("task_type") == domain)
        if sub.height < 30:
            print(f"  {domain}: skip (n={sub.height})"); continue
        # pick the outcome column that actually has values for THIS domain
        outcome = None
        for cand in ("success", "quality_score"):
            if cand in sub.columns and sub[cand].drop_nulls().len() >= 30:
                outcome = cand; break
        if outcome is None:
            print(f"  {domain}: no usable outcome column (success/quality_score all null)"); continue
        # keep only rows with a non-null outcome
        sub = sub.filter(pl.col(outcome).is_not_null())
        c, f = fit_domain(sub, domain, outcome)
        all_coef += c; all_fit += f
        if f:
            best = f[-1]
            r2 = best.get("pseudo_R2", best.get("R2"))
            note = best.get("note", "")
            print(f"  {domain}: n={best.get('n','?')} outcome={best.get('outcome', outcome)} "
                  f"fit_R2={r2} {note}")
        else:
            print(f"  {domain}: no model fit")
    pl.DataFrame(all_coef).write_csv(f"{out_dir}/mechanism_coefs.csv")
    pl.DataFrame(all_fit).write_csv(f"{out_dir}/mechanism_model_fit.csv")
    # interpretation guard: these are difficulty-controlled ASSOCIATIONS, not causal effects.
    # negative behavior coefficients most plausibly mark within-problem struggle (the model
    # backtracks/re-decomposes more on problems it is failing), not that the behavior causes failure.
    with open(f"{out_dir}/INTERPRETATION.txt", "w") as fh:
        fh.write("Mechanism coefficients are difficulty-controlled ASSOCIATIONS, not causal effects.\n"
                 "Deliberation is endogenous to within-problem difficulty; difficulty_raw controls are\n"
                 "coarse (esp. math, where success is ~flat across levels for this model). A negative\n"
                 "coefficient (e.g. backtracking in math) most plausibly indicates the behavior MARKS a\n"
                 "trace where the model is struggling, not that the behavior CAUSES failure. Causal claims\n"
                 "require intervention (behavior ablation/steering), which is future work.\n")
    print(f"-> {out_dir}/mechanism_coefs.csv  +  mechanism_model_fit.csv  +  INTERPRETATION.txt")
    return pl.DataFrame(all_coef), pl.DataFrame(all_fit)


if __name__ == "__main__":
    import argparse, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--grades", nargs="+", required=True, help="one or more grade parquets to concat")
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out-dir", default="data/perf")
    a = ap.parse_args()
    feats = pl.read_parquet(a.features)
    grades = pl.concat([pl.read_parquet(g) for g in a.grades], how="diagonal_relaxed")
    tr = pl.concat([pl.read_parquet(p) for p in glob.glob(a.traces_glob)], how="diagonal_relaxed")
    run(feats, grades, tr, a.out_dir)