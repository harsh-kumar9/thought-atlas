"""src/analysis/aggregate.py — Track A aggregate analysis (domain x behavior).

Cell stats (mean + bootstrap CI per (domain, model, behavior)) for the headline heatmap,
and the GLMM design. Full NB-GLMM with crossed random effects (problem_id, seed) is best
fit in R/lme4 (glmer.nb) per the pre-reg formula; here we provide:
  * the tidy long table + cell means/CIs (figure-ready),
  * a Poisson/NB GEE with cluster-robust SE on problem_id (statsmodels) as the in-Python
    approximation, with the lme4 formula emitted for the canonical fit.
"""
from __future__ import annotations
import json
import numpy as np
import polars as pl

BEHAVIORS = ["Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives",
             "Reconciliation", "verification", "backtracking", "subgoal", "backward_chaining"]


def join_counts(trackA: pl.DataFrame, traces: pl.DataFrame) -> pl.DataFrame:
    meta = traces.select(["trace_id", "gen_model", "task_type", "seed", "instance_id",
                          "failure_mode", "completed", "n_new_tokens"])
    return trackA.join(meta, on="trace_id", how="inner")


def _boot_ci(x: np.ndarray, n=2000, seed=0):
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cell_table(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (model, task), g in df.group_by(["gen_model", "task_type"]):
        for b in BEHAVIORS:
            x = g[b].to_numpy().astype(float)
            lo, hi = _boot_ci(x)
            rows.append({"gen_model": model, "task_type": task, "behavior": b,
                         "n": len(x), "mean": float(x.mean()) if len(x) else float("nan"),
                         "ci_lo": lo, "ci_hi": hi})
    return pl.DataFrame(rows)


def lme4_formula(behavior: str) -> str:
    # canonical (fit in R): negative binomial GLMM
    return (f"{behavior} ~ task_type * gen_model + (1|instance_id) + (1|seed)  # glmer.nb")


def gee_cluster_robust(df: pl.DataFrame, behavior: str):
    """Poisson GEE clustered on instance_id with task_type*gen_model fixed effects."""
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        pdf = df.select([behavior, "task_type", "gen_model", "instance_id"]).to_pandas()
        pdf = pdf.rename(columns={behavior: "y"})
        m = smf.gee("y ~ C(task_type) * C(gen_model)", groups="instance_id", data=pdf,
                    family=sm.families.Poisson())
        return m.fit()
    except Exception as e:  # noqa
        return f"GEE unavailable ({e}); fit {lme4_formula(behavior)} in R/lme4"
