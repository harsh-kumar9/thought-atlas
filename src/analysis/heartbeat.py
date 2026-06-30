"""src/analysis/heartbeat.py — Track B temporal "heartbeat" (ThinkARM Fig.3, cross-domain).

For each behavior: normalized-position curve (mean normalized frequency per position bin),
one line per domain, mean +/- SD across gen models. Plus the shuffle-within-trace null and
a functional-ANOVA hook (scikit-fda) testing whether curves differ by domain.
"""
from __future__ import annotations
import numpy as np
import polars as pl

BEHAVIORS = ["Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives",
             "Reconciliation", "verification", "backtracking", "subgoal", "backward_chaining"]


def curve(segments: pl.DataFrame, behavior: str, *, n_bins=20,
          section: str | None = None) -> pl.DataFrame:
    """Mean presence of `behavior` per normalized-position bin, per (task_type, gen_model)."""
    df = segments
    if section:
        df = df.filter(pl.col("section_type") == section)
    df = df.with_columns((pl.col("norm_pos") * (n_bins - 1)).round().cast(pl.Int32).alias("bin"))
    return (df.group_by(["task_type", "gen_model", "bin"])
              .agg(pl.col(behavior).mean().alias("freq"), pl.len().alias("n"))
              .sort(["task_type", "gen_model", "bin"]))


def shuffle_null(segments: pl.DataFrame, behavior: str, *, n_bins=20, seed=0) -> pl.DataFrame:
    """Shuffle behavior labels within each trace (preserve counts), recompute the curve."""
    rng = np.random.default_rng(seed)
    parts = []
    for tid, g in segments.group_by("trace_id"):
        vals = g[behavior].to_numpy().copy()
        rng.shuffle(vals)
        parts.append(g.with_columns(pl.Series(behavior, vals)))
    sh = pl.concat(parts)
    return curve(sh, behavior, n_bins=n_bins)


def functional_anova(segments: pl.DataFrame, behavior: str, *, n_bins=20):
    """Test whether per-domain heartbeat curves differ (functional ANOVA). Needs scikit-fda."""
    try:
        from skfda import FDataGrid
        from skfda.inference.anova import oneway_anova
        # one curve per (trace) as the functional datum, grouped by task_type
        grid = np.linspace(0, 1, n_bins)
        curves, groups = [], []
        seg = segments.with_columns((pl.col("norm_pos") * (n_bins - 1)).round().cast(pl.Int32).alias("bin"))
        for (tid, task), g in seg.group_by(["trace_id", "task_type"]):
            per_bin = (g.group_by("bin").agg(pl.col(behavior).mean()).sort("bin"))
            y = np.full(n_bins, np.nan)
            for b, v in zip(per_bin["bin"].to_list(), per_bin[behavior].to_list()):
                if 0 <= b < n_bins:
                    y[b] = v
            y = np.nan_to_num(y)
            curves.append(y); groups.append(task)
        fd = FDataGrid(np.array(curves), grid_points=grid)
        groups = np.array(groups)
        fdatas = [fd[groups == gname] for gname in np.unique(groups)]
        stat, pval = oneway_anova(*fdatas)
        return {"behavior": behavior, "fanova_stat": float(stat), "p_value": float(pval)}
    except Exception as e:  # noqa
        return {"behavior": behavior, "error": str(e),
                "note": "pip install scikit-fda; or fit functional ANOVA in R::fda"}
