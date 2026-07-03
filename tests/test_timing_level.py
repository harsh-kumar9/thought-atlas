import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.timing_level import K, dashboard_payload, decompose  # noqa: E402


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))


def synthetic_trace_bins(
    *,
    p_good,
    p_bad,
    n_traces=200,
    segs_per_bin=3,
    seed=0,
    sigma=0.0,
    sparse=False,
):
    rng = np.random.default_rng(seed)
    rows = []
    probs = {"good": np.asarray(p_good, dtype=float), "bad": np.asarray(p_bad, dtype=float)}
    for outcome in ["good", "bad"]:
        for trace_idx in range(n_traces):
            intercept = rng.normal(0, sigma) if sigma else 0.0
            for bin_idx in range(K):
                n = int(rng.poisson(segs_per_bin)) if sparse else int(segs_per_bin)
                if n <= 0:
                    continue
                p = float(_expit(_logit(probs[outcome][bin_idx]) + intercept))
                rows.append(
                    {
                        "gen_model": "m",
                        "task_type": "math",
                        "outcome_class": outcome,
                        "behavior": "verification",
                        "trace_id": f"{outcome}-{trace_idx}",
                        "bin": bin_idx,
                        "X": int(rng.binomial(n, p)),
                        "n": n,
                    }
                )
    return pl.DataFrame(rows)


def test_constant_gap_is_level_not_timing():
    table = synthetic_trace_bins(
        p_good=np.full(K, 0.10),
        p_bad=np.full(K, 0.06),
        n_traces=240,
        segs_per_bin=4,
        seed=1,
    )
    row = decompose(table, boot=200, seed=1)[0]
    assert abs(row["level_z_robust"]) > 4
    assert row["timing_I2_robust"] < 0.15
    assert row["timing_p_robust"] > 0.05


def test_timing_only_shape_is_detected():
    p_good = np.full(K, 0.10)
    p_bad = np.array([0.135] * 8 + [0.10] * 8 + [0.075] * 8, dtype=float)
    table = synthetic_trace_bins(
        p_good=p_good,
        p_bad=p_bad,
        n_traces=260,
        segs_per_bin=5,
        seed=2,
    )
    row = decompose(table, boot=200, seed=2)[0]
    assert row["timing_p_robust"] <= 0.05
    assert row["timing_I2_robust"] > 0.5
    assert abs(row["level_dbar_robust_pp"]) < 1.5
    assert row["level_p_robust"] > 0.05


def test_trace_bootstrap_tempers_clustered_false_timing():
    naive_rejects = 0
    robust_rejects = 0
    p = np.full(K, 0.10)
    for seed in range(20):
        table = synthetic_trace_bins(
            p_good=p,
            p_bad=p,
            n_traces=90,
            segs_per_bin=1.5,
            seed=100 + seed,
            sigma=1.0,
            sparse=True,
        )
        row = decompose(table, boot=120, seed=200 + seed)[0]
        naive_rejects += row["timing_p"] <= 0.05
        robust_rejects += row["timing_p_robust"] <= 0.05
    assert naive_rejects > 1
    assert robust_rejects <= 2


def test_dashboard_schema_and_guards():
    table = synthetic_trace_bins(
        p_good=np.full(K, 0.10),
        p_bad=np.full(K, 0.06),
        n_traces=4,
        segs_per_bin=1,
        seed=3,
    )
    rows = decompose(table, boot=20, seed=3)
    payload = dashboard_payload(rows)
    pair = payload["pairs"][0]
    assert pair["class"] == "insufficient"
    assert pair["status"] == "insufficient"
    assert len(pair["d_pp"]) == K
    assert len(pair["d_se_pp"]) == K
    assert all(value is None for value in pair["d_pp"])
    assert payload["meta"]["K"] == K
