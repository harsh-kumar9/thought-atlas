import numpy as np
import pandas as pd

from src.analysis.paper.signature_analysis import (
    _per_trace_predictive_gain,
    add_timing_contrasts,
    assign_quality_classes,
    exact_shapley_trace_gains,
    ridge_multinomial_predict,
)


def _phase_frame(scale: float) -> pd.DataFrame:
    row = {}
    for behavior in [
        "Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives", "Reconciliation",
        "verification", "backtracking", "subgoal", "backward_chaining",
    ]:
        row[f"{behavior}__early"] = 1 * scale
        row[f"{behavior}__middle"] = 2 * scale
        row[f"{behavior}__late"] = 4 * scale
    return pd.DataFrame([row])


def test_timing_contrasts_are_invariant_to_amount_scaling():
    first = add_timing_contrasts(_phase_frame(1.0))
    second = add_timing_contrasts(_phase_frame(11.0))
    timing = [column for column in first if "__timing_" in column]
    np.testing.assert_allclose(first[timing], second[timing])


def test_multinomial_probabilities_are_valid_and_learn_separation():
    x = np.r_[np.linspace(-2, -.2, 40), np.linspace(.2, 2, 40)]
    design = np.column_stack([np.ones(len(x)), x])
    y = np.r_[np.zeros(40, dtype=int), np.ones(40, dtype=int)]
    probability = ridge_multinomial_predict(design, y, design, 2)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    assert (probability.argmax(axis=1) == y).mean() > .9


def test_quality_classes_keep_missing_completed_outcomes_missing():
    frame = pd.DataFrame({
        "completed": [False, True, True, True, True],
        "outcome_available": [False, True, True, True, False],
        "outcome_type": [None, "binary", "binary", "continuous", None],
        "outcome_higher_is_better": [np.nan, 1.0, 0.0, .4, np.nan],
        "gen_model": ["m"] * 5,
        "task_type": ["d", "d", "d", "d", "d"],
    })
    result = assign_quality_classes(frame)
    assert result.quality_class.tolist()[:4] == ["incomplete", "high", "low", "high"]
    assert pd.isna(result.quality_class.iloc[4])


def test_exact_shapley_contributions_sum_to_full_predictive_gain():
    y = np.array([0, 1, 0, 1])
    predictions = {}
    for mask in range(16):
        confidence = 0.55 + 0.025 * mask.bit_count()
        predictions[mask] = np.where(y == 1, confidence, 1 - confidence)
    contributions = exact_shapley_trace_gains(
        y,
        predictions,
        ["a", "b", "c", "d"],
        "binary",
    )
    observed = np.sum(np.stack(list(contributions.values())), axis=0)
    expected = _per_trace_predictive_gain(y, predictions[0], predictions[15], "binary")
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_continuous_per_trace_gain_mean_equals_delta_r2():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    baseline = np.repeat(y.mean(), len(y))
    improved = np.array([0.1, 1.1, 1.9, 2.9])
    gain = _per_trace_predictive_gain(y, baseline, improved, "continuous")
    denominator = np.sum((y - y.mean()) ** 2)
    baseline_r2 = 1 - np.sum((y - baseline) ** 2) / denominator
    improved_r2 = 1 - np.sum((y - improved) ** 2) / denominator
    np.testing.assert_allclose(gain.mean(), improved_r2 - baseline_r2)
