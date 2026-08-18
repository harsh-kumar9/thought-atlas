import numpy as np

from src.analysis.paper.final_analysis import _js


def test_uniform_scaling_changes_amount_not_conditional_shape():
    early = np.array([2.0, 1.0, 0.0])
    scaled = early * 7
    assert _js(early, scaled) == 0.0
    assert scaled.sum() != early.sum()


def test_moving_occurrences_changes_shape_not_amount():
    early = np.array([3.0, 0.0, 0.0])
    late = np.array([0.0, 0.0, 3.0])
    assert early.sum() == late.sum()
    assert _js(early, late) > 0.9


def test_null_mass_has_no_fabricated_shape():
    assert np.isnan(_js(np.zeros(3), np.ones(3)))

