"""Tests for bootstrap CI and calibration helpers."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval import (
    bootstrap_metric_ci,
    auc_ci,
    ap_ci,
    brier_ci,
    expected_calibration_error,
    reliability_curve,
)


def _synthetic_classification(n=1000, prevalence=0.05, separation=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    scores = rng.normal(loc=0.0, scale=1.0, size=n)
    scores[y == 1] += separation
    # Map to [0, 1] probabilities via logistic
    probs = 1 / (1 + np.exp(-scores))
    return y, probs


def test_bootstrap_ci_brackets_point_estimate():
    y, p = _synthetic_classification()
    point = roc_auc_score(y, p)
    lo, mean, hi = bootstrap_metric_ci(y, p, roc_auc_score, n_bootstraps=500)
    assert lo <= point <= hi, f"point {point} not in CI [{lo}, {hi}]"
    assert lo < hi, "CI is degenerate"
    assert 0.0 <= lo and hi <= 1.0, f"AUC CI out of range: [{lo}, {hi}]"


def test_bootstrap_ci_widens_with_smaller_n():
    """CI on a smaller dataset must be at least as wide as on a larger one.
    Uses low separation so neither CI hits the [0, 1] ceiling."""
    y_big, p_big = _synthetic_classification(n=4000, prevalence=0.2, separation=0.5, seed=1)
    y_small, p_small = _synthetic_classification(n=200, prevalence=0.2, separation=0.5, seed=2)
    lo_b, _, hi_b = bootstrap_metric_ci(y_big, p_big, roc_auc_score, n_bootstraps=500)
    lo_s, _, hi_s = bootstrap_metric_ci(y_small, p_small, roc_auc_score, n_bootstraps=500)
    width_big = hi_b - lo_b
    width_small = hi_s - lo_s
    assert width_small > width_big, (
        f"CI width should widen with smaller n: big={width_big:.3f}, small={width_small:.3f}"
    )


def test_auc_ap_brier_wrappers_match_underlying_metric():
    y, p = _synthetic_classification()
    lo_a, mean_a, hi_a = auc_ci(y, p, n_bootstraps=300)
    lo_p, mean_p, hi_p = ap_ci(y, p, n_bootstraps=300)
    lo_b, mean_b, hi_b = brier_ci(y, p, n_bootstraps=300)
    # Means should be close to the unbootstrapped point estimate
    assert abs(mean_a - roc_auc_score(y, p)) < 0.05
    assert abs(mean_p - average_precision_score(y, p)) < 0.05
    assert abs(mean_b - brier_score_loss(y, p)) < 0.05


def test_ece_zero_for_perfectly_calibrated_predictions():
    """If predictions exactly equal binary labels (0 or 1), ECE is 0."""
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 20)
    p = y.astype(float)
    assert expected_calibration_error(y, p, n_bins=10) == 0.0


def test_ece_nonzero_for_uniform_predictions_on_skewed_labels():
    """If every prediction is 0.5 but the base rate is 0.1, ECE should be near 0.4."""
    n = 1000
    y = (np.arange(n) < 100).astype(int)  # 10% positives
    p = np.full(n, 0.5)
    ece = expected_calibration_error(y, p, n_bins=10)
    # All predictions land in bin 5 (around 0.5). Bin mean pred = 0.5, bin observed = 0.1
    # ECE = 1.0 * |0.5 - 0.1| = 0.4
    assert 0.35 < ece < 0.45, f"ECE was {ece}, expected ~0.4"


def test_reliability_curve_returns_consistent_shapes():
    y, p = _synthetic_classification()
    pred, obs, counts = reliability_curve(y, p, n_bins=10)
    assert pred.shape == obs.shape == counts.shape == (10,)
    assert counts.sum() == len(y)


def test_bootstrap_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        bootstrap_metric_ci(np.array([0, 1]), np.array([0.1, 0.2, 0.3]), roc_auc_score)


def test_ece_rejects_out_of_range_probabilities():
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0, 1]), np.array([0.5, 1.5]))
