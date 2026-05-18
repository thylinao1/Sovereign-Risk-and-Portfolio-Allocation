"""Evaluation helpers: bootstrap confidence intervals on classifier metrics
and calibration diagnostics. Kept dependency-light (numpy + sklearn only)
so the test suite runs in CI without TensorFlow.

The bootstrap_metric_ci helper resamples (y_true, y_score) with replacement
n_bootstraps times and returns the (lower, mean, upper) percentile triple.
The expected_calibration_error helper bins predictions and returns the
weighted absolute difference between mean prediction and observed frequency
inside each bin.
"""
from __future__ import annotations

from typing import Callable, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Return (lower, mean, upper) for `metric(y_true, y_score)` via the
    percentile bootstrap. Resamples are draw with replacement at the row
    level. Skips resamples that contain only one class (which would crash
    metrics like roc_auc_score)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}"
        )

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    estimates = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        sub_true = y_true[idx]
        sub_score = y_score[idx]
        # Skip resamples that don't contain both classes
        if len(np.unique(sub_true)) < 2:
            continue
        try:
            estimates.append(metric(sub_true, sub_score))
        except ValueError:
            continue

    if not estimates:
        raise RuntimeError(
            "no valid bootstrap resamples produced a metric value; "
            "check that y_true contains both classes"
        )
    estimates = np.array(estimates)
    lo = float(np.quantile(estimates, alpha / 2))
    hi = float(np.quantile(estimates, 1 - alpha / 2))
    mean = float(estimates.mean())
    return lo, mean, hi


def auc_ci(y_true, y_score, **kwargs):
    return bootstrap_metric_ci(y_true, y_score, roc_auc_score, **kwargs)


def ap_ci(y_true, y_score, **kwargs):
    return bootstrap_metric_ci(y_true, y_score, average_precision_score, **kwargs)


def brier_ci(y_true, y_score, **kwargs):
    return bootstrap_metric_ci(y_true, y_score, brier_score_loss, **kwargs)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Standard equal-width-bin ECE. Returns the size-weighted absolute gap
    between mean predicted probability and observed frequency within bin.

    Values range from 0 (perfectly calibrated) to roughly 0.5 (catastrophic)."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob shapes must match")
    if not np.all((0 <= y_prob) & (y_prob <= 1)):
        raise ValueError("y_prob must lie in [0, 1]")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=False gives bin index in [1, n_bins+1]
    # Clamp predictions exactly at 1.0 into the top bin.
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_pred = y_prob[mask].mean()
        bin_true = y_true[mask].mean()
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(bin_pred - bin_true)
    return float(ece)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
):
    """Return (mean_pred_per_bin, mean_true_per_bin, count_per_bin) for plotting."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)
    means_pred, means_true, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            means_pred.append(np.nan)
            means_true.append(np.nan)
            counts.append(0)
            continue
        means_pred.append(float(y_prob[mask].mean()))
        means_true.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(means_pred), np.array(means_true), np.array(counts)
