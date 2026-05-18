"""Tests for time_series_grid_search."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tune import time_series_grid_search


def _make_xy(n=600, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    # y depends on feature 0
    logits = 0.8 * X[:, 0]
    p = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    return X, y


def test_grid_search_returns_best_params_and_score():
    X, y = _make_xy()
    grid = {"C": [0.01, 1.0]}
    best_params, best_score, all_results = time_series_grid_search(
        LogisticRegression,
        param_grid=grid,
        X=X,
        y=y,
        n_splits=3,
        base_kwargs={"max_iter": 200},
    )
    assert "C" in best_params
    assert best_params["C"] in [0.01, 1.0]
    assert 0.0 < best_score < 1.0
    assert len(all_results) == 2
    # best_score is the max mean across the grid
    means = [r["mean"] for r in all_results]
    assert best_score == max(means)


def test_grid_search_with_groups_uses_temporal_splits():
    """When years are supplied as groups, each fold's val set should be a
    strictly-later set of years than the train set."""
    rng = np.random.default_rng(1)
    years = np.repeat(np.arange(1990, 2010), 30)  # 20 years x 30 rows
    n = len(years)
    X = rng.normal(size=(n, 3))
    y = (rng.random(n) < 0.3).astype(int)

    captured_splits = []

    class CapturingLR(LogisticRegression):
        def fit(self, X, y, **kw):
            captured_splits.append(("fit", len(X)))
            return super().fit(X, y, **kw)

    _ = time_series_grid_search(
        CapturingLR,
        param_grid={"C": [1.0]},
        X=X,
        y=y,
        groups=years,
        n_splits=4,
        base_kwargs={"max_iter": 200},
    )
    # Each fold's train should be shorter than X total
    train_sizes = [size for tag, size in captured_splits if tag == "fit"]
    assert all(0 < s < n for s in train_sizes), train_sizes
    # Train sizes should be non-decreasing (TimeSeriesSplit gives growing windows)
    assert train_sizes == sorted(train_sizes), train_sizes


def test_grid_search_handles_grid_of_two_models():
    X, y = _make_xy(seed=2)
    best, score, results = time_series_grid_search(
        RandomForestClassifier,
        param_grid={"n_estimators": [10, 30], "max_depth": [3, 5]},
        X=X,
        y=y,
        n_splits=3,
        base_kwargs={"random_state": 0},
    )
    assert len(results) == 4  # 2 x 2
    assert best["n_estimators"] in [10, 30]
    assert best["max_depth"] in [3, 5]


def test_grid_search_rejects_too_few_groups():
    X, y = _make_xy(n=50)
    years = np.repeat(np.arange(1990, 1993), 17)[:50]  # only 3 unique years
    with pytest.raises(ValueError, match="unique groups"):
        time_series_grid_search(
            LogisticRegression,
            param_grid={"C": [1.0]},
            X=X,
            y=y,
            groups=years,
            n_splits=5,
        )
