"""Time-series hyperparameter search.

An earlier configuration compared LR / RF / sklearn GB / Two-Tower NN using
constructor-default hyperparameters and ranked them by test-set AUC. With
only 18 positive cases in the test set, this comparison is uninformative.

This module runs a TimeSeriesSplit CV on the train years and selects the
hyperparameter combination that maximises mean AUC across folds. The
notebook then refits the chosen configuration on all train years and scores
once on the held-out 2015-2023 test set.

Kept dependency-light: numpy + sklearn only. XGBoost is imported lazily so
the test suite can run without it.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def _iter_param_grid(grid: Dict[str, Iterable]) -> Iterable[Dict[str, Any]]:
    """Cartesian product of a sklearn-style param grid (dict of name -> values)."""
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    values = [list(v) for v in grid.values()]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def time_series_grid_search(
    estimator_cls,
    param_grid: Dict[str, Iterable],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    metric=roc_auc_score,
    base_kwargs: Dict[str, Any] | None = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    """Return (best_params, best_mean_score, all_results).

    `groups` is optional. If supplied, splits are made on unique group values
    in their sorted order (use for year-based splitting so each fold is a
    contiguous block of years rather than a random row sample).

    `all_results` is a list of dicts {params, fold_scores, mean, std} suitable
    for inspection or tabulation in a notebook.
    """
    base_kwargs = base_kwargs or {}
    X = np.asarray(X)
    y = np.asarray(y)

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.sort(np.unique(groups))
        if len(unique_groups) <= n_splits:
            raise ValueError(
                f"need at least n_splits+1 unique groups, got {len(unique_groups)}"
            )
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # Split on group indices, then map back to row indices
        group_splits = list(tscv.split(unique_groups))
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for train_g_idx, val_g_idx in group_splits:
            train_groups = unique_groups[train_g_idx]
            val_groups = unique_groups[val_g_idx]
            train_mask = np.isin(groups, train_groups)
            val_mask = np.isin(groups, val_groups)
            splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(X))

    results: List[Dict[str, Any]] = []
    for params in _iter_param_grid(param_grid):
        fold_scores = []
        for fold, (tr, va) in enumerate(splits):
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]
            if len(np.unique(yva)) < 2:
                if verbose:
                    print(f"  fold {fold}: single-class val set; skipping")
                continue
            model = estimator_cls(**{**base_kwargs, **params})
            try:
                model.fit(Xtr, ytr)
            except Exception as exc:
                if verbose:
                    print(f"  fold {fold} fit failed for {params}: {exc}")
                continue
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(Xva)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(Xva)
            else:
                scores = model.predict(Xva)
            fold_scores.append(float(metric(yva, scores)))
        if not fold_scores:
            mean = float("nan")
            std = float("nan")
        else:
            mean = float(np.mean(fold_scores))
            std = float(np.std(fold_scores))
        results.append({
            "params": params,
            "fold_scores": fold_scores,
            "mean": mean,
            "std": std,
        })
        if verbose:
            print(f"  {params}: mean={mean:.4f} std={std:.4f}")

    # Select best by mean (skip NaN)
    valid = [r for r in results if not np.isnan(r["mean"])]
    if not valid:
        raise RuntimeError("no valid hyperparameter combinations evaluated")
    best = max(valid, key=lambda r: r["mean"])
    return best["params"], best["mean"], results
