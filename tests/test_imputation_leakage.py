"""Regression test for imputer leakage that existed in an earlier version of the pipeline.

The previous implementation called `imputer.fit_transform(full_data[feature_cols])`
before the temporal train/test split in cell 19, which meant the median used
to fill 1990-2014 rows depended on 2015-2023 observations (and vice versa).

The fix is to fit the imputer on the train slice only and apply `.transform`
to the test slice. These tests assert that property holds.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def _make_panel():
    """Synthetic panel with deliberately divergent test-window distribution
    and balanced train/test sizes. No NaNs - we control the statistics exactly."""
    rng = np.random.default_rng(42)
    train_rows, test_rows = [], []
    # 500 train rows centred around (0, 10)
    for _ in range(500):
        train_rows.append({"year": 2000, "feature_a": rng.normal(0.0, 1.0),
                           "feature_b": rng.normal(10.0, 1.0)})
    # 500 test rows centred around (100, 1000)
    for _ in range(500):
        test_rows.append({"year": 2020, "feature_a": rng.normal(100.0, 1.0),
                          "feature_b": rng.normal(1000.0, 1.0)})
    return pd.DataFrame(train_rows + test_rows)


FEATURE_COLS = ["feature_a", "feature_b"]


def test_imputer_statistics_match_train_median_exactly():
    """Load-bearing: imputer.statistics_ must equal train.median()."""
    df = _make_panel()
    train = df[df["year"] < 2015]
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train[FEATURE_COLS])
    expected = train[FEATURE_COLS].median().values
    assert np.allclose(imputer.statistics_, expected), (
        f"got {imputer.statistics_}, expected {expected}"
    )


def test_correct_and_buggy_flows_produce_different_statistics():
    """Direct head-to-head: the correct flow fits on train only, the buggy
    flow fits on the full panel. With train and test distributions disjoint,
    the two imputers must learn meaningfully different medians."""
    df = _make_panel()
    train = df[df["year"] < 2015]

    correct = SimpleImputer(strategy="median").fit(train[FEATURE_COLS])
    buggy = SimpleImputer(strategy="median").fit(df[FEATURE_COLS])

    # Train means are (0, 10), test means are (100, 1000), balanced sizes.
    # Full-panel medians sit far above train-only medians for both features.
    gap = np.abs(correct.statistics_ - buggy.statistics_)
    assert gap[0] > 10.0, (
        f"feature_a: correct={correct.statistics_[0]:.3f}, "
        f"buggy={buggy.statistics_[0]:.3f}, gap={gap[0]:.3f}"
    )
    assert gap[1] > 100.0, (
        f"feature_b: correct={correct.statistics_[1]:.3f}, "
        f"buggy={buggy.statistics_[1]:.3f}, gap={gap[1]:.3f}"
    )


def test_test_set_imputation_uses_train_statistics_only():
    """When the test slice has missing values, the imputer.transform call must
    fill them with train medians, not with test or full-panel medians."""
    df = _make_panel()
    train = df[df["year"] < 2015]
    test = df[df["year"] >= 2015].copy()
    # Inject NaNs into test only
    test.loc[test.index[:50], "feature_a"] = np.nan

    imputer = SimpleImputer(strategy="median").fit(train[FEATURE_COLS])
    filled = imputer.transform(test[FEATURE_COLS])

    train_median_a = train["feature_a"].median()
    # All formerly-NaN positions in feature_a should now hold the train median
    assert np.allclose(filled[:50, 0], train_median_a), (
        f"test NaNs filled with {filled[:50, 0][0]} but train median is "
        f"{train_median_a}; imputer may have been refit on test"
    )
