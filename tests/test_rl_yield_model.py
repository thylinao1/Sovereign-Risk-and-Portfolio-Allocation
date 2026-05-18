"""Tests for stylised yield / recovery / cost helpers."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rl.yield_model import compute_spread, compute_recovery, transaction_cost


def test_spread_increases_with_debt():
    base = compute_spread(50.0, 5.0)
    higher = compute_spread(100.0, 5.0)
    assert higher > base, f"spread should rise with debt: {base} -> {higher}"


def test_spread_decreases_with_reserves():
    base = compute_spread(50.0, 2.0)
    more_reserves = compute_spread(50.0, 6.0)
    assert more_reserves < base, (
        f"spread should fall with reserves: {base} -> {more_reserves}"
    )


def test_spread_clipped_to_floor_for_negative_inputs():
    # debt=0, reserves=100 (extreme) drives spread very negative; should clip to floor
    s = compute_spread(0.0, 100.0)
    assert s == 0.005, f"expected floor 0.005, got {s}"


def test_spread_clipped_to_ceiling_for_extreme_debt():
    s = compute_spread(10000.0, 0.0)
    assert s == 0.25, f"expected ceiling 0.25, got {s}"


def test_spread_with_negative_inputs_treats_them_as_zero():
    # Both negatives clamped to 0; spread should equal the intercept
    s = compute_spread(-50.0, -5.0)
    assert abs(s - 0.02) < 1e-9, f"expected ~0.02, got {s}"


def test_recovery_at_zero_gdp_equals_floor():
    assert compute_recovery(0.0) == pytest.approx(0.35)


def test_recovery_saturates_at_gdp_norm():
    assert compute_recovery(40000.0) == pytest.approx(0.50)
    assert compute_recovery(80000.0) == pytest.approx(0.50)  # clipped


def test_recovery_monotone_in_gdp():
    r_low = compute_recovery(10000.0)
    r_high = compute_recovery(30000.0)
    assert r_low < r_high


def test_transaction_cost_is_linear_in_turnover():
    assert transaction_cost(0.0) == 0.0
    assert transaction_cost(1.0) == pytest.approx(0.003)
    assert transaction_cost(2.0) == pytest.approx(0.006)


def test_transaction_cost_clips_negative_turnover():
    assert transaction_cost(-5.0) == 0.0


def test_custom_coefficients_propagate():
    # Override spread parameters; floor should follow
    s = compute_spread(0.0, 0.0, spread_intercept=0.5, spread_ceiling=1.0)
    assert s == 0.5
    s = compute_spread(0.0, 0.0, spread_intercept=2.0, spread_ceiling=1.0)
    assert s == 1.0  # clipped to new ceiling
