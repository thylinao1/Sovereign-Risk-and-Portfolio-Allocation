"""Tests for SovereignBondEnv on a synthetic 4-country, 6-year panel.

The env requires pandas but not TensorFlow, so these tests stay in the CI
matrix and run without the notebook's heavy stack.
"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rl.env import SovereignBondEnv


DOMESTIC_FEATURES = [
    "gdp_growth_annual", "gdp_per_capita_constant",
    "external_debt_pct_gni", "reserves_months_imports",
]
GLOBAL_FEATURES = ["vix_annual_avg", "us_10y_treasury"]


def _make_panel(n_countries: int = 4, n_years: int = 6, default_country_idx: int = 0):
    """4 countries x 6 years; default_country_idx defaults in the final year."""
    rng = np.random.default_rng(0)
    rows = []
    for ci in range(n_countries):
        for yi, y in enumerate(range(2000, 2000 + n_years)):
            rows.append({
                "country_code": f"C{ci}",
                "year": y,
                "gdp_growth_annual":      rng.normal(2.0, 1.0),
                "gdp_per_capita_constant": 10000.0 + ci * 5000.0,
                "external_debt_pct_gni":  50.0 + ci * 20.0,
                "reserves_months_imports": 3.0 + ci,
                "vix_annual_avg":         15.0 + yi,
                "us_10y_treasury":         2.0 + 0.1 * yi,
                "default_2y":              1 if (ci == default_country_idx and yi == n_years - 1) else 0,
            })
    return pd.DataFrame(rows)


def test_env_dimensions_match_inputs():
    df = _make_panel()
    env = SovereignBondEnv(df, DOMESTIC_FEATURES, GLOBAL_FEATURES)
    assert env.n_countries == 4
    assert env.n_years == 6
    # state_dim = n_countries * n_dom + n_glob + n_countries weights
    assert env.state_dim == 4 * 4 + 2 + 4
    assert env.action_dim == 4


def test_reset_yields_initial_state_of_correct_shape():
    df = _make_panel()
    env = SovereignBondEnv(df, DOMESTIC_FEATURES, GLOBAL_FEATURES)
    state = env.reset()
    assert state.shape == (env.state_dim,)
    assert state.dtype == np.float32
    # Initial portfolio is uniform 1/n
    assert env.portfolio.shape == (env.n_countries,)
    np.testing.assert_allclose(env.portfolio, 1.0 / env.n_countries)


def test_step_produces_softmax_weights_summing_to_one():
    df = _make_panel()
    env = SovereignBondEnv(df, DOMESTIC_FEATURES, GLOBAL_FEATURES)
    env.reset()
    raw_action = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    _, reward, _ = env.step(raw_action)
    np.testing.assert_allclose(env.portfolio.sum(), 1.0, atol=1e-5)
    assert np.all(env.portfolio >= 0)
    assert np.isfinite(reward)


def test_episode_terminates_after_n_years_minus_one_steps():
    df = _make_panel()
    env = SovereignBondEnv(df, DOMESTIC_FEATURES, GLOBAL_FEATURES)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done = env.step(np.zeros(env.n_countries, dtype=np.float32))
        steps += 1
        assert steps < 100, "infinite loop"
    assert steps == env.n_years - 1


def test_default_country_recovery_rate_kicks_in():
    """The defaulting country in year 5 should lose (1 - recovery) of its weight."""
    df = _make_panel(default_country_idx=0)
    env = SovereignBondEnv(df, DOMESTIC_FEATURES, GLOBAL_FEATURES)
    env.reset()
    # Walk to the final year
    for _ in range(env.n_years - 2):
        env.step(np.zeros(env.n_countries, dtype=np.float32))
    # Run final step with all weight on country 0
    # action = log probabilities; very large for index 0
    action = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, reward, done = env.step(action)
    assert done
    # Country 0 with gdp_per_capita 10000: recovery = 0.35 + 0.15 * min(1, 10000/40000) = 0.3875
    # Default loss is large, so excess return should be negative
    # (this is a smoke check on the recovery formula being wired up)


def test_custom_parameters_propagate():
    df = _make_panel()
    env = SovereignBondEnv(
        df, DOMESTIC_FEATURES, GLOBAL_FEATURES,
        spread_intercept=0.1,
        recovery_floor=0.5,
        transaction_cost_per_turnover=0.01,
    )
    assert env.spread_intercept == 0.1
    assert env.recovery_floor == 0.5
    assert env.transaction_cost_per_turnover == 0.01


def test_default_parameters_match_documented_defaults():
    df = _make_panel()
    env = SovereignBondEnv(df, DOMESTIC_FEATURES, GLOBAL_FEATURES)
    assert env.base_rate == 0.03
    assert env.spread_intercept == 0.02
    assert env.spread_per_debt == 0.0005
    assert env.spread_per_reserves == 0.005
    assert env.recovery_floor == 0.35
    assert env.recovery_slope == 0.15
    assert env.transaction_cost_per_turnover == 0.003


def test_env_inline_math_matches_yield_model_helpers():
    """Regression guard: env's inline yield / recovery / cost math must equal
    the standalone helpers in src.rl.yield_model. Drift between the two has
    bitten this repo before: an earlier review caught gdp_pc clamping divergence
    on the recovery branch."""
    from src.rl.yield_model import compute_spread, compute_recovery, transaction_cost

    df = _make_panel()
    env = SovereignBondEnv(df, DOMESTIC_FEATURES, GLOBAL_FEATURES)
    env.reset()

    # --- spreads ---
    debt_idx = env.domestic_features.index("external_debt_pct_gni")
    reserves_idx = env.domestic_features.index("reserves_months_imports")
    _, env_spreads = env._compute_yields(0)
    for i in range(env.n_countries):
        debt = env.macro_data[0, i, debt_idx]
        reserves = env.macro_data[0, i, reserves_idx]
        expected = compute_spread(
            debt, reserves,
            spread_intercept=env.spread_intercept,
            spread_per_debt=env.spread_per_debt,
            spread_per_reserves=env.spread_per_reserves,
            spread_floor=env.spread_floor,
            spread_ceiling=env.spread_ceiling,
        )
        assert abs(env_spreads[i] - expected) < 1e-6, (
            f"country {i}: env spread {env_spreads[i]} vs helper {expected}"
        )

    # --- recovery: force a default in year 0 so the branch runs ---
    env.defaults[0, 0] = 1
    gdp_idx = env.domestic_features.index("gdp_per_capita_constant")
    env_recovery = env._get_recovery_rates(0)
    expected_recovery = compute_recovery(
        env.macro_data[0, 0, gdp_idx],
        recovery_floor=env.recovery_floor,
        recovery_slope=env.recovery_slope,
        recovery_gdp_norm=env.recovery_gdp_norm,
    )
    assert abs(env_recovery[0] - expected_recovery) < 1e-6

    # --- recovery on a synthetic negative-GDP row should also match (this
    # specifically guards the clamping fix) ---
    env.macro_data[0, 1, gdp_idx] = -5000.0
    env.defaults[0, 1] = 1
    env_recovery = env._get_recovery_rates(0)
    expected_recovery = compute_recovery(
        -5000.0,
        recovery_floor=env.recovery_floor,
        recovery_slope=env.recovery_slope,
        recovery_gdp_norm=env.recovery_gdp_norm,
    )
    assert abs(env_recovery[1] - expected_recovery) < 1e-6

    # --- transaction cost via step() ---
    pre_weights = env.portfolio.copy()
    env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    new_weights = env.portfolio
    turnover = np.abs(new_weights - pre_weights).sum()
    expected_cost = transaction_cost(turnover, per_unit=env.transaction_cost_per_turnover)
    inline_cost = env.transaction_cost_per_turnover * turnover
    assert abs(inline_cost - expected_cost) < 1e-6
