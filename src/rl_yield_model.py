"""Stylised yield, spread, and recovery model used by SovereignBondEnv.

Extracted as a module so the math can be tested without spinning up the
full RL environment (which needs pandas + a populated DataFrame).

The coefficients here are the defaults; the env passes its own values
through when computing per-step yields. Parameter sensitivity is exercised
in the notebook 'Parameter sensitivity' cell.
"""
from __future__ import annotations


def compute_spread(
    debt: float,
    reserves_months: float,
    *,
    spread_intercept: float = 0.02,
    spread_per_debt: float = 0.0005,
    spread_per_reserves: float = 0.005,
    spread_floor: float = 0.005,
    spread_ceiling: float = 0.25,
) -> float:
    """Per-country spread on external debt and reserves coverage.

    The signs follow Borensztein & Panizza (2009): higher debt-to-GNI
    raises the spread; higher reserves coverage lowers it. Both raw
    inputs are clipped to >= 0 to avoid sign-flips on a missing value.
    Output is clipped to [spread_floor, spread_ceiling].
    """
    debt = max(0.0, float(debt))
    reserves_months = max(0.0, float(reserves_months))
    s = (spread_intercept
         + spread_per_debt * debt
         - spread_per_reserves * reserves_months)
    if s < spread_floor:
        return spread_floor
    if s > spread_ceiling:
        return spread_ceiling
    return s


def compute_recovery(
    gdp_per_capita: float,
    *,
    recovery_floor: float = 0.35,
    recovery_slope: float = 0.15,
    recovery_gdp_norm: float = 40000.0,
) -> float:
    """Per-country recovery rate on default.

    Slope on GDP per capita is consistent with Cruces and Trebesch (2013):
    higher-income defaulters tend to settle at higher recovery rates.
    """
    gdp = max(0.0, float(gdp_per_capita))
    normalised = min(1.0, gdp / recovery_gdp_norm)
    return recovery_floor + recovery_slope * normalised


def transaction_cost(turnover: float, *, per_unit: float = 0.003) -> float:
    """Symmetric round-trip turnover cost. Turnover is sum(|new - old|)."""
    return per_unit * max(0.0, float(turnover))
