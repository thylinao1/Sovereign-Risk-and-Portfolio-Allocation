"""SovereignBondEnv: a toy single-step-per-year sovereign bond allocation
environment used by the PPO agent. Extracted from the notebook so the
class is importable, unit-testable, and reusable.

The yield / spread / recovery / transaction-cost coefficients are stylised
rather than calibrated. Defaults follow Borensztein and Panizza (2009)
for the debt and reserves signs on the spread, and Cruces and Trebesch
(2013) for the recovery-on-GDP slope. Parameter sensitivity is exercised
in the notebook section labelled "Parameter sensitivity".
"""
from __future__ import annotations

import numpy as np


class SovereignBondEnv:
    """Environment for sovereign bond allocation.

    State uses observable macro fundamentals only (no model predictions).
    Rewards based on yields minus default losses minus transaction costs.

    The yield, recovery, and transaction-cost coefficients are stylised
    rather than calibrated to historical data. Sensitivity to each is
    measured in the cell labelled 'Parameter sensitivity' further down. The
    spread-on-debt and spread-on-reserves slopes follow the qualitative
    sign reported in Borensztein and Panizza (2009); the GDP-per-capita
    recovery slope is consistent with the developing-country range in
    Cruces and Trebesch (2013).
    """

    def __init__(
        self,
        data,
        domestic_features,
        global_features,
        # Yield model: yield_i = base_rate + spread_i,
        #   spread_i = spread_intercept + spread_per_debt * debt
        #              - spread_per_reserves * reserves_months
        # All values clipped to [spread_floor, spread_ceiling].
        base_rate: float = 0.03,
        spread_intercept: float = 0.02,
        spread_per_debt: float = 0.0005,
        spread_per_reserves: float = 0.005,
        spread_floor: float = 0.005,
        spread_ceiling: float = 0.25,
        # Recovery rate on default: recovery_floor + recovery_slope * min(1, gdp_pc / recovery_gdp_norm)
        recovery_floor: float = 0.35,
        recovery_slope: float = 0.15,
        recovery_gdp_norm: float = 40000.0,
        # Transaction cost per unit of turnover (round-trip basis points)
        transaction_cost_per_turnover: float = 0.003,
    ):
        self.data = data.reset_index(drop=True)
        self.domestic_features = domestic_features
        self.global_features = global_features

        # Save parameters
        self.base_rate = base_rate
        self.spread_intercept = spread_intercept
        self.spread_per_debt = spread_per_debt
        self.spread_per_reserves = spread_per_reserves
        self.spread_floor = spread_floor
        self.spread_ceiling = spread_ceiling
        self.recovery_floor = recovery_floor
        self.recovery_slope = recovery_slope
        self.recovery_gdp_norm = recovery_gdp_norm
        self.transaction_cost_per_turnover = transaction_cost_per_turnover

        self.countries = np.sort(self.data['country_code'].unique())
        self.n_countries = len(self.countries)
        self.years = np.sort(self.data['year'].unique())
        self.n_years = len(self.years)

        self.country_to_idx = {c: i for i, c in enumerate(self.countries)}

        n_dom = len(domestic_features)
        n_glob = len(global_features)

        self.state_dim = self.n_countries * n_dom + n_glob + self.n_countries
        self.action_dim = self.n_countries

        self.macro_data = np.zeros((self.n_years, self.n_countries, n_dom), dtype=np.float32)
        self.global_data_array = np.zeros((self.n_years, n_glob), dtype=np.float32)
        self.defaults = np.zeros((self.n_years, self.n_countries), dtype=np.int8)

        for _, row in self.data.iterrows():
            yi = np.where(self.years == row['year'])[0][0]
            ci = self.country_to_idx[row['country_code']]
            self.macro_data[yi, ci, :] = row[domestic_features].values.astype(float)
            self.global_data_array[yi, :] = row[global_features].values.astype(float)
            self.defaults[yi, ci] = int(row['default_2y'])

        self.current_step = 0
        self.portfolio = np.ones(self.n_countries, dtype=np.float32) / self.n_countries
        self.return_history = []

    def reset(self):
        self.current_step = 0
        self.portfolio = np.ones(self.n_countries, dtype=np.float32) / self.n_countries
        self.return_history = []
        return self._get_state()

    def _get_state(self):
        t = self.current_step
        macro_flat = self.macro_data[t].flatten()
        global_t = self.global_data_array[t]
        weights = self.portfolio
        return np.concatenate([macro_flat, global_t, weights]).astype(np.float32)

    def _compute_yields(self, year_idx):
        spreads = np.zeros(self.n_countries, dtype=np.float32)
        debt_idx = self.domestic_features.index('external_debt_pct_gni')
        reserves_idx = self.domestic_features.index('reserves_months_imports')
        for i in range(self.n_countries):
            debt = self.macro_data[year_idx, i, debt_idx]
            reserves = self.macro_data[year_idx, i, reserves_idx]
            spread = (self.spread_intercept
                      + self.spread_per_debt * max(0, debt)
                      - self.spread_per_reserves * max(0, reserves))
            spreads[i] = max(self.spread_floor, min(self.spread_ceiling, spread))
        return self.base_rate, spreads

    def _get_recovery_rates(self, year_idx):
        recovery = np.ones(self.n_countries, dtype=np.float32)
        gdp_pc_idx = self.domestic_features.index('gdp_per_capita_constant')
        for i in range(self.n_countries):
            if self.defaults[year_idx, i] == 1:
                # Clamp negative imputed/scaled GDP to zero so this matches
                # compute_recovery() in src/rl/yield_model.py, which does the
                # same. Without the clamp the env produced recovery rates
                # below recovery_floor for any negative GDP row.
                gdp_pc = max(0.0, float(self.macro_data[year_idx, i, gdp_pc_idx]))
                recovery[i] = (self.recovery_floor
                               + self.recovery_slope * min(1.0, gdp_pc / self.recovery_gdp_norm))
        return recovery

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        exp_a = np.exp(action - action.max())
        new_weights = exp_a / exp_a.sum()

        turnover = np.abs(new_weights - self.portfolio).sum()
        transaction_cost = self.transaction_cost_per_turnover * turnover

        base_rate, spreads = self._compute_yields(self.current_step)
        total_yields = base_rate + spreads

        gross_return = np.sum(new_weights * total_yields)

        defaults_t = self.defaults[self.current_step].astype(np.float32)
        recovery_rates = self._get_recovery_rates(self.current_step)
        default_loss = np.sum(new_weights * defaults_t * (1 - recovery_rates))

        net_return = gross_return - default_loss - transaction_cost
        excess_return = net_return - base_rate
        self.return_history.append(net_return)

        if len(self.return_history) > 3:
            vol = np.std(self.return_history[-20:])
            reward = excess_return / (vol + 0.01)
        else:
            reward = excess_return * 10

        self.portfolio = new_weights
        self.current_step += 1
        done = self.current_step >= self.n_years - 1
        return self._get_state(), reward, done

