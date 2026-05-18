# Sovereign Risk ML

Predicting sovereign defaults and optimizing bond portfolios using machine learning on macroeconomic fundamentals.

## Overview

This project builds an end-to-end pipeline for sovereign credit risk analysis:
1. **Data Collection**: Automated fetching from World Bank and FRED APIs
2. **Default Prediction**: Comparison of neural networks vs. traditional ML models
3. **Portfolio Optimization**: Reinforcement learning for risk-aware bond allocation

The analysis covers 117 countries from 1990-2023, with temporal train/test splits to prevent data leakage.

## Repository layout

```
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt          # Notebook runtime stack
├── requirements-test.txt     # Lightweight test stack
├── .gitignore
├── .github/workflows/ci.yml  # pytest + notebook parse + banned-phrasing scan
├── data/
│   └── README.md             # Indicator catalogue, FRED_API_KEY setup
├── paper/
│   └── limitations-of-deep-learning-for-sovereign-default-prediction.pdf
├── notebooks/
│   └── 01_sovereign_default_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── eval.py               # Bootstrap CIs, ECE, reliability curves
│   ├── tune.py               # TimeSeriesSplit grid search
│   ├── data.py               # World Bank + FRED fetchers
│   └── rl/
│       ├── __init__.py
│       ├── env.py            # SovereignBondEnv
│       ├── ppo.py            # PPOAgent, PPOAgentNorm
│       ├── normaliser.py     # Welford-running StateNormaliser
│       └── yield_model.py    # spread / recovery / cost helpers
└── tests/
    ├── test_imputation_leakage.py
    ├── test_eval_helpers.py
    ├── test_tune.py
    ├── test_rl_yield_model.py
    ├── test_state_normaliser.py
    ├── test_rl_env.py
    └── test_data.py
```

## Installation

The notebook needs TensorFlow and XGBoost; the test suite does not. Install
the lightweight stack first:

```bash
git clone https://github.com/thylinao1/Sovereign-Risk-and-Portfolio-Allocation.git
cd Sovereign-Risk-and-Portfolio-Allocation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt
pytest tests/        # should print 46 passed
```

For running the notebook itself, install the full stack:

```bash
pip install -r requirements.txt
```

On Apple Silicon, swap `tensorflow` for `tensorflow-macos` and `tensorflow-metal`.

## Reproduce

If you cannot run the full notebook (TensorFlow + PPO training is heavy on
small machines), use the standalone script for the classical baselines only:

```bash
export FRED_API_KEY=<your_key>
pip install -r requirements-test.txt xgboost
python scripts/refresh_baseline_numbers.py
```

This skips the Two-Tower NN and PPO entirely. It runs the corrected,
leakage-fixed, CV-tuned pipeline for LR / RF / sklearn GB / XGBoost in under
five minutes and prints a table of AUC / AP / Brier / ECE with bootstrap 95%
CIs. Paste the resulting table into the README Prediction Performance section.

For the full pipeline (including TF-based Two-Tower and PPO):


1. Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html.
2. `export FRED_API_KEY=<your_key>`.
3. `jupyter notebook notebooks/01_sovereign_default_modeling.ipynb` and Cell -> Run All,
   or non-interactively:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace \
       notebooks/01_sovereign_default_modeling.ipynb
   ```

4. World Bank data is fetched without authentication; FRED needs the env var
   from step 2. See `data/README.md` for the indicator catalogue and source
   citations.


## Key Findings

### Prediction Performance
- **Random Forest achieves best AUC (0.828)** on out-of-sample test data
- Two-tower neural network underperforms simpler models (AUC 0.675)
- Neural embeddings collapse to near 1-dimensional representation (94% variance in first PC)
- Class imbalance (2.4% default rate) requires careful handling with focal loss and class weights

### Reinforcement Learning Results
- PPO agent learns a fixed allocation policy from macro fundamentals.
  Equal-weight baseline 16.67; RL policy single-run reward 18.78 (+12.6% vs equal-weight).
- Across environments: deterministic single-run +12.6% (range 9-20% across seeds),
  stochastic +9%, contagion +5%. The "+13-20%" range previously headlined was the
  best deterministic environment only and included seed-to-seed variation.
- Beats a low-volatility heuristic (+5.9%) and a low-debt heuristic (-26.7%).
- Correlation between default rate and portfolio weight: -0.585.
- The learned policy is effectively static. A 10% perturbation of every macro
  feature moves portfolio weights by 0.0000 (sensitivity test, notebook
  "State-level sensitivity check" section). Both static-trained and
  stochastic-trained agents have zero weight variance across years. The gain
  over equal-weight comes from learning to underweight serial defaulters by
  about 1.4 percentage points, not from temporal risk management.
## Technical Implementation

### Data Collection

**Domestic Vulnerability Indicators (World Bank API)**
- GDP growth rate (annual %)
- GDP per capita (constant 2015 USD)
- Inflation (CPI annual %)
- Unemployment rate (% of total labor force)
- Current account balance (% of GDP)
- Total reserves (months of imports)
- Trade openness (% of GDP)
- FDI net inflows (% of GDP)
- External debt stocks (% of GNI)
- Debt service (% of exports)
- Central government debt (% of GDP)
- Government revenue and expenditure (% of GDP)
- Broad money (% of GDP)
- Domestic credit (% of GDP)

**Global Stress Factors (FRED API)**
- VIX (annual average)
- US 10-year Treasury yield
- USD broad index
- High yield credit spread
- TED spread
- Yield curve slope (10Y-2Y)

**Sovereign Default Events**

Curated database from multiple authoritative sources:
- Reinhart & Rogoff "This Time is Different" database
- S&P Global Ratings sovereign default history
- Moody's sovereign default studies
- Bank of Canada / Bank of England sovereign default database

Default definition includes: missed payments, debt restructuring with haircuts, IMF bailouts with debt relief, and selective default ratings. Total of 88 default events across 63 countries.

### Prediction Models

**1. Logistic Regression (Baseline)**
- L2 regularization; C selected by TimeSeriesSplit CV on training years
- Balanced class weights
- AUC, AP, Brier and bootstrap 95% CIs reported in the notebook after re-execution

**2. Random Forest**
- n_estimators and max_depth selected by TimeSeriesSplit CV on training years
- Balanced class weights
- Per-metric ranking (AUC, AP, Brier, ECE) reported with bootstrap 95% CIs in the notebook

**3. Gradient Boosting (sklearn)**
- n_estimators, max_depth, learning_rate selected by TimeSeriesSplit CV on training years

**4. XGBoost**
- n_estimators, max_depth, learning_rate selected by TimeSeriesSplit CV on training years
- scale_pos_weight set to neg/pos ratio in train

**5. Two-Tower Neural Network**
- Architecture inspired by recommender systems (MovieLens coursework)
- Separate embedding towers for domestic and global features
- L2-normalized embeddings with dot product interaction
- Focal loss (gamma=2.0, alpha=0.75) for class imbalance
- AUC: 0.675, Average Precision: 0.064
- **Underperforms traditional models**

The two-tower hypothesis: P(Default) = f(Vulnerability · Stress), where the dot product in learned latent space captures interaction effects. Countries with high "vulnerability embeddings" only default when global "stress embeddings" are elevated.

**Why it failed**: PCA analysis on learned embeddings shows 94% of variance captured by first principal component. The network collapsed to a trivial 1-dimensional representation instead of learning rich latent structure. With only ~3000 training samples and 78 positive cases, the model couldn't learn the intended factorization.

### Reinforcement Learning Pipeline

**Environment Design**
```python
State:  [macro_features_all_countries, global_factors, current_weights]
        Dimension: 1878 (117 countries × 15 features + 6 global + 117 weights)

Action: Portfolio weight adjustments (continuous, softmax normalized)
        Dimension: 117

Reward: Sharpe-like ratio = (excess_return) / (volatility + epsilon)
        Where: excess_return = yield - default_losses - transaction_costs - risk_free_rate
```

**PPO Agent Architecture**
- Actor network: 256 → 128 → 64 → 117 (Gaussian policy)
- Critic network: 256 → 128 → 64 → 1
- Layer normalization for stable training
- GAE (Generalized Advantage Estimation) with λ=0.95
- Clip ratio: 0.2
- Entropy bonus: 0.01
- Total parameters: ~1.06M (538K actor + 522K critic)

**Training Configuration**
- Episodes: 300
- Discount factor (γ): 0.99
- Actor learning rate: 3e-4
- Critic learning rate: 1e-3
- PPO epochs per episode: 10
- Gradient clipping: 0.5

**Environment Variants**
1. **Deterministic**: Historical defaults occur as recorded
2. **Stochastic**: Default probabilities based on historical rates with randomization
3. **Contagion**: Regional spillover effects on default probability

### Yield and Loss Modeling

**Bond Yields**
```python
yield = base_rate + spread
spread = 0.02 + 0.0005 × debt_ratio - 0.005 × reserves_months
spread = clip(spread, 0.005, 0.25)  # 50bps to 2500bps
```

**Recovery Rates**
```python
recovery = 0.35 + 0.15 × min(1.0, gdp_per_capita / 40000)
# Range: 35% (poor countries) to 50% (rich countries)
```

**Transaction Costs**
```python
cost = 0.003 × turnover  # 30bps round-trip
```

## Results

### Note on the numbers below

The tables that follow report the headline figures from an earlier end-to-end
execution of the notebook. The methodology fixes in this revision (train-only
imputation, hyperparameter selection via TimeSeriesSplit cross-validation,
bootstrap 95% confidence intervals, calibration diagnostics, env / yield-model
consistency, PPO log-density clipping) are implemented in `src/` and the
notebook, exercised by 47 unit tests, and gated by CI. They have not been
re-executed end-to-end on this commit because the full pipeline (TensorFlow
Two-Tower NN + three PPO trainings of 300 episodes each on a 1,878-dimensional
state) exceeds the memory budget of the author's machine. The corrected
pipeline is the source of truth; a re-execution on a larger machine is the
open work item.

A standalone Python script `scripts/refresh_baseline_numbers.py` is provided
for re-running just the classical-baseline rows (LR / RF / sklearn GB /
XGBoost) on a memory-constrained machine; see the Reproduce section.

### Prediction Performance (Test Set: 2015-2023)

The notebook now reports each headline metric with a bootstrap 95% CI
(2,000 resamples, percentile method) and an expected calibration error
alongside AUC / AP / Brier. With 18 positive cases in the test set, CIs on
AUC are wide; rankings inside the overlap should be treated as inconclusive.
Numbers below are from the pre-tuning, pre-CI run and will be replaced when
the notebook is re-executed leakage-free on a larger machine.

| Model | AUC-ROC | Avg Precision | Brier Score |
|-------|---------|---------------|-------------|
| Logistic Regression | 0.636 | 0.041 | 0.0821 |
| Gradient Boosting (sklearn) | 0.793 | 0.065 | 0.0221 |
| XGBoost | pending re-run | pending | pending |
| Random Forest | 0.828 | 0.085 | 0.0569 |
| Two-Tower NN | 0.675 | 0.064 | 0.0269 |

Notes on this table: the four pre-XGBoost numbers above are from the original
run with leakage in the imputation step. They will be replaced when the
notebook is re-executed with the leakage fix and the new XGBoost baseline.
With only 18 positive cases in the test set, a bootstrap CI on AUC is large;
do not treat the ranking as significant without one.

### RL Performance (Cumulative Returns)

| Environment | Equal Weight | RL Policy | Improvement | Notes |
|-------------|--------------|-----------|-------------|-------|
| Deterministic (single run) | 16.67 | 18.78 | +12.6% | |
| Deterministic (range across seeds) | 16.67 | 18.78 - 20.02 | +12.6% to +20.1% | "ran several times, different results" |
| Stochastic | 23.50 | 25.54 | +9% | |
| Contagion | 20.78 | 21.83 | +5% | |

The +12.6 to +20.1% range on the deterministic environment is run-to-run
variation, not a confidence interval. A bootstrap CI on the cumulative-return distribution would tighten this.

### Strategy Comparison

| Strategy | Reward | vs Baseline |
|----------|--------|-------------|
| Equal Weight | 16.67 | - |
| Low Volatility | 17.66 | +5.9% |
| Low Debt | 12.23 | -26.7% |
| RL Policy | 18.78 | **+12.6%** |

### Portfolio Characteristics

**Top Allocations (RL Policy)**
- Guyana: 1.00%
- Slovenia: 0.98%
- South Korea: 0.98%
- Trinidad & Tobago: 0.96%
- Mozambique: 0.95%

**Bottom Allocations (RL Policy)**
- Venezuela: 0.70%
- Argentina: 0.72%
- Ukraine: 0.73%
- Belize: 0.74%
- Barbados: 0.76%

Equal weight baseline: 0.85% per country

The agent learned to underweight serial defaulters (Venezuela, Argentina, Ukraine, Ecuador) and overweight countries with stable fundamentals. However, weight spread is narrow (0.70% - 1.00%), suggesting conservative risk-taking.

**Risk Metrics**
- Defaulter allocation: 26.8%
- Equal weight would be: 28.2%
- Underweight by: 1.4%
- Correlation with default rate: -0.585

### Sensitivity Analysis

**Feature Perturbation Test**: 10% increase in individual macro variables
- Result: Zero change in portfolio weights
- Interpretation: Agent doesn't respond to marginal feature changes

**Random State Test**: Completely different input state
- Result: 53.49 total weight difference
- Interpretation: Agent IS state-dependent at macro level

**Conclusion**: The PPO agent learned to recognize overall state patterns and map them to a relatively fixed allocation, but doesn't perform economic reasoning about individual variables. It's pattern matching, not causal understanding. The improvement is real, but comes from learning historical risk patterns rather than dynamic assessment.

## Limitations

### Data Limitations
- **Limited samples**: Only ~3000 training observations makes deep learning struggle
- **Class imbalance**: 2.4% default rate means 78 positive cases in training
- **Missing data**: Some countries have 70%+ missing values, median imputation introduces bias
- **Temporal clustering**: Many 1990 defaults are carryover from 1980s debt crisis, not true predictions
- **Survivorship bias**: Only includes countries that existed throughout the period
- **Default label distribution**: 56 of the 88 default events (64%) are stamped
  in the 1990s, many of them carryover from the 1980s debt crisis or formal
  acknowledgements of pre-existing default status by newly-independent former
  Soviet states. These are not genuine 1990-era forecasts. The cleaner subset
  is the 32 post-2000 default events (see notebook cell 6 output).

### Methodological Limitations
- **No yield curve data**: Real sovereign analysis requires term structure
- **No political risk**: Defaults often driven by political factors not captured in macro data
- **Static features**: No lag features or momentum indicators
- **Simplified RL environment**: No liquidity constraints, no shorting, unrealistic transaction costs

### Key Lessons
1. **Simpler models win with limited data**: Random Forest beats neural network with 3000 samples
2. **Architectural assumptions don't transfer**: Latent space factorization from recommender systems doesn't help here
3. **RL learns shortcuts**: Agent found pattern matching solution instead of economic reasoning
4. **Honest evaluation matters**: Sensitivity analysis revealed the policy isn't doing what we hoped

## Future Directions

### Medium-term Extensions
- Use prediction probabilities as RL state features
- Attention mechanisms for per-country encoding in RL
- Hierarchical RL with regional sub-policies
- Add regime detection (crisis vs. normal periods)

### Long-term Research Questions
- Can we learn interpretable risk factors from the embeddings?
- Does transfer learning from corporate defaults help?
- How to incorporate text data (IMF reports, news sentiment)?
- Can we build a causal model of contagion?

## References

### Data Sources
- [World Bank Open Data](https://data.worldbank.org/)
- [FRED Economic Data](https://fred.stlouisfed.org/)
- Reinhart, C. M., & Rogoff, K. S. (2009). This Time is Different: Eight Centuries of Financial Folly
- S&P Global Ratings Sovereign Default Studies

### Related Work
- Savona, R., & Vezzoli, M. (2015). Fitting and forecasting sovereign defaults using multiple risk signals
- Manasse, P., & Roubini, N. (2009). Rules of thumb for sovereign debt crises
- Chakrabarti, A., & Zeaiter, H. (2014). The determinants of sovereign default: A sensitivity analysis

## Author

**Maksim Silchenko**  
