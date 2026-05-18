# Sovereign Risk ML

Predicting sovereign defaults and optimizing bond portfolios using machine learning on macroeconomic fundamentals.

## Overview

This project builds an end-to-end pipeline for sovereign credit risk analysis:
1. **Data Collection**: Automated fetching from World Bank and FRED APIs
2. **Default Prediction**: Comparison of neural networks vs. traditional ML models
3. **Portfolio Optimization**: Reinforcement learning for risk-aware bond allocation

The analysis covers 117 countries from 1990-2023, with temporal train/test splits to prevent data leakage.

## Key Findings

### Prediction Performance
- **Random Forest achieves best AUC (0.828)** on out-of-sample test data
- Two-tower neural network underperforms simpler models (AUC 0.675)
- Neural embeddings collapse to near 1-dimensional representation (94% variance in first PC)
- Class imbalance (2.4% default rate) requires careful handling with focal loss and class weights

### Reinforcement Learning Results
- **PPO agent achieves 13-20% improvement** over equal-weight baseline
- Consistent outperformance across deterministic, stochastic, and contagion scenarios
- Correlation between default rate and portfolio weight: -0.585
- However, learned policy is mostly static (doesn't respond dynamically to feature perturbations)
- Agent learns historical risk patterns rather than economic reasoning about individual variables
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
- L2 regularization with C=0.1
- Balanced class weights
- AUC: 0.636, Average Precision: 0.041

**2. Random Forest**
- 100 trees, max depth 5
- Balanced class weights
- AUC: 0.828, Average Precision: 0.085
- **Best performer for discrimination**

**3. Gradient Boosting**
- 100 estimators, learning rate 0.1
- Max depth 3
- AUC: 0.793, Average Precision: 0.065

**4. Two-Tower Neural Network**
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

### Prediction Performance (Test Set: 2015-2023)

| Model | AUC-ROC | Avg Precision | Brier Score |
|-------|---------|---------------|-------------|
| Logistic Regression | 0.636 | 0.041 | 0.0821 |
| Gradient Boosting | 0.793 | 0.065 | 0.0221 |
| Random Forest | **0.828** | **0.085** | 0.0569 |
| Two-Tower NN | 0.675 | 0.064 | 0.0269 |

### RL Performance (Cumulative Returns)

| Environment | Equal Weight | RL Policy | Improvement |
|-------------|--------------|-----------|-------------|
| Deterministic | 16.67 | 18.78 - 20.02 | **+13-20%** |
| Stochastic | 23.50 | 25.54 | +9% |
| Contagion | 20.78 | 21.83 | +5% |

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
