#!/usr/bin/env python3
"""Refresh classical-baseline numbers for the README.

Runs the leakage-fixed, CV-tuned pipeline for LR / RF / sklearn GB / XGBoost
on the World Bank + FRED panel. Skips the Two-Tower NN and PPO entirely so
this runs on a memory-constrained machine (under 1GB peak RAM, ~5 minutes).

Required:  FRED_API_KEY env var. Get a free key at
           https://fred.stlouisfed.org/docs/api/api_key.html.
Deps:      numpy, pandas, requests, scikit-learn, scipy. Optionally xgboost.

Usage:     export FRED_API_KEY=<your_key>
           python scripts/refresh_baseline_numbers.py

Output:    A single block of numbers (AUC / AP / Brier / ECE with bootstrap
           95% CIs) that can be pasted into the README Prediction Performance
           table.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.data import fetch_all_indicators_batch, fetch_fred_series, get_fred_api_key
from src.tune import time_series_grid_search
from src.eval import auc_ci, ap_ci, brier_ci, expected_calibration_error


START_YEAR = 1990
END_YEAR = 2023

# Country codes kept in sync with notebooks/01_sovereign_default_modeling.ipynb
# cell 3. If you change one, change the other.
COUNTRIES = [
    # Historical defaulters (our positive class)
    'ARG', 'GRC', 'ECU', 'VEN', 'UKR', 'RUS', 'PAK', 'JAM', 'BLZ', 'SUR',
    'MOZ', 'COG', 'ZMB', 'MLI', 'NER', 'TGO', 'SEN', 'CIV', 'CMR', 'GAB',
    'NGA', 'GHA', 'KEN', 'ETH', 'TZA', 'UGA', 'MDG', 'MWI', 'ZWE', 'AGO',
    'PER', 'BOL', 'PRY', 'URY', 'CRI', 'PAN', 'DOM', 'SLV', 'GTM', 'HND',
    'NIC', 'CUB', 'HTI', 'TTO', 'GUY', 'BRB', 'SRB', 'BIH', 'MKD', 'ALB',
    'MDA', 'BLR', 'GEO', 'ARM', 'AZE', 'KAZ', 'UZB', 'TKM', 'KGZ', 'TJK',

    # Stable developed economies (negative class)
    'USA', 'DEU', 'JPN', 'GBR', 'CAN', 'AUS', 'FRA', 'ITA', 'ESP', 'NLD',
    'BEL', 'CHE', 'AUT', 'SWE', 'NOR', 'DNK', 'FIN', 'IRL', 'PRT', 'NZL',
    'SGP', 'HKG', 'KOR', 'TWN', 'ISR', 'CZE', 'POL', 'HUN', 'SVK', 'SVN',

    # Emerging markets (mixed outcomes)
    'BRA', 'MEX', 'CHL', 'COL', 'CHN', 'IND', 'IDN', 'THA', 'MYS', 'PHL',
    'VNM', 'BGD', 'LKA', 'TUR', 'ZAF', 'EGY', 'MAR', 'TUN', 'SAU', 'ARE',
    'QAT', 'KWT', 'OMN', 'BHR', 'JOR', 'LBN', 'IRQ', 'IRN']

DOMESTIC_INDICATORS = {
    'GC.DOD.TOTL.GD.ZS':    'central_govt_debt_pct_gdp',
    'DT.DOD.DECT.GN.ZS':    'external_debt_pct_gni',
    'DT.TDS.DECT.EX.ZS':    'debt_service_pct_exports',
    'GC.BAL.CASH.GD.ZS':    'cash_surplus_deficit_pct_gdp',
    'GC.REV.XGRT.GD.ZS':    'govt_revenue_pct_gdp',
    'GC.XPN.TOTL.GD.ZS':    'govt_expenditure_pct_gdp',
    'BN.CAB.XOKA.GD.ZS':    'current_account_pct_gdp',
    'FI.RES.TOTL.MO':       'reserves_months_imports',
    'NE.TRD.GNFS.ZS':       'trade_pct_gdp',
    'BX.KLT.DINV.WD.GD.ZS': 'fdi_inflows_pct_gdp',
    'NY.GDP.MKTP.KD.ZG':    'gdp_growth_annual',
    'NY.GDP.PCAP.KD':       'gdp_per_capita_constant',
    'FP.CPI.TOTL.ZG':       'inflation_cpi_annual',
    'SL.UEM.TOTL.ZS':       'unemployment_rate',
    'FM.LBL.BMNY.GD.ZS':    'broad_money_pct_gdp',
    'FS.AST.DOMS.GD.ZS':    'domestic_credit_pct_gdp',
}

GLOBAL_SERIES = {
    'VIXCLS':       'vix_annual_avg',
    'DGS10':        'us_10y_treasury',
    'DTWEXBGS':     'usd_broad_index',
    'BAMLH0A0HYM2': 'high_yield_spread',
    'TEDRATE':      'ted_spread',
    'T10Y2Y':       'yield_curve_slope',
}

DEFAULT_EVENTS = {
    # Latin America
    'ARG': [1989, 2001, 2014, 2020],
    'ECU': [1999, 2008, 2020],
    'VEN': [1995, 1998, 2004, 2017],
    'PER': [1990],
    'BRA': [1990],
    'URY': [1990, 2003],
    'PRY': [2003],
    'DOM': [1999, 2005],
    'CRI': [1990],
    'PAN': [1990],
    'BOL': [1990],
    'NIC': [1990],
    'GTM': [1990],
    'HND': [1990],
    'SLV': [1990],
    'JAM': [2010, 2013],
    'BLZ': [2006, 2012, 2017],
    'SUR': [2020],
    'GUY': [1990],
    'TTO': [1990],
    'CUB': [1990],

    # Europe
    'GRC': [2012, 2015],
    'RUS': [1991, 1998, 2022],
    'UKR': [1998, 2015, 2022],
    'SRB': [1992, 2000],
    'ALB': [1991],
    'MDA': [1998, 2002],
    'BLR': [1998],
    'MKD': [1992],
    'BIH': [1992],

    # Former Soviet States
    'GEO': [1991],
    'ARM': [1991],
    'AZE': [1991],
    'KAZ': [1991],
    'UZB': [1991],
    'TKM': [1991],
    'KGZ': [1991],
    'TJK': [1991],

    # Africa
    'NGA': [2002, 2005],
    'GHA': [1990],
    'ZAF': [1990],
    'ZMB': [1990],
    'ZWE': [1990, 2000, 2006],
    'CIV': [1990, 2011],
    'CMR': [1990],
    'COG': [1990],
    'GAB': [1990, 2002],
    'MOZ': [1990],
    'MDG': [1990],
    'KEN': [1990],
    'ETH': [1990],
    'TZA': [1990],
    'UGA': [1990],
    'SEN': [1990],
    'MLI': [1990],
    'NER': [1990],
    'TGO': [1990],
    'AGO': [1990],
    'MWI': [1990],

    # Asia
    'PAK': [1999],
    'LKA': [2022],
    'LBN': [2020],
    'IRQ': [1990],}


def build_default_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Add default_1y and default_2y labels using DEFAULT_EVENTS.

    A row (country, year) is labelled default_2y=1 if there is a default event
    for that country within the next 1 or 2 years.
    """
    panel = panel.copy()
    panel['default_1y'] = 0
    panel['default_2y'] = 0
    for country, years in DEFAULT_EVENTS.items():
        for ev_year in years:
            mask_1y = (panel['country_code'] == country) & (panel['year'] == ev_year - 1)
            panel.loc[mask_1y, 'default_1y'] = 1
            mask_2y = (panel['country_code'] == country) & (
                panel['year'].isin([ev_year - 1, ev_year - 2])
            )
            panel.loc[mask_2y, 'default_2y'] = 1
    return panel


def main() -> int:
    print("=" * 80)
    print("Refresh classical baseline numbers")
    print("=" * 80)

    get_fred_api_key()  # raises with a useful message if FRED_API_KEY is unset

    print("\nFetching World Bank indicators (~3 min)...")
    domestic = fetch_all_indicators_batch(
        COUNTRIES, START_YEAR, END_YEAR, DOMESTIC_INDICATORS,
    )
    print(f"  domestic shape: {domestic.shape}")

    print("\nFetching FRED series...")
    global_data = pd.DataFrame()
    for sid, col in GLOBAL_SERIES.items():
        df = fetch_fred_series(sid, START_YEAR, END_YEAR)
        if df.empty:
            print(f"  WARN: FRED returned empty for {sid}; dropping {col}")
            continue
        df = df.rename(columns={'value': col})
        global_data = df if global_data.empty else global_data.merge(df, on='year', how='outer')
        time.sleep(0.5)
    print(f"  global shape:   {global_data.shape}")

    full = domestic.merge(global_data, on='year', how='left')
    full = build_default_labels(full)
    print(f"\nMerged panel: {full.shape}, {int(full['default_2y'].sum())} positive labels "
          f"({full['default_2y'].mean()*100:.2f}%)")

    # Temporal split + train-only imputation
    train = full[full['year'] < 2015].copy()
    test  = full[full['year'] >= 2015].copy()
    feature_cols = [c for c in full.columns
                    if c not in ('country_code', 'year', 'default_1y', 'default_2y')]
    imputer = SimpleImputer(strategy='median')
    train.loc[:, feature_cols] = imputer.fit_transform(train[feature_cols])
    test.loc[:, feature_cols]  = imputer.transform(test[feature_cols])

    n_train_pos = int(train['default_2y'].sum())
    n_test_pos  = int(test['default_2y'].sum())
    print(f"Train: {len(train)} rows, {n_train_pos} positives ({train['default_2y'].mean()*100:.2f}%)")
    print(f"Test:  {len(test)} rows, {n_test_pos} positives ({test['default_2y'].mean()*100:.2f}%)")

    domestic_features = list(DOMESTIC_INDICATORS.values())
    global_features = [c for c in GLOBAL_SERIES.values() if c in train.columns]
    dropped = [c for c in GLOBAL_SERIES.values() if c not in global_features]
    if dropped:
        print(f"Dropped global features (not in panel): {dropped}")

    scaler_dom = StandardScaler()
    scaler_glob = StandardScaler()
    X_train = np.hstack([
        scaler_dom.fit_transform(train[domestic_features].values),
        scaler_glob.fit_transform(train[global_features].values),
    ])
    X_test = np.hstack([
        scaler_dom.transform(test[domestic_features].values),
        scaler_glob.transform(test[global_features].values),
    ])
    y_train = train['default_2y'].values
    y_test  = test['default_2y'].values
    years_train = train['year'].values

    print("\nCV-tuning hyperparameters (TimeSeriesSplit on training years)...")
    tuned = {}

    best, score, _ = time_series_grid_search(
        LogisticRegression,
        param_grid={'C': [0.01, 0.1, 1.0, 10.0]},
        X=X_train, y=y_train, groups=years_train, n_splits=5,
        base_kwargs={'class_weight': 'balanced', 'max_iter': 2000, 'random_state': 42},
    )
    tuned['LR']  = (LogisticRegression,
                    {**best, 'class_weight': 'balanced', 'max_iter': 2000, 'random_state': 42})
    print(f"  LR  best: {best}  CV AUC {score:.4f}")

    best, score, _ = time_series_grid_search(
        RandomForestClassifier,
        param_grid={'n_estimators': [100, 300], 'max_depth': [3, 5, 7]},
        X=X_train, y=y_train, groups=years_train, n_splits=5,
        base_kwargs={'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
    )
    tuned['RF']  = (RandomForestClassifier,
                    {**best, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
    print(f"  RF  best: {best}  CV AUC {score:.4f}")

    best, score, _ = time_series_grid_search(
        GradientBoostingClassifier,
        param_grid={'n_estimators': [100, 200], 'max_depth': [2, 3],
                    'learning_rate': [0.05, 0.1]},
        X=X_train, y=y_train, groups=years_train, n_splits=5,
        base_kwargs={'random_state': 42},
    )
    tuned['GB']  = (GradientBoostingClassifier, {**best, 'random_state': 42})
    print(f"  GB  best: {best}  CV AUC {score:.4f}")

    try:
        from xgboost import XGBClassifier
        neg_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        best, score, _ = time_series_grid_search(
            XGBClassifier,
            param_grid={'n_estimators': [100, 200], 'max_depth': [2, 3],
                        'learning_rate': [0.05, 0.1]},
            X=X_train, y=y_train, groups=years_train, n_splits=5,
            base_kwargs={'scale_pos_weight': neg_pos, 'eval_metric': 'logloss',
                         'random_state': 42, 'n_jobs': -1},
        )
        tuned['XGB'] = (XGBClassifier,
                        {**best, 'scale_pos_weight': neg_pos, 'eval_metric': 'logloss',
                         'random_state': 42, 'n_jobs': -1})
        print(f"  XGB best: {best}  CV AUC {score:.4f}")
    except ImportError:
        print("  xgboost not installed; skipping")

    print("\nFitting tuned models on full train and scoring on 2015-2023 test...")
    results = {}
    for name, (cls, kwargs) in tuned.items():
        model = cls(**kwargs)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        auc_lo, _, auc_hi = auc_ci(y_test, probs, n_bootstraps=2000, random_state=42)
        ap_lo,  _, ap_hi  = ap_ci(y_test, probs,  n_bootstraps=2000, random_state=42)
        br_lo,  _, br_hi  = brier_ci(y_test, probs, n_bootstraps=2000, random_state=42)
        results[name] = {
            'AUC':   roc_auc_score(y_test, probs),
            'AUC_CI': (auc_lo, auc_hi),
            'AP':    average_precision_score(y_test, probs),
            'AP_CI':  (ap_lo, ap_hi),
            'Brier': brier_score_loss(y_test, probs),
            'Brier_CI': (br_lo, br_hi),
            'ECE':   expected_calibration_error(y_test, probs, n_bins=10),
        }

    print()
    print("=" * 100)
    print(f"RESULTS (test set: 2015-2023, {n_test_pos} positives / {len(y_test)} total)")
    print("=" * 100)
    header = f"{'Model':<6} {'AUC (95% CI)':<26} {'AP (95% CI)':<26} {'Brier (95% CI)':<26} {'ECE':<8}"
    print(header)
    print("-" * 100)
    for name, m in results.items():
        auc_s = f"{m['AUC']:.3f} [{m['AUC_CI'][0]:.3f}, {m['AUC_CI'][1]:.3f}]"
        ap_s  = f"{m['AP']:.3f} [{m['AP_CI'][0]:.3f}, {m['AP_CI'][1]:.3f}]"
        br_s  = f"{m['Brier']:.4f} [{m['Brier_CI'][0]:.4f}, {m['Brier_CI'][1]:.4f}]"
        ece_s = f"{m['ECE']:.4f}"
        print(f"{name:<6} {auc_s:<26} {ap_s:<26} {br_s:<26} {ece_s:<8}")
    print("=" * 100)
    print("\nDone. Paste the table above into a reply and I'll write the README update.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
