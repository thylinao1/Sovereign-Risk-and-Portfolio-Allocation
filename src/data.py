"""Data fetchers for the sovereign-risk panel.

The notebook builds its dataset on the fly from two public APIs. These
helpers wrap each API in a function that returns a pandas DataFrame. None of
the raw data is committed to the repo. See data/README.md for the indicator
catalogue and the FRED_API_KEY pattern.

All functions return an empty DataFrame on error rather than raising, so
the notebook can keep going if a single indicator is temporarily missing.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List

import pandas as pd
import requests


WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2/country"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def get_fred_api_key() -> str:
    """Read the FRED API key from the FRED_API_KEY env var. Raises if missing.

    Replaces the previous hard-coded placeholder. See data/README.md for how
    to obtain a key.
    """
    key = os.environ.get("FRED_API_KEY")
    if not key or key == "----":
        raise RuntimeError(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and "
            "'export FRED_API_KEY=<key>' before running the notebook."
        )
    return key


def fetch_worldbank_indicator(
    indicator_code: str,
    countries: Iterable[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch a single World Bank indicator for a list of country codes.

    Returns DataFrame with columns [country_code, country_name, year, value,
    indicator]. Empty on error.
    """
    countries_str = ";".join(countries)
    url = f"{WORLD_BANK_BASE_URL}/{countries_str}/indicator/{indicator_code}"
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": 10000,
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if len(data) < 2 or data[1] is None:
            print(f"No data for {indicator_code}")
            return pd.DataFrame()
        records = []
        for entry in data[1]:
            records.append({
                "country_code": entry["country"]["id"],
                "country_name": entry["country"]["value"],
                "year": int(entry["date"]),
                "value": entry["value"],
            })
        df = pd.DataFrame(records)
        df["indicator"] = indicator_code
        return df
    except Exception as e:
        print(f"Error fetching {indicator_code}: {e}")
        return pd.DataFrame()


def fetch_all_indicators_batch(
    countries: List[str],
    start_year: int,
    end_year: int,
    indicators: Dict[str, str],
    batch_size: int = 20,
    sleep_between_indicators: float = 0.3,
    sleep_between_batches: float = 1.0,
) -> pd.DataFrame:
    """Fetch every indicator in `indicators` (code -> column-name) for every
    country in `countries`, in batches of `batch_size`. Returns a long-form
    panel with one row per (country, year) and one column per indicator."""
    country_year_data: Dict = {}
    batches = [countries[i:i + batch_size] for i in range(0, len(countries), batch_size)]
    for batch_idx, batch in enumerate(batches):
        print(f"Batch {batch_idx + 1}/{len(batches)}: {', '.join(batch[:3])}...")
        for indicator_code, indicator_name in indicators.items():
            df = fetch_worldbank_indicator(indicator_code, batch, start_year, end_year)
            if not df.empty:
                for _, row in df.iterrows():
                    key = (row["country_code"], row["year"])
                    if key not in country_year_data:
                        country_year_data[key] = {
                            "country_code": row["country_code"],
                            "year": row["year"],
                        }
                    country_year_data[key][indicator_name] = row["value"]
            time.sleep(sleep_between_indicators)
        time.sleep(sleep_between_batches)
    if not country_year_data:
        return pd.DataFrame()
    return pd.DataFrame(list(country_year_data.values()))


def fetch_fred_series(
    series_id: str,
    start_year: int,
    end_year: int,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch a single FRED series at annual frequency (average aggregation).

    Returns DataFrame with columns [year, value]. Empty on error.
    """
    key = api_key or get_fred_api_key()
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": f"{start_year}-01-01",
        "observation_end": f"{end_year}-12-31",
        "frequency": "a",
        "aggregation_method": "avg",
    }
    try:
        response = requests.get(FRED_BASE_URL, params=params, timeout=30)
        data = response.json()
        if "observations" not in data:
            print(f"No observations for {series_id}")
            return pd.DataFrame()
        records = []
        for obs in data["observations"]:
            if obs["value"] != ".":
                records.append({
                    "year": int(obs["date"][:4]),
                    "value": float(obs["value"]),
                })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return pd.DataFrame()
