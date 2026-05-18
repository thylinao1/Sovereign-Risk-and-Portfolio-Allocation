"""Tests for src/data.py. Network is mocked so the suite runs offline."""
import os
import json
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import (
    get_fred_api_key,
    fetch_worldbank_indicator,
    fetch_fred_series,
)


def test_get_fred_api_key_returns_env_value(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    assert get_fred_api_key() == "abc123"


def test_get_fred_api_key_raises_when_unset(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY is not set"):
        get_fred_api_key()


def test_get_fred_api_key_raises_on_placeholder(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "----")
    with pytest.raises(RuntimeError, match="FRED_API_KEY is not set"):
        get_fred_api_key()


def test_worldbank_fetch_parses_response_correctly():
    fake_response = MagicMock()
    fake_response.json.return_value = [
        {"page": 1, "pages": 1, "per_page": 50, "total": 2},
        [
            {"country": {"id": "ARG", "value": "Argentina"},
             "date": "2000", "value": 1.5},
            {"country": {"id": "ARG", "value": "Argentina"},
             "date": "2001", "value": -10.9},
        ]
    ]
    fake_response.raise_for_status = MagicMock()
    with patch("src.data.requests.get", return_value=fake_response):
        df = fetch_worldbank_indicator("NY.GDP.MKTP.KD.ZG", ["ARG"], 2000, 2001)
    assert len(df) == 2
    assert set(df.columns) == {"country_code", "country_name", "year", "value", "indicator"}
    assert df["year"].tolist() == [2000, 2001]
    assert df["indicator"].iloc[0] == "NY.GDP.MKTP.KD.ZG"


def test_worldbank_fetch_returns_empty_on_no_data():
    fake_response = MagicMock()
    fake_response.json.return_value = [{"message": "no data"}, None]
    fake_response.raise_for_status = MagicMock()
    with patch("src.data.requests.get", return_value=fake_response):
        df = fetch_worldbank_indicator("XYZ", ["ARG"], 2000, 2001)
    assert df.empty


def test_worldbank_fetch_returns_empty_on_exception():
    with patch("src.data.requests.get", side_effect=ConnectionError("oops")):
        df = fetch_worldbank_indicator("NY.GDP.MKTP.KD.ZG", ["ARG"], 2000, 2001)
    assert df.empty


def test_fred_fetch_parses_observations():
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "observations": [
            {"date": "2000-01-01", "value": "20.5"},
            {"date": "2001-01-01", "value": "."},  # missing value, filtered out
            {"date": "2002-01-01", "value": "18.3"},
        ]
    }
    with patch("src.data.requests.get", return_value=fake_response):
        df = fetch_fred_series("VIXCLS", 2000, 2002, api_key="fake")
    assert len(df) == 2
    assert df["year"].tolist() == [2000, 2002]
    assert df["value"].tolist() == [20.5, 18.3]


def test_fred_fetch_uses_env_key_when_not_supplied(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "envkey")
    captured = {}

    def fake_get(url, params, timeout):
        captured["params"] = params
        m = MagicMock()
        m.json.return_value = {"observations": []}
        return m

    with patch("src.data.requests.get", side_effect=fake_get):
        fetch_fred_series("VIXCLS", 2000, 2001)
    assert captured["params"]["api_key"] == "envkey"


def test_fred_fetch_uses_supplied_key_over_env(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "envkey")
    captured = {}

    def fake_get(url, params, timeout):
        captured["params"] = params
        m = MagicMock()
        m.json.return_value = {"observations": []}
        return m

    with patch("src.data.requests.get", side_effect=fake_get):
        fetch_fred_series("VIXCLS", 2000, 2001, api_key="explicit")
    assert captured["params"]["api_key"] == "explicit"
