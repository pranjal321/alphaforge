"""Smoke tests for indicators + feature pipeline."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.indicators import rsi, macd, bollinger, atr, build_features


@pytest.fixture
def ohlcv():
    np.random.seed(0)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.5,
        "high": close + np.abs(np.random.randn(n)),
        "low": close - np.abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(1e5, 1e6, n),
    }, index=pd.date_range("2022-01-01", periods=n, freq="B"))


def test_rsi_bounds(ohlcv):
    r = rsi(ohlcv["close"]).dropna()
    assert (r.between(0, 100)).all()


def test_macd_shapes(ohlcv):
    m, s, h = macd(ohlcv["close"])
    assert len(m) == len(s) == len(h) == len(ohlcv)


def test_bollinger_order(ohlcv):
    _, up, lo, _, _ = bollinger(ohlcv["close"])
    valid = up.dropna().index.intersection(lo.dropna().index)
    assert (up.loc[valid] >= lo.loc[valid]).all()


def test_atr_positive(ohlcv):
    a = atr(ohlcv).dropna()
    assert (a > 0).all()


def test_build_features_no_nan(ohlcv):
    cfg = {"features": {
        "ma_windows": [5, 20], "rsi_period": 14,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "bb_period": 20, "bb_std": 2, "atr_period": 14,
    }}
    out = build_features(ohlcv, cfg)
    assert not out.isna().any().any()
    assert "target" in out.columns
    assert out["target"].isin([0, 1]).all()
