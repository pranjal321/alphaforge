"""Synthetic OHLCV generator for offline testing.

Generates realistic price series with momentum + mean reversion regimes
so the ML pipeline has learnable patterns. Used when yfinance is rate-limited
or you have no internet.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def generate_ohlcv(
    ticker: str = "SYNTH",
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    seed: int | None = None,
    initial_price: float = 1000.0,
    annual_vol: float = 0.25,
    annual_drift: float = 0.10,
    momentum_strength: float = 0.15,
) -> pd.DataFrame:
    """Generate plausible daily OHLCV.

    Uses GBM + AR(1) momentum component so the direction signal is
    partially predictable from past returns (realistic but learnable).
    """
    if seed is None:
        seed = abs(hash(ticker)) % (2**31 - 1)
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    dt = 1.0 / 252.0

    eps = rng.standard_normal(n)
    momentum = np.zeros(n)
    for i in range(1, n):
        momentum[i] = momentum_strength * momentum[i - 1] + eps[i]

    daily_drift = (annual_drift - 0.5 * annual_vol**2) * dt
    daily_vol = annual_vol * np.sqrt(dt)
    log_ret = daily_drift + daily_vol * (0.7 * eps + 0.3 * momentum)
    close = initial_price * np.exp(np.cumsum(log_ret))

    intraday_range = np.abs(rng.standard_normal(n)) * close * 0.012
    open_offset = rng.standard_normal(n) * close * 0.004
    open_ = close - open_offset
    high = np.maximum(close, open_) + intraday_range * 0.5
    low = np.minimum(close, open_) - intraday_range * 0.5
    volume = rng.integers(5e5, 5e6, n) * (1 + 0.3 * np.abs(log_ret) / daily_vol)

    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume.astype(int),
    }, index=pd.Index(dates, name="date"))


def generate_universe(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    return {t: generate_ohlcv(t, start, end) for t in tickers}
