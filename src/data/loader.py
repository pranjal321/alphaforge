"""OHLCV data loader with on-disk caching.

Tries yfinance first; falls back to synthetic data if Yahoo is unreachable
or rate-limiting. Pass `source='synthetic'` to skip yfinance entirely.
"""
from __future__ import annotations
import time
import warnings
from pathlib import Path
import pandas as pd
import yfinance as yf

from .synthetic import generate_ohlcv


class DataLoader:
    def __init__(self, cache_dir: str = "data/", source: str = "auto"):
        """source: 'auto' | 'yfinance' | 'synthetic'."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.source = source

    def _cache_path(self, ticker: str, start: str, end: str) -> Path:
        safe = ticker.replace(".", "_")
        return self.cache_dir / f"{safe}_{start}_{end}.parquet"

    def _fetch_yfinance(self, ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
                if df.empty:
                    raise ValueError("empty frame")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                df.index.name = "date"
                return df
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"yfinance failed for {ticker}: {last_err}")

    def fetch(self, ticker: str, start: str, end: str, force: bool = False) -> pd.DataFrame:
        path = self._cache_path(ticker, start, end)
        if path.exists() and not force:
            return pd.read_parquet(path)

        if self.source == "synthetic":
            df = generate_ohlcv(ticker, start, end)
        else:
            try:
                df = self._fetch_yfinance(ticker, start, end)
            except Exception as e:
                if self.source == "yfinance":
                    raise
                warnings.warn(f"yfinance unavailable ({e}); using synthetic data for {ticker}")
                df = generate_ohlcv(ticker, start, end)

        df.to_parquet(path)
        return df

    def fetch_many(self, tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        return {t: self.fetch(t, start, end) for t in tickers}
