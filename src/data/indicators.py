"""Vectorized technical indicators (no TA-Lib dependency)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger(close: pd.Series, n: int = 20, k: float = 2):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper, lower = mid + k * std, mid - k * std
    width = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, width, pct_b


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_k = df["low"].rolling(k).min()
    high_k = df["high"].rolling(k).max()
    k_line = 100 * (df["close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    return k_line, k_line.rolling(d).mean()


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Generate full feature matrix used by the model.

    Returns: dataframe with engineered features + a binary target
    `target` = 1 if next-day close > today's close, else 0.
    """
    out = df.copy()
    close = out["close"]

    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["log_ret"] = np.log(close / close.shift(1))
    out["volatility_10"] = out["ret_1"].rolling(10).std()
    out["volatility_20"] = out["ret_1"].rolling(20).std()

    for n in cfg["features"]["ma_windows"]:
        out[f"sma_{n}"] = sma(close, n)
        out[f"ema_{n}"] = ema(close, n)
        out[f"close_over_sma_{n}"] = close / out[f"sma_{n}"] - 1

    out["rsi"] = rsi(close, cfg["features"]["rsi_period"])

    macd_line, sig, hist = macd(
        close,
        cfg["features"]["macd_fast"],
        cfg["features"]["macd_slow"],
        cfg["features"]["macd_signal"],
    )
    out["macd"], out["macd_signal"], out["macd_hist"] = macd_line, sig, hist

    mid, up, lo, width, pctb = bollinger(
        close, cfg["features"]["bb_period"], cfg["features"]["bb_std"]
    )
    out["bb_width"], out["bb_pct"] = width, pctb

    out["atr"] = atr(out, cfg["features"]["atr_period"])
    out["atr_pct"] = out["atr"] / close
    out["obv"] = obv(out)
    out["obv_chg_5"] = out["obv"].pct_change(5)
    out["vwap"] = vwap(out)
    out["close_over_vwap"] = close / out["vwap"] - 1

    k, d = stochastic(out)
    out["stoch_k"], out["stoch_d"] = k, d

    out["volume_chg"] = out["volume"].pct_change()
    out["volume_sma_20"] = out["volume"].rolling(20).mean()
    out["rel_volume"] = out["volume"] / out["volume_sma_20"]

    out["target"] = (close.shift(-1) > close).astype(int)
    return out.dropna()
