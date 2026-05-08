"""Risk-adjusted performance metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd


TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    n_years = len(equity) / TRADING_DAYS
    if n_years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    std = excess.std()
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / std) if std > 0 else 0.0


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0].std()
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / downside) if downside > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def calmar(equity: pd.Series, returns: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    return float(cagr(equity) / mdd) if mdd > 0 else 0.0


def summarize(equity: pd.Series, returns: pd.Series, trades: pd.DataFrame) -> dict:
    wins = trades[trades["pnl"] > 0] if not trades.empty else trades
    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": float(equity.iloc[-1] / equity.iloc[0] - 1) * 100,
        "cagr_pct": cagr(equity) * 100,
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown_pct": max_drawdown(equity) * 100,
        "calmar": calmar(equity, returns),
        "n_trades": int(len(trades)),
        "win_rate_pct": float(len(wins) / len(trades) * 100) if len(trades) else 0.0,
        "avg_trade_pnl": float(trades["pnl"].mean()) if len(trades) else 0.0,
    }
