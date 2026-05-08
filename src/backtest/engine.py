"""Vectorized event-driven backtester with brokerage + slippage modeling."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .metrics import summarize


class Backtester:
    """Long-only signal-based backtester.

    Position sizing: at each BUY signal, allocate `position_size_pct` of
    current cash to the asset at the next bar's open (avoids look-ahead).
    Exits when signal flips to 0.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        brokerage_pct: float = 0.0003,
        slippage_pct: float = 0.0005,
        position_size_pct: float = 0.95,
        signal_threshold: float = 0.55,
    ):
        self.cap0 = initial_capital
        self.brokerage = brokerage_pct
        self.slippage = slippage_pct
        self.size_pct = position_size_pct
        self.threshold = signal_threshold

    def run(self, prices: pd.DataFrame, proba: np.ndarray) -> dict:
        """`prices` must contain at least open & close, indexed by date.

        proba is the model's P(up) aligned to `prices.index`.
        Trades execute at next-bar open to avoid look-ahead.
        """
        df = prices.copy()
        df["proba"] = proba
        df["signal"] = (df["proba"] > self.threshold).astype(int)
        df["next_open"] = df["open"].shift(-1)
        df = df.dropna(subset=["next_open"])

        cash = self.cap0
        shares = 0.0
        equity_curve, trades = [], []
        in_pos = False
        entry_price, entry_date, entry_shares = 0.0, None, 0.0

        for date, row in df.iterrows():
            sig = int(row["signal"])
            exec_price = float(row["next_open"])

            if not in_pos and sig == 1:
                buy_price = exec_price * (1 + self.slippage)
                alloc = cash * self.size_pct
                shares = alloc / buy_price
                cost = shares * buy_price * (1 + self.brokerage)
                cash -= cost
                in_pos = True
                entry_price, entry_date, entry_shares = buy_price, date, shares

            elif in_pos and sig == 0:
                sell_price = exec_price * (1 - self.slippage)
                proceeds = entry_shares * sell_price * (1 - self.brokerage)
                cash += proceeds
                pnl = proceeds - (entry_shares * entry_price * (1 + self.brokerage))
                trades.append({
                    "entry_date": entry_date, "exit_date": date,
                    "entry_price": entry_price, "exit_price": sell_price,
                    "shares": entry_shares, "pnl": pnl,
                    "return_pct": (sell_price / entry_price - 1) * 100,
                })
                in_pos, shares, entry_shares = False, 0.0, 0.0

            mark = float(row["close"])
            equity = cash + (entry_shares * mark if in_pos else 0.0)
            equity_curve.append((date, equity))

        if in_pos:
            last_price = float(df["close"].iloc[-1]) * (1 - self.slippage)
            proceeds = entry_shares * last_price * (1 - self.brokerage)
            cash += proceeds
            pnl = proceeds - (entry_shares * entry_price * (1 + self.brokerage))
            trades.append({
                "entry_date": entry_date, "exit_date": df.index[-1],
                "entry_price": entry_price, "exit_price": last_price,
                "shares": entry_shares, "pnl": pnl,
                "return_pct": (last_price / entry_price - 1) * 100,
            })

        equity = pd.Series(dict(equity_curve), name="equity")
        returns = equity.pct_change().fillna(0.0)
        trades_df = pd.DataFrame(trades)

        bh = self.cap0 * (df["close"] / df["close"].iloc[0])
        bh_returns = bh.pct_change().fillna(0.0)

        return {
            "equity": equity,
            "returns": returns,
            "trades": trades_df,
            "buy_hold_equity": bh,
            "metrics": summarize(equity, returns, trades_df),
            "buy_hold_metrics": summarize(bh, bh_returns, pd.DataFrame()),
        }
