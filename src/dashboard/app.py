"""Streamlit dashboard. Run: streamlit run src/dashboard/app.py"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch  # noqa: F401  (load before xgboost on Windows)

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import DataLoader
from src.data.indicators import build_features
from src.models.xgboost_model import XGBDirectionModel
from src.backtest.engine import Backtester


CFG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
CFG = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

st.set_page_config(page_title="AlphaForge", layout="wide", page_icon="📈")
st.title("AlphaForge — ML Equity Forecasting")
st.caption("Production-grade direction prediction + backtesting on NSE equities")

DATA_SOURCE = os.environ.get("ALPHAFORGE_DATA_SOURCE", "auto")

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Ticker", CFG["data"]["tickers"], index=0)
    start = st.date_input("Start", value=date(2020, 1, 1))
    end = st.date_input("End", value=date.today() - timedelta(days=1))
    threshold = st.slider("Signal threshold", 0.50, 0.70, 0.55, 0.01)
    source = st.selectbox("Data source", ["auto", "yfinance", "synthetic"],
                          index=["auto", "yfinance", "synthetic"].index(DATA_SOURCE))
    run = st.button("Run analysis", use_container_width=True)


@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: str, end: str, source: str) -> pd.DataFrame:
    return DataLoader(CFG["data"]["cache_dir"], source=source).fetch(ticker, start, end)


@st.cache_resource
def load_model():
    return XGBDirectionModel.load(CFG["api"]["model_path"])


if run:
    try:
        with st.spinner("Loading data & generating predictions..."):
            df = load_data(ticker, start.isoformat(), end.isoformat(), source)
            feats = build_features(df, CFG)
            model = load_model()
            proba = model.predict_proba(feats)

            bt = Backtester(
                CFG["backtest"]["initial_capital"],
                CFG["backtest"]["brokerage_pct"],
                CFG["backtest"]["slippage_pct"],
                CFG["backtest"]["position_size_pct"],
                threshold,
            )
            r = bt.run(feats[["open", "high", "low", "close"]], proba)

        m, bm = r["metrics"], r["buy_hold_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{m['total_return_pct']:.2f}%", f"{m['total_return_pct'] - bm['total_return_pct']:.2f}% vs B&H")
        c2.metric("Sharpe", f"{m['sharpe']:.2f}", f"{m['sharpe'] - bm['sharpe']:.2f}")
        c3.metric("Max Drawdown", f"{m['max_drawdown_pct']:.2f}%", f"{m['max_drawdown_pct'] - bm['max_drawdown_pct']:.2f}%")
        c4.metric("Win Rate", f"{m['win_rate_pct']:.1f}%", f"{m['n_trades']} trades")

        st.subheader("Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r["equity"].index, y=r["equity"].values, name="AlphaForge", line=dict(color="#00cc88", width=2)))
        fig.add_trace(go.Scatter(x=r["buy_hold_equity"].index, y=r["buy_hold_equity"].values, name="Buy & Hold", line=dict(color="#888", dash="dash")))
        fig.update_layout(height=420, hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Latest Signal")
        latest_proba = float(proba[-1])
        latest_signal = "BUY" if latest_proba > threshold else "HOLD/SELL"
        s1, s2, s3 = st.columns(3)
        s1.metric("As of", str(feats.index[-1].date()))
        s2.metric("P(up)", f"{latest_proba:.4f}")
        s3.metric("Signal", latest_signal)

        st.subheader("Recent Trades")
        st.dataframe(r["trades"].tail(20), use_container_width=True)

        with st.expander("Full metrics"):
            st.json({"strategy": m, "buy_hold": bm})

    except FileNotFoundError:
        st.error("Model not trained. Run: `python scripts/train.py` first.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Choose a ticker and click **Run analysis**. Train the model first with `python scripts/train.py`.")
