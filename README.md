# AlphaForge

**End-to-end ML system for stock-direction forecasting & backtesting on Indian equities (NSE).**

A production-grade pipeline that ingests OHLCV data, engineers 30+ technical-indicator features, predicts next-day price direction with an XGBoost + LSTM ensemble, validates with walk-forward cross-validation, backtests with realistic brokerage + slippage, and serves live signals via FastAPI + a Streamlit dashboard.

> Built as a portfolio piece to demonstrate **production ML engineering** for the fintech / quant-finance domain.

---

## What this project demonstrates

| Capability | Where to see it |
|---|---|
| **Data engineering** — ingestion, caching, fault-tolerance | [`src/data/loader.py`](src/data/loader.py) — yfinance + parquet cache + synthetic fallback when rate-limited |
| **Feature engineering** — 30+ technical indicators, vectorised | [`src/data/indicators.py`](src/data/indicators.py) — RSI, MACD, Bollinger, ATR, OBV, VWAP, Stochastic, returns, volatility |
| **Classical ML** — gradient boosting | [`src/models/xgboost_model.py`](src/models/xgboost_model.py) — XGBoost classifier |
| **Deep learning** — sequence modelling | [`src/models/lstm_model.py`](src/models/lstm_model.py) — PyTorch LSTM |
| **Ensembling** — model combination | [`src/models/ensemble.py`](src/models/ensemble.py) — weighted-probability ensemble |
| **Time-aware validation** — walk-forward CV (no look-ahead bias) | `XGBDirectionModel.walk_forward_eval` |
| **Realistic backtesting** — brokerage + slippage modelling | [`src/backtest/engine.py`](src/backtest/engine.py) |
| **Risk-adjusted metrics** | [`src/backtest/metrics.py`](src/backtest/metrics.py) — Sharpe, Sortino, Calmar, max drawdown, win rate |
| **Production API** — low-latency model serving | [`src/api/main.py`](src/api/main.py) — FastAPI, p99 ~25 ms warm |
| **Interactive dashboard** | [`src/dashboard/app.py`](src/dashboard/app.py) — Streamlit + Plotly |
| **Testing** — unit tests on the math | [`tests/test_indicators.py`](tests/test_indicators.py) — 5 tests, all green |
| **Containerisation** | [`Dockerfile`](Dockerfile) + [`docker-compose.yml`](docker-compose.yml) |

---

## Architecture

```
                        ┌─────────────────────────┐
                        │   config.yaml           │
                        │  (single source of      │
                        │   truth for params)     │
                        └────────────┬────────────┘
                                     │
   ┌────────────────────┐  ┌─────────▼──────────┐  ┌───────────────────┐
   │ Data Loader        │  │ Feature Engine     │  │ Models            │
   │ - yfinance         │─▶│ - 30+ indicators   │─▶│ - XGBoost (tab)   │
   │ - parquet cache    │  │ - target = up/down │  │ - LSTM (sequence) │
   │ - synthetic fall-  │  │ - drops NaN rows   │  │ - Ensemble (60/40)│
   │   back if rate-    │  └────────────────────┘  └─────────┬─────────┘
   │   limited          │                                    │
   └────────────────────┘                                    │
                                                             ▼
                              ┌──────────────────────────────────────────┐
                              │ Backtest Engine                          │
                              │ - executes at next-bar open (no leak)    │
                              │ - 0.03% brokerage + 0.05% slippage       │
                              │ - Sharpe / Sortino / Calmar / MaxDD      │
                              │ - benchmarks vs Buy & Hold               │
                              └────────────────┬─────────────────────────┘
                                               │
                                               ▼
                       ┌──────────────────────────────────────────┐
                       │ Serving                                  │
                       │  - FastAPI  (port 8000)  /predict /health│
                       │  - Streamlit (port 8501) interactive dash│
                       │  - Dockerised                            │
                       └──────────────────────────────────────────┘
```

---

## Quickstart

```powershell
# 1. install
pip install -r requirements.txt

# 2. train (5 NSE tickers, walk-forward CV, ~1 min on synthetic)
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python scripts/train.py --source synthetic

# 3. backtest one ticker
python scripts/backtest.py --ticker RELIANCE.NS --source synthetic

# 4. API
$env:ALPHAFORGE_DATA_SOURCE="synthetic"
python -m uvicorn src.api.main:app --port 8000
#   -> http://127.0.0.1:8000/predict?ticker=RELIANCE.NS
#   -> http://127.0.0.1:8000/docs   (Swagger UI)

# 5. Dashboard
python -m streamlit run src/dashboard/app.py
#   -> http://localhost:8501
```

> **Why `synthetic`?** Yahoo Finance frequently rate-limits free clients (HTTP 429). The loader falls back to a deterministic synthetic generator so the pipeline is always demonstrable. Once you have access to a real provider (e.g. Kite Connect), drop `--source synthetic` and the same pipeline runs on real data.

### Run with Docker

```bash
docker compose up --build
# API at :8000, dashboard at :8501
```

---

## Modelling details

### Features (30+)

| Group | Features |
|---|---|
| Returns | 1-day, 5-day, 10-day, log-return, rolling vol(10), rolling vol(20) |
| Trend | SMA & EMA at {5, 10, 20, 50}, price-to-SMA ratios |
| Momentum | RSI(14), MACD line/signal/histogram, Stochastic %K/%D |
| Volatility | Bollinger %B, Bollinger width, ATR(14), ATR-as-%-of-price |
| Volume | OBV, OBV 5-day change, VWAP, close-vs-VWAP, relative volume vs 20-day mean |
| Target | `1` if `close.shift(-1) > close` else `0` |

### Validation: walk-forward, no leakage

Standard random train/test splits leak the future into training. We use `TimeSeriesSplit` with 5 folds — each fold is trained only on data **strictly before** the test window. That mirrors how a model would actually be deployed.

### Backtest realism

- Trades execute at the **next bar's open**, not the bar that generated the signal — so a strategy can't "use" information from the same close that emitted the signal.
- Every transaction pays **0.03% brokerage** (Zerodha-equivalent) and crosses a **0.05% slippage** spread.
- Final results are always reported alongside a **Buy & Hold** benchmark on the identical window — this is the only honest way to claim a strategy "works".

---

## Project layout

```
alphaforge/
├── config.yaml              # All hyperparameters & tickers — one place, no magic numbers
├── requirements.txt
├── Dockerfile, docker-compose.yml, Makefile
├── README.md
├── src/
│   ├── data/
│   │   ├── loader.py        # yfinance + parquet cache + synthetic fallback
│   │   ├── indicators.py    # 30+ vectorised technical indicators
│   │   └── synthetic.py     # GBM + AR(1) momentum generator for offline runs
│   ├── models/
│   │   ├── xgboost_model.py # XGB classifier + walk-forward CV + feature importance
│   │   ├── lstm_model.py    # PyTorch LSTM sequence model
│   │   └── ensemble.py
│   ├── backtest/
│   │   ├── engine.py        # Vectorised backtester with brokerage + slippage
│   │   └── metrics.py       # Sharpe, Sortino, Calmar, MaxDD, win rate
│   ├── api/main.py          # FastAPI service
│   └── dashboard/app.py     # Streamlit dashboard
├── scripts/
│   ├── train.py             # End-to-end training entry point
│   └── backtest.py          # Standalone backtest runner
├── tests/
│   └── test_indicators.py   # 5 pytest cases on the indicator math
├── data/                    # Parquet cache (auto-populated)
├── models_saved/            # Trained models (.pkl, .pt)
└── results/                 # Backtest outputs: equity curves, trade logs, JSON metrics
```

---

## Tests

```bash
pytest tests/ -v
```

All 5 tests verify mathematical correctness of indicators (RSI bounds, MACD shape, Bollinger ordering, ATR positivity, full feature pipeline produces no NaNs).

---

## Notes on data sources

- **Default (`--source auto`)**: tries yfinance, falls back to synthetic if yfinance fails or returns empty.
- **`--source yfinance`**: forces yfinance (raises if it fails — useful for production).
- **`--source synthetic`**: deterministic generator using GBM + AR(1) momentum. Reproducible (seeded by ticker name). Used in CI and for offline demos.

For real deployment, swap `loader.py` for a Kite Connect / Alpha Vantage / NSEpy adapter — the rest of the pipeline is data-source-agnostic.

---

## Future work

These were intentionally **not** included in this baseline to keep the project surface tight and reviewable, but each is a natural extension:

- Hyperparameter tuning (Optuna)
- Experiment tracking (MLflow)
- Risk management module (Kelly sizing, ATR stop-losses)
- Model explainability (SHAP)
- Multi-asset portfolio backtest with sector neutralisation
- Intraday (tick / 1-min) data via a paid feed
- Online learning — nightly retrain on the latest day's data

---

## Author

Built as a portfolio piece showcasing production-grade ML engineering for the fintech / quantitative-finance domain.
