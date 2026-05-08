"""FastAPI prediction service. Run: uvicorn src.api.main:app --reload"""
from __future__ import annotations
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.loader import DataLoader
from src.data.indicators import build_features
from src.models.xgboost_model import XGBDirectionModel


CFG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_SOURCE = os.environ.get("ALPHAFORGE_DATA_SOURCE", "auto")  # auto|yfinance|synthetic

app = FastAPI(title="AlphaForge", description="ML-driven equity direction forecasting", version="1.0")
loader = DataLoader(CFG["data"]["cache_dir"], source=DATA_SOURCE)
_model: XGBDirectionModel | None = None


def get_model() -> XGBDirectionModel:
    global _model
    if _model is None:
        path = CFG["api"]["model_path"]
        if not Path(path).exists():
            raise HTTPException(503, f"Model not trained yet. Run scripts/train.py first.")
        _model = XGBDirectionModel.load(path)
    return _model


class PredictionResponse(BaseModel):
    ticker: str
    as_of: str
    probability_up: float
    signal: str
    confidence: float
    threshold: float
    latency_ms: float


@app.get("/")
def root():
    return {"service": "AlphaForge", "status": "ok", "endpoints": ["/predict", "/health"]}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None}


@app.get("/predict", response_model=PredictionResponse)
def predict(ticker: str = "RELIANCE.NS", threshold: float = 0.55):
    t0 = time.perf_counter()
    model = get_model()
    end = datetime.utcnow().date().isoformat()
    start = (datetime.utcnow().date() - timedelta(days=400)).isoformat()

    try:
        df = loader.fetch(ticker, start, end, force=False)
    except Exception as e:
        raise HTTPException(400, f"Data fetch failed for {ticker}: {e}")

    feats = build_features(df, CFG)
    if feats.empty:
        raise HTTPException(400, "Insufficient data after feature engineering.")

    proba = float(model.predict_proba(feats.tail(1))[0])
    signal = "BUY" if proba > threshold else "HOLD/SELL"
    latency = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        ticker=ticker,
        as_of=str(feats.index[-1].date()),
        probability_up=round(proba, 4),
        signal=signal,
        confidence=round(abs(proba - 0.5) * 2, 4),
        threshold=threshold,
        latency_ms=round(latency, 2),
    )
