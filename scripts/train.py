"""Train ensemble (XGBoost + optional LSTM) with walk-forward CV.

Usage:
    python scripts/train.py
    python scripts/train.py --ticker RELIANCE.NS --no-lstm
"""
from __future__ import annotations

# OpenMP fix MUST run before torch/xgboost are imported.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch  # noqa: F401  (load torch's OpenMP first)

import argparse
import json
import sys
from pathlib import Path

import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import DataLoader
from src.data.indicators import build_features
from src.models.xgboost_model import XGBDirectionModel
from src.models.lstm_model import LSTMDirectionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None, help="Single ticker; default = all in config")
    ap.add_argument("--no-lstm", action="store_true", help="Skip LSTM training")
    ap.add_argument("--source", default="auto", choices=["auto", "yfinance", "synthetic"],
                    help="Data source. 'auto' tries yfinance, falls back to synthetic.")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    tickers = [args.ticker] if args.ticker else cfg["data"]["tickers"]
    loader = DataLoader(cfg["data"]["cache_dir"], source=args.source)

    print(f"Loading data for {len(tickers)} ticker(s)...")
    frames = []
    for t in tickers:
        df = loader.fetch(t, cfg["data"]["start_date"], cfg["data"]["end_date"])
        f = build_features(df, cfg)
        f["ticker"] = t
        frames.append(f)
        print(f"  {t}: {len(f)} rows")

    full = pd.concat(frames).sort_index()
    feature_df = full.drop(columns=["ticker"])

    print(f"\nTraining XGBoost on {len(feature_df)} samples...")
    xgb = XGBDirectionModel(cfg["model"]["xgboost"])
    cv = xgb.walk_forward_eval(feature_df, n_splits=cfg["backtest"]["walk_forward_splits"])
    print(f"  mean acc={cv['mean_accuracy']:.4f}  mean auc={cv['mean_auc']:.4f}")
    xgb.fit(feature_df)
    xgb.save(ROOT / "models_saved" / "xgb_model.pkl")
    print("  saved -> models_saved/xgb_model.pkl")

    print("\nTop 15 features:")
    print(xgb.feature_importance(15).to_string(index=False))

    if not args.no_lstm:
        print("\nTraining LSTM (single ticker for sequence integrity)...")
        primary = frames[0].drop(columns=["ticker"])
        lstm = LSTMDirectionModel(cfg["model"]["lstm"])
        lstm.fit(primary)
        lstm.save(ROOT / "models_saved" / "lstm_model.pt")
        print("  saved -> models_saved/lstm_model.pt")

    report = {
        "tickers": tickers,
        "n_samples": int(len(feature_df)),
        "xgb_mean_accuracy": cv["mean_accuracy"],
        "xgb_mean_auc": cv["mean_auc"],
        "xgb_fold_accuracy": cv["fold_accuracy"],
        "xgb_fold_auc": cv["fold_auc"],
        "top_features": xgb.feature_importance(15).to_dict(orient="records"),
    }
    out = ROOT / "results" / "training_report.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(report, open(out, "w"), indent=2, default=str)
    print(f"\nReport saved -> {out}")


if __name__ == "__main__":
    main()
