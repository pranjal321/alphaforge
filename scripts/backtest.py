"""Run backtest with the trained model.

Usage:
    python scripts/backtest.py --ticker RELIANCE.NS
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import DataLoader
from src.data.indicators import build_features
from src.models.xgboost_model import XGBDirectionModel
from src.backtest.engine import Backtester


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="RELIANCE.NS")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--source", default="auto", choices=["auto", "yfinance", "synthetic"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml", "r", encoding="utf-8"))
    threshold = args.threshold if args.threshold is not None else cfg["backtest"]["signal_threshold"]

    df = DataLoader(cfg["data"]["cache_dir"], source=args.source).fetch(args.ticker, args.start, args.end)
    feats = build_features(df, cfg)

    model = XGBDirectionModel.load(ROOT / "models_saved" / "xgb_model.pkl")
    proba = model.predict_proba(feats)

    bt = Backtester(
        cfg["backtest"]["initial_capital"],
        cfg["backtest"]["brokerage_pct"],
        cfg["backtest"]["slippage_pct"],
        cfg["backtest"]["position_size_pct"],
        threshold,
    )
    result = bt.run(feats[["open", "high", "low", "close"]], proba)

    print("\n=== AlphaForge Strategy ===")
    for k, v in result["metrics"].items():
        print(f"  {k:>22}: {v}")
    print("\n=== Buy & Hold Benchmark ===")
    for k, v in result["buy_hold_metrics"].items():
        print(f"  {k:>22}: {v}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    result["equity"].to_csv(out / f"{args.ticker}_equity.csv")
    result["trades"].to_csv(out / f"{args.ticker}_trades.csv", index=False)
    json.dump(
        {"strategy": result["metrics"], "buy_hold": result["buy_hold_metrics"]},
        open(out / f"{args.ticker}_metrics.json", "w"), indent=2, default=str,
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(result["equity"].index, result["equity"].values, label="AlphaForge", lw=2, color="#00aa66")
    ax.plot(result["buy_hold_equity"].index, result["buy_hold_equity"].values, label="Buy & Hold", lw=1.5, color="#888", ls="--")
    ax.set_title(f"AlphaForge vs Buy & Hold — {args.ticker}")
    ax.set_ylabel("Equity (INR)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{args.ticker}_equity.png", dpi=120)
    print(f"\nResults saved -> {out}/")


if __name__ == "__main__":
    main()
