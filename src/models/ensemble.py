"""Weighted ensemble of XGBoost + LSTM probabilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .xgboost_model import XGBDirectionModel
from .lstm_model import LSTMDirectionModel


class Ensemble:
    def __init__(self, xgb: XGBDirectionModel, lstm: LSTMDirectionModel | None, weights=(0.6, 0.4)):
        self.xgb, self.lstm = xgb, lstm
        self.weights = weights

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p_xgb = self.xgb.predict_proba(df)
        if self.lstm is None:
            return p_xgb
        p_lstm = self.lstm.predict_proba(df)
        n = min(len(p_xgb), len(p_lstm))
        wx, wl = self.weights
        return wx * p_xgb[-n:] + wl * p_lstm[-n:]

    def predict(self, df: pd.DataFrame, threshold: float = 0.55) -> np.ndarray:
        return (self.predict_proba(df) > threshold).astype(int)
