"""XGBoost direction classifier with walk-forward CV."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


FEATURE_COLS_EXCLUDE = {"open", "high", "low", "close", "volume", "target"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]


class XGBDirectionModel:
    def __init__(self, params: dict):
        self.params = params
        self.model: XGBClassifier | None = None
        self.features: list[str] = []

    def fit(self, df: pd.DataFrame) -> "XGBDirectionModel":
        self.features = feature_columns(df)
        X, y = df[self.features].values, df["target"].values
        self.model = XGBClassifier(
            **self.params, eval_metric="logloss", tree_method="hist"
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(df[self.features].values)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) > threshold).astype(int)

    def walk_forward_eval(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        feats = feature_columns(df)
        X, y = df[feats].values, df["target"].values
        tscv = TimeSeriesSplit(n_splits=n_splits)
        accs, aucs, oof = [], [], np.full(len(df), np.nan)

        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            m = XGBClassifier(**self.params, eval_metric="logloss", tree_method="hist")
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            oof[te] = p
            accs.append(accuracy_score(y[te], p > 0.5))
            aucs.append(roc_auc_score(y[te], p))
            print(f"  fold {fold}: acc={accs[-1]:.4f}  auc={aucs[-1]:.4f}")

        return {
            "fold_accuracy": accs,
            "fold_auc": aucs,
            "mean_accuracy": float(np.mean(accs)),
            "mean_auc": float(np.mean(aucs)),
            "oof_proba": oof,
        }

    def feature_importance(self, top_k: int = 15) -> pd.DataFrame:
        assert self.model is not None
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.features, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features, "params": self.params}, path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBDirectionModel":
        d = joblib.load(path)
        obj = cls(d["params"])
        obj.model = d["model"]
        obj.features = d["features"]
        return obj
