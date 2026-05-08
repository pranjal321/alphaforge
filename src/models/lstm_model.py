"""LSTM sequence model for next-day direction prediction."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .xgboost_model import feature_columns


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMDirectionModel:
    def __init__(self, params: dict):
        self.p = params
        self.net: LSTMNet | None = None
        self.features: list[str] = []
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _scale(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0) + 1e-8
        return (X - self.mean) / self.std

    def fit(self, df: pd.DataFrame) -> "LSTMDirectionModel":
        self.features = feature_columns(df)
        X = self._scale(df[self.features].values.astype(np.float32), fit=True)
        y = df["target"].values.astype(np.float32)

        Xs, ys = make_sequences(X, y, self.p["sequence_length"])
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
        dl = DataLoader(ds, batch_size=self.p["batch_size"], shuffle=True)

        self.net = LSTMNet(
            len(self.features), self.p["hidden_size"], self.p["num_layers"], self.p["dropout"]
        ).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.p["lr"])
        loss_fn = nn.BCEWithLogitsLoss()

        for ep in range(self.p["epochs"]):
            self.net.train()
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            if (ep + 1) % 5 == 0:
                print(f"  epoch {ep+1}/{self.p['epochs']}  loss={tot/len(ds):.4f}")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        assert self.net is not None
        X = self._scale(df[self.features].values.astype(np.float32), fit=False)
        seq = self.p["sequence_length"]
        Xs, _ = make_sequences(X, np.zeros(len(X)), seq)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(Xs).to(self.device)).cpu().numpy()
        proba = 1 / (1 + np.exp(-logits))
        # pad first `seq` rows with 0.5 so output aligns with df length
        return np.concatenate([np.full(seq, 0.5), proba])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "features": self.features,
            "mean": self.mean, "std": self.std, "params": self.p,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMDirectionModel":
        d = torch.load(path, map_location="cpu")
        obj = cls(d["params"])
        obj.features = d["features"]
        obj.mean, obj.std = d["mean"], d["std"]
        obj.net = LSTMNet(len(obj.features), obj.p["hidden_size"], obj.p["num_layers"], obj.p["dropout"])
        obj.net.load_state_dict(d["state_dict"])
        obj.net.to(obj.device)
        return obj
