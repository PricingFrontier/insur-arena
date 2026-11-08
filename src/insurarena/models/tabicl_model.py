"""TabICL — in-context tabular foundation model, BLACK-BOX comparator. Classification only.

Exposure-agnostic; categoricals ordinal-encoded; train capped. CPU + no auth. Optional dep `tabicl`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.models.base import Explanation, ModelAdapter
from insurarena.scoring import task_targets
from insurarena.tasks import TaskType

_MAX_ROWS = 100_000   # TabICL classification limit


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class TabICL(ModelAdapter):
    name = "tabicl"
    family = "foundation"
    needs_es_probe = False               # in-context model, no iterative early stopping -> skip probe
    supported_tasks = (TaskType.CLF,)

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self._columns].copy()
        for c in self._cat:
            X[c] = X[c].astype("category").cat.codes.astype(float)
        return np.nan_to_num(X.astype(float).to_numpy(), nan=0.0)

    def fit(self, train: DatasetBundle, val: DatasetBundle | None = None) -> "TabICL":
        from tabicl import TabICLClassifier

        self._columns = list(train.X.columns)
        self._cat = [c for c in train.cat_features if c in self._columns]
        y, _, _ = task_targets(train)
        X = self._encode(train.X)
        y = np.asarray(y, dtype=float)
        if len(X) > _MAX_ROWS:
            rng = np.random.default_rng(int(self.hp.get("seed", 0)))
            idx = rng.choice(len(X), _MAX_ROWS, replace=False)
            X, y = X[idx], y[idx]
        dev = self.hp.get("device") or _auto_device()        # GPU when available, else CPU
        try:
            self.model = TabICLClassifier(device=dev)
        except TypeError:
            self.model = TabICLClassifier()
        self.model.fit(X, y)
        self.best_iteration_ = 0
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, exposure: np.ndarray | None = None) -> np.ndarray:
        return np.clip(np.asarray(self.model.predict_proba(self._encode(X))[:, 1], dtype=float), 0.0, 1.0)

    def explain(self) -> Explanation:
        return Explanation(kind="none")

    @staticmethod
    def default_hp() -> dict:
        return {}

    @staticmethod
    def sample_hp(rng: np.random.Generator) -> dict:
        return {}
