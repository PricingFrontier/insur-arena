"""TabDPT — tabular foundation model (retrieval-based), BLACK-BOX comparator.

Like all foundation models here it is exposure-agnostic: no Poisson/Gamma/Tweedie loss and no sample
weights, so for freq/sev/pp it fits a generic regressor on the rate/severity target and ignores
exposure weighting. Categoricals are ordinal-encoded; train is capped for tractability on full data.
Works on CPU with no auth (unlike TabPFN). Lazy import; optional dependency `tabdpt`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.models.base import Explanation, ModelAdapter
from insurarena.scoring import task_targets
from insurarena.tasks import TaskType

_MAX_ROWS = 20_000   # cap training context — CPU retrieval gets slow with large context


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class TabDPT(ModelAdapter):
    name = "tabdpt"
    family = "foundation"
    needs_es_probe = False               # in-context model, no iterative early stopping -> skip probe
    supported_tasks = (TaskType.FREQ, TaskType.SEV, TaskType.PP, TaskType.CLF)

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self._columns].copy()
        for c in self._cat:
            X[c] = X[c].astype("category").cat.codes.astype(float)
        return np.nan_to_num(X.astype(float).to_numpy(), nan=0.0)

    def fit(self, train: DatasetBundle, val: DatasetBundle | None = None) -> "TabDPT":
        from tabdpt import TabDPTClassifier, TabDPTRegressor

        self._columns = list(train.X.columns)
        self._cat = [c for c in train.cat_features if c in self._columns]
        self._task = train.task.task
        y, _, _ = task_targets(train)              # exposure weight intentionally dropped
        X = self._encode(train.X)
        y = np.asarray(y, dtype=float)
        if len(X) > _MAX_ROWS:
            rng = np.random.default_rng(int(self.hp.get("seed", 0)))
            idx = rng.choice(len(X), _MAX_ROWS, replace=False)
            X, y = X[idx], y[idx]
        self._is_clf = self._task == TaskType.CLF
        Model = TabDPTClassifier if self._is_clf else TabDPTRegressor
        dev = self.hp.get("device") or _auto_device()        # GPU when available, else CPU
        try:
            self.model = Model(device=dev)
        except TypeError:
            self.model = Model()
        self.model.fit(X, y)
        self.best_iteration_ = 0
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, exposure: np.ndarray | None = None) -> np.ndarray:
        Xp = self._encode(X)
        if self._is_clf:
            return np.clip(np.asarray(self.model.predict_proba(Xp)[:, 1], dtype=float), 0.0, 1.0)
        return np.clip(np.asarray(self.model.predict(Xp), dtype=float), 1e-9, None)

    def explain(self) -> Explanation:
        return Explanation(kind="none")

    @staticmethod
    def default_hp() -> dict:
        return {}

    @staticmethod
    def sample_hp(rng: np.random.Generator) -> dict:
        return {}   # foundation model — default only (no HPO), as in TabArena
