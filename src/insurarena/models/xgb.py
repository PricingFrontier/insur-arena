"""XGBoost — single fitted booster, task-aware (count:poisson / reg:gamma / reg:tweedie / binary)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.models.base import Explanation, ModelAdapter, filter_tiny_exposure_fit_rows
from insurarena.scoring import task_targets
from insurarena.tasks import TaskType

# task -> (objective, is_classifier)
_OBJ = {
    TaskType.FREQ: ("count:poisson", False),
    TaskType.SEV: ("reg:gamma", False),
    TaskType.PP: ("reg:tweedie", False),
    TaskType.CLF: ("binary:logistic", True),
    TaskType.MULTICLF: ("multi:softprob", True),
}


class XGBoost(ModelAdapter):
    name = "xgboost"
    family = "gbm"
    adapter_version = "1"   # reset 2026-09-09 (arena rebuilt from scratch); prior history in git
    supported_tasks = (TaskType.FREQ, TaskType.SEV, TaskType.PP, TaskType.CLF, TaskType.MULTICLF)

    def _prep_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """Set categoricals to pandas 'category' dtype (native XGBoost handling), learning the levels."""
        X = X.loc[:, self._columns].copy(deep=False)
        self._cat_levels = {}
        for c in self._cat:
            cc = pd.Categorical(X[c].astype(str))
            self._cat_levels[c] = list(cc.categories)
            X[c] = cc
        return X

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        """Coerce to the TRAINING levels so unseen categories become NaN (missing) — no native crash."""
        X = X.loc[:, self._columns].copy(deep=False)
        for c in self._cat:
            X[c] = pd.Categorical(X[c].astype(str), categories=self._cat_levels[c])
        return X

    def fit(self, train: DatasetBundle, val: DatasetBundle | None = None) -> "XGBoost":
        import xgboost as xgb

        train = filter_tiny_exposure_fit_rows(train)
        val = filter_tiny_exposure_fit_rows(val) if val is not None else None
        self._columns = list(train.X.columns)
        self._cat = [c for c in train.cat_features if c in self._columns]
        self._task = train.task.task
        obj, is_clf = _OBJ[self._task]
        self._is_multi = self._task == TaskType.MULTICLF
        y, w, _ = task_targets(train)
        if is_clf:
            y = y.astype(int)                         # XGBClassifier wants integer class codes
        params = dict(
            n_estimators=int(self.hp.get("n_estimators", 2000)),
            learning_rate=float(self.hp.get("learning_rate", 0.05)),
            max_depth=int(self.hp.get("max_depth", 6)),
            min_child_weight=float(self.hp.get("min_child_weight", 1.0)),
            subsample=float(self.hp.get("subsample", 1.0)),
            reg_alpha=float(self.hp.get("reg_alpha", 0.0)),
            reg_lambda=float(self.hp.get("reg_lambda", 1.0)),
            tree_method="hist", enable_categorical=True,     # native categorical splits (like TabArena)
            random_state=int(self.hp.get("seed", 0)), verbosity=0,
        )
        _t = self.hp.get("n_jobs") or os.environ.get("OMP_NUM_THREADS")
        if _t:
            params["n_jobs"] = int(_t)
        # TabArena C.3 extras (colsample by level+node, grow_policy, max_leaves, max_cat_to_onehot) —
        # added ONLY when a tuned config sets them, so the `default` variant keeps XGBoost's own defaults.
        for k in ("colsample_bylevel", "colsample_bynode", "grow_policy", "max_leaves", "max_cat_to_onehot"):
            if k in self.hp:
                params[k] = self.hp[k]
        Xt = self._prep_fit(train.X)
        eval_set = None
        eval_weights = None
        if val is not None and len(val.X):
            yv, wv, _ = task_targets(val)
            eval_set = [(self._prep(val.X), yv.astype(int) if is_clf else yv)]
            eval_weights = [wv]
            params["early_stopping_rounds"] = 50
        self._is_clf = is_clf
        if is_clf:
            self.model = xgb.XGBClassifier(objective=obj, **params)
        else:
            if self._task == TaskType.PP:
                params["tweedie_variance_power"] = float(train.task.tweedie_power or 1.5)
            self.model = xgb.XGBRegressor(objective=obj, **params)
        self.model.fit(Xt, y, sample_weight=w, eval_set=eval_set,
                       sample_weight_eval_set=eval_weights, verbose=False)
        self.best_iteration_ = int(getattr(self.model, "best_iteration", 0) or 0)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, exposure: np.ndarray | None = None) -> np.ndarray:
        Xp = self._prep(X)
        if self._is_multi:
            return np.asarray(self.model.predict_proba(Xp), dtype=float)   # (n, K)
        if self._is_clf:
            return np.asarray(self.model.predict_proba(Xp)[:, 1], dtype=float)
        return np.asarray(self.model.predict(Xp), dtype=float)

    def explain(self) -> Explanation:
        imp = {f: float(v) for f, v in zip(self._columns, self.model.feature_importances_)}
        return Explanation(kind="gain", global_importance=imp)

    @staticmethod
    def default_hp() -> dict:
        return {"n_estimators": 2000, "learning_rate": 0.05, "max_depth": 6}

    @staticmethod
    def sample_hp(rng: np.random.Generator) -> dict:
        # Full TabArena search space (Table C.3); n_estimators left large + early-stopped.
        gp = str(rng.choice(["depthwise", "lossguide"]))
        return {
            "n_estimators": 2000,
            "learning_rate": float(np.exp(rng.uniform(np.log(0.005), np.log(0.1)))),
            "max_depth": int(np.exp(rng.uniform(np.log(4), np.log(10)))),
            "min_child_weight": float(np.exp(rng.uniform(np.log(0.001), np.log(5.0)))),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bylevel": float(rng.uniform(0.6, 1.0)),
            "colsample_bynode": float(rng.uniform(0.6, 1.0)),
            "grow_policy": gp,
            "max_leaves": int(np.exp(rng.uniform(np.log(8), np.log(1024)))) if gp == "lossguide" else 0,
            "max_cat_to_onehot": int(np.exp(rng.uniform(np.log(8), np.log(100)))),
            "reg_alpha": float(rng.uniform(1e-4, 5.0)),
            "reg_lambda": float(rng.uniform(1e-4, 5.0)),
        }
