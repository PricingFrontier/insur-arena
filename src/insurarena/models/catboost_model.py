"""CatBoost — single fitted booster, task-aware (Poisson / Tweedie / Logloss).

No native Gamma loss; severity runs on the standard proxy Tweedie:variance_power=1.99
(the compound-Poisson-Gamma family converges to Gamma as the power -> 2, and CatBoost
rejects 2.0 itself). Scoring is unchanged — Gamma deviance on the predictions — so the
comparison stays apples-to-apples with the native-Gamma techniques.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.models.base import Explanation, ModelAdapter, filter_tiny_exposure_fit_rows
from insurarena.scoring import task_targets
from insurarena.tasks import TaskType


class CatBoost(ModelAdapter):
    name = "catboost"
    family = "gbm"
    adapter_version = "1"   # reset 2026-09-09 (arena rebuilt from scratch); prior history in git
    supported_tasks = (TaskType.FREQ, TaskType.SEV, TaskType.PP, TaskType.CLF, TaskType.MULTICLF)

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.loc[:, self._columns]
        needs_copy = False
        for c in self._cat:
            if c in X.columns and (X[c].isna().any() or not pd.api.types.is_object_dtype(X[c])):
                needs_copy = True
                break
        if needs_copy:
            X = X.copy(deep=False)
            for c in self._cat:
                if c in X.columns:
                    X[c] = X[c].astype(str).fillna("nan")
        return X

    def fit(self, train: DatasetBundle, val: DatasetBundle | None = None) -> "CatBoost":
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        train = filter_tiny_exposure_fit_rows(train)
        val = filter_tiny_exposure_fit_rows(val) if val is not None else None
        self._columns = list(train.X.columns)
        self._cat = [c for c in train.cat_features if c in self._columns]
        self._task = train.task.task
        tweedie_power = train.task.tweedie_power
        y, w, _ = task_targets(train)
        common = dict(
            # honor the harness's early-stopped count: fit_one (and the tuned OOF) hand the chosen
            # round count as `n_estimators` (sklearn name); fall back to the native `iterations`, then 2000.
            iterations=int(self.hp.get("n_estimators", self.hp.get("iterations", 2000))),
            learning_rate=float(self.hp.get("learning_rate", 0.05)),
            depth=int(self.hp.get("depth", 6)),
            l2_leaf_reg=float(self.hp.get("l2_leaf_reg", 3.0)),
            random_seed=int(self.hp.get("seed", 0)), verbose=False, allow_writing_files=False,
        )
        _t = self.hp.get("thread_count") or os.environ.get("OMP_NUM_THREADS")
        if _t:
            common["thread_count"] = int(_t)
        # TabArena C.1 extras — added ONLY when a tuned config sets them, so the `default` variant keeps
        # CatBoost's own defaults (its cached default cells stay byte-identical).
        for hp_key, cb_key, cast in [
            ("bootstrap_type", "bootstrap_type", str), ("subsample", "subsample", float),
            ("grow_policy", "grow_policy", str), ("colsample_bylevel", "rsm", float),
            ("leaf_estimation_iterations", "leaf_estimation_iterations", int),
            ("one_hot_max_size", "one_hot_max_size", int), ("model_size_reg", "model_size_reg", float),
            ("max_ctr_complexity", "max_ctr_complexity", int), ("boosting_type", "boosting_type", str),
            ("max_bin", "border_count", int)]:
            if hp_key in self.hp:
                common[cb_key] = cast(self.hp[hp_key])
        X_train = self._prep(train.X)
        pool = Pool(X_train, label=y, weight=w, cat_features=self._cat)
        X_train = y = w = train = None
        eval_pool, fit_kw = None, {}
        if val is not None and len(val.X):
            yv, wv, _ = task_targets(val)
            X_val = self._prep(val.X)
            eval_pool = Pool(X_val, label=yv, weight=wv, cat_features=self._cat)
            X_val = yv = wv = val = None
            fit_kw = {"eval_set": eval_pool, "early_stopping_rounds": 50}
        self._is_multi = self._task == TaskType.MULTICLF
        self._is_clf = self._task in (TaskType.CLF, TaskType.MULTICLF)
        if self._is_clf:
            self.model = CatBoostClassifier(
                loss_function="MultiClass" if self._is_multi else "Logloss", **common)
        else:
            loss = ("Poisson" if self._task == TaskType.FREQ
                    # Gamma proxy: Tweedie -> Gamma as variance_power -> 2 (CatBoost has no
                    # native Gamma and rejects 2.0); scored on Gamma deviance like everyone.
                    else "Tweedie:variance_power=1.99" if self._task == TaskType.SEV
                    else f"Tweedie:variance_power={tweedie_power or 1.5}")
            self.model = CatBoostRegressor(loss_function=loss, **common)
        self.model.fit(pool, **fit_kw)
        self.best_iteration_ = int(self.model.get_best_iteration() or 0) if eval_pool is not None else 0
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
        imp = {f: float(v) for f, v in zip(self._columns, self.model.get_feature_importance())}
        return Explanation(kind="gain", global_importance=imp)

    @staticmethod
    def default_hp() -> dict:
        return {"iterations": 2000, "learning_rate": 0.05, "depth": 6}

    @staticmethod
    def sample_hp(rng: np.random.Generator) -> dict:
        # Full TabArena search space (Table C.1); iterations early-stopped. Bernoulli bootstrap + Plain
        # boosting + fixed border_count=254 per the table; the rest are searched.
        return {
            "iterations": 2000,
            "learning_rate": float(np.exp(rng.uniform(np.log(0.005), np.log(0.1)))),
            "bootstrap_type": "Bernoulli",
            "subsample": float(rng.uniform(0.7, 1.0)),
            "grow_policy": str(rng.choice(["SymmetricTree", "Depthwise"])),
            "depth": int(rng.integers(4, 9)),                                    # [4, 8]
            "colsample_bylevel": float(rng.uniform(0.85, 1.0)),                  # -> rsm
            "l2_leaf_reg": float(np.exp(rng.uniform(np.log(1e-4), np.log(5.0)))),
            "leaf_estimation_iterations": int(np.exp(rng.uniform(np.log(1), np.log(20)))),
            "one_hot_max_size": int(np.exp(rng.uniform(np.log(8), np.log(100)))),
            "model_size_reg": float(np.exp(rng.uniform(np.log(0.1), np.log(1.5)))),
            "max_ctr_complexity": int(rng.integers(2, 6)),                       # [2, 5]
            "boosting_type": "Plain",
            "max_bin": 254,
        }
