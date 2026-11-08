"""LightGBM — a single fitted booster, task-aware (poisson / gamma / tweedie / binary).

Internal boosting is fine — it's one fitted artifact; what's excluded is benchmark-level
bagging/stacking. LightGBM is imported lazily so the package + metric/split tests work without it.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.models.base import Explanation, ModelAdapter
from insurarena.scoring import task_targets
from insurarena.tasks import TaskType

# task -> (objective, eval metric)
_OBJECTIVE = {
    TaskType.FREQ: ("poisson", "poisson"),
    TaskType.SEV: ("gamma", "gamma"),
    TaskType.PP: ("tweedie", "tweedie"),
    TaskType.CLF: ("binary", "binary_logloss"),
    TaskType.MULTICLF: ("multiclass", "multi_logloss"),
}


class LightGBM(ModelAdapter):
    name = "lightgbm"
    family = "gbm"
    supported_tasks = (TaskType.FREQ, TaskType.SEV, TaskType.PP, TaskType.CLF, TaskType.MULTICLF)

    @staticmethod
    def _safe_names(columns: list[str]) -> dict[str, str]:
        """LightGBM silently rewrites whitespace in feature names (and rejects JSON-special
        characters), which desyncs an explicit ``categorical_feature`` list -> TypeError.
        Sanitize up front; identity for already-clean names so cached cells are unaffected."""
        out: dict[str, str] = {}
        seen: set[str] = set()
        for c in columns:
            safe = re.sub(r"[^\w]", "_", str(c))
            while safe in seen:
                safe += "_"
            seen.add(safe)
            out[c] = safe
        return out

    def _prep_X(self, X: pd.DataFrame, cat_features) -> pd.DataFrame:
        X = X.loc[:, self._columns].copy(deep=False)
        for c in cat_features:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X.rename(columns=self._names)

    def fit(self, train: DatasetBundle, val: DatasetBundle | None = None) -> "LightGBM":
        import lightgbm as lgb

        self._columns = list(train.X.columns)
        self._names = self._safe_names(self._columns)
        self._cat = [c for c in train.cat_features if c in self._columns]
        self._task = train.task.task
        obj, metric = _OBJECTIVE[self._task]
        y, w, _ = task_targets(train)

        params = {
            "objective": obj,
            "metric": metric,
            "learning_rate": float(self.hp.get("learning_rate", 0.05)),
            "num_leaves": int(self.hp.get("num_leaves", 31)),
            "min_data_in_leaf": int(self.hp.get("min_data_in_leaf", 100)),
            "feature_fraction": float(self.hp.get("feature_fraction", 0.9)),
            "bagging_fraction": float(self.hp.get("bagging_fraction", 1.0)),
            "bagging_freq": 1,            # activates row subsampling (within-model, not model bagging)
            "lambda_l1": float(self.hp.get("lambda_l1", 0.0)),
            "lambda_l2": float(self.hp.get("lambda_l2", 0.0)),
            "verbosity": -1,
            "seed": int(self.hp.get("seed", 0)),
        }
        _t = self.hp.get("num_threads") or os.environ.get("OMP_NUM_THREADS")
        if _t:
            params["num_threads"] = int(_t)
        # TabArena C.2 extras — added ONLY when a tuned config sets them, so the `default` variant keeps
        # LightGBM's own defaults (its cached default cells stay byte-identical).
        for k in ("extra_trees", "min_data_per_group", "cat_l2", "cat_smooth", "max_cat_to_onehot"):
            if k in self.hp:
                params[k] = self.hp[k]
        if self._task == TaskType.FREQ:
            params["poisson_max_delta_step"] = 0.7
        if self._task == TaskType.PP:
            params["tweedie_variance_power"] = float(train.task.tweedie_power or 1.5)
        if self._task == TaskType.MULTICLF:
            params["num_class"] = int(train.meta.get("n_classes") or (int(np.max(y)) + 1))

        has_val = val is not None and len(val.X)
        keep_raw_for_cats = bool(self._cat) and has_val

        # The explicit pre-train construct() (to free raw arrays) bakes the feature pre-filter into
        # the Dataset, so the binning-relevant threshold must match training: constructing at the
        # library default (20) and then training with a smaller min_data_in_leaf hard-errors when
        # free_raw_data prevents re-construction (any tuned config with min_data_in_leaf < 20).
        # Benchmark defaults use 20 == library default, so their cells are unaffected.
        ds_params = {"min_data_in_leaf": params["min_data_in_leaf"]}
        X_train = self._prep_X(train.X, self._cat)
        dtrain = lgb.Dataset(X_train, label=y, weight=w, params=ds_params,
                             categorical_feature=[self._names[c] for c in self._cat] or "auto",
                             free_raw_data=not keep_raw_for_cats)
        dtrain.construct()
        X_train = y = w = train = None
        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = None
        if has_val:
            yv, wv, _ = task_targets(val)
            X_val = self._prep_X(val.X, self._cat)
            dval = lgb.Dataset(X_val, label=yv, weight=wv, reference=dtrain, params=ds_params,
                               free_raw_data=not keep_raw_for_cats)
            dval.construct()
            X_val = yv = wv = val = None
            valid_sets = [dval]
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

        num_round = int(self.hp.get("n_estimators", 2000))
        self.booster = lgb.train(params, dtrain, num_boost_round=num_round,
                                 valid_sets=valid_sets, callbacks=callbacks)
        self.best_iteration_ = int(self.booster.best_iteration or 0)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, exposure: np.ndarray | None = None) -> np.ndarray:
        Xp = self._prep_X(X[self._columns], self._cat)
        best = getattr(self.booster, "best_iteration", None) or 0
        return np.asarray(self.booster.predict(Xp, num_iteration=best or None), dtype=float)

    def explain(self) -> Explanation:
        gains = self.booster.feature_importance(importance_type="gain")
        imp = {f: float(g) for f, g in zip(self._columns, gains)}
        return Explanation(kind="gain", global_importance=imp, artifact=self.booster.model_to_string())

    @staticmethod
    def default_hp() -> dict:
        return {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20, "n_estimators": 2000}

    @staticmethod
    def sample_hp(rng: np.random.Generator) -> dict:
        # Full TabArena search space (Table C.2); n_estimators left large + early-stopped.
        return {
            "learning_rate": float(np.exp(rng.uniform(np.log(0.005), np.log(0.1)))),
            "num_leaves": int(np.exp(rng.uniform(np.log(2), np.log(200)))),
            "min_data_in_leaf": int(np.exp(rng.uniform(np.log(1), np.log(64)))),
            "feature_fraction": float(rng.uniform(0.4, 1.0)),
            "bagging_fraction": float(rng.uniform(0.7, 1.0)),
            "extra_trees": bool(rng.integers(0, 2)),
            "min_data_per_group": int(np.exp(rng.uniform(np.log(2), np.log(100)))),
            "cat_l2": float(np.exp(rng.uniform(np.log(0.005), np.log(2.0)))),
            "cat_smooth": float(np.exp(rng.uniform(np.log(0.001), np.log(100.0)))),
            "max_cat_to_onehot": int(np.exp(rng.uniform(np.log(8), np.log(100)))),
            "lambda_l1": float(rng.uniform(1e-4, 1.0)),
            "lambda_l2": float(rng.uniform(1e-4, 2.0)),
            "n_estimators": 2000,
        }
