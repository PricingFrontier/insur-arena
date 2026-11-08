"""EBM (Explainable Boosting Machine / GA2M) — a GAM-style technique benchmarked alongside the rest.

Task-aware via the regressor objective (poisson/gamma/tweedie deviance) + the classifier.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.models.base import Explanation, ModelAdapter
from insurarena.scoring import task_targets
from insurarena.tasks import TaskType

_OBJ = {TaskType.FREQ: "poisson_deviance", TaskType.SEV: "gamma_deviance"}

# EBM's outer_bags AVERAGES per-feature shape functions across bags -> the result is still ONE GA2M
# (one shape function per feature / interaction pair), so interpretability is fully preserved
# (analogous to tri-boost's exactness-preserving bagging). We enable it, but only on the *deployed*
# model: during the tuned OOF search we fit a single bag for speed (bagging is a fixed structural
# choice, not a tuned hyper-parameter). Library default is 14; 4 keeps most of the variance-reduction
# benefit (diminishing returns past ~4) while staying tractable with interactions on wide datasets
# (e.g. catelematic13's 49 features -> C(49,2)=1176 candidate pairs scanned per bag).
_EBM_BAGS = 4

# Params forwarded to the EBM constructor when present in hp; anything absent uses the LIBRARY default,
# so the `default` variant matches TabArena's EBM-D (interactions='5x', lr=0.04, ...). Interactions ON.
_EBM_KEYS = ("learning_rate", "interactions", "max_leaves", "smoothing_rounds", "min_samples_leaf",
             "interaction_smoothing_rounds", "min_hessian", "validation_size",
             "early_stopping_tolerance", "gain_scale", "max_bins", "max_rounds")


_N_INTERACTIONS = 16                     # fixed COUNT of pairwise interactions (GA2M). TabArena's
                                         # '5x'/0.95-0.999 FRACTION -> ~100 pairs on our wide, high-
                                         # cardinality insurance data = intractable; a bounded count
                                         # keeps EBM a real GA2M (interaction capacity ~ tri-boost's) yet
                                         # runnable. Necessary deviation from Table C.6's fraction.


class EBM(ModelAdapter):
    name = "ebm"
    family = "gam"
    adapter_version = "1"                # reset 2026-09-09 (arena rebuilt from scratch); prior history in git
    hpo_search_version = "1"             # reset 2026-09-09 (arena rebuilt from scratch); prior history in git
    needs_es_probe = False               # EBM early-stops internally; skip fit_one's probe (no wasted refit)
    supported_tasks = (TaskType.FREQ, TaskType.SEV, TaskType.PP, TaskType.CLF, TaskType.MULTICLF)

    def fit(self, train: DatasetBundle, val: DatasetBundle | None = None) -> "EBM":
        from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

        self._columns = list(train.X.columns)
        self._task = train.task.task
        y, w, _ = task_targets(train)
        X = train.X[self._columns]
        # full bags for the deployed model (val is None); a single bag while searching configs (val given)
        bags = int(self.hp.get("outer_bags", _EBM_BAGS)) if val is None else 1
        _t = os.environ.get("OMP_NUM_THREADS")            # respect the parallel-worker thread budget
        # Single-bag search fits (val given) force n_jobs=1: one bag gives the joblib pool nothing
        # to parallelize, and keeping loky worker processes out of the long-lived search coordinator
        # avoids the stale-pool stall that hung the brvehins1 search (07-17). Scheduling only —
        # fits are seed-deterministic at any n_jobs, so cached scores are unaffected.
        common = {"outer_bags": bags, "random_state": int(self.hp.get("seed", 0)),
                  "n_jobs": 1 if val is not None else (int(_t) if _t else -1)}
        for k in _EBM_KEYS:              # absent -> library default (interactions='5x' etc.)
            if k in self.hp:
                common[k] = self.hp[k]
        self._is_multi = self._task == TaskType.MULTICLF
        self._is_clf = self._task in (TaskType.CLF, TaskType.MULTICLF)
        if self._is_multi:
            common["interactions"] = 0        # GA2M interactions unsupported for multiclass -> pure GAM
        if self._is_clf:
            self.model = ExplainableBoostingClassifier(**common)
        else:
            obj = (f"tweedie_deviance:variance_power={train.task.tweedie_power or 1.5}"
                   if self._task == TaskType.PP else _OBJ[self._task])
            self.model = ExplainableBoostingRegressor(objective=obj, **common)
        self.model.fit(X, y, sample_weight=w)
        self.best_iteration_ = 0     # EBM early-stops internally; no external iteration to refit
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, exposure: np.ndarray | None = None) -> np.ndarray:
        Xp = X[self._columns]
        if self._is_multi:
            return np.asarray(self.model.predict_proba(Xp), dtype=float)   # (n, K)
        if self._is_clf:
            return np.asarray(self.model.predict_proba(Xp)[:, 1], dtype=float)
        return np.asarray(self.model.predict(Xp), dtype=float)

    def explain(self) -> Explanation:
        try:
            names = list(self.model.term_names_)
            imp = {n: float(s) for n, s in zip(names, self.model.term_importances())}
        except Exception:
            imp = {}
        return Explanation(kind="ebm_shape", global_importance=imp)

    @staticmethod
    def default_hp() -> dict:
        # library defaults for everything EXCEPT interactions, which we bound to a fixed count (GA2M).
        return {"interactions": _N_INTERACTIONS}

    @staticmethod
    def sample_hp(rng: np.random.Generator) -> dict:
        # TabArena search space (Table C.6), but interactions as a bounded COUNT (0 = additive .. 32)
        # instead of the fraction — so tuning can find the additive sweet spot AND stay tractable on
        # wide/high-cardinality insurance data.
        return {
            "max_leaves": int(rng.integers(2, 4)),                                    # [2, 3]
            "smoothing_rounds": int(rng.integers(0, 1001)),                           # [0, 1000]
            "learning_rate": float(np.exp(rng.uniform(np.log(0.0025), np.log(0.2)))),
            "interactions": int(rng.integers(0, 33)),                                # 0 (additive) .. 32
            "interaction_smoothing_rounds": int(rng.integers(0, 201)),               # [0, 200]
            "min_hessian": float(np.exp(rng.uniform(np.log(1e-10), np.log(1e-2)))),
            "min_samples_leaf": int(rng.integers(2, 21)),                            # [2, 20]
            "validation_size": float(rng.uniform(0.05, 0.25)),
            "early_stopping_tolerance": float(np.exp(rng.uniform(np.log(1e-10), np.log(1e-5)))),
            "gain_scale": float(np.exp(rng.uniform(np.log(0.5), np.log(5.0)))),
        }
