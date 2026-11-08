"""Glass-box GLM — the actuarial baseline and frozen skill denominator.

Task-aware: picks the distribution from the dataset's task (Poisson freq / Gamma severity /
Tweedie pure-premium / logistic classification). Fully AUTOMATED preprocessing — the feature
pipeline is derived from the data (quantile-binned continuous + one-hot with rare-level grouping),
no per-dataset recipe; the L2 penalty `alpha` is tuned by random search. glum is a drop-in upgrade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.models.base import Explanation, ModelAdapter
from insurarena.scoring import task_targets
from insurarena.tasks import TaskSpec, TaskType

_N_BINS = 10
_LOW_CARD = 12
_MIN_FREQ = 0.005
_MAX_CATS = 100


def _auto_roles(X: pd.DataFrame, cat_features, low_card: int = _LOW_CARD):
    """Detect categorical vs continuous columns from dtype + cardinality (no recipe needed)."""
    declared = set(cat_features or [])
    cat, cont = [], []
    for c in X.columns:
        s = X[c]
        if c in declared or s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
            cat.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            (cat if int(s.nunique(dropna=True)) <= low_card else cont).append(c)
        else:
            cat.append(c)
    return cat, cont


def _build_preprocessor(bundle: DatasetBundle, hp: dict | None = None):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

    hp = hp or {}
    n_bins = int(hp.get("n_bins", _N_BINS))
    cat, cont = _auto_roles(bundle.X, bundle.cat_features, int(hp.get("low_card", _LOW_CARD)))

    transformers = []
    if cont:
        transformers.append(
            ("cont", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                # sparse one-hot of quantile bins -> scales to full datasets without dense blow-up
                ("bin", KBinsDiscretizer(n_bins=n_bins, encode="onehot", strategy="quantile")),
            ]), cont)
        )
    if cat:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=float(hp.get("min_frequency", _MIN_FREQ)),
                                  max_categories=int(hp.get("max_categories", _MAX_CATS)),
                                  sparse_output=True), cat)
        )
    if not transformers:
        transformers.append(("passthrough", "passthrough", list(bundle.X.columns)))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


def _estimator(task: TaskSpec, alpha: float):
    t = task.task
    if t == TaskType.FREQ:
        from sklearn.linear_model import PoissonRegressor
        return PoissonRegressor(alpha=alpha, max_iter=400)
    if t == TaskType.SEV:
        from sklearn.linear_model import GammaRegressor
        return GammaRegressor(alpha=alpha, max_iter=400)
    if t == TaskType.PP:
        from sklearn.linear_model import TweedieRegressor
        return TweedieRegressor(power=task.tweedie_power or 1.5, alpha=alpha, max_iter=400)
    if t in (TaskType.CLF, TaskType.MULTICLF):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0 / max(alpha, 1e-9), max_iter=400)  # multinomial for K>2 labels
    raise ValueError(f"unsupported task {t}")


class GLM(ModelAdapter):
    name = "glm"
    family = "glm"
    needs_es_probe = False               # single closed-form fit, no early stopping -> skip fit_one's probe
    supported_tasks = (TaskType.FREQ, TaskType.SEV, TaskType.PP, TaskType.CLF, TaskType.MULTICLF)

    def fit(self, train: DatasetBundle, val: DatasetBundle | None = None) -> "GLM":
        from sklearn.pipeline import Pipeline

        self._task = train.task.task
        est = _estimator(train.task, float(self.hp.get("alpha", 1e-4)))
        self.pipe = Pipeline([("pre", _build_preprocessor(train, self.hp)), ("est", est)])
        y, w, _ = task_targets(train)
        self.pipe.fit(train.X, y, est__sample_weight=w)
        self.best_iteration_ = 0
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, exposure: np.ndarray | None = None) -> np.ndarray:
        if self._task == TaskType.MULTICLF:
            return np.asarray(self.pipe.predict_proba(X), dtype=float)   # (n, K) class probabilities
        if self._task == TaskType.CLF:
            return np.asarray(self.pipe.predict_proba(X)[:, 1], dtype=float)
        return np.asarray(self.pipe.predict(X), dtype=float)  # rate / cost (natural scale)

    def explain(self) -> Explanation:
        est = self.pipe.named_steps["est"]
        coef = np.ravel(est.coef_)
        try:
            names = self.pipe.named_steps["pre"].get_feature_names_out()
            coefs = dict(zip(map(str, names), map(float, coef)))
        except Exception:
            coefs = {"coef_": [float(c) for c in coef]}
        coefs["intercept_"] = float(np.ravel(est.intercept_)[0])
        return Explanation(kind="coefficients", global_importance=coefs, artifact=coefs)

    @staticmethod
    def default_hp() -> dict:
        return {"alpha": 1e-4}

    @staticmethod
    def sample_hp(rng: np.random.Generator) -> dict:
        return {"alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(10.0))))}
