"""Synthetic datasets — deterministic, offline — one per task, for tests + `smoke`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.tasks import CLF_SPEC, FREQ_SPEC, MULTICLF_SPEC, PP_SPEC, SEV_SPEC


def _features(rng, n):
    df = pd.DataFrame({
        "DrivAge": rng.integers(18, 80, n).astype(float),
        "BonusMalus": rng.integers(50, 230, n).astype(float),
        "VehPower": rng.integers(4, 12, n).astype(float),
        "Region": rng.choice(list("ABCDE"), n),
    })
    region = pd.Series(df["Region"]).map({"A": 0.0, "B": 0.1, "C": 0.2, "D": 0.3, "E": 0.4}).to_numpy()
    lin = 0.004 * (df["BonusMalus"].to_numpy() - 100) + 0.05 * df["VehPower"].to_numpy() + region
    return df, lin


class SyntheticFreqLoader:
    name = "synthetic_freq"

    def __init__(self, n: int = 8000, seed: int = 0, subsample: int | None = None):
        self.n, self.seed = subsample or n, seed

    def load(self) -> DatasetBundle:
        rng = np.random.default_rng(self.seed)
        X, lin = _features(rng, self.n)
        exposure = np.clip(rng.beta(2, 2, self.n), 0.05, 1.0)
        rate = np.exp(-2.4 + lin)
        counts = rng.poisson(rate * exposure).astype(float)
        return DatasetBundle("synthetic_freq", X, counts, exposure, exposure, FREQ_SPEC,
                             cat_features=["Region"], meta=_meta(self.n, "claim_occurrence", self.source_hash()))

    def source_hash(self) -> str:
        return f"synthetic_freq-{self.seed}-{self.n}"


class SyntheticSevLoader:
    name = "synthetic_sev"

    def __init__(self, n: int = 4000, seed: int = 1, subsample: int | None = None):
        self.n, self.seed = subsample or n, seed

    def load(self) -> DatasetBundle:
        rng = np.random.default_rng(self.seed)
        X, lin = _features(rng, self.n)
        mean = np.exp(7.5 + lin)                       # Gamma mean (cost per claim)
        severity = rng.gamma(shape=2.0, scale=mean / 2.0)  # > 0
        weight = rng.integers(1, 4, self.n).astype(float)  # claim count weight
        return DatasetBundle("synthetic_sev", X, severity, None, weight, SEV_SPEC,
                             cat_features=["Region"], meta=_meta(self.n, None, self.source_hash()))

    def source_hash(self) -> str:
        return f"synthetic_sev-{self.seed}-{self.n}"


class SyntheticPPLoader:
    name = "synthetic_pp"

    def __init__(self, n: int = 8000, seed: int = 2, subsample: int | None = None):
        self.n, self.seed = subsample or n, seed

    def load(self) -> DatasetBundle:
        rng = np.random.default_rng(self.seed)
        X, lin = _features(rng, self.n)
        exposure = np.clip(rng.beta(2, 2, self.n), 0.05, 1.0)
        counts = rng.poisson(np.exp(-2.4 + lin) * exposure)
        sev = rng.gamma(shape=2.0, scale=np.exp(7.5 + lin) / 2.0)
        total_loss = counts * sev                      # compound Poisson-Gamma; 0 when no claim
        return DatasetBundle("synthetic_pp", X, total_loss.astype(float), exposure, exposure, PP_SPEC,
                             cat_features=["Region"], meta=_meta(self.n, None, self.source_hash()))

    def source_hash(self) -> str:
        return f"synthetic_pp-{self.seed}-{self.n}"


class SyntheticClfLoader:
    name = "synthetic_clf"

    def __init__(self, n: int = 6000, seed: int = 3, subsample: int | None = None):
        self.n, self.seed = subsample or n, seed

    def load(self) -> DatasetBundle:
        rng = np.random.default_rng(self.seed)
        X, lin = _features(rng, self.n)
        prob = 1.0 / (1.0 + np.exp(-(-2.0 + 3.0 * lin)))
        y = (rng.uniform(size=self.n) < prob).astype(float)
        return DatasetBundle("synthetic_clf", X, y, None, np.ones(self.n), CLF_SPEC,
                             cat_features=["Region"], meta=_meta(self.n, "claim_occurrence", self.source_hash()))

    def source_hash(self) -> str:
        return f"synthetic_clf-{self.seed}-{self.n}"


class SyntheticMulticlfLoader:
    name = "synthetic_multiclf"

    def __init__(self, n: int = 6000, seed: int = 4, subsample: int | None = None, n_classes: int = 4):
        self.n, self.seed, self.k = subsample or n, seed, n_classes

    def load(self) -> DatasetBundle:
        rng = np.random.default_rng(self.seed)
        X, lin = _features(rng, self.n)
        # K class scores as ordered shifts of the linear predictor + balancing intercepts, then
        # Gumbel-max sampling = draw from softmax(scores) (vectorised, deterministic given seed).
        scores = np.stack([(i - (self.k - 1) / 2) * lin for i in range(self.k)], axis=1)
        scores = scores + np.linspace(-0.4, 0.4, self.k)
        y = (scores + rng.gumbel(size=(self.n, self.k))).argmax(axis=1).astype(float)
        meta = {**_meta(self.n, "class_label", self.source_hash()), "n_classes": self.k}
        return DatasetBundle("synthetic_multiclf", X, y, None, np.ones(self.n), MULTICLF_SPEC,
                             cat_features=["Region"], meta=meta)

    def source_hash(self) -> str:
        return f"synthetic_multiclf-{self.seed}-{self.n}-{self.k}"


def _meta(n, stratify_on, src_hash):
    return {"source": "synthetic", "license": "n/a", "n_rows": n,
            "stratify_on": stratify_on, "hash": src_hash}
