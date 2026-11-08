"""Accuracy axis — strictly-consistent deviances + D² (never RMSE/MAE for these targets)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_tweedie_deviance,
)


def _w(y, sample_weight):
    y = np.asarray(y, dtype=float)
    if sample_weight is None:
        return np.ones_like(y)
    return np.asarray(sample_weight, dtype=float)


def poisson_deviance(y_true, y_pred, sample_weight=None, **_) -> float:
    return float(mean_poisson_deviance(y_true, y_pred, sample_weight=sample_weight))


def gamma_deviance(y_true, y_pred, sample_weight=None, **_) -> float:
    return float(mean_gamma_deviance(y_true, y_pred, sample_weight=sample_weight))


def tweedie_deviance(y_true, y_pred, sample_weight=None, tweedie_power: float = 1.9, **_) -> float:
    return float(mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=tweedie_power))


def _d2(y_true, y_pred, sample_weight, dev_fn, **kw) -> float:
    """Deviance explained vs the exposure-weighted intercept-only model."""
    y_true = np.asarray(y_true, dtype=float)
    w = _w(y_true, sample_weight)
    dev_model = dev_fn(y_true, y_pred, sample_weight=sample_weight, **kw)
    ybar = float(np.average(y_true, weights=w))
    y_null = np.full_like(y_true, max(ybar, 1e-12))
    dev_null = dev_fn(y_true, y_null, sample_weight=sample_weight, **kw)
    if dev_null <= 0:
        return float("nan")
    return 1.0 - dev_model / dev_null


def d2_poisson(y_true, y_pred, sample_weight=None, **_) -> float:
    return _d2(y_true, y_pred, sample_weight, poisson_deviance)


def d2_gamma(y_true, y_pred, sample_weight=None, **_) -> float:
    return _d2(y_true, y_pred, sample_weight, gamma_deviance)


def d2_tweedie(y_true, y_pred, sample_weight=None, tweedie_power: float = 1.9, **_) -> float:
    return _d2(y_true, y_pred, sample_weight, tweedie_deviance, tweedie_power=tweedie_power)


def log_loss(y_true, y_pred, sample_weight=None, **_) -> float:
    from sklearn.metrics import log_loss as _ll

    return float(_ll(y_true, np.clip(y_pred, 1e-15, 1 - 1e-15), sample_weight=sample_weight, labels=[0, 1]))


def multiclass_log_loss(y_true, y_pred, sample_weight=None, **_) -> float:
    """Multinomial cross-entropy. y_pred is an (n, K) class-probability matrix; y_true integer codes."""
    from sklearn.metrics import log_loss as _ll

    p = np.asarray(y_pred, dtype=float)
    if p.ndim == 1:                                   # degenerate 1-col edge case
        p = np.column_stack([1.0 - p, p])
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)              # renormalise after clipping
    return float(_ll(np.asarray(y_true, dtype=int), p,
                     sample_weight=sample_weight, labels=list(range(p.shape[1]))))


def multiclass_accuracy(y_true, y_pred, sample_weight=None, **_) -> float:
    """Weighted top-1 accuracy of argmax(y_pred) vs y_true (integer codes). Higher is better."""
    p = np.asarray(y_pred, dtype=float)
    pred = p.argmax(axis=1) if p.ndim > 1 else (p > 0.5).astype(int)
    hit = (pred == np.asarray(y_true, dtype=int)).astype(float)
    w = _w(hit, sample_weight)
    return float(np.average(hit, weights=w))


def skill_vs_baseline(dev_model: float, dev_baseline: float) -> float:
    """Skill score 1 - D_model/D_baseline (SPEC §3.2). Baseline is the frozen GLM."""
    if dev_baseline is None or not np.isfinite(dev_baseline) or dev_baseline <= 0:
        return float("nan")
    return 1.0 - dev_model / dev_baseline
