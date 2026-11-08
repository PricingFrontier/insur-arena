"""Discrimination axis — exposure-weighted Lorenz / Gini (SPEC §3.3).

Cumulate exposure on x and exposure*rate (= loss euros) on y. Row-weighting silently
misranks, so these refuse to be computed without exposure on insurance tasks.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, roc_auc_score


def lorenz_curve(y_true, y_pred, exposure):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    order = np.argsort(y_pred, kind="mergesort")  # ascending predicted risk; stable
    e = exposure[order]
    loss = y_true[order] * e  # loss euros = rate * exposure
    cum_loss = np.cumsum(loss)
    cum_exp = np.cumsum(e)
    if cum_loss[-1] <= 0 or cum_exp[-1] <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cum_loss = np.insert(cum_loss / cum_loss[-1], 0, 0.0)
    cum_exp = np.insert(cum_exp / cum_exp[-1], 0, 0.0)
    return cum_exp, cum_loss


def gini_exposure(y_true, y_pred, sample_weight=None, exposure=None, **_) -> float:
    exp = exposure if exposure is not None else sample_weight
    if exp is None:
        raise ValueError("exposure-weighted Gini requires exposure/sample_weight — refusing unweighted")
    cx, cy = lorenz_curve(y_true, y_pred, exp)
    return float(1.0 - 2.0 * auc(cx, cy))


def normalized_gini(y_true, y_pred, sample_weight=None, exposure=None, **_) -> float:
    """model Gini / oracle Gini; oracle = sort by actual outcome (empirical upper bound)."""
    exp = exposure if exposure is not None else sample_weight
    g_model = gini_exposure(y_true, y_pred, exposure=exp)
    g_oracle = gini_exposure(y_true, y_true, exposure=exp)
    if g_oracle == 0:
        return float("nan")
    return g_model / g_oracle


def roc_auc(y_true, y_pred, sample_weight=None, **_) -> float:
    return float(roc_auc_score(y_true, y_pred, sample_weight=sample_weight))
