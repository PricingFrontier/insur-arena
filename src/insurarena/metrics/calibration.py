"""Calibration axis — balance (Sigma w*pred / Sigma w*actual -> 1) + per-decile deviation."""
from __future__ import annotations

import numpy as np


def _w(y, sample_weight):
    y = np.asarray(y, dtype=float)
    return np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)


def balance(y_true, y_pred, sample_weight=None, **_) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = _w(y_true, sample_weight)
    denom = float(np.sum(w * y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(w * y_pred) / denom)


def max_decile_balance_dev(y_true, y_pred, sample_weight=None, n_bins: int = 10, **_) -> float:
    """Max |balance - 1| across exposure-weighted prediction deciles (segment calibration)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = _w(y_true, sample_weight)
    order = np.argsort(y_pred, kind="mergesort")
    yt, yp, ww = y_true[order], y_pred[order], w[order]
    cum_w = np.cumsum(ww)
    total = cum_w[-1]
    if total <= 0:
        return float("nan")
    edges = np.searchsorted(cum_w, np.linspace(0, total, n_bins + 1)[1:-1])
    worst = 0.0
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(yt)]):
        if hi <= lo:
            continue
        denom = float(np.sum(ww[lo:hi] * yt[lo:hi]))
        if denom == 0:
            continue
        bal = float(np.sum(ww[lo:hi] * yp[lo:hi]) / denom)
        worst = max(worst, abs(bal - 1.0))
    return worst
