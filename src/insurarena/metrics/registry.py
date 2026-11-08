"""Metric registry — name -> (callable, direction, weighting)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from insurarena.metrics import calibration as _cal
from insurarena.metrics import deviance as _dev
from insurarena.metrics import discrimination as _disc


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable
    direction: str            # "lower" | "higher" | "target1"
    weighting: str            # "exposure" | "claims" | "none"
    needs: tuple[str, ...] = field(default_factory=tuple)  # e.g. ("tweedie_power",)


METRICS: dict[str, Metric] = {}


def register(m: Metric) -> None:
    METRICS[m.name] = m


def get_metric(name: str) -> Metric:
    return METRICS[name]


for _m in [
    Metric("poisson_deviance", _dev.poisson_deviance, "lower", "exposure"),
    Metric("gamma_deviance", _dev.gamma_deviance, "lower", "claims"),
    Metric("tweedie_deviance", _dev.tweedie_deviance, "lower", "exposure", ("tweedie_power",)),
    Metric("log_loss", _dev.log_loss, "lower", "exposure"),
    Metric("multiclass_log_loss", _dev.multiclass_log_loss, "lower", "none"),
    Metric("multiclass_accuracy", _dev.multiclass_accuracy, "higher", "none"),
    Metric("d2_poisson", _dev.d2_poisson, "higher", "exposure"),
    Metric("d2_gamma", _dev.d2_gamma, "higher", "claims"),
    Metric("d2_tweedie", _dev.d2_tweedie, "higher", "exposure", ("tweedie_power",)),
    Metric("gini_exposure", _disc.gini_exposure, "higher", "exposure"),
    Metric("normalized_gini", _disc.normalized_gini, "higher", "exposure"),
    Metric("roc_auc", _disc.roc_auc, "higher", "exposure"),
    Metric("balance", _cal.balance, "target1", "exposure"),
    Metric("max_decile_balance_dev", _cal.max_decile_balance_dev, "lower", "exposure"),
]:
    register(_m)


def score_all(
    metric_names,
    y_true,
    y_pred,
    sample_weight=None,
    exposure=None,
    tweedie_power: float | None = None,
) -> dict[str, float]:
    """Evaluate a list of metrics, passing exposure/power where a metric needs them."""
    out: dict[str, float] = {}
    for name in metric_names:
        m = METRICS[name]
        kw: dict = {}
        if "tweedie_power" in m.needs and tweedie_power is not None:
            kw["tweedie_power"] = tweedie_power
        try:
            out[name] = float(
                m.fn(y_true, y_pred, sample_weight=sample_weight, exposure=exposure, **kw)
            )
        except TypeError:
            # metric does not accept exposure kwarg
            out[name] = float(m.fn(y_true, y_pred, sample_weight=sample_weight, **kw))
    return out
