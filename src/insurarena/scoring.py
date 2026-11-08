"""Resolve a task's scoring target (y_true, sample_weight, exposure) and score predictions."""
from __future__ import annotations

import numpy as np

from insurarena.datasets.base import DatasetBundle
from insurarena.metrics.registry import score_all
from insurarena.tasks import TaskType


def validate_bundle(bundle: DatasetBundle) -> None:
    """Validate dataset bundle targets, exposure, and sample weights per task convention.

    Enforces finite positive exposure (no silent clipping), finite nonnegative positive-total
    weights, and valid task-specific target domains. Do not silently drop rows.
    """
    for name in ("y", "sample_weight", "exposure"):
        value = getattr(bundle, name)
        if value is not None and np.asarray(value).shape != (len(bundle.X),):
            raise ValueError(f"{name} must be an aligned one-dimensional vector")
    if bundle.exposure is not None:
        exp = np.asarray(bundle.exposure, dtype=float)
        if not np.all(np.isfinite(exp)):
            raise ValueError("exposure must be finite, contains NaN or Inf")
        if not np.all(exp > 0.0):
            raise ValueError("exposure must be strictly positive (> 0), no silent clipping")

    w = np.asarray(bundle.sample_weight, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight must be finite, contains NaN or Inf")
    if not np.all(w >= 0.0):
        raise ValueError("sample_weight must be nonnegative (>= 0)")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("sample_weight total must be strictly positive (> 0)")

    y = np.asarray(bundle.y, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("target y must be finite, contains NaN or Inf")

    t = bundle.task.task
    if t == TaskType.FREQ:
        if (y < 0.0).any():
            raise ValueError("FREQ target y must be nonnegative (>= 0)")
    elif t == TaskType.PP:
        power = bundle.task.tweedie_power
        if power is None or not np.isfinite(power) or not 1 < power < 2:
            raise ValueError("pure-premium Tweedie scoring power must be fixed in (1, 2)")
        if (y < 0.0).any():
            raise ValueError("PP target y must be nonnegative (>= 0)")
    elif t == TaskType.SEV:
        if (y <= 0.0).any():
            raise ValueError("SEV target y must be strictly positive (> 0)")
    elif t == TaskType.CLF:
        unique_y = np.unique(y)
        if not np.all(np.isin(unique_y, [0.0, 1.0])):
            raise ValueError(f"CLF target y must be binary {{0, 1}}, got {unique_y}")
    elif t == TaskType.MULTICLF:
        if not (np.all(y >= 0.0) and np.all(y == np.round(y))):
            raise ValueError("MULTICLF target y must be nonnegative integer class codes")
        n_classes = bundle.meta.get("n_classes")
        if n_classes is not None and (y >= int(n_classes)).any():
            raise ValueError(f"MULTICLF target y contains class codes >= n_classes={n_classes}")
    else:
        raise ValueError(f"unsupported task {t}")


def validate_predictions(bundle: DatasetBundle, y_pred) -> None:
    """Validate prediction shapes, finite ranges, and task-appropriate values without dropping rows."""
    if y_pred is None:
        raise ValueError("predictions cannot be None")

    n_expected = len(bundle.y)
    pred_arr = np.asarray(y_pred, dtype=float)

    if pred_arr.ndim == 0:
        raise ValueError("predictions must have a row dimension")
    if len(pred_arr) != n_expected:
        raise ValueError(
            f"prediction row count ({len(pred_arr)}) does not match dataset bundle ({n_expected}); do not silently drop rows"
        )
    if not np.all(np.isfinite(pred_arr)):
        raise ValueError("predictions must be finite, contain NaN or Inf")

    t = bundle.task.task
    if t != TaskType.MULTICLF and pred_arr.ndim != 1:
        raise ValueError("scalar predictions must have shape (n,)")
    if t == TaskType.MULTICLF:
        if pred_arr.ndim != 2:
            raise ValueError(f"MULTICLF predictions must be 2D array of shape (n, K), got ndim={pred_arr.ndim}")
        if pred_arr.shape[0] != n_expected:
            raise ValueError(f"MULTICLF predictions row count ({pred_arr.shape[0]}) != expected ({n_expected})")
        n_classes = bundle.meta.get("n_classes")
        if n_classes is not None and pred_arr.shape[1] != int(n_classes):
            raise ValueError("probability columns must match the fixed class mapping")
        if len(bundle.y) and np.max(bundle.y) >= pred_arr.shape[1]:
            raise ValueError("probability columns do not cover target classes")
        if pred_arr.shape[1] < 2:
            raise ValueError(f"MULTICLF predictions must have at least 2 classes, got {pred_arr.shape[1]}")
        if (pred_arr < -1e-6).any() or (pred_arr > 1.0 + 1e-6).any():
            raise ValueError("MULTICLF predicted probabilities must be in range [0, 1]")
        row_sums = pred_arr.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError("MULTICLF predicted probabilities rows must sum to 1.0")
    elif t == TaskType.CLF:
        if pred_arr.ndim != 1:
            if pred_arr.ndim == 2 and pred_arr.shape[1] == 1:
                pred_arr = pred_arr.ravel()
            else:
                raise ValueError(f"CLF predictions must be 1D array of shape (n,), got shape {pred_arr.shape}")
        if (pred_arr < -1e-6).any() or (pred_arr > 1.0 + 1e-6).any():
            raise ValueError("CLF predicted probabilities must be in range [0, 1]")
    elif t in (TaskType.FREQ, TaskType.PP):
        if pred_arr.ndim != 1:
            if pred_arr.ndim == 2 and pred_arr.shape[1] == 1:
                pred_arr = pred_arr.ravel()
            else:
                raise ValueError(f"{t.value.upper()} predictions must be 1D rate array of shape (n,), got shape {pred_arr.shape}")
        if (pred_arr < 0.0).any():
            raise ValueError(f"{t.value.upper()} predictions must be nonnegative rates (>= 0)")
    elif t == TaskType.SEV:
        if pred_arr.ndim != 1:
            if pred_arr.ndim == 2 and pred_arr.shape[1] == 1:
                pred_arr = pred_arr.ravel()
            else:
                raise ValueError(f"SEV predictions must be 1D severity array of shape (n,), got shape {pred_arr.shape}")
        if (pred_arr <= 0.0).any():
            raise ValueError("SEV predictions must be strictly positive mean severities (> 0)")
    else:
        raise ValueError(f"unsupported task {t}")


def target_contract(bundle: DatasetBundle) -> dict:
    """Return canonical target conventions dict for the bundle's task with fixed Tweedie power.

    Describes canonical conventions:
    - FREQ / PP: canonical rate predictions and responses with exposure weighting and total = prediction * exposure
    - SEV: mean severity predictions claim-weighted by claim count and total = prediction * claim_count
    - CLF: probabilities with binary label response
    - MULTICLF: class probabilities (n, K) with class code response
    """
    validate_bundle(bundle)
    t = bundle.task.task
    tweedie_power = bundle.task.tweedie_power if t == TaskType.PP else None

    if t in (TaskType.FREQ, TaskType.PP):
        response_conv = "rate"
        pred_conv = "rate"
        weight_conv = "exposure"
        total_conv = "prediction * exposure"
    elif t == TaskType.SEV:
        response_conv = "mean_severity"
        pred_conv = "mean_severity"
        weight_conv = "claim_count"
        total_conv = "prediction * claim_count"
    elif t == TaskType.CLF:
        response_conv = "binary_label"
        pred_conv = "probability"
        weight_conv = "sample_weight"
        total_conv = "prediction * weight"
    elif t == TaskType.MULTICLF:
        response_conv = "class_code"
        pred_conv = "probabilities"
        weight_conv = "sample_weight"
        total_conv = "prediction * weight"
    else:
        raise ValueError(f"unsupported task {t}")

    proxy_objectives: dict[str, str] = {}
    if t == TaskType.SEV:
        proxy_objectives["catboost"] = "tweedie:variance_power=1.99"

    return {
        "task": t.value,
        "distribution": bundle.task.distribution,
        "link": bundle.task.link,
        "response": response_conv,
        "prediction": pred_conv,
        "weight": weight_conv,
        "total": total_conv,
        "tweedie_power": tweedie_power,
        "fixed_tweedie_power": tweedie_power,
        "proxy_objectives": proxy_objectives,
    }


def task_targets(bundle: DatasetBundle):
    """Return (y_true, sample_weight, exposure) on the scale predictions live on."""
    validate_bundle(bundle)
    t = bundle.task.task
    if t in (TaskType.FREQ, TaskType.PP):
        # Exposure is validated finite and strictly positive (> 0); no silent clipping.
        # sample_weight in DatasetBundle is already exposure for freq/pp; do not multiply twice.
        exp = np.asarray(bundle.exposure, dtype=float) if bundle.exposure is not None else np.ones(len(bundle.y), dtype=float)
        return np.asarray(bundle.y, dtype=float) / exp, exp, exp  # rate, weight=exposure, exposure
    if t == TaskType.SEV:
        return bundle.y, bundle.sample_weight, None
    if t == TaskType.CLF:
        return bundle.y, bundle.sample_weight, bundle.exposure
    if t == TaskType.MULTICLF:  # y = integer class codes; preds are (n, K)
        return bundle.y, bundle.sample_weight, None
    raise ValueError(f"unsupported task {t}")


def score_bundle(bundle: DatasetBundle, y_pred, metric_names) -> dict[str, float]:
    validate_predictions(bundle, y_pred)
    y_true, w, exp = task_targets(bundle)
    return score_all(
        metric_names, y_true, y_pred, sample_weight=w, exposure=exp,
        tweedie_power=bundle.task.tweedie_power,
    )


def primary_value(bundle: DatasetBundle, y_pred) -> float:
    from insurarena.metrics.registry import METRICS

    validate_predictions(bundle, y_pred)
    y_true, w, exp = task_targets(bundle)
    m = METRICS[bundle.task.primary_metric]
    kw = {"tweedie_power": bundle.task.tweedie_power} if "tweedie_power" in m.needs else {}
    try:
        return float(m.fn(y_true, y_pred, sample_weight=w, exposure=exp, **kw))
    except TypeError:
        return float(m.fn(y_true, y_pred, sample_weight=w, **kw))
