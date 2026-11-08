"""Evidence and timing support for model evaluation and benchmark reporting.

Functions:
  - collect_evidence: Collects JSON-serializable diagnostic evidence (portfolio A/E,
    target units, row hashes, safe model artifact size, TBoost table/box census
    and raw export agreement, allowlisted feature segments).
  - prediction_payload: Constructs dict of non-object numpy arrays for safe archival
    (y_true canonical, original y, predictions, weights, exposure, row positions,
    group identifiers, approved segment columns).
  - measure_prediction: Benchmarks single-row and batch inference latency and throughput
    (warmup, pure predict time, median and p95).
  - write_evidence: Atomically persists evidence JSON and optional compressed npz arrays
    keyed by cell signature.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, NamedTuple, Sequence
import uuid

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.scoring import task_targets
from insurarena.tasks import TaskType


class EvidencePaths(NamedTuple):
    """Container for written evidence paths supporting attr, key, and tuple access."""

    json: Path
    npz: Path | None

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item in ("json", "json_path"):
                return self.json
            if item in ("npz", "npz_path"):
                return self.npz
            raise KeyError(f"Invalid evidence path key: {item!r}")
        return tuple.__getitem__(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _clean_for_json(obj: Any) -> Any:
    """Recursively convert objects to standard JSON-serializable types, replacing non-finites with None."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _clean_for_json(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value"):  # Enum support
        return _clean_for_json(obj.value)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _clean_for_json(obj.to_dict())
        except Exception:
            pass
    return str(obj)


def _compute_row_hashes(train_rows: Sequence[int] | np.ndarray | None,
                        test_rows: Sequence[int] | np.ndarray | None) -> dict[str, Any]:
    """Compute sha256 hashes for train and test row index sets."""
    tr_arr = np.ascontiguousarray(train_rows if train_rows is not None else [], dtype=np.int64)
    te_arr = np.ascontiguousarray(test_rows if test_rows is not None else [], dtype=np.int64)
    return {
        "train_rows_hash": hashlib.sha256(tr_arr.tobytes()).hexdigest(),
        "test_rows_hash": hashlib.sha256(te_arr.tobytes()).hexdigest(),
        "train_rows_count": int(len(tr_arr)),
        "test_rows_count": int(len(te_arr)),
    }


def _inspect_safe_model_artifact_size(model: Any) -> dict[str, Any]:
    """Inspect model artifact size via safe serialization where available.
    
    Native formats are preferred; known local adapters may be measured with pickle
    protocol 5. No pickle payload is loaded or retained.
    """
    # 1. Check if model has native to_bytes / to_json (e.g. TBoost / t_boost / inner model)
    est = getattr(model, "model", model)
    tm = getattr(est, "_model", None) or getattr(model, "_model", None)

    for target in (est, tm, model):
        if target is not None and hasattr(target, "to_bytes") and callable(target.to_bytes):
            try:
                b = bytes(target.to_bytes())
                return {
                    "artifact_size_bytes": len(b),
                    "serialization_method": "native_to_bytes",
                    "is_exact_bytes": True,
                    "unsupported_reason": None,
                }
            except Exception:
                pass

    for target in (est, tm, model):
        if target is not None and hasattr(target, "to_json") and callable(target.to_json):
            try:
                s = target.to_json()
                if isinstance(s, str):
                    return {
                        "artifact_size_bytes": len(s.encode("utf-8")),
                        "serialization_method": "native_to_json",
                        "is_exact_bytes": True,
                        "unsupported_reason": None,
                    }
            except Exception:
                pass

    # 2. LightGBM booster
    booster = getattr(model, "booster", None) or getattr(model, "booster_", None) or getattr(est, "booster_", None)
    if booster is not None and hasattr(booster, "model_to_string"):
        try:
            s = booster.model_to_string()
            return {
                "artifact_size_bytes": len(s.encode("utf-8")),
                "serialization_method": "lightgbm_model_to_string",
                "is_exact_bytes": True,
                "unsupported_reason": None,
            }
        except Exception:
            pass

    # 3. XGBoost booster
    if hasattr(est, "get_booster"):
        try:
            raw = est.get_booster().save_raw(raw_format="json")
            return {
                "artifact_size_bytes": len(bytes(raw)),
                "serialization_method": "xgboost_save_raw_json",
                "is_exact_bytes": True,
                "unsupported_reason": None,
            }
        except Exception:
            pass

    # Standard Python estimators (GLM/EBM) have no native portable model format.
    # Measure only a locally fitted adapter from this package; no pickle is loaded.
    if type(model).__module__.startswith('insurarena.models.'):
        try:
            import pickle
            blob = pickle.dumps(model, protocol=5)
            return {"artifact_size_bytes": len(blob), "serialization_method": "python_pickle_protocol5_size_only",
                    "is_exact_bytes": True, "unsupported_reason": None,
                    "scope": "complete fitted adapter; size measurement only"}
        except Exception:
            pass
    # No serializable artifact available.
    return {
        "artifact_size_bytes": None,
        "serialization_method": None,
        "is_exact_bytes": False,
        "unsupported_reason": "No safe native serialization method available without object pickling",
    }


def _resolve_tboost_components(model: Any) -> tuple[Any, Any]:
    """Resolve (estimator, inner_model) for TBoost adapter or raw t_boost model."""
    inner_est = getattr(model, "model", None)
    if inner_est is not None:
        tm = getattr(inner_est, "_model", None) or getattr(inner_est, "_multi_model", None)
        return inner_est, tm
    if hasattr(model, "_model"):
        return model, getattr(model, "_model")
    if hasattr(model, "tables") or hasattr(model, "to_json"):
        return model, model
    return None, None


def _score_tboost_numeric_tables_independently(bank, X):
    """Score the public rating export, including every factored box, in float64."""
    raw = np.full(len(X), float(bank['f0']), dtype=float)
    for table in bank['tables']:
        coords = []
        for axis in table['axes']:
            name = axis.get('name') or X.columns[axis['raw']]
            v = np.asarray(X[name], dtype=np.float32)
            borders = np.asarray(axis['borders'], dtype=np.float32)
            coords.append(np.where(np.isnan(v), 0, 1 + np.searchsorted(borders, v, side='left')))
        raw += np.asarray(table['values'])[np.ravel_multi_index(coords, table['shape'])]
    for effect in bank.get('factored', []):
        for box in effect['boxes']:
            index = np.zeros(len(X), dtype=int)
            for d, name in enumerate(effect['feature_names']):
                if box.get('categorical_low_cells') and box['categorical_low_cells'][d] is not None:
                    raise ValueError('categorical box requires persisted encoder')
                v = np.asarray(X[name], dtype=np.float32)
                low = np.where(np.isnan(v), box['missing_left'][d], v <= np.float32(box['thresholds'][d]))
                index |= low.astype(int) << d
            raw += np.asarray(box['octants'])[index]
    return raw, len(bank['tables']), len(bank.get('factored', []))


def _inspect_tboost_evidence(model, test):
    """Census from the public rating export; unsupported agreement is explicit."""
    is_tb = getattr(model, 'name', '') == 't-boost' or 'TBoost' in type(model).__name__
    report = {'is_triboost': is_tb, 'table_counts': None, 'box_counts': None,
              'exported_bytes': None, 'census': getattr(model, 'table_census_', None),
              'raw_export_agreement': {'tested': False, 'agreement': None, 'max_abs_diff': None,
                 'dense_tables_scored': 0, 'factored_tables_scored': 0, 'unsupported_reason': None}}
    agreement = report['raw_export_agreement']
    if not is_tb:
        agreement['unsupported_reason'] = 'Not a t-boost estimator'
        return report
    estimator, _ = _resolve_tboost_components(model)
    try:
        prep = getattr(model, '_prep_X', lambda X: X)
        design = prep(test.X)
        # Explicit rating export; never interpret the different internal wire schema as this schema.
        payload = estimator.tables(design)
        bank = json.loads(payload)
        report['exported_bytes'] = len(payload.encode())
        def banks(value):
            if isinstance(value, dict):
                if 'tables' in value and 'f0' in value:
                    yield value
                else:
                    for child in value.values():
                        yield from banks(child)
            elif isinstance(value, list):
                for child in value:
                    yield from banks(child)
        all_banks = list(banks(bank))
        if not all_banks:
            raise ValueError('no rating banks found in export')
        dense_terms = [t for b in all_banks for t in b['tables']]
        factored_terms = [t for b in all_banks for t in b.get('factored', [])]
        terms = dense_terms + factored_terms
        by_order = {}
        for term in terms:
            order = str(len(term['feature_set']))
            by_order[order] = by_order.get(order, 0) + 1
        report['table_counts'] = {'dense_tables': len(dense_terms),
            'factored_tables': len(factored_terms), 'total_tables': len(terms),
            'census_by_arity': by_order}
        report['box_counts'] = {
            'dense_cells': sum(len(t['values']) for t in dense_terms),
            'factored_boxes': sum(len(t['boxes']) for t in factored_terms)}
        if 'tables' not in bank or test.task.task == TaskType.MULTICLF:
            agreement['unsupported_reason'] = 'Multiclass export agreement not implemented'
            return report
        if test.cat_features or getattr(model, '_cat', []):
            agreement['unsupported_reason'] = 'Categorical features require persisted encoder; independent scoring not implemented'
            return report
        native = np.asarray(estimator.predict_raw(design), dtype=float)
        independent, dense, factored = _score_tboost_numeric_tables_independently(bank, test.X)
        error = float(np.max(np.abs(native - independent)))
        # Compare an f64 sum with the deployed f32 score using a stated scale-aware tolerance.
        tolerance = 64 * np.finfo(np.float32).eps * max(1., float(np.max(np.abs(native))))
        agreement.update(tested=True, agreement=bool(error <= tolerance), max_abs_diff=error,
                         tolerance=float(tolerance), dense_tables_scored=dense, factored_tables_scored=factored)
    except Exception as exc:
        agreement['unsupported_reason'] = f'{type(exc).__name__}: {exc}'
    return report


def _evaluate_feature_segments(
    test: DatasetBundle,
    pred: np.ndarray,
    segment_columns: Sequence[str],
) -> dict[str, Any]:
    """Compute segment-level actual/expected aggregates only on explicitly allowlisted columns."""
    if not segment_columns:
        return {}

    missing = [col for col in segment_columns if col not in test.X.columns]
    if missing:
        raise KeyError(f"Explicitly requested segment columns missing from test set: {missing}")

    task_type = test.task.task
    is_multiclass = task_type == TaskType.MULTICLF

    # Resolve exposure/weighting
    exp = np.asarray(test.exposure, dtype=float) if test.exposure is not None else np.ones(len(test.y))
    weight = np.asarray(test.sample_weight, dtype=float)

    segments_out: dict[str, list[dict[str, Any]]] = {}

    for col in segment_columns:
        series = test.X[col].reset_index(drop=True)
        # For discrete or low-cardinality, group directly; for continuous, decile bins
        unique_vals = series.unique()
        if len(unique_vals) <= 30 or series.dtype == "object" or isinstance(series.dtype, pd.CategoricalDtype):
            groups = series.astype(str)
        else:
            # Numeric continuous: quantile bins
            try:
                groups = pd.qcut(series, q=min(10, len(unique_vals)), duplicates="drop").astype(str)
            except Exception:
                groups = pd.cut(series, bins=min(10, len(unique_vals))).astype(str)

        col_records: list[dict[str, Any]] = []
        for level, mask_series in groups.groupby(groups).groups.items():
            idx = mask_series.to_numpy()
            n_sub = len(idx)
            sub_y = test.y[idx]
            sub_pred = pred[idx]
            sub_exp = exp[idx]
            sub_w = weight[idx]

            total_exp = float(np.sum(sub_exp))
            total_w = float(np.sum(sub_w))

            if is_multiclass:
                col_records.append({
                    "level": str(level),
                    "n_rows": n_sub,
                    "exposure": total_exp,
                    "weight": total_w,
                    "actual": None,
                    "expected": None,
                    "ae_ratio": None,
                })
            elif task_type in (TaskType.FREQ, TaskType.PP):
                actual = float(np.sum(sub_y))
                expected = float(np.sum(sub_pred * sub_exp))
                ae = (actual / expected) if (expected > 0 and math.isfinite(expected)) else None
                col_records.append({
                    "level": str(level),
                    "n_rows": n_sub,
                    "exposure": total_exp,
                    "actual": actual,
                    "expected": expected,
                    "ae_ratio": ae,
                    "actual_rate": (actual / total_exp) if total_exp > 0 else None,
                    "expected_rate": (expected / total_exp) if total_exp > 0 else None,
                })
            else:  # SEV or CLF
                actual = float(np.sum(sub_y * sub_w))
                expected = float(np.sum(sub_pred * sub_w))
                ae = (actual / expected) if (expected > 0 and math.isfinite(expected)) else None
                col_records.append({
                    "level": str(level),
                    "n_rows": n_sub,
                    "weight": total_w,
                    "actual": actual,
                    "expected": expected,
                    "ae_ratio": ae,
                    "actual_mean": (actual / total_w) if total_w > 0 else None,
                    "expected_mean": (expected / total_w) if total_w > 0 else None,
                })

        segments_out[col] = col_records

    return segments_out


def collect_evidence(
    model: Any,
    train: DatasetBundle,
    test: DatasetBundle,
    pred: Any,
    train_rows: Sequence[int] | np.ndarray,
    test_rows: Sequence[int] | np.ndarray,
    segment_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Collect serializable JSON evidence (non-finite values mapped to null).
    
    Includes target units, portfolio actual/expected A/E for scalar tasks, multiclass
    without bogus scalar A/E, deterministic row hashes, model artifact size via safe
    serialization, TBoost table/box inspection and raw export agreement, and
    allowlisted feature segment aggregates. Never records raw features by default.
    """
    pred_arr = np.asarray(pred)
    task_type = test.task.task
    is_multiclass = task_type == TaskType.MULTICLF

    # 1. Target units and portfolio actual/expected A/E
    actual: float | None = None
    expected: float | None = None
    ae_ratio: float | None = None
    actual_mean: float | None = None
    expected_mean: float | None = None
    total_exposure: float | None = None
    multiclass_details: dict[str, Any] | None = None

    if is_multiclass:
        target_units = "class_probabilities"
        # "multiclass no bogus A/E": leave actual, expected, ae_ratio as None
        k = pred_arr.shape[1] if pred_arr.ndim == 2 else int(test.meta.get("n_classes", len(np.unique(test.y))))
        y_int = np.asarray(test.y, dtype=int)
        class_counts = {str(c): int(np.sum(y_int == c)) for c in range(k)}
        weights = np.asarray(test.sample_weight, dtype=float)
        actual_shares = {str(c): float(np.average(y_int == c, weights=weights)) for c in range(k)}
        expected_shares = (
            {str(c): float(np.average(pred_arr[:, c], weights=weights)) for c in range(k)}
            if pred_arr.ndim == 2 and pred_arr.shape[1] == k else None
        )
        total_weight = float(np.sum(test.sample_weight)) if test.sample_weight is not None else float(len(test.y))
        multiclass_details = {
            "n_classes": k,
            "actual_class_counts": class_counts,
            "actual_class_shares": actual_shares,
            "expected_class_shares": expected_shares,
            "multiclass_note": "Multiclass task produces probability matrices; scalar portfolio A/E is not defined.",
        }
    elif task_type in (TaskType.FREQ, TaskType.PP):
        target_units = "claim_frequency" if task_type == TaskType.FREQ else "pure_premium"
        exp = np.asarray(test.exposure, dtype=float) if test.exposure is not None else np.ones(len(test.y))
        actual = float(np.sum(test.y))
        expected = float(np.sum(pred_arr * exp))
        total_exposure = float(np.sum(exp))
        total_weight = total_exposure
        ae_ratio = (actual / expected) if (expected > 0 and math.isfinite(expected)) else None
        actual_mean = (actual / total_exposure) if total_exposure > 0 else None
        expected_mean = (expected / total_exposure) if total_exposure > 0 else None
    elif task_type == TaskType.SEV:
        target_units = "claim_severity"
        w = np.asarray(test.sample_weight, dtype=float)
        actual = float(np.sum(test.y * w))
        expected = float(np.sum(pred_arr * w))
        total_weight = float(np.sum(w))
        ae_ratio = (actual / expected) if (expected > 0 and math.isfinite(expected)) else None
        actual_mean = (actual / total_weight) if total_weight > 0 else None
        expected_mean = (expected / total_weight) if total_weight > 0 else None
    elif task_type == TaskType.CLF:
        target_units = "probability"
        w = np.asarray(test.sample_weight, dtype=float) if test.sample_weight is not None else np.ones(len(test.y))
        actual = float(np.sum(test.y * w))
        expected = float(np.sum(pred_arr * w))
        total_weight = float(np.sum(w))
        ae_ratio = (actual / expected) if (expected > 0 and math.isfinite(expected)) else None
        actual_mean = (actual / total_weight) if total_weight > 0 else None
        expected_mean = (expected / total_weight) if total_weight > 0 else None
    else:
        target_units = str(task_type)
        total_weight = float(np.sum(test.sample_weight))

    portfolio_record = {
        "task": str(task_type.value if hasattr(task_type, "value") else task_type),
        "target_units": target_units,
        "distribution": getattr(test.task, "distribution", None),
        "actual": actual,
        "expected": expected,
        "ae_ratio": ae_ratio,
        "actual_mean": actual_mean,
        "expected_mean": expected_mean,
        "total_weight": total_weight,
        "total_exposure": total_exposure,
        "n_rows": int(len(test.y)),
        "multiclass_details": multiclass_details,
    }

    # 2. Deterministic row hashes
    row_hashes = _compute_row_hashes(train_rows, test_rows)

    # 3. Model artifact size via safe serialization
    safe_size = _inspect_safe_model_artifact_size(model)

    # 4. TBoost-specific evidence (table/box census, raw export agreement)
    tboost_evidence = _inspect_tboost_evidence(model, test)

    # 5. Allowlisted feature segments (reject missing, record no raw features)
    feature_segments = _evaluate_feature_segments(test, pred_arr, segment_columns)

    raw_evidence = {
        "task": str(task_type.value if hasattr(task_type, "value") else task_type),
        "target_units": target_units,
        "portfolio": portfolio_record,
        "row_hashes": row_hashes,
        "model_artifact": safe_size,
        # key FROZEN as "triboost" (and "is_triboost" inside): the retained evidence files of the
        # cached board use it, and one schema is worth more than the new spelling here.
        "triboost": tboost_evidence,
        "feature_segments": feature_segments,
    }

    return _clean_for_json(raw_evidence)


def prediction_payload(
    test: DatasetBundle,
    pred: Any,
    test_rows: Sequence[int] | np.ndarray,
    segment_columns: Sequence[str] = (),
) -> dict[str, np.ndarray]:
    """Construct a payload dict of numpy arrays for compressed npz storage without object pickling.
    
    Includes canonical y_true, original y, predictions, weights, exposure, row positions,
    group identifiers (safe unicode strings), and approved segment columns (safe unicode strings).
    """
    missing = [c for c in segment_columns if c not in test.X.columns]
    if missing:
        raise KeyError(f"Explicitly requested segment columns missing from test features: {missing}")

    y_canon, w, exp = task_targets(test)

    # Base arrays
    payload: dict[str, np.ndarray] = {
        "y_true": np.asarray(y_canon, dtype=np.float64),
        "original_y": np.asarray(test.y, dtype=np.float64),
        "predictions": np.asarray(pred, dtype=np.float64),
        "weights": np.asarray(w, dtype=np.float64),
        "exposure": np.asarray(
            exp if exp is not None else np.full(len(test.y), np.nan),
            dtype=np.float64,
        ),
        "row_positions": np.asarray(test_rows, dtype=np.int64),
    }

    # Group identifiers safe strings
    if test.group_ids is not None:
        groups = [str(g) for g in test.group_ids]
    else:
        groups = [f"row_{r}" for r in test_rows]
    payload["group_identifiers"] = np.asarray(groups, dtype=np.str_)

    # Approved segment columns as safe strings
    for col in segment_columns:
        payload[f"segment_{col}"] = np.asarray([str(v) for v in test.X[col]], dtype=np.str_)

    return payload


def measure_prediction(
    model: Any,
    test: DatasetBundle,
    repeats: int = 7,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Benchmark prediction latency for single row and batch.
    
    Prepares input rows BEFORE timing, executes warmups, and times strictly model.predict.
    Computes median and p95 latency in seconds.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if len(test.X) < 1:
        raise ValueError("test dataset cannot be empty")

    n_test = len(test.X)
    actual_batch_size = min(int(batch_size), n_test)

    # 1. Prepare single row and batch BEFORE timing
    single_X = test.X.iloc[:1].copy()
    single_exp = test.exposure[:1] if test.exposure is not None else None

    batch_X = test.X.iloc[:actual_batch_size].copy()
    batch_exp = test.exposure[:actual_batch_size] if test.exposure is not None else None

    # Helper to call predict respecting signature
    def _predict(X: pd.DataFrame, exposure: np.ndarray | None) -> Any:
        return model.predict(X, exposure)

    # 2. Warmups (unmeasured)
    _predict(single_X, single_exp)
    _predict(batch_X, batch_exp)

    # 3. Time single row predictions
    single_times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _predict(single_X, single_exp)
        t1 = time.perf_counter()
        single_times.append(t1 - t0)

    # 4. Time batch predictions
    batch_times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _predict(batch_X, batch_exp)
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)

    single_median = float(np.median(single_times))
    single_p95 = float(np.percentile(single_times, 95))
    batch_median = float(np.median(batch_times))
    batch_p95 = float(np.percentile(batch_times, 95))
    throughput = (actual_batch_size / batch_median) if batch_median > 0 else None

    timing_result = {
        "repeats": int(repeats),
        "batch_rows": int(actual_batch_size),
        "single_median_s": single_median,
        "single_p95_s": single_p95,
        "batch_median_s": batch_median,
        "batch_p95_s": batch_p95,
        "batch_rows_per_s": throughput,
        "single_times_s": single_times,
        "batch_times_s": batch_times,
        "isolation_note": "no process isolation claim; caller orchestrates environment",
    }
    return _clean_for_json(timing_result)


def write_evidence(
    directory: str | Path,
    key: str,
    metadata: dict[str, Any],
    predictions: dict[str, np.ndarray] | None = None,
) -> EvidencePaths:
    """Atomically write evidence metadata to JSON and optional predictions to compressed npz.
    
    Uses signature-based file naming to prevent collisions and overwrite hazards.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    safe_key = str(key).strip().replace("/", "_").replace("\\", "_")
    json_path = dir_path / f"{safe_key}.json"
    npz_path = (dir_path / f"{safe_key}.npz") if predictions is not None else None

    # Merge caller metadata and references
    clean_meta = _clean_for_json(metadata)
    if npz_path is not None:
        clean_meta["predictions_file"] = npz_path.name

    # Write compressed npz atomically first if present
    if npz_path is not None and predictions is not None:
        tmp_npz = npz_path.with_name(f"{npz_path.name}.tmp.{uuid.uuid4().hex}")
        with open(tmp_npz, "wb") as f:
            np.savez_compressed(f, **predictions)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_npz, npz_path)

    # Write JSON atomically
    tmp_json = json_path.with_name(f"{json_path.name}.tmp.{uuid.uuid4().hex}")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(clean_meta, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_json, json_path)

    return EvidencePaths(json=json_path, npz=npz_path)
