"""Tests for evidence collection, timing measurement, and atomic serialization.

Covers:
  - collect_evidence on scalar tasks (FREQ, PP, SEV, CLF) with portfolio actual,
    expected, and A/E ratios.
  - collect_evidence on MULTICLF task enforcing "multiclass no bogus A/E" and class distributions.
  - Non-finite (NaN / inf / -inf) sanitization to JSON null.
  - Deterministic row hashes.
  - Model artifact size inspection via safe serialization (no unconstrained pickle).
  - TBoost table and box census, exported bytes, and independent raw export agreement.
  - Feature segment allowlist enforcement and missing column rejection.
  - Record no raw features by default.
  - prediction_payload dictionary with canonical targets, weights, exposure, group IDs,
    safe strings, and no object pickling.
  - measure_prediction timing single row and batch latency, warmups, and argument validation.
  - write_evidence atomic JSON and optional compressed npz with signature keying and metadata.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insurarena.datasets.base import DatasetBundle
from insurarena.datasets.registry import build_loader
from insurarena.eval.evidence import (
    EvidencePaths,
    collect_evidence,
    measure_prediction,
    prediction_payload,
    write_evidence,
)
from insurarena.models.registry import build_model
from insurarena.tasks import CLF_SPEC, FREQ_SPEC, MULTICLF_SPEC, PP_SPEC, SEV_SPEC, TaskType


# --- 1. Portfolio A/E, Units, and Multiclass contract -------------------------------------


def test_collect_evidence_freq_portfolio_ae():
    """FREQ task produces rate target_units and valid actual, expected, and A/E."""
    bundle = build_loader("synthetic_freq", subsample=500).load()
    tr_idx = np.arange(0, 350)
    te_idx = np.arange(350, 500)
    tr, te = bundle.subset(tr_idx), bundle.subset(te_idx)

    model = build_model("glm").fit(tr)
    pred = model.predict(te.X, te.exposure)

    ev = collect_evidence(model, tr, te, pred, tr_idx, te_idx)

    assert ev["target_units"] == "claim_frequency"
    portfolio = ev["portfolio"]
    assert portfolio["task"] == "freq"
    assert portfolio["target_units"] == "claim_frequency"
    assert portfolio["actual"] is not None and portfolio["actual"] >= 0
    assert portfolio["expected"] is not None and portfolio["expected"] > 0
    assert portfolio["ae_ratio"] is not None and portfolio["ae_ratio"] > 0
    # A/E ratio is actual / expected
    assert abs(portfolio["ae_ratio"] - (portfolio["actual"] / portfolio["expected"])) < 1e-6
    assert portfolio["total_exposure"] == float(np.sum(te.exposure))

    # Clean JSON roundtrip
    dumped = json.dumps(ev)
    loaded = json.loads(dumped)
    assert loaded["portfolio"]["ae_ratio"] == portfolio["ae_ratio"]


@pytest.mark.parametrize(
    "dataset_name,expected_task,expected_units",
    [
        ("synthetic_pp", "pp", "pure_premium"),
        ("synthetic_sev", "sev", "claim_severity"),
        ("synthetic_clf", "clf", "probability"),
    ],
)
def test_collect_evidence_scalar_tasks(dataset_name, expected_task, expected_units):
    """PP, SEV, and CLF produce expected target units and scalar A/E ratios."""
    bundle = build_loader(dataset_name, subsample=500).load()
    tr_idx = np.arange(0, 350)
    te_idx = np.arange(350, 500)
    tr, te = bundle.subset(tr_idx), bundle.subset(te_idx)

    model = build_model("glm").fit(tr)
    pred = model.predict(te.X, te.exposure)

    ev = collect_evidence(model, tr, te, pred, tr_idx, te_idx)
    assert ev["target_units"] == expected_units
    portfolio = ev["portfolio"]
    assert portfolio["task"] == expected_task
    assert portfolio["actual"] is not None
    assert portfolio["expected"] is not None
    assert portfolio["ae_ratio"] is not None
    assert json.dumps(ev)  # serializable


def test_collect_evidence_multiclass_no_bogus_ae():
    """MULTICLF produces probability units, class distributions, and strictly no scalar A/E."""
    bundle = build_loader("synthetic_multiclf", subsample=500).load()
    tr_idx = np.arange(0, 350)
    te_idx = np.arange(350, 500)
    tr, te = bundle.subset(tr_idx), bundle.subset(te_idx)

    model = build_model("glm").fit(tr)
    pred = model.predict(te.X)  # shape (150, K)
    assert pred.ndim == 2

    ev = collect_evidence(model, tr, te, pred, tr_idx, te_idx)

    assert ev["target_units"] == "class_probabilities"
    portfolio = ev["portfolio"]
    assert portfolio["task"] == "multiclf"
    # STRICT REQUIREMENT: multiclass no bogus A/E
    assert portfolio["actual"] is None
    assert portfolio["expected"] is None
    assert portfolio["ae_ratio"] is None

    details = portfolio["multiclass_details"]
    assert details is not None
    assert details["n_classes"] == pred.shape[1]
    assert "actual_class_counts" in details
    assert "actual_class_shares" in details
    assert "expected_class_shares" in details

    dumped = json.dumps(ev)
    assert '"ae_ratio": null' in dumped


# --- 2. Non-finite (NaN / Inf) Sanitization to JSON null -----------------------------------


def test_collect_evidence_nonfinite_mapped_to_null():
    """Non-finite floats (NaN, +inf, -inf) anywhere in evidence are converted to null."""
    bundle = build_loader("synthetic_freq", subsample=100).load()
    tr_idx = np.arange(0, 70)
    te_idx = np.arange(70, 100)
    tr, te = bundle.subset(tr_idx), bundle.subset(te_idx)

    class MockModel:
        def predict(self, X, exposure=None):
            # inject NaNs and Infs into predictions
            p = np.full(len(X), 0.1)
            p[0] = np.nan
            p[1] = np.inf
            p[2] = -np.inf
            return p

    model = MockModel()
    pred = model.predict(te.X, te.exposure)

    ev = collect_evidence(model, tr, te, pred, tr_idx, te_idx)

    # Standard json.dumps will fail on NaN/Inf unless allow_nan=True (which emits invalid JSON),
    # but our nonfinite mapping must make it standard-compliant without non-standard tokens.
    raw_json = json.dumps(ev, allow_nan=False)
    assert "NaN" not in raw_json
    assert "Infinity" not in raw_json
    loaded = json.loads(raw_json)
    assert loaded is not None


# --- 3. Deterministic Row Hashes -----------------------------------------------------------


def test_row_hashes_deterministic_and_sensitive():
    """Row hashes are sha256 digests that depend deterministically on indices."""
    bundle = build_loader("synthetic_freq", subsample=100).load()
    tr_idx1 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    te_idx1 = np.array([5, 6, 7], dtype=np.int64)

    tr, te = bundle.subset(tr_idx1), bundle.subset(te_idx1)
    model = build_model("glm").fit(tr)
    pred = model.predict(te.X, te.exposure)

    ev1 = collect_evidence(model, tr, te, pred, tr_idx1, te_idx1)
    ev2 = collect_evidence(model, tr, te, pred, tr_idx1, te_idx1)

    assert ev1["row_hashes"]["train_rows_hash"] == ev2["row_hashes"]["train_rows_hash"]
    assert ev1["row_hashes"]["test_rows_hash"] == ev2["row_hashes"]["test_rows_hash"]
    assert len(ev1["row_hashes"]["train_rows_hash"]) == 64
    assert ev1["row_hashes"]["train_rows_count"] == 5
    assert ev1["row_hashes"]["test_rows_count"] == 3

    # Changing rows changes the hash
    tr_idx_diff = np.array([0, 1, 2, 3, 9], dtype=np.int64)
    ev3 = collect_evidence(model, tr, te, pred, tr_idx_diff, te_idx1)
    assert ev3["row_hashes"]["train_rows_hash"] != ev1["row_hashes"]["train_rows_hash"]


# --- 4. Safe Model Artifact Size (No Object Pickling) --------------------------------------


def test_safe_model_artifact_size_no_pickling():
    """Artifact size is determined via safe serializers, never unconstrained pickle."""
    bundle = build_loader("synthetic_freq", subsample=100).load()
    model = build_model("glm").fit(bundle)
    pred = model.predict(bundle.X, bundle.exposure)

    ev = collect_evidence(model, bundle, bundle, pred, np.arange(100), np.arange(100))
    art = ev["model_artifact"]
    assert "artifact_size_bytes" in art
    assert "serialization_method" in art

    # Mock a model with native to_bytes
    class MockNativeModel:
        def to_bytes(self):
            return b"serialized_binary_bytes_12345"

        def predict(self, X, exposure=None):
            return np.ones(len(X))

    native_m = MockNativeModel()
    ev_native = collect_evidence(native_m, bundle, bundle, pred, np.arange(100), np.arange(100))
    assert ev_native["model_artifact"]["artifact_size_bytes"] == len(b"serialized_binary_bytes_12345")
    assert ev_native["model_artifact"]["serialization_method"] == "native_to_bytes"
    assert ev_native["model_artifact"]["is_exact_bytes"] is True


# --- 5. Feature Segments Allowlist and Missing Rejection ------------------------------------


def test_feature_segments_allowlist_and_missing_rejection():
    """Only explicitly allowlisted columns are segmented; missing columns are rejected."""
    bundle = build_loader("synthetic_freq", subsample=200).load()
    cols = list(bundle.X.columns)
    valid_col = cols[0]

    tr_idx, te_idx = np.arange(150), np.arange(150, 200)
    tr, te = bundle.subset(tr_idx), bundle.subset(te_idx)
    model = build_model("glm").fit(tr)
    pred = model.predict(te.X, te.exposure)

    # 1. Valid allowlist segment
    ev = collect_evidence(model, tr, te, pred, tr_idx, te_idx, segment_columns=(valid_col,))
    assert valid_col in ev["feature_segments"]
    seg_records = ev["feature_segments"][valid_col]
    assert len(seg_records) > 0
    assert "n_rows" in seg_records[0]
    assert "exposure" in seg_records[0]
    assert "ae_ratio" in seg_records[0]

    # Invariant: Record no raw features by default
    assert "X" not in ev
    assert "features" not in ev
    for key in ev:
        assert not isinstance(ev[key], pd.DataFrame)

    # 2. Reject missing column
    with pytest.raises(KeyError, match="missing from test set"):
        collect_evidence(model, tr, te, pred, tr_idx, te_idx, segment_columns=("non_existent_column_xyz",))


# --- 6. prediction_payload Contract and No Object Pickling ---------------------------------


def test_prediction_payload_no_object_pickling(tmp_path):
    """prediction_payload returns non-object numpy arrays that save cleanly with allow_pickle=False."""
    bundle = build_loader("synthetic_freq", subsample=200).load()
    cols = list(bundle.X.columns)
    valid_col = cols[0]

    tr_idx, te_idx = np.arange(150), np.arange(150, 200)
    tr, te = bundle.subset(tr_idx), bundle.subset(te_idx)
    model = build_model("glm").fit(tr)
    pred = model.predict(te.X, te.exposure)

    payload = prediction_payload(te, pred, te_idx, segment_columns=(valid_col,))

    required_keys = {
        "y_true",
        "original_y",
        "predictions",
        "weights",
        "exposure",
        "row_positions",
        "group_identifiers",
        f"segment_{valid_col}",
    }
    assert required_keys.issubset(set(payload.keys()))

    # Verify no object arrays (all concrete numeric or unicode string)
    for k, arr in payload.items():
        assert isinstance(arr, np.ndarray)
        assert arr.dtype != object, f"Array {k} has object dtype, which requires pickling!"

    # Invariant: np.savez_compressed with allow_pickle=False must succeed
    npz_file = tmp_path / "preds.npz"
    with open(npz_file, "wb") as f:
        np.savez_compressed(f, allow_pickle=False, **payload)

    # Verify loading back
    loaded = np.load(npz_file, allow_pickle=False)
    assert np.allclose(loaded["predictions"], pred)
    assert np.array_equal(loaded["row_positions"], te_idx)
    assert len(loaded[f"segment_{valid_col}"]) == len(te_idx)

    # Reject missing column in prediction_payload
    with pytest.raises(KeyError, match="missing from test features"):
        prediction_payload(te, pred, te_idx, segment_columns=("bogus_feature",))


# --- 7. measure_prediction Benchmarking ----------------------------------------------------


def test_measure_prediction_validation_and_timing():
    """measure_prediction executes warmups, isolates predict call, and returns median/p95 latency."""
    bundle = build_loader("synthetic_freq", subsample=150).load()
    model = build_model("glm").fit(bundle)

    # Argument validation
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        measure_prediction(model, bundle, repeats=0)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        measure_prediction(model, bundle, batch_size=0)

    # Successful measurement
    timings = measure_prediction(model, bundle, repeats=5, batch_size=64)

    assert timings["repeats"] == 5
    assert timings["batch_rows"] == 64
    assert timings["single_median_s"] > 0
    assert timings["single_p95_s"] >= timings["single_median_s"]
    assert timings["batch_median_s"] > 0
    assert timings["batch_p95_s"] >= timings["batch_median_s"]
    assert timings["batch_rows_per_s"] is not None and timings["batch_rows_per_s"] > 0
    assert len(timings["single_times_s"]) == 5
    assert len(timings["batch_times_s"]) == 5
    assert "no process isolation" in timings["isolation_note"]

    # Verify serializable
    assert json.dumps(timings)


# --- 8. write_evidence Atomic Persistence --------------------------------------------------


def test_write_evidence_atomic_and_npz_support(tmp_path):
    """write_evidence atomically writes JSON metadata and optional compressed npz."""
    metadata = {
        "dataset": "synthetic_freq",
        "split_id": 1,
        "variant": "default",
        "score": 0.1234,
        "nonfinite_check": float("nan"),
    }
    payload = {
        "pred": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "rows": np.array([10, 20, 30], dtype=np.int64),
    }

    key = "syn_freq__glm__default__1"
    paths = write_evidence(tmp_path, key, metadata, predictions=payload)

    # Accessible via attributes, dict keys, and tuple unpacking
    assert isinstance(paths, EvidencePaths)
    assert paths.json.exists()
    assert paths.npz is not None and paths.npz.exists()
    assert paths["json"] == paths.json
    assert paths["npz"] == paths.npz
    assert paths.get("json") == paths.json
    json_p, npz_p = paths
    assert json_p == paths.json
    assert npz_p == paths.npz

    # Verify JSON content
    loaded_json = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded_json["dataset"] == "synthetic_freq"
    assert loaded_json["split_id"] == 1
    assert loaded_json["nonfinite_check"] is None  # mapped from nan to null
    assert loaded_json["predictions_file"] == paths.npz.name

    # Verify NPZ content
    loaded_npz = np.load(paths.npz, allow_pickle=False)
    assert np.allclose(loaded_npz["pred"], [1.0, 2.0, 3.0])

    # Optional predictions=None
    key2 = "syn_freq__glm__default__2"
    paths_no_npz = write_evidence(tmp_path, key2, metadata, predictions=None)
    assert paths_no_npz.json.exists()
    assert paths_no_npz.npz is None


# --- 9. TBoost Table Census and Raw Export Agreement -------------------------------------


def test_triboost_evidence_structure_and_agreement():
    """TBoost model collects census, exported bytes, and checks raw export agreement."""
    pytest.importorskip("t_boost")
    try:
        from insurarena.models.tboost_model import TBoost
    except ImportError:
        pytest.skip("TBoost adapter not available")

    # Fit small numeric TBoost
    n = 200
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    w = np.ones(n)
    bundle = DatasetBundle(
        name="test_tri_num",
        X=X,
        y=rng.poisson(0.2, size=n).astype(float),
        exposure=w,
        sample_weight=w,
        task=FREQ_SPEC,
    )

    tr_idx, te_idx = np.arange(140), np.arange(140, 200)
    tr, te = bundle.subset(tr_idx), bundle.subset(te_idx)

    model = TBoost(hp={"budget": 20, "n_bags": 1}).fit(tr)
    pred = model.predict(te.X, te.exposure)

    ev = collect_evidence(model, tr, te, pred, tr_idx, te_idx)

    tri = ev["triboost"]
    assert tri["is_triboost"] is True
    assert tri["table_counts"] is not None
    assert tri["box_counts"] is not None
    assert tri["exported_bytes"] is not None and tri["exported_bytes"] > 0

    agreement = tri["raw_export_agreement"]
    # For purely numeric features, raw export agreement should be tested and agree
    assert agreement["tested"] is True
    assert agreement["agreement"] is True
    assert agreement["max_abs_diff"] is not None and agreement["max_abs_diff"] < 1e-4
    assert agreement["unsupported_reason"] is None


def test_triboost_categorical_unsupported_reason():
    """When categoricals are present in TBoost, agreement is not claimed and unsupported_reason is set."""
    pytest.importorskip("t_boost")
    try:
        from insurarena.models.tboost_model import TBoost
    except ImportError:
        pytest.skip("TBoost adapter not available")

    n = 200
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "cat1": [f"c{i % 5}" for i in range(n)],
    })
    w = np.ones(n)
    bundle = DatasetBundle(
        name="test_tri_cat",
        X=X,
        y=rng.poisson(0.2, size=n).astype(float),
        exposure=w,
        sample_weight=w,
        task=FREQ_SPEC,
        cat_features=["cat1"],
    )

    tr_idx, te_idx = np.arange(140), np.arange(140, 200)
    tr, te = bundle.subset(tr_idx), bundle.subset(te_idx)

    model = TBoost(hp={"budget": 20, "n_bags": 1}).fit(tr)
    pred = model.predict(te.X, te.exposure)

    ev = collect_evidence(model, tr, te, pred, tr_idx, te_idx)
    agreement = ev["triboost"]["raw_export_agreement"]

    # Invariant: Do not claim tested if not
    assert agreement["tested"] is False
    assert agreement["agreement"] is None
    assert agreement["unsupported_reason"] is not None
    assert "Categorical features" in agreement["unsupported_reason"]


def test_numeric_reconstruction_includes_real_three_way_boxes_and_boundaries():
    from t_boost import TBoostRegressor
    from insurarena.eval.evidence import _score_tboost_numeric_tables_independently
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 4)).astype(np.float32)
    y = (1.2 * (x[:, 0] > 0) - .8 * (x[:, 1] > 0)
         + 1.5 * ((x[:, 0] > 0) & (x[:, 1] > 0) & (x[:, 2] > 0))).astype(np.float32)
    X = pd.DataFrame(x, columns=['a', 'b', 'c', 'd'])
    model = TBoostRegressor(n_trees=40, n_bags=1, validation_fraction=None,
                           prune=False, interaction_gain_hurdle=0,
                           colsample_bytree=1, leaf_refine_steps=0).fit(X, y)
    bank = json.loads(model.tables(X))
    assert bank['factored']
    rows = [r.astype(float) for r in x[:20]]
    for table in bank['tables']:
        for axis in table['axes']:
            for border in axis['borders']:
                b = float(np.float32(border))
                for value in (b, np.nextafter(b, -np.inf), np.nextafter(b, np.inf)):
                    row = np.zeros(4)
                    row[axis['raw']] = value
                    rows.append(row)
    rows.extend([np.full(4, np.nan), np.full(4, -1e20), np.full(4, 1e20)])
    audit = pd.DataFrame(rows, columns=X.columns)
    independent, dense, factored = _score_tboost_numeric_tables_independently(bank, audit)
    assert dense == len(bank['tables']) and factored == len(bank['factored'])
    np.testing.assert_allclose(independent, model.predict_raw(audit), rtol=0,
                               atol=4 * 40 * np.finfo(np.float32).eps)


def test_segment_aggregation_uses_positions_not_original_index():
    from insurarena.eval.evidence import _evaluate_feature_segments
    X = pd.DataFrame({'region': ['a', 'b', 'a']}, index=[50, 70, 90])
    b = DatasetBundle('segment', X, np.array([1., 2., 3.]), np.ones(3), np.ones(3), FREQ_SPEC,
                      cat_features=['region'])
    rows = _evaluate_feature_segments(b, np.ones(3), ['region'])['region']
    assert sum(r['actual'] for r in rows) == 6
