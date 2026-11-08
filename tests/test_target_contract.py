"""Tests for target conventions, validation contracts, and TBoost frequency formulations."""
from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_poisson_deviance

from insurarena.datasets.base import DatasetBundle
from insurarena.scoring import (
    primary_value,
    score_bundle,
    target_contract,
    task_targets,
    validate_bundle,
    validate_predictions,
)
from insurarena.tasks import (
    CLF_SPEC,
    FREQ_SPEC,
    MULTICLF_SPEC,
    PP_SPEC,
    SEV_SPEC,
    TaskType,
)


# ---------------------------------------------------------------------------
# Synthetic DatasetBundle fixtures
# ---------------------------------------------------------------------------

def _make_freq_bundle(
    n: int = 50,
    *,
    counts: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
) -> DatasetBundle:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.integers(0, 5, size=n)})
    exp = exposure if exposure is not None else rng.uniform(0.1, 2.0, size=n)
    y = counts if counts is not None else rng.poisson(0.3 * exp).astype(float)
    w = weights if weights is not None else exp.copy()
    return DatasetBundle(
        name="synthetic_freq",
        X=X,
        y=np.asarray(y, dtype=float),
        exposure=np.asarray(exp, dtype=float) if exp is not None else None,
        sample_weight=np.asarray(w, dtype=float),
        task=FREQ_SPEC,
        group_ids=group_ids,
        cat_features=[],
        meta={"_subset": True},
    )


def _make_pp_bundle(n: int = 50) -> DatasetBundle:
    rng = np.random.default_rng(43)
    X = pd.DataFrame({"x1": rng.normal(size=n)})
    exp = rng.uniform(0.2, 1.5, size=n)
    # Compound Poisson-Gamma total loss
    claims = rng.poisson(0.2 * exp)
    loss = np.where(claims > 0, rng.gamma(2.0, 500.0, size=n), 0.0)
    return DatasetBundle(
        name="synthetic_pp",
        X=X,
        y=loss.astype(float),
        exposure=exp,
        sample_weight=exp.copy(),
        task=PP_SPEC,
        cat_features=[],
        meta={"_subset": True},
    )


def _make_sev_bundle(n: int = 50) -> DatasetBundle:
    rng = np.random.default_rng(44)
    X = pd.DataFrame({"x1": rng.normal(size=n)})
    claim_count = rng.integers(1, 4, size=n).astype(float)
    severity = rng.gamma(3.0, 200.0, size=n)  # strictly positive
    return DatasetBundle(
        name="synthetic_sev",
        X=X,
        y=severity,
        exposure=None,
        sample_weight=claim_count,
        task=SEV_SPEC,
        cat_features=[],
        meta={"_subset": True},
    )


def _make_clf_bundle(n: int = 50) -> DatasetBundle:
    rng = np.random.default_rng(45)
    X = pd.DataFrame({"x1": rng.normal(size=n)})
    labels = rng.integers(0, 2, size=n).astype(float)
    exp = rng.uniform(0.5, 1.0, size=n)
    return DatasetBundle(
        name="synthetic_clf",
        X=X,
        y=labels,
        exposure=exp,
        sample_weight=exp.copy(),
        task=CLF_SPEC,
        cat_features=[],
        meta={"_subset": True},
    )


def _make_multiclf_bundle(n: int = 60, k: int = 3) -> DatasetBundle:
    rng = np.random.default_rng(46)
    X = pd.DataFrame({"x1": rng.normal(size=n)})
    codes = rng.integers(0, k, size=n).astype(float)
    w = np.ones(n, dtype=float)
    return DatasetBundle(
        name="synthetic_multiclf",
        X=X,
        y=codes,
        exposure=None,
        sample_weight=w,
        task=MULTICLF_SPEC,
        cat_features=[],
        meta={"_subset": True, "n_classes": k},
    )


# ---------------------------------------------------------------------------
# Target convention contracts
# ---------------------------------------------------------------------------

def test_target_contract_conventions_all_tasks():
    # FREQ
    c_freq = target_contract(_make_freq_bundle())
    assert c_freq["task"] == "freq"
    assert c_freq["distribution"] == "poisson"
    assert c_freq["link"] == "log"
    assert c_freq["response"] == "rate"
    assert c_freq["prediction"] == "rate"
    assert c_freq["weight"] == "exposure"
    assert "prediction * exposure" in c_freq["total"]
    assert c_freq["tweedie_power"] is None
    assert c_freq["fixed_tweedie_power"] is None

    # PP
    c_pp = target_contract(_make_pp_bundle())
    assert c_pp["task"] == "pp"
    assert c_pp["distribution"] == "tweedie"
    assert c_pp["link"] == "log"
    assert c_pp["response"] == "rate"
    assert c_pp["prediction"] == "rate"
    assert c_pp["weight"] == "exposure"
    assert "prediction * exposure" in c_pp["total"]
    assert c_pp["tweedie_power"] == 1.9
    assert c_pp["fixed_tweedie_power"] == 1.9

    # SEV
    c_sev = target_contract(_make_sev_bundle())
    assert c_sev["task"] == "sev"
    assert c_sev["distribution"] == "gamma"
    assert c_sev["link"] == "log"
    assert c_sev["response"] == "mean_severity"
    assert c_sev["prediction"] == "mean_severity"
    assert c_sev["weight"] == "claim_count"
    assert "claim_count" in c_sev["total"]
    assert c_sev["proxy_objectives"].get("catboost") == "tweedie:variance_power=1.99"

    # CLF
    c_clf = target_contract(_make_clf_bundle())
    assert c_clf["task"] == "clf"
    assert c_clf["distribution"] == "bernoulli"
    assert c_clf["response"] == "binary_label"
    assert c_clf["prediction"] == "probability"
    assert c_clf["total"] == "prediction * weight"

    # MULTICLF
    c_multi = target_contract(_make_multiclf_bundle())
    assert c_multi["task"] == "multiclf"
    assert c_multi["distribution"] == "multinomial"
    assert c_multi["response"] == "class_code"
    assert c_multi["prediction"] == "probabilities"
    assert c_multi["total"] == "prediction * weight"


# ---------------------------------------------------------------------------
# Audit task_targets against sample_weight=exposure (no multiplying twice)
# ---------------------------------------------------------------------------

def test_task_targets_no_double_exposure_multiplication():
    bundle = _make_freq_bundle(n=20)
    y_rate, w, exp = task_targets(bundle)

    # In bundle, sample_weight was set to exposure
    np.testing.assert_array_equal(bundle.sample_weight, bundle.exposure)

    # Rate is y / exp
    np.testing.assert_allclose(y_rate, bundle.y / bundle.exposure)

    # w is exposure once, NOT exposure ** 2
    np.testing.assert_array_equal(w, bundle.exposure)
    np.testing.assert_array_equal(exp, bundle.exposure)


# ---------------------------------------------------------------------------
# Validation: exposure, sample weights, domains, no silent clipping
# ---------------------------------------------------------------------------

def test_validate_bundle_rejects_zero_or_negative_exposure():
    bundle = _make_freq_bundle(n=5)

    # Zero exposure -> must raise, no silent clipping to 1e-9
    bundle.exposure[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_bundle(bundle)

    # Negative exposure -> must raise
    bundle.exposure[0] = -0.5
    with pytest.raises(ValueError, match="strictly positive"):
        validate_bundle(bundle)

    # NaN exposure -> must raise
    bundle.exposure[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_bundle(bundle)


def test_validate_bundle_rejects_negative_or_zero_total_weights():
    bundle = _make_freq_bundle(n=5)

    bundle.sample_weight[0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        validate_bundle(bundle)

    bundle.sample_weight = np.zeros(5)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_bundle(bundle)


def test_validate_bundle_target_domains():
    # FREQ rejects negative counts
    bundle_freq = _make_freq_bundle(n=5)
    bundle_freq.y[0] = -1.0
    with pytest.raises(ValueError, match="FREQ target y must be nonnegative"):
        validate_bundle(bundle_freq)

    # PP rejects negative loss
    bundle_pp = _make_pp_bundle(n=5)
    bundle_pp.y[0] = -0.01
    with pytest.raises(ValueError, match="PP target y must be nonnegative"):
        validate_bundle(bundle_pp)

    # SEV rejects non-positive claim severity
    bundle_sev = _make_sev_bundle(n=5)
    bundle_sev.y[0] = 0.0
    with pytest.raises(ValueError, match="SEV target y must be strictly positive"):
        validate_bundle(bundle_sev)

    bundle_sev.y[0] = -10.0
    with pytest.raises(ValueError, match="SEV target y must be strictly positive"):
        validate_bundle(bundle_sev)

    # CLF rejects non-binary labels
    bundle_clf = _make_clf_bundle(n=5)
    bundle_clf.y[0] = 2.0
    with pytest.raises(ValueError, match="binary"):
        validate_bundle(bundle_clf)

    # MULTICLF rejects negative or non-integer class codes
    bundle_multi = _make_multiclf_bundle(n=6, k=3)
    bundle_multi.y[0] = -1.0
    with pytest.raises(ValueError, match="nonnegative integer"):
        validate_bundle(bundle_multi)

    bundle_multi.y[0] = 0.5
    with pytest.raises(ValueError, match="nonnegative integer"):
        validate_bundle(bundle_multi)

    bundle_multi.y[0] = 5.0  # >= n_classes
    with pytest.raises(ValueError, match=">= n_classes"):
        validate_bundle(bundle_multi)


# ---------------------------------------------------------------------------
# Prediction validation: shapes, ranges, do not silently drop rows
# ---------------------------------------------------------------------------

def test_validate_predictions_does_not_drop_rows():
    bundle = _make_freq_bundle(n=10)
    # Shorter predictions -> raises
    with pytest.raises(ValueError, match="do not silently drop rows"):
        validate_predictions(bundle, np.ones(8))

    # Extra rows -> raises
    with pytest.raises(ValueError, match="do not silently drop rows"):
        validate_predictions(bundle, np.ones(12))


def test_validate_predictions_non_finite():
    bundle = _make_freq_bundle(n=5)
    preds = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(bundle, preds)

    preds[2] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(bundle, preds)


def test_validate_predictions_domain_ranges():
    # CLF probabilities must be in [0, 1]
    b_clf = _make_clf_bundle(n=5)
    with pytest.raises(ValueError, match="probabilities must be in range"):
        validate_predictions(b_clf, np.array([0.1, 0.2, -0.05, 0.4, 0.5]))
    with pytest.raises(ValueError, match="probabilities must be in range"):
        validate_predictions(b_clf, np.array([0.1, 0.2, 1.05, 0.4, 0.5]))

    # FREQ and PP rates must be >= 0
    b_freq = _make_freq_bundle(n=5)
    with pytest.raises(ValueError, match="nonnegative rates"):
        validate_predictions(b_freq, np.array([0.1, 0.2, -0.01, 0.4, 0.5]))

    b_pp = _make_pp_bundle(n=5)
    with pytest.raises(ValueError, match="nonnegative rates"):
        validate_predictions(b_pp, np.array([10.0, 20.0, -1.0, 40.0, 50.0]))

    # SEV severities must be > 0
    b_sev = _make_sev_bundle(n=5)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_predictions(b_sev, np.array([100.0, 0.0, 200.0, 300.0, 400.0]))

    # MULTICLF requires 2D probabilities summing to 1
    b_multi = _make_multiclf_bundle(n=5, k=3)
    with pytest.raises(ValueError, match="2D array"):
        validate_predictions(b_multi, np.array([0, 1, 2, 0, 1]))

    bad_sum = np.full((5, 3), 0.5)  # row sum = 1.5
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_predictions(b_multi, bad_sum)


# ---------------------------------------------------------------------------
# Scoring integration: score_bundle and primary_value validate inputs
# ---------------------------------------------------------------------------

def test_score_bundle_and_primary_value_validate():
    bundle = _make_freq_bundle(n=10)
    good_preds = np.full(10, 0.25)
    val = primary_value(bundle, good_preds)
    assert np.isfinite(val)

    scores = score_bundle(bundle, good_preds, ("poisson_deviance", "balance"))
    assert "poisson_deviance" in scores
    assert "balance" in scores

    # Invalid prediction shape caught by score_bundle
    with pytest.raises(ValueError, match="do not silently drop rows"):
        score_bundle(bundle, np.ones(5), ("poisson_deviance",))


# ---------------------------------------------------------------------------
# TBoost: frequency_formulation monkeypatch & contract tests
# ---------------------------------------------------------------------------

class _DummyTBoostModel:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.fit_args = None
        self.fit_kwargs = None

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.fit_args = (X, y, sample_weight)
        self.fit_kwargs = kwargs
        return self

    def predict(self, X):
        return np.full(len(X), 0.15)


def test_tboost_frequency_formulation_rate_weight_monkeypatch(monkeypatch):
    """Verify rate_weight (default): passed y is rate, sample_weight is exposure, no exposure arg."""
    from insurarena.models import tboost_model as TM

    dummy = _DummyTBoostModel()
    import t_boost
    def recipe(*args, **kwargs):
        dummy.init_kwargs = kwargs
        return dummy
    monkeypatch.setattr(t_boost, "recommended_recipe", recipe)

    bundle = _make_freq_bundle(n=20)
    model = TM.TBoost(hp={"frequency_formulation": "rate_weight", "budget": 10})
    model.fit(bundle)

    assert dummy.fit_args is not None
    _X, passed_y, passed_w = dummy.fit_args
    expected_rate = bundle.y / bundle.exposure
    np.testing.assert_allclose(passed_y, expected_rate)
    np.testing.assert_allclose(passed_w, bundle.exposure)
    assert "exposure" not in dummy.fit_kwargs

    # hp key was popped before constructor
    assert "frequency_formulation" not in dummy.init_kwargs
    assert "frequency_formulation" in model.hp


def test_tboost_frequency_formulation_count_offset_monkeypatch(monkeypatch):
    """Verify count_offset: passed y is raw counts, sample_weight is unit weights, exposure is passed."""
    from insurarena.models import tboost_model as TM

    dummy = _DummyTBoostModel()
    import t_boost
    def recipe(*args, **kwargs):
        dummy.init_kwargs = kwargs
        return dummy
    monkeypatch.setattr(t_boost, "recommended_recipe", recipe)

    bundle = _make_freq_bundle(n=25, group_ids=np.arange(25))
    model = TM.TBoost(hp={"frequency_formulation": "count_offset", "budget": 10})
    model.fit(bundle)

    assert dummy.fit_args is not None
    _X, passed_y, passed_w = dummy.fit_args
    # y is raw counts
    np.testing.assert_allclose(passed_y, bundle.y)
    # sample_weight is policy unitweights (1.0)
    np.testing.assert_allclose(passed_w, np.ones(25))
    # exposure is passed
    assert "exposure" in dummy.fit_kwargs
    np.testing.assert_allclose(dummy.fit_kwargs["exposure"], bundle.exposure)
    # group handling is preserved
    assert "groups" in dummy.fit_kwargs
    np.testing.assert_array_equal(dummy.fit_kwargs["groups"], bundle.group_ids)

    # hp key was popped before constructor
    assert "frequency_formulation" not in dummy.init_kwargs
    assert "frequency_formulation" in model.hp

    # predict outputs rate
    preds = model.predict(bundle.X)
    assert preds.shape == (25,)
    assert (preds > 0).all()


def test_tboost_frequency_formulation_guard_nonfreq():
    """Verify count_offset is explicitly barred on non-FREQ tasks and invalid formulations raise."""
    from insurarena.models.tboost_model import TBoost

    # Non-FREQ tasks raise
    for b_maker in (_make_sev_bundle, _make_pp_bundle, _make_clf_bundle, _make_multiclf_bundle):
        bundle = b_maker()
        model = TBoost(hp={"frequency_formulation": "count_offset"})
        with pytest.raises(ValueError, match="only valid for FREQ"):
            model.fit(bundle)

    # Unknown formulation raises
    model_invalid = TBoost(hp={"frequency_formulation": "magic_formulation"})
    with pytest.raises(ValueError, match="unsupported frequency_formulation"):
        model_invalid.fit(_make_freq_bundle())


def test_tboost_actual_small_fit():
    """Test actual small fit if t_boost is importable/installed."""
    pytest.importorskip("t_boost")
    from insurarena.models.tboost_model import TBoost

    bundle = _make_freq_bundle(n=30)

    # rate_weight fit
    m_rw = TBoost(hp={"frequency_formulation": "rate_weight", "budget": 10, "n_bags": 1})
    m_rw.fit(bundle)
    pred_rw = m_rw.predict(bundle.X)
    assert pred_rw.shape == (30,)
    assert (pred_rw > 0).all()

    # count_offset fit
    m_co = TBoost(hp={"frequency_formulation": "count_offset", "budget": 10, "n_bags": 1})
    m_co.fit(bundle)
    pred_co = m_co.predict(bundle.X)
    assert pred_co.shape == (30,)
    assert (pred_co > 0).all()


# ---------------------------------------------------------------------------
# Independent scoring identity: Poisson count offset vs rate exposures
# ---------------------------------------------------------------------------

def test_poisson_count_offset_vs_rate_exposure_scoring_identity():
    """Verify the mathematical identity between Poisson count with offset vs Poisson rate with exposure weights.

    Let Y_i be raw claim counts, E_i be exposures (> 0).
    Rate target is y_i = Y_i / E_i.
    Let lambda_i be predicted rates, so expected counts are mu_i = E_i * lambda_i.

    1. Total deviance is identical:
       sum(2 * [ Y_i * log(Y_i / mu_i) - (Y_i - mu_i) ])
       == sum(E_i * 2 * [ y_i * log(y_i / lambda_i) - (y_i - lambda_i) ])

    2. Mean deviance relation:
       D_rate = D_total / sum(E)    (mean weighted by exposure)
       D_count = D_total / N        (mean weighted by unit weights)
       => D_rate = D_count * (N / sum(E))
    """
    rng = np.random.default_rng(123)
    n = 100
    exp = rng.uniform(0.1, 2.5, size=n)
    # Raw counts
    counts = rng.poisson(0.4 * exp).astype(float)
    rates = counts / exp

    # Rate predictions from an arbitrary model
    pred_rates = rng.uniform(0.05, 0.8, size=n)
    pred_counts = pred_rates * exp

    # Weighted rate deviance (benchmark canonical metric)
    dev_rate_mean = mean_poisson_deviance(rates, pred_rates, sample_weight=exp)
    tot_dev_rate = dev_rate_mean * float(np.sum(exp))

    # Unweighted count deviance (raw count scale)
    dev_count_mean = mean_poisson_deviance(counts, pred_counts, sample_weight=np.ones(n))
    tot_dev_count = dev_count_mean * n

    # 1. Total deviances are identical to machine precision
    assert tot_dev_rate == pytest.approx(tot_dev_count, rel=1e-12)

    # 2. Denominator conversion identity holds
    expected_rate_dev = dev_count_mean * (n / float(np.sum(exp)))
    assert dev_rate_mean == pytest.approx(expected_rate_dev, rel=1e-12)
