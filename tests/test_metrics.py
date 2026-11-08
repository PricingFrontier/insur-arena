import numpy as np
import pytest
from sklearn.metrics import mean_poisson_deviance

from insurarena.metrics import calibration as C
from insurarena.metrics import deviance as D
from insurarena.metrics import discrimination as G


def test_poisson_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.poisson(0.3, 500).astype(float)
    p = np.clip(rng.gamma(2, 0.15, 500), 1e-3, None)
    w = rng.uniform(0.1, 1, 500)
    assert abs(D.poisson_deviance(y, p, w) - mean_poisson_deviance(y, p, sample_weight=w)) < 1e-9


def test_d2_perfect_is_one():
    rng = np.random.default_rng(1)
    y = rng.poisson(0.4, 400).astype(float) + 1e-6
    assert D.d2_poisson(y, y, np.ones_like(y)) > 0.999


def test_balance_unit_when_pred_equals_actual():
    y = np.array([1.0, 2.0, 3.0])
    assert abs(C.balance(y, y, np.array([1.0, 1.0, 1.0])) - 1.0) < 1e-12


def test_gini_bounds_and_oracle():
    rng = np.random.default_rng(2)
    n = 3000
    exp = rng.uniform(0.1, 1, n)
    rate = np.exp(rng.normal(-1, 0.5, n))
    y = rng.poisson(rate * exp) / exp
    pred = rate * np.exp(rng.normal(0, 0.3, n))
    gm = G.gini_exposure(y, pred, exposure=exp)
    go = G.gini_exposure(y, y, exposure=exp)
    assert -1.0 <= gm <= 1.0
    assert go >= gm - 1e-9            # oracle is an upper bound


def test_gini_refuses_unweighted():
    with pytest.raises(ValueError):
        G.gini_exposure([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])


def test_skill_sign():
    # model better than baseline -> positive skill
    assert D.skill_vs_baseline(0.8, 1.0) == pytest.approx(0.2)
    assert np.isnan(D.skill_vs_baseline(0.8, 0.0))
