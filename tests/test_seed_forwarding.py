"""fit_one must forward the per-cell seed into the model hp (was silently dropped -> every
deployed cell fit with seed=0 regardless of the seed recorded on its ResultRow)."""
import numpy as np
import pandas as pd
import pytest

from insurarena.datasets.base import DatasetBundle
from insurarena.eval.hpo import fit_one
from insurarena.tasks import FREQ_SPEC

pytest.importorskip("t_boost")


def _bundle(n: int = 300, data_seed: int = 0) -> DatasetBundle:
    rng = np.random.default_rng(data_seed)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    exposure = np.ones(n)
    y = rng.poisson(0.2, size=n).astype(float)          # poisson-ish counts
    return DatasetBundle(
        name="seed_forwarding_probe", X=X, y=y, exposure=exposure,
        sample_weight=exposure, task=FREQ_SPEC,
    )


def test_fit_one_forwards_cell_seed_to_triboost():
    model = fit_one("t-boost", {"budget": 50, "n_bags": 1}, _bundle(), seed=123)

    assert model.model.get_params()["seed"] == 123


def test_fit_one_explicit_hp_seed_beats_cell_seed():
    model = fit_one("t-boost", {"budget": 50, "seed": 7, "n_bags": 1}, _bundle(), seed=123)

    assert model.model.get_params()["seed"] == 7


def test_fit_one_triboost_is_deterministic_per_seed():
    bundle = _bundle()

    # n_bags=1: only the seed-forwarding/determinism behavior is under test here, not the (now
    # 8-bag default) recipe's accuracy — single-bag fits are the fast, sufficient check.
    m_a = fit_one("t-boost", {"budget": 50, "n_bags": 1}, bundle, seed=42)
    m_b = fit_one("t-boost", {"budget": 50, "n_bags": 1}, bundle, seed=42)
    m_c = fit_one("t-boost", {"budget": 50, "n_bags": 1}, bundle, seed=99)

    p_a = m_a.predict(bundle.X, bundle.exposure)
    p_b = m_b.predict(bundle.X, bundle.exposure)
    p_c = m_c.predict(bundle.X, bundle.exposure)

    assert np.array_equal(p_a, p_b)          # same seed -> byte-identical predictions
    assert not np.array_equal(p_a, p_c)      # different seed -> different predictions
