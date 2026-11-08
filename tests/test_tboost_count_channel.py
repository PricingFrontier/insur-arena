"""av32 switchover: the count/rarity channel is now a NATIVE t-boost feature. The adapter no
longer adds __share numeric columns; it enables the library's cat_channels=["mean","count"] with
cat_count_min_levels=20 (the retired shim's gate) so t-boost adds a target-free count axis to
high-card categoricals and collapses it losslessly into one rating table per categorical."""
import numpy as np
import pandas as pd
import pytest

from insurarena.datasets.base import DatasetBundle
from insurarena.models.tboost_model import TBoost
from insurarena.tasks import FREQ_SPEC

pytest.importorskip("t_boost")


def _bundle(n: int = 600, levels: int = 30, seed: int = 0) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "hicard": pd.Series(rng.integers(0, levels, size=n)).astype(str),
        "locard": pd.Series(rng.integers(0, 4, size=n)).astype(str),
    })
    exposure = np.ones(n)
    y = rng.poisson(0.2, size=n).astype(float)
    return DatasetBundle(
        name="count_channel_probe", X=X, y=y, exposure=exposure,
        sample_weight=exposure, task=FREQ_SPEC, cat_features=["hicard", "locard"],
    )


def test_adapter_enables_library_count_channel_and_adds_no_share_columns():
    b = _bundle()
    m = TBoost(hp={"budget": 30, "n_bags": 1}).fit(b)

    # _prep_X is back to pure dtype coercion — NO derived __share columns, no shim state.
    Xp = m._prep_X(b.X)
    assert not any(c.endswith("__share") for c in Xp.columns)
    assert not hasattr(m, "_share_maps")

    # the switchover wiring: the library estimator was built with the count channel + the ≥20 gate.
    params = m.model.get_params()
    assert params["cat_channels"] == ["mean", "count"]
    assert params["cat_count_min_levels"] == 20


def test_purely_numeric_dataset_does_not_enable_cat_channels():
    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    b = DatasetBundle(name="numeric_only", X=X, y=rng.poisson(0.2, n).astype(float),
                      exposure=np.ones(n), sample_weight=np.ones(n), task=FREQ_SPEC)
    m = TBoost(hp={"budget": 30, "n_bags": 1}).fit(b)
    assert m.model.get_params()["cat_channels"] is None   # inert without categoricals


def test_high_card_fit_predicts_finite_and_serves_unseen_levels():
    b = _bundle()
    m = TBoost(hp={"budget": 30, "n_bags": 1}).fit(b)
    X_new = b.X.iloc[:5].copy()
    X_new.loc[X_new.index[0], "hicard"] = "never_seen_level"
    assert np.isfinite(m.predict(X_new, np.ones(5))).all()
