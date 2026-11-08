import numpy as np
import pandas as pd
import pytest

from insurarena.datasets.base import DatasetBundle
from insurarena.datasets.registry import build_loader
from insurarena.models.rustystats_model import RustyStatsGLM
from insurarena.tasks import PP_SPEC, SEV_SPEC


def test_rustystats_terms_use_conservative_defaults():
    n = 40
    bundle = DatasetBundle(
        name="terms",
        X=pd.DataFrame(
            {
                "x": np.linspace(0.0, 1.0, n),
                "low": np.where(np.arange(n) % 2 == 0, "A", "B"),
                "high": [f"L{i % 8}" for i in range(n)],
            }
        ),
        y=np.linspace(10.0, 20.0, n),
        exposure=None,
        sample_weight=np.ones(n),
        task=SEV_SPEC,
        cat_features=["low", "high"],
    )
    model = RustyStatsGLM(
        hp={
            "spline_min_rows": 10,
            "spline_min_signal": 1,
            "low_card_categorical": 3,
            "max_categorical_dummy_features": 10,
        }
    )
    model._task = SEV_SPEC.task
    model._learn_schema(bundle)

    terms = model._terms(bundle)

    assert terms["x"] == {"type": "ns", "df": 4}
    assert terms["low"] == {"type": "categorical"}
    assert terms["high"] == {"type": "target_encoding", "prior_weight": "auto"}


def test_rustystats_continuous_preprocessing_is_train_fold_only():
    n = 120
    train = DatasetBundle(
        name="robust",
        X=pd.DataFrame(
            {"heavy": np.r_[np.arange(100, dtype=float), np.linspace(10_000.0, 1_000_000.0, 20)]}
        ),
        y=np.linspace(10.0, 20.0, n),
        exposure=None,
        sample_weight=np.ones(n),
        task=SEV_SPEC,
        cat_features=[],
    )
    model = RustyStatsGLM(
        hp={
            "continuous_winsor_lower": 0.10,
            "continuous_winsor_upper": 0.90,
        }
    )
    model._task = SEV_SPEC.task
    model._learn_schema(train)

    transform = model._cont_transforms["heavy"]
    prepared = model._prep_X(pd.DataFrame({"heavy": [-10.0, 50.0, 1_000_000_000.0]}))
    values = prepared["heavy"].to_numpy()

    assert transform["log1p"]
    assert values[0] == pytest.approx(transform["lower"])
    assert values[1] == pytest.approx(np.log1p(50.0))
    assert values[2] == pytest.approx(transform["upper"])


def test_rustystats_default_predict_policy_recovers_finite_extreme_eta():
    model = RustyStatsGLM()

    assert model._predict_kwargs()["on_extreme_eta"] == "clip"


def test_rustystats_pp_categoricals_can_use_target_encoding():
    n = 40
    bundle = DatasetBundle(
        name="pp_terms",
        X=pd.DataFrame(
            {
                "x": np.linspace(0.0, 1.0, n),
                "zone": np.where(np.arange(n) % 2 == 0, "A", "B"),
            }
        ),
        y=np.linspace(0.0, 20.0, n),
        exposure=np.ones(n),
        sample_weight=np.ones(n),
        task=PP_SPEC,
        cat_features=["zone"],
    )
    model = RustyStatsGLM(hp={"pp_target_encode_categoricals": True})
    model._task = PP_SPEC.task
    model._learn_schema(bundle)

    terms = model._terms(bundle)

    assert terms["zone"] == {"type": "target_encoding", "prior_weight": "auto"}


def test_rustystats_pp_auto_smooths_sparse_positive_levels():
    n = 60
    zone = np.array(["A"] * 25 + ["B"] * 20 + ["C"] * 15)
    y = np.zeros(n, dtype=float)
    y[3] = 100.0
    y[25:45] = 10.0
    bundle = DatasetBundle(
        name="pp_sparse_positive_terms",
        X=pd.DataFrame(
            {
                "x": np.linspace(0.0, 1.0, n),
                "zone": zone,
                "stable": np.where(np.arange(n) % 2 == 0, "L", "R"),
            }
        ),
        y=y,
        exposure=np.ones(n),
        sample_weight=np.ones(n),
        task=PP_SPEC,
        cat_features=["zone", "stable"],
    )
    model = RustyStatsGLM()
    model._task = PP_SPEC.task
    model._learn_schema(bundle)
    model._pp_force_target_encoding = False
    model._pp_target_encode_columns = model._pp_auto_target_encode_columns(bundle)

    terms = model._terms(bundle)

    assert model._pp_target_encode_columns == {"stable", "zone"}
    assert terms["stable"] == {"type": "target_encoding", "prior_weight": "auto"}
    assert terms["zone"] == {"type": "target_encoding", "prior_weight": "auto"}


def test_rustystats_pp_auto_keeps_structural_zero_dummy_levels():
    n = 60
    zone = np.array(["A"] * 25 + ["B"] * 20 + ["C"] * 15)
    y = np.zeros(n, dtype=float)
    y[25:45] = 10.0
    bundle = DatasetBundle(
        name="pp_structural_zero_terms",
        X=pd.DataFrame(
            {
                "x": np.linspace(0.0, 1.0, n),
                "zone": zone,
            }
        ),
        y=y,
        exposure=np.ones(n),
        sample_weight=np.ones(n),
        task=PP_SPEC,
        cat_features=["zone"],
    )
    model = RustyStatsGLM()
    model._task = PP_SPEC.task
    model._learn_schema(bundle)
    model._pp_force_target_encoding = False
    model._pp_target_encode_columns = model._pp_auto_target_encode_columns(bundle)

    terms = model._terms(bundle)

    assert model._pp_target_encode_columns == set()
    assert terms["zone"] == {"type": "categorical"}


def test_rustystats_multiclf_uses_native_multinomial():
    pytest.importorskip("rustystats")
    bundle = build_loader("synthetic_multiclf", subsample=320).load()
    train = bundle.subset(np.arange(0, 240))
    test = bundle.subset(np.arange(240, 320))

    model = RustyStatsGLM(
        hp={
            "alpha": 0.03,
            "max_iter": 80,
            "multiclass_solver": "dense",
        }
    ).fit(train)
    pred = model.predict(test.X)

    assert hasattr(model.model, "predict_proba")
    assert not isinstance(model.model, list)
    assert pred.shape == (len(test.X), bundle.meta["n_classes"])
    assert np.isfinite(pred).all()
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-6)
