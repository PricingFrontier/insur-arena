"""Multinomial (MULTICLF) task: loader shaping, (n,K) predictions, metric, and a runner smoke."""
import numpy as np
import pytest

from insurarena.datasets.registry import build_loader
from insurarena.metrics.deviance import multiclass_accuracy, multiclass_log_loss
from insurarena.models.registry import build_model
from insurarena.scoring import primary_value, score_bundle
from insurarena.tasks import TaskType


def test_loader_shapes_multiclass_target():
    b = build_loader("synthetic_multiclf", subsample=2000).load()
    assert b.task.task is TaskType.MULTICLF
    k = b.meta["n_classes"]
    assert k >= 3
    codes = np.unique(b.y)
    assert set(codes) <= set(range(k))          # dense 0..K-1 codes
    assert b.meta["stratify_on"] == "class_label"
    assert b.exposure is None


def test_multiclass_metric_beats_uniform():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    k = 4
    uniform = np.full((len(y), k), 1.0 / k)
    confident = np.full((len(y), k), 0.01)
    confident[np.arange(len(y)), y] = 0.97
    assert multiclass_log_loss(y, confident) < multiclass_log_loss(y, uniform)
    assert multiclass_accuracy(y, confident) == 1.0
    assert abs(multiclass_log_loss(y, uniform) - np.log(k)) < 1e-6


def test_glm_predicts_probability_matrix():
    b = build_loader("synthetic_multiclf", subsample=1500).load()
    k = b.meta["n_classes"]
    tr, te = b.subset(np.arange(0, 1200)), b.subset(np.arange(1200, 1500))
    pred = build_model("glm").fit(tr).predict(te.X)
    assert pred.shape == (len(te.X), k)
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-6)
    # a fitted GLM should beat the uniform baseline log-loss
    assert primary_value(te, pred) < np.log(k)
    sec = score_bundle(te, pred, b.task.secondary_metrics)
    assert "multiclass_accuracy" in sec and 0.0 <= sec["multiclass_accuracy"] <= 1.0


def test_tboost_predicts_multiclass_probability_matrix():
    pytest.importorskip("t_boost")
    b = build_loader("synthetic_multiclf", subsample=900).load()
    k = b.meta["n_classes"]
    tr, te = b.subset(np.arange(0, 700)), b.subset(np.arange(700, 900))
    pred = build_model("t-boost", hp={"budget": 40}).fit(tr).predict(te.X)
    assert pred.shape == (len(te.X), k)
    assert np.isfinite(pred).all()
    assert ((pred >= 0.0) & (pred <= 1.0)).all()
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-6)


def test_runner_smoke_multiclass():
    """Full cell path on the multinomial task: frozen-GLM baseline, skill-vs-GLM, secondary metrics,
    default + tuned (OOF config selection). GLM-only keeps it fast + deterministic; booster multiclass
    support is covered by test_adapters conformance."""
    from insurarena.config import Config
    from insurarena.eval.runner import run_benchmark
    from insurarena.results.store import ResultStore

    cfg = Config(
        datasets=["synthetic_multiclf"],
        models=[{"name": "glm", "variants": ["default", "tuned"]}],
        split={"small_threshold": 2500, "small": {"folds": 3, "repeats": 1},
               "large": {"folds": 3, "repeats": 1}},
        n_configs=3, inner_folds=3,
        subsample_rows=1500,
        cache_dir="/tmp/insurarena-multiclf-test",
        store_predictions=True,
        use_cache=False,
    )
    df = run_benchmark(cfg, store=ResultStore(cfg.cache_dir, store_predictions=True), verbose=False)
    for variant, n in [("default", 3), ("tuned", 3)]:
        cells = df[(df["model"] == "glm") & (df["variant"] == variant)]
        assert len(cells) == n and cells["failed"].eq("").all()
        assert np.isfinite(cells["primary_value"]).all()
        assert cells["primary_metric"].eq("multiclass_log_loss").all()
    # default GLM == the frozen GLM baseline -> skill ~ 0
    assert np.allclose(df[df["variant"] == "default"]["skill_vs_glm"].to_numpy(), 0.0, atol=1e-9)
