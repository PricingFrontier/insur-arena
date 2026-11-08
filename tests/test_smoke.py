import numpy as np
import pytest

from insurarena.config import Config
from insurarena.eval.runner import run_benchmark
from insurarena.results.leaderboard import leaderboard
from insurarena.results.store import ResultStore


def _cfg(tmp_path, models):
    return Config(
        threads_per_worker=1,
        datasets=["synthetic_freq"],
        models=models,
        subsample_rows=3000,
        split={"small_threshold": 100000, "small": {"folds": 3, "repeats": 1}, "large": {"folds": 3, "repeats": 1}},
        n_configs=3,
        cache_dir=str(tmp_path),
        store_predictions=True,
    )


def test_glm_only_end_to_end(tmp_path):
    cfg = _cfg(tmp_path, [{"name": "glm", "variants": ["default"]}])
    df = run_benchmark(cfg, verbose=False)
    assert len(df) == 3                      # 3 outer splits
    lb = leaderboard(df)
    assert "skill_vs_glm_mean" in lb.columns
    # the ranked GLM default vs the frozen-GLM default baseline -> ~0 skill
    assert abs(float(lb["skill_vs_glm_mean"].iloc[0])) < 0.05


def test_with_lightgbm(tmp_path):
    pytest.importorskip("lightgbm")
    cfg = _cfg(
        tmp_path,
        [
            {"name": "glm", "variants": ["default"]},
            # small cap keeps this a smoke test, not an accuracy check (early-stopped ES probe would
            # otherwise chase the full 2000-round default)
            {"name": "lightgbm", "variants": ["default"], "hp": {"n_estimators": 40}},
        ],
    )
    df = run_benchmark(cfg, verbose=False)
    assert set(df["model"]) >= {"glm", "lightgbm"}
    assert set(df["family"]) >= {"glm", "gbm"}
    assert np.isfinite(df["skill_vs_glm"]).any()


def test_tuned_oof_selection(tmp_path):
    """The `tuned` variant selects a config by k-fold OOF and refits a single model."""
    cfg = Config(
        threads_per_worker=1,
        datasets=["synthetic_freq"],
        models=[{"name": "glm", "variants": ["default", "tuned"]}],
        subsample_rows=2500, n_configs=3, inner_folds=2,
        split={"small_threshold": 100000, "small": {"folds": 3, "repeats": 1}, "large": {"folds": 3, "repeats": 1}},
        cache_dir=str(tmp_path), store_predictions=False,
    )
    store = ResultStore(str(tmp_path), store_predictions=False)
    df = run_benchmark(cfg, store=store, verbose=False)
    assert set(df["variant"]) == {"default", "tuned"}
    assert (df["failed"] == "").all()
    assert store.trajectories                       # tuned trajectory recorded
    assert store.hpo_evaluations                    # scored tuned configs recorded
    store.save()
    assert (tmp_path / "hpo_evaluations.json").exists()
    # tuned should not be worse than default's frozen-GLM baseline by much (>= ~0)
    assert df[df.variant == "tuned"]["skill_vs_glm"].mean() > -0.05


@pytest.mark.parametrize("model,lib", [("xgboost", "xgboost"), ("catboost", "catboost"), ("ebm", "interpret")])
def test_extra_techniques_run(tmp_path, model, lib):
    pytest.importorskip(lib)
    # one shared cap for whichever technique is under test this run (a smoke test, not an accuracy
    # check); unread keys are harmlessly ignored by adapters that don't have them. EBM in particular
    # fits at full library defaults (unbounded rounds, 4 bags, fractional interactions) without these.
    hp = {"n_estimators": 40, "iterations": 40, "interactions": 0, "smoothing_rounds": 1, "outer_bags": 1}
    cfg = _cfg(
        tmp_path, [{"name": "glm", "variants": ["default"]}, {"name": model, "variants": ["default"], "hp": hp}]
    )
    df = run_benchmark(cfg, verbose=False)
    assert model in set(df["model"])
    assert np.isfinite(df["skill_vs_glm"]).any()


@pytest.mark.parametrize("ds", ["synthetic_sev", "synthetic_pp", "synthetic_clf"])
def test_nonfreq_tasks_run(tmp_path, ds):
    """GLM + LightGBM are task-aware: severity (Gamma), pure premium (Tweedie), classification."""
    pytest.importorskip("lightgbm")
    cfg = Config(
        threads_per_worker=1,
        datasets=[ds],
        models=[
            {"name": "glm", "variants": ["default"]},
            # small cap keeps this a smoke test, not an accuracy check (matches test_with_lightgbm)
            {"name": "lightgbm", "variants": ["default"], "hp": {"n_estimators": 40}},
        ],
        subsample_rows=2500,
        split={"small_threshold": 100000, "small": {"folds": 3, "repeats": 1}, "large": {"folds": 3, "repeats": 1}},
        n_configs=2,
        cache_dir=str(tmp_path),
        store_predictions=False,
    )
    df = run_benchmark(cfg, verbose=False)
    assert set(df["model"]) >= {"glm", "lightgbm"}
    assert np.isfinite(df["skill_vs_glm"]).any()
