"""Conformance: every registered technique fits + predicts sanely on each task it supports."""
import numpy as np
import pytest

from insurarena.datasets.registry import build_loader
from insurarena.models.registry import build_model, model_class
from insurarena.tasks import TaskType

_SYN = {
    TaskType.FREQ: "synthetic_freq",
    TaskType.SEV: "synthetic_sev",
    TaskType.PP: "synthetic_pp",
    TaskType.CLF: "synthetic_clf",
    TaskType.MULTICLF: "synthetic_multiclf",
}
_LIB = {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost", "ebm": "interpret",
        "rustystats": "rustystats",
        "tabpfn3": "tabpfn", "t-boost": "t_boost"}


@pytest.mark.parametrize("name", ["glm", "lightgbm", "xgboost", "catboost", "ebm", "t-boost",
                                  "rustystats", "tabdpt", "tabicl", "tabpfn3"])
def test_adapter_conformance(name):
    if name in _LIB:
        pytest.importorskip(_LIB[name])
    # small budgets keep the conformance check fast (no early stopping here — val is None). This is
    # an API-contract check, not an accuracy check, so every adapter gets capped to a tiny budget —
    # not just the classic GBMs: EBM only reads keys in _EBM_KEYS (+outer_bags/seed), so without an
    # explicit cap it fits at full library defaults (unbounded rounds, fractional interactions, 4
    # bags, n_jobs=-1); tri-boost defaults to 8 in-model bags unless overridden. Extra keys are
    # harmlessly ignored by adapters that don't read them (each reads hp via .get()/allowlist).
    hp = {
        "n_estimators": 40, "iterations": 40,                      # classic GBMs
        "n_bags": 1, "budget": 40,                                  # tri-boost: single bag, tiny cap
        "interactions": 0, "smoothing_rounds": 1, "outer_bags": 1,  # EBM: without these it fits at full library defaults
        "backend": "local",                                         # tabpfn3: conformance must not hit the network
    }
    for task in model_class(name).supported_tasks:
        bundle = build_loader(_SYN[task], subsample=800).load()
        try:
            model = build_model(name, hp=hp).fit(bundle, None)
        except Exception as e:  # noqa: BLE001 - only tabpfn3's license gate is special-cased below
            if name == "tabpfn3":
                from tabpfn.errors import TabPFNLicenseError

                if isinstance(e, TabPFNLicenseError):
                    pytest.skip(f"tabpfn3: model weights unavailable locally ({e})")
            raise
        pred = model.predict(bundle.X, bundle.exposure)
        assert pred.shape[0] == len(bundle.X)
        assert np.isfinite(pred).all(), f"{name} on {task.value} produced non-finite predictions"
        if task == TaskType.CLF:
            assert ((pred >= 0) & (pred <= 1)).all()


def test_lightgbm_small_min_data_in_leaf_refits_without_val():
    # Tuned configs sample min_data_in_leaf down to 1; the refit path (val=None, raw data freed)
    # used to hard-error because the pre-train construct() pre-filtered at the library default 20.
    pytest.importorskip("lightgbm")
    bundle = build_loader("synthetic_freq", subsample=800).load()
    model = build_model("lightgbm", hp={"n_estimators": 40, "min_data_in_leaf": 5}).fit(bundle, None)
    pred = model.predict(bundle.X, bundle.exposure)
    assert np.isfinite(pred).all()
