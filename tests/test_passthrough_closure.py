"""The `_PASSTHROUGH` closure invariant.

`tboost_model.fit` forwards `self.hp` to the estimator through exactly one filter:

    overrides = {c: self.hp[c] for c in _PASSTHROUGH if c in self.hp}

Before 2026-09-04 that filter dropped an unlisted key SILENTLY, which is the one failure a
version-keyed cache cannot detect: a tune-grid or coordinate-probe dimension that is not named in
`_PASSTHROUGH` still enters the candidate hp dict, still moves `hpo_config_set`, still forces a
full tuned recompute — and never reaches the fit, so every candidate is scored as though the
dimension did not exist. `max_bin` sat in exactly that state until hsv 18.

These tests assert the closure over every producer of hp keys, so the state is unreachable rather
than merely documented.
"""
from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.eval import hpo
from insurarena.models.tboost_model import (
    _HP_NOT_FORWARDED,
    _PASSTHROUGH,
    TBoost,
)
from insurarena.tasks import FREQ_SPEC


def _tiny_freq_bundle(n: int = 600, seed: int = 0) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    w = np.ones(n)
    return DatasetBundle(
        name="closure_probe", X=X, y=rng.poisson(0.2, n).astype(float),
        exposure=w, sample_weight=w, task=FREQ_SPEC, cat_features=[],
    )

_N_CAMPAIGN = 25            # configs/t-boost.yaml


def _every_hp_key_any_producer_can_emit() -> set[str]:
    """The static closure set — deliberately computed from `hpo.static_configs` (the real
    candidate stream) rather than from `_own_hp_keys`, so the two derivations must agree."""
    keys: set[str] = set()
    for hp in hpo.static_configs("t-boost", _N_CAMPAIGN):   # anchor + probes + sampled basis
        keys |= set(hp)
    keys |= set(TBoost.default_hp())
    keys |= set(TBoost.tuning_base_hp())
    keys |= set(TBoost.selection_hp())
    finale = getattr(TBoost, "finale_hp", None)
    if callable(finale):
        keys |= set(finale())
    return keys


def test_passthrough_is_closed_over_every_hp_producer():
    leaked = sorted(_every_hp_key_any_producer_can_emit()
                    - set(_PASSTHROUGH) - set(_HP_NOT_FORWARDED))
    assert not leaked, (
        f"hp keys {leaked} are produced but reach neither `_PASSTHROUGH` (forwarded to the "
        f"estimator) nor `_HP_NOT_FORWARDED` (deliberately harness-only). They would be dropped "
        f"silently while still re-keying every tuned cell."
    )


def test_the_two_sets_are_disjoint():
    # A key in both is ambiguous about whether the fit is meant to see it.
    both = sorted(set(_PASSTHROUGH) & set(_HP_NOT_FORWARDED))
    assert not both, f"{both} are in both _PASSTHROUGH and _HP_NOT_FORWARDED"


def test_a_key_this_adapter_searches_but_does_not_forward_raises(monkeypatch):
    # The guard itself, at its real scope: a dimension t-boost's OWN machinery can emit that is
    # not wired through `_PASSTHROUGH`. Simulated by widening the producible set, because the
    # closure currently holds — which is the point of the test above.
    from insurarena.models import tboost_model as TM

    widened = frozenset(TM._own_hp_keys() | {"nonesuch_knob"})   # capture BEFORE patching
    monkeypatch.setattr(TM, "_own_hp_keys", lambda: widened)
    with pytest.raises(ValueError, match="nonesuch_knob"):
        TBoost(hp={"budget": 20, "n_bags": 1, "nonesuch_knob": 3}).fit(_tiny_freq_bundle())


def test_a_foreign_models_hp_key_is_tolerated_not_raised():
    # `tests/test_adapters.py`'s conformance sweep hands EVERY adapter one shared hp dict carrying
    # other models' knobs (`outer_bags` and `smoothing_rounds` are EBM's). Ignoring those is the
    # contract; only keys this adapter itself can produce are policed.
    TBoost(hp={"budget": 20, "n_bags": 1, "outer_bags": 8, "smoothing_rounds": 200}).fit(
        _tiny_freq_bundle())


def test_a_listed_key_and_an_allowlisted_key_both_fit_fine():
    # The guard must not fire on legitimate hp. `max_bin` is forwarded; `budget` is harness-only.
    TBoost(hp={"budget": 20, "n_bags": 1, "max_bin": 64}).fit(_tiny_freq_bundle())


def test_gate_written_keys_need_not_be_listed_for_the_fit_to_be_correct():
    # Scope note, asserted so the closure is not over-read. The av40 size gate writes
    # `overrides[k] = v` DIRECTLY, after this filter, so a gate-written key reaches the estimator
    # whether or not it is in `_PASSTHROUGH`. It is listed so an explicit `hp` value can OUTRANK
    # the gate — a different contract, covered by tests/test_tboost_fidelity_gate.py.
    for k in ("prune_fold_fidelity", "prune_keep_budget"):
        assert k in _PASSTHROUGH, (
            f"{k} is gate-written; it is listed so explicit hp can outrank the gate"
        )
