"""av45 size-gated FOLD FIDELITY + KEEP BUDGET: the per-dataset `ffull` config, folded into the model.

Until av44 the five sub-2500 datasets got `prune_fold_fidelity=True` and an unbounded
`prune_keep_budget` from a SECOND campaign config (`configs/tuned_triboost_small_ffull.yaml`), so
tri-boost was the only model on the board resolving from more than one config while every
comparator ran exactly one. Those knobs are chosen by DATASET SIZE, which the adapter can read for
itself, so they are now a fit-time gate and the config is deleted.

What these tests defend, in the order the fairness argument needs them:

  * the gate fires on n_rows < 2500 and nowhere else, at the SAME threshold as the av38 bags gate
    (the arena's own `split.small_threshold`);
  * it reads DATASET-level n from the bundle meta, not fit-slice rows — so a selection fold on
    ~4n/9 rows and the deploy fit on 2n/3 reach the same verdict, which is the property that keeps
    the ranked model and the shipped model in one regime;
  * explicit `hp` outranks it in BOTH directions, the same precedence every other tri-boost gate
    uses, so an ablation is one config line;
  * the decision is recorded on `fidelity_gate_` whether or not it fires, and reaches the estimator.
"""
import numpy as np
import pandas as pd
import pytest

from insurarena.datasets.base import DatasetBundle
from insurarena.models.tboost_model import (
    _BAGS_SMALL_THRESHOLD,
    _FIDELITY_KEEP_BUDGET,
    TBoost,
)
from insurarena.tasks import FREQ_SPEC

#: the five catalogue sets the deleted config listed, with their real row counts. No catalogue set
#: sits in 2500 <= n < 3000, so this threshold and "n < 2500" fire on exactly these five.
_GATED_SETS = {"credit_g": 1000, "brautocoll": 1314, "autobi_us": 1340,
               "swautoins": 2182, "swautoins_pp": 2182}
#: the smallest sets on the other side of the line, for the negative half of the same rule
_UNGATED_SETS = {"freclaimdam": 3948, "autoclaims_us": 6773, "tic2000": 9822, "eusavingul": 21921}


def _bundle(n_rows: int, *, fit_rows: int | None = None, seed: int = 0) -> DatasetBundle:
    """A small FREQ bundle whose `meta["n_rows"]` is the DATASET size, like a real loader's.

    `fit_rows` lets a test hand the adapter a slice SMALLER than the dataset — which is what an
    inner-CV fold is — while `meta` keeps saying how big the dataset is.
    """
    n = fit_rows if fit_rows is not None else n_rows
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=n),
                      "cat": [f"l{i % 12}" for i in range(n)]})
    w = np.ones(n)
    return DatasetBundle(name="fidelity_probe", X=X, y=rng.poisson(0.3, n).astype(float),
                         exposure=w, sample_weight=w, task=FREQ_SPEC, cat_features=["cat"],
                         meta={"n_rows": n_rows})


def _fit(bundle: DatasetBundle, **hp) -> TBoost:
    return TBoost(hp={"budget": 20, "n_bags": 2, **hp}).fit(bundle)


# ------------------------------------------------------------------- the size rule and its line


@pytest.mark.parametrize("dataset,n_rows", sorted(_GATED_SETS.items()))
def test_the_gate_fires_on_every_set_the_deleted_config_listed(dataset, n_rows):
    g = _fit(_bundle(n_rows)).fidelity_gate_
    assert g["fires"] is True, dataset
    assert g["applied"] == {"prune_fold_fidelity": True,
                            "prune_keep_budget": _FIDELITY_KEEP_BUDGET}


@pytest.mark.parametrize("dataset,n_rows", sorted(_UNGATED_SETS.items()))
def test_the_gate_declines_on_the_sets_it_never_covered(dataset, n_rows):
    g = _fit(_bundle(n_rows)).fidelity_gate_
    assert g["fires"] is False, dataset
    assert g["applied"] == {}


def test_the_boundary_is_the_arena_small_threshold_and_the_bags_gates():
    # One threshold, shared with the av38 bags gate on purpose: the two gates describe the same
    # "this dataset is small" regime and must never disagree about which sets are in it.
    assert _BAGS_SMALL_THRESHOLD == 2500
    below = _fit(_bundle(_BAGS_SMALL_THRESHOLD - 1))
    at = _fit(_bundle(_BAGS_SMALL_THRESHOLD))
    assert below.fidelity_gate_["fires"] is True
    assert at.fidelity_gate_["fires"] is False
    assert below.fidelity_gate_["fires"] is below.bags_gate_["fires"]
    assert at.fidelity_gate_["fires"] is at.bags_gate_["fires"]


def test_the_verdict_is_dataset_level_not_fit_slice_level():
    # The property the av38 bags-gate note argues for at length: a selection fold sees ~4n/9 rows
    # and the deploy fit sees 2n/3, so a SLICE rule would gate the two differently on a set just
    # above the line. `meta["n_rows"]` survives `.subset()`, so both reach the same verdict.
    big_set_small_slice = _bundle(9822, fit_rows=1200)       # tic2000-shaped: a fold, not a set
    assert len(big_set_small_slice.X) < _BAGS_SMALL_THRESHOLD
    assert _fit(big_set_small_slice).fidelity_gate_["fires"] is False

    small_set = _bundle(1000)                                 # credit_g-shaped
    assert _fit(small_set).fidelity_gate_["fires"] is True
    assert _fit(small_set.subset(np.arange(400))).fidelity_gate_["fires"] is True


# --------------------------------------------------------------------------- explicit-hp precedence


def test_explicit_hp_outranks_the_gate_in_both_directions():
    # Same precedence as the rare-floor, bags, cell-refit and depth gates: a config that names the
    # knob gets what it asked for, and the gate still RECORDS that it fired. One config line is
    # therefore a complete ablation of the av45 behavior.
    m = _fit(_bundle(1000), prune_fold_fidelity=False, prune_keep_budget=32)
    assert m.fidelity_gate_["fires"] is True                  # measured and reported ...
    assert m.fidelity_gate_["applied"] == {}                  # ... but not applied
    assert m.model.get_params()["prune_fold_fidelity"] is False
    assert m.model.get_params()["prune_keep_budget"] == 32

    # ...and one knob at a time: the gate still supplies the other.
    half = _fit(_bundle(1000), prune_keep_budget=160)
    assert half.fidelity_gate_["applied"] == {"prune_fold_fidelity": True}
    assert half.model.get_params()["prune_keep_budget"] == 160
    assert half.model.get_params()["prune_fold_fidelity"] is True

    # An ungated set can still opt IN from a config, which is how the battery arms were measured.
    opt_in = _fit(_bundle(9822), prune_fold_fidelity=True,
                  prune_keep_budget=_FIDELITY_KEEP_BUDGET)
    assert opt_in.fidelity_gate_["fires"] is False
    assert opt_in.model.get_params()["prune_fold_fidelity"] is True


# ------------------------------------------------------------------------------ adapter wiring


def test_the_gated_values_reach_the_estimator():
    m = _fit(_bundle(1000))
    params = m.model.get_params()
    assert params["prune_fold_fidelity"] is True
    assert params["prune_keep_budget"] == _FIDELITY_KEEP_BUDGET == 1_000_000


def test_an_ungated_fit_keeps_the_library_defaults():
    m = _fit(_bundle(9822))
    params = m.model.get_params()
    assert params["prune_fold_fidelity"] is False
    assert params["prune_keep_budget"] != _FIDELITY_KEEP_BUDGET


def test_the_decision_is_recorded_and_deterministic():
    b = _bundle(1000)
    a, c = _fit(b).fidelity_gate_, _fit(b).fidelity_gate_
    assert a == c
    assert a["n_rows"] == 1000
    assert set(a) == {"fires", "n_rows", "task", "prune_fold_fidelity", "prune_keep_budget",
                      "applied"}


def test_the_gate_never_fires_on_multiclf():
    # tri-boost REFUSES `prune_fold_fidelity` on the K>=3 prune path (one train/select split, no
    # per-fold refits to pin a bank into) and that path takes no keep-budget argument at all, so a
    # gate that fired there would hand the fit knobs it cannot honor — and, on the fidelity knob,
    # would RAISE. Inert on today's catalog (all five gated sets are CLF/SEV/FREQ/PP), asserted so
    # a small K>=3 dataset joining the catalog cannot crash a campaign.
    from insurarena.tasks import MULTICLF_SPEC, TaskType

    rng = np.random.default_rng(3)
    n = 900
    X = pd.DataFrame({"x1": rng.normal(size=n), "cat": [f"l{i % 8}" for i in range(n)]})
    w = np.ones(n)
    b = DatasetBundle(name="multiclf_probe", X=X, y=rng.integers(0, 3, n).astype(float),
                      exposure=w, sample_weight=w, task=MULTICLF_SPEC,
                      cat_features=["cat"], meta={"n_rows": n})
    m = _fit(b)
    assert m.fidelity_gate_["fires"] is False
    assert m.fidelity_gate_["task"] == TaskType.MULTICLF.value
    assert m.fidelity_gate_["n_rows"] < _BAGS_SMALL_THRESHOLD    # size alone WOULD have fired
    assert m.bags_gate_["fires"] is True                          # ...as the bags gate shows


def test_the_gate_also_applies_to_selection_fits():
    # The deleted config's `hp:` block reached EVERY candidate, scout evals included, so the gate
    # must too — otherwise the small sets would be ranked in one regime and shipped in another.
    # (The scout overlay turns pruning off, which makes the knobs inert rather than absent; what
    # matters is that the adapter does not condition the gate on `val`.)
    overlay = TBoost.selection_hp()
    m = TBoost(hp={"budget": 20, **overlay}).fit(_bundle(1000), _bundle(1000, seed=1))
    assert m.fidelity_gate_["fires"] is True
    assert m.fidelity_gate_["applied"] == {"prune_fold_fidelity": True,
                                           "prune_keep_budget": _FIDELITY_KEEP_BUDGET}
