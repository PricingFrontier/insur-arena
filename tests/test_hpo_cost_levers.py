"""Search-cost levers (SPEC §5.5, t-boost design/tuning-speed-roadmap.md):
A3 selection subsampling, cost-aware candidate evaluation order, and the nominal-order
budget curve that keeps its meaning under both.
"""

import unittest.mock as mock

import numpy as np
import pytest

from insurarena.config import Config
from insurarena.datasets.registry import build_loader
from insurarena.eval import hpo
from insurarena.eval.runner import _hpo_config_proto, _proto
from insurarena.eval.splitting import selection_subsample


def _bundle(n=1200, name="synthetic_freq"):
    return build_loader(name, subsample=n).load()


# --- A3: selection subsampling -------------------------------------------------------------------


def test_subsample_is_a_no_op_at_or_below_the_cap():
    b = _bundle(500)
    idx = np.arange(len(b.X))
    assert selection_subsample(b, idx, None, 7) is idx        # off
    assert selection_subsample(b, idx, 500, 7) is idx         # exactly at the cap
    assert selection_subsample(b, idx, 9999, 7) is idx        # above


def test_subsample_caps_rows_and_is_deterministic():
    b = _bundle(1200)
    idx = np.arange(len(b.X))
    a = selection_subsample(b, idx, 400, 7)
    c = selection_subsample(b, idx, 400, 7)
    assert np.array_equal(a, c)                               # deterministic in (idx, cap, seed)
    assert not np.array_equal(a, selection_subsample(b, idx, 400, 8))  # and seed actually moves it
    assert len(a) <= 1.25 * 400                               # stratum rounding may overshoot mildly
    assert np.array_equal(a, np.sort(a)) and len(set(a.tolist())) == len(a)
    assert set(a.tolist()) <= set(idx.tolist())


def test_subsample_preserves_the_positive_event_share():
    b = _bundle(2000)
    idx = np.arange(len(b.X))
    y = np.asarray(b.y)
    full = float((y > 0).mean())
    sub = selection_subsample(b, idx, 500, 3)
    got = float((y[sub] > 0).mean())
    assert abs(got - full) < 0.05, f"event share drifted {full:.3f} -> {got:.3f}"


def test_subsample_never_splits_a_group():
    b = _bundle(1500)
    idx = np.arange(len(b.X))
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 150, size=len(idx))              # ~10 rows per entity
    b.group_ids = groups
    b.meta = {**b.meta, "stratify_on": "claim_occurrence"}
    sub = selection_subsample(b, idx, 500, 11)
    kept = set(groups[sub].tolist())
    for g in kept:
        rows_in_g = set(np.flatnonzero(groups == g).tolist())
        assert rows_in_g <= set(sub.tolist()), f"group {g} was split across the carve"


def test_selection_max_rows_parses_and_keys_only_tuned_cells(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text(
        "datasets: [synthetic_freq]\nmodels: [{name: lightgbm, variants: [tuned]}]\n"
        "hpo: {n_configs: 4, selection_max_rows: 200000, racing_se: 2.0}\n"
    )
    cfg = Config.from_yaml(str(y))
    assert cfg.selection_max_rows == 200000 and cfg.racing_se == 2.0

    b = _bundle(300)
    tuned = _proto(cfg, b, "lightgbm", "tuned", {})
    default = _proto(cfg, b, "lightgbm", "default", {})
    cand = _hpo_config_proto(cfg, b, "lightgbm", {})
    assert tuned["selection_max_rows"] == 200000            # selection-affecting -> keyed
    assert cand["selection_max_rows"] == 200000             # candidate scores come from the sample
    assert "selection_max_rows" not in default              # deploy fits never see the cap

    y.write_text(
        "datasets: [synthetic_freq]\nmodels: [{name: lightgbm, variants: [tuned]}]\n"
        "hpo: {n_configs: 4}\n"
    )
    off = Config.from_yaml(str(y))
    assert off.selection_max_rows is None
    assert "selection_max_rows" not in _proto(off, b, "lightgbm", "tuned", {})  # off -> unkeyed


# --- cost-aware evaluation order -----------------------------------------------------------------


def test_evaluation_order_is_nominal_without_a_cost_hint():
    configs = [{"seed": i, "learning_rate": lr} for i, lr in enumerate([0.05, 0.01, 0.1, 0.05])]
    assert hpo._evaluation_order("lightgbm", configs) == list(enumerate(configs))


def test_evaluation_order_is_cheap_first_and_keeps_nominal_indices():
    pytest.importorskip("t_boost")
    configs = [
        {"seed": 0, "learning_rate": 0.05},   # anchor: always first
        {"seed": 1, "learning_rate": 0.01},   # most expensive
        {"seed": 2, "learning_rate": 0.10},   # cheapest
        {"seed": 3, "learning_rate": 0.05},
    ]
    order = hpo._evaluation_order("t-boost", configs)
    assert order[0][0] == 0, "the default anchor must stay first (ship-gate reference/incumbent)"
    assert [i for i, _ in order] == [0, 2, 3, 1], "expected cheap-first after the anchor"
    # the permutation is a permutation: same configs, same nominal index attached to each
    assert sorted(i for i, _ in order) == [0, 1, 2, 3]
    for i, hp in order:
        assert hp is configs[i], "a config must keep the index that seeds and caches it"


def test_cost_hint_ranks_the_learning_rate_grid():
    pytest.importorskip("t_boost")
    from insurarena.models.tboost_model import TBoost

    c = {lr: TBoost.hp_cost_hint({"learning_rate": lr, "max_interaction_order": 3})
         for lr in (0.1, 0.05, 0.01)}
    assert c[0.01] > c[0.05] > c[0.1], f"cost must rise as lr falls, got {c}"


def test_bad_cost_hint_falls_back_to_nominal_order():
    configs = [{"seed": i} for i in range(4)]

    class Boom:
        @staticmethod
        def hp_cost_hint(hp):
            raise RuntimeError("bad hint")

    with mock.patch.object(hpo, "model_class", lambda name: Boom):
        assert hpo._evaluation_order("whatever", configs) == list(enumerate(configs))


# --- the budget curve survives both --------------------------------------------------------------


def test_budget_curve_is_rebuilt_over_nominal_order():
    # Evaluated cheapest-first (2, 3, 1) but the curve must read as nominal 0,1,2,3.
    evals = [
        {"config_idx": 0, "score": 1.00},
        {"config_idx": 2, "score": 0.80},
        {"config_idx": 3, "score": 0.90},
        {"config_idx": 1, "score": 0.70},
    ]
    traj = hpo._nominal_trajectory(evals, 4, lower_better=True)
    assert traj == [1.00, 0.70, 0.70, 0.70]


def test_budget_curve_carries_the_incumbent_through_raced_candidates():
    evals = [
        {"config_idx": 0, "score": 1.00},
        {"config_idx": 1, "score": None},          # raced: known worse than the incumbent
        {"config_idx": 2, "score": 0.50},
    ]
    assert hpo._nominal_trajectory(evals, 3, lower_better=True) == [1.00, 1.00, 0.50]


def test_cost_ordering_selects_the_same_winner_as_nominal_order():
    # The whole point: order changes cost, not the answer. Same faked scores, both orders.
    # candidate 2 wins by far more than one paired SE, so the ship-gate actually ships it
    fold_map = {0: [1.00, 1.01, 0.99], 1: [1.20, 1.20, 1.20], 2: [0.50, 0.51, 0.49],
                3: [1.10, 1.10, 1.10], 4: [1.05, 1.05, 1.05]}
    lrs = {0: 0.05, 1: 0.01, 2: 0.01, 3: 0.10, 4: 0.05}

    def fake_oof(model_name, otr, hp, k, seed, repeats=1, racing=None, **_kw):
        folds = fold_map[int(hp["seed"])]
        if racing is not None:
            if folds[0] - racing["inc_fold1"] > racing["mult"] * racing["sd"]:
                return None, 0, [folds[0]], True
        return float(np.mean(folds)), 5, list(folds), False

    def fake_configs(model_name, n, hp_override=None):
        return [{"seed": i, "learning_rate": lrs[i]} for i in range(5)]

    def run(order_fn):
        with mock.patch.object(hpo, "_oof_score", fake_oof), \
             mock.patch.object(hpo, "static_configs", fake_configs), \
             mock.patch.object(hpo, "_evaluation_order", order_fn), \
             mock.patch.object(hpo, "_model_selection_hp", lambda name: {}):
            return hpo._select_config_static("lightgbm", _bundle(400), 5, 3, 123, racing_se=2.0)

    nominal = run(lambda name, cfgs: list(enumerate(cfgs)))
    # anchor first, then cheap-first by lr: 0.10 (idx 3), 0.05 (idx 4), then the two lr=0.01
    cheap_first = run(lambda name, cfgs: [(i, cfgs[i]) for i in (0, 3, 4, 1, 2)])
    assert nominal[0] == cheap_first[0] == {"seed": 2, "learning_rate": 0.01}
    # ...and the reported budget curve is identical despite the different evaluation order
    assert nominal[2] == cheap_first[2]


# --- A2 safety gate: racing is unsound on tune splits the protocol calls too noisy ---------------


def _cfg(tmp_path, extra: str):
    y = tmp_path / "c.yaml"
    y.write_text(
        "datasets: [synthetic_freq]\nmodels: [{name: lightgbm, variants: [tuned]}]\n"
        f"hpo: {{n_configs: 4, {extra}}}\n"
    )
    return Config.from_yaml(str(y))


def test_racing_min_rows_suppresses_racing_below_the_floor(tmp_path):
    from insurarena.eval.runner import _racing_se_for

    cfg = _cfg(tmp_path, "racing_se: 2.0, racing_min_rows: 45000")
    assert cfg.racing_min_rows == 45000
    small, big = _bundle(2182), _bundle(50000)
    assert _racing_se_for(cfg, small) is None, "racing must be OFF below the floor"
    assert _racing_se_for(cfg, big) == 2.0, "racing must stay ON above the floor"


def test_racing_gate_rekeys_only_the_datasets_it_changes(tmp_path):
    # The whole point of keying the EFFECTIVE threshold: adding the gate must invalidate the small
    # datasets (whose selection changes) while leaving large ones cache-valid.
    from insurarena.eval.runner import _proto

    ungated = _cfg(tmp_path, "racing_se: 2.0")
    gated = _cfg(tmp_path, "racing_se: 2.0, racing_min_rows: 45000")
    small, big = _bundle(2182), _bundle(50000)

    assert _proto(ungated, big, "lightgbm", "tuned", {}) == _proto(gated, big, "lightgbm", "tuned", {}), \
        "a dataset above the floor must keep its key (campaign stays cached)"
    assert _proto(ungated, small, "lightgbm", "tuned", {}) != _proto(gated, small, "lightgbm", "tuned", {}), \
        "a dataset below the floor must re-key (its raced selection must not be reused)"
    # and the suppressed key is exactly the no-racing key
    no_racing = _cfg(tmp_path, "n_configs: 4")
    assert "racing_se" not in _proto(gated, small, "lightgbm", "tuned", {})
    assert _proto(gated, small, "lightgbm", "tuned", {}) == _proto(no_racing, small, "lightgbm", "tuned", {})


def test_racing_gate_is_inert_when_unset(tmp_path):
    from insurarena.eval.runner import _racing_se_for

    cfg = _cfg(tmp_path, "racing_se: 2.0")          # no racing_min_rows -> current behaviour
    assert cfg.racing_min_rows is None
    assert _racing_se_for(cfg, _bundle(500)) == 2.0
    off = _cfg(tmp_path, "racing_min_rows: 45000")  # gate without racing_se -> still no racing
    assert _racing_se_for(off, _bundle(50000)) is None
