"""hsv 18 (2026-08-23) — the tuned-selection upgrade, three parts:

  (a) `max_bin` becomes a searchable dimension of tri-boost's static grid;
  (b) a deterministic coordinate-probe block replaces the head of the random basis;
  (c) the protocol-v3 pooled finale reaches large sets, but only on a near-tie gate.

The house bar these tests defend: the selection must never ship a champion the pooled evidence
does not support. Bias toward the anchor is acceptable; harmful ships are not.
"""
import pathlib
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from insurarena.eval import hpo, runner
from insurarena.models.registry import model_class
from insurarena.models.tboost_model import (
    _COORDINATE_PROBES,
    _STATIC_TUNE_GRID,
    _is_structural_hp,
    _probe_moves,
    TBoost,
)

# The budget THE campaign config runs at (configs/t-boost.yaml — since av45 there is only
# one; the per-dataset `ffull` and eusavingul configs are deleted). 25 through hsv 21; 27 at
# hsv 22; 29 at hsv 23, each of those buying a probe by raising the budget because the probe block
# sat ON the joint-coverage floor; and back to 25 at hsv 24, which is the FAIRNESS number — the
# budget lightgbm, xgboost, catboost and ebm each race. Kept as one name so a future budget move is
# a one-line edit here and the tests that pin the campaign's own balance/coverage cannot silently
# drift off it.
_N_CAMPAIGN = 25
#: every other tunable model's budget. tri's must equal it — see
#: `test_tri_races_the_same_budget_as_every_comparator`.
_N_COMPARATOR = 25


def _fake_bundle(n_rows: int = 200_000):
    return SimpleNamespace(
        task=SimpleNamespace(primary_metric="poisson_deviance", tweedie_power=None),
        X=list(range(n_rows)),
    )


# --- (a) max_bin joins the searchable grid -------------------------------------------------------


def test_max_bin_is_searchable_and_leads_with_the_recipe_default():
    # The forensics finding: max_bin was in _PASSTHROUGH (live, reachable by an hp: block) but not
    # in the tuned grid, so no tuned cell could ever move it off the recipe's 254.
    from insurarena.models.tboost_model import _PASSTHROUGH

    assert "max_bin" in _PASSTHROUGH
    assert _STATIC_TUNE_GRID["max_bin"] == [254, 128, 64, 32]
    assert _STATIC_TUNE_GRID["max_bin"][0] == 254, "the recipe default must lead the list"

    # ...and the two references the ship gate depends on stay on 254 by NOT carrying the key:
    assert "max_bin" not in TBoost.default_hp()       # candidate 0, the gate's anchor
    assert "max_bin" not in TBoost.tuning_base_hp()   # the tuned family's base recipe


def test_max_bin_marginal_is_balanced_across_the_sampled_basis():
    sampled = hpo.static_configs("t-boost", _N_CAMPAIGN)[1 + len(_COORDINATE_PROBES):]
    counts = {v: sum(1 for h in sampled if h["max_bin"] == v)
              for v in _STATIC_TUNE_GRID["max_bin"]}
    assert sorted(counts.values()) == [3, 4, 4, 4], counts       # 15 slots over 4 values
    # 15 at hsv 19 (n=25, 6 probes -> 19), hsv 20 (16), hsv 21 (15), hsv 22 (n=27, 12 probes -> 15)
    # and hsv 24 (n=25, 10 probes -> 15 again). Every probe through hsv 21 DISPLACED a basis draw;
    # hsv 21 hit the floor of 15, hsv 22/23 raised n_configs with the block instead, and hsv 24
    # retired four probes to get the BUDGET back to comparator parity without touching the tail.
    # The cycle stays balanced to within one slot, which is the guarantee `sample_hp_set` actually
    # makes at a count that is not a multiple of 4.


# --- (b) the coordinate-probe block --------------------------------------------------------------


def test_probe_block_is_deterministic_and_moves_exactly_one_knob():
    base = TBoost.tuning_base_hp()
    probes = TBoost.coordinate_probe_hp_set()

    assert probes == TBoost.coordinate_probe_hp_set()          # seedless: same list every call
    assert len(probes) == len(_COORDINATE_PROBES) == 10
    for entry, hp in zip(_COORDINATE_PROBES, probes):
        moves = _probe_moves(entry)
        moved = {k for k in set(hp) | set(base) if hp.get(k, object()) != base.get(k, object())}
        assert moved == {name for name, _ in moves}, f"probe {entry} moved {moved}"
        for name, value in moves:
            assert hp[name] == value

    # every curated direction has campaign evidence behind it; none is a no-op against the base
    assert [p for p in _COORDINATE_PROBES] == [
        ("max_interaction_order", 1),
        ("max_interaction_order", 2),
        ("max_bin", 64),
        ("max_bin", 128),
        ("learning_rate", 0.02),
        ("interaction_gain_hurdle", 5.0),
        ("max_depth", 5),
        ("max_depth", 6),
        # hsv 22 (av43): the two joint shapes a 9-split forensic measured. hsv 24 retired the four
        # probes that used to sit between and after them — `cell_refit_base` 20000.0 (inert at
        # scout fidelity by construction), `max_depth` 4 (outer-test rung evidence that never led
        # an inner-CV cell) and `random_strength` 1.0/4.0 (no smoked cell ships one) — which is
        # what puts the budget back on comparator parity.
        (("max_depth", 8), ("interaction_gain_hurdle", 0.0)),
        (("max_depth", 5), ("max_interaction_order", 4)),
    ]


def test_probe_block_occupies_the_indices_after_the_anchor():
    configs = hpo.static_configs("t-boost", _N_CAMPAIGN)
    probes = TBoost.coordinate_probe_hp_set()

    assert configs[0] == TBoost.default_hp()                   # anchor untouched
    for j, probe in enumerate(probes, start=1):
        assert configs[j] == {**probe, "seed": j}                # seeded by NOMINAL index


def test_probes_replace_basis_draws_rather_than_extending_the_candidate_count():
    # The headroom scan's warning: naive space enrichment backfires through the winner's curse
    # (it fired 2 harmful ships on enriched champions). Holding len == n_configs + 1 holds the
    # max-of-N the ship gate has to defend against.
    for n in (1, 4, 6, 7, 25, 27, 40):
        assert len(hpo.static_configs("t-boost", n)) == n + 1, n

    # the sampled tail is exactly the sampler's own prefix, so its balance guarantees survive
    n_probes = len(_COORDINATE_PROBES)
    sampled = hpo.static_configs("t-boost", _N_CAMPAIGN)[1 + n_probes:]
    raw = TBoost.sample_hp_set(_N_CAMPAIGN - n_probes, hpo._CONFIG_SEED)
    assert len(sampled) == len(raw) == 15
    assert all(s == {**r, "seed": i} for i, (s, r) in enumerate(zip(sampled, raw), start=1 + n_probes))


def test_sampled_tail_still_covers_every_joint_order_hurdle_combination():
    # sample_hp_set's joint cycle needs n >= 15 to see all 15 (order, hurdle) pairs. hsv 21 hit
    # that floor EXACTLY at n_configs=25 (25 - 10 probes = 15) and recorded that the budget line
    # was there: an eleventh probe would have cost joint coverage, not a basis draw.
    #
    # hsv 22 and hsv 23 then bought their probes by RAISING n_configs (25 -> 27 -> 29) rather than
    # paying that cost. hsv 24 (av45) unwinds the budget instead of the coverage: four probes go,
    # the block is back to 10, and the tail is 25 - 10 = 15 — the floor again, at the comparators'
    # budget. This is the invariant to hold at every future hsv, and it now has a second one beside
    # it (`test_tri_races_the_same_budget_as_every_comparator`):
    # n_configs - len(_COORDINATE_PROBES) >= 15.
    assert _N_CAMPAIGN - len(_COORDINATE_PROBES) == 15
    sampled = hpo.static_configs("t-boost", _N_CAMPAIGN)[1 + len(_COORDINATE_PROBES):]
    joint = {(h["max_interaction_order"], h["interaction_gain_hurdle"]) for h in sampled}
    assert len(joint) == 15


def test_candidate_prefix_is_stable_as_n_configs_grows():
    # The whole cache/append contract: candidate i depends only on i, never on n.
    base = hpo.static_configs("t-boost", _N_CAMPAIGN)
    for n in (27, 29, 32, 40, 60):
        assert hpo.static_configs("t-boost", n)[: len(base)] == base, n
    # ...and shrinking truncates rather than reshuffling
    assert hpo.static_configs("t-boost", 12) == base[:13]


def test_no_candidate_is_a_duplicate_of_another():
    import json

    configs = hpo.static_configs("t-boost", _N_CAMPAIGN)
    keys = [json.dumps({k: v for k, v in h.items() if k != "seed"}, sort_keys=True, default=str)
            for h in configs]
    assert len(set(keys)) == len(keys), "a duplicated candidate would burn an eval for nothing"


def test_models_without_the_hook_are_byte_untouched():
    for name in ("lightgbm", "xgboost", "catboost", "ebm", "glm"):
        assert hpo.coordinate_probes(name, 25) == []
        cfgs = hpo.static_configs(name, 25)
        assert len(cfgs) == 26
        assert all(c["seed"] == i for i, c in enumerate(cfgs) if i)


def test_probes_are_exempt_from_racing_like_the_anchor():
    # A racing discard on a noisy fold 1 would deny a probe the clean look at the gate that is the
    # entire point of the block — the same burial the joint basis was already inflicting.
    n_probes = len(_COORDINATE_PROBES)
    raced_indices, racing_offered = [], []

    def fake_oof(model_name, otr, hp, k, seed, repeats=1, racing=None, **_kw):
        idx = int(hp["seed"]) if "seed" in hp else 0
        racing_offered.append((idx, racing is not None))
        folds = [1.0 + 0.5 * (idx > 0), 1.0, 1.0]      # every non-anchor is far worse on fold 1
        if racing is not None and folds[0] - racing["inc_fold1"] > racing["mult"] * racing["sd"]:
            raced_indices.append(idx)
            return None, 0, [folds[0]], True
        return float(np.mean(folds)), 5, list(folds), False

    with mock.patch.object(hpo, "_oof_score", fake_oof), \
         mock.patch.object(hpo, "_model_selection_hp", lambda name: {}), \
         mock.patch.object(hpo, "_model_selection_finale_hp", lambda name: {}):
        hpo._select_config_static("t-boost", _fake_bundle(), _N_CAMPAIGN, 3, 123, racing_se=2.0)

    offered = dict(racing_offered)
    assert offered[0] is False, "the anchor is never raced"
    for j in range(1, n_probes + 1):
        assert offered[j] is False, f"probe {j} must not be raced"
    assert any(offered[i] for i in range(n_probes + 1, _N_CAMPAIGN + 1)), \
        "basis candidates still race"
    assert all(i > n_probes for i in raced_indices)


# --- (c) the adaptive pooled finale on large sets ------------------------------------------------


def test_pooled_finale_wanted_band():
    assert hpo._ADAPTIVE_POOL_SE_MULT == 2.0

    def gate(margin, se):
        return {"gate_margin": margin, "gate_se": se}

    # "always" (small tune sets, protocol v3 as measured at hsv 16) ignores the margin entirely
    assert hpo._pooled_finale_wanted(gate(10.0, 0.001), "always") is True

    # adaptive: two-sided +/-2 SE. norauto's real numbers sit at 0.88 SE -> pooled.
    assert hpo._pooled_finale_wanted(gate(5.81e-5, 6.57e-5), "adaptive") is True
    assert hpo._pooled_finale_wanted(gate(-1.5, 1.0), "adaptive") is True     # near-tie, other side
    assert hpo._pooled_finale_wanted(gate(1.99, 1.0), "adaptive") is True
    assert hpo._pooled_finale_wanted(gate(2.01, 1.0), "adaptive") is False    # decisive champion
    assert hpo._pooled_finale_wanted(gate(-2.01, 1.0), "adaptive") is False   # decisive default

    # unresolvable evidence pools — the conservative side (more evidence for a gate that has none)
    assert hpo._pooled_finale_wanted({}, "adaptive") is True
    assert hpo._pooled_finale_wanted(gate(np.nan, 1.0), "adaptive") is True


def _pooled_run(primary_folds, extra_folds, mode, n_configs=1):
    """Drive t-boost's static search with faked scores; returns (hp, selection, evals, n_built)."""
    built = []

    def fake_oof(model_name, otr, hp, k, seed, repeats=1, **_kw):
        champion = "seed" in hp
        folds = (extra_folds if getattr(otr, "_extra", False) else primary_folds)[bool(champion)]
        return float(np.mean(folds)), 9 if champion else 7, list(folds), False

    def make_extra(tag):
        """A thunk, as the runner now supplies: `built` records whether it was ever called."""
        def thunk():
            b = _fake_bundle()
            b._extra = True
            built.append(tag)
            return b
        return thunk

    extras = [make_extra("a"), make_extra("b")]

    with mock.patch.object(hpo, "_oof_score", fake_oof):
        hp, best_iter, _, evals, sel = hpo.select_config(
            "t-boost", _fake_bundle(), n_configs, 3, 123,
            extra_finale_otrs=extras, pooled_finale_mode=mode,
        )
    return hp, sel, evals, len(built)


def test_adaptive_finale_pools_on_a_near_tie_and_reverts_the_champion():
    # norauto's shape: the champion wins the primary tune split by LESS than 1 SE (so today's gate
    # already ships the default) but well within 2 SE — so the extra evidence is worth buying.
    # Here the pooled evidence says the champion is genuinely worse: the default must ship.
    hp, sel, evals, n_built = _pooled_run(
        primary_folds={False: [1.00, 1.02, 0.98], True: [0.99, 1.03, 0.97]},
        extra_folds={False: [1.00, 1.00, 1.00], True: [1.05, 1.05, 1.05]},
        mode="adaptive",
    )
    assert sel["rule"] == "static_ship_gate_1se_msplit"
    assert sel["pooled_finale"] == "adaptive"
    assert sel["n_finale_splits"] == 3
    assert sel["shipped"] == "default"
    assert sel["reason"] == "within_1se_of_default_pooled"
    assert n_built == 2, "the extra outer-trains are materialized exactly once each"
    assert sum(1 for e in evals if (e.get("param") or "").startswith("pool_")) == 4
    # the superseded single-split verdict stays on the record so the change is auditable
    assert sel["single_split_shipped"] == "default"
    assert 0 < sel["single_split_gate_margin"] < sel["single_split_gate_se"]
    assert hp == TBoost.default_hp()


def test_adaptive_finale_can_promote_a_champion_the_single_split_gate_lost():
    # The other half of the norauto lesson: the gate's refusals are not automatically right either.
    # A champion just under 1 SE on one split, confirmed by two more, SHIPS on the pooled evidence.
    hp, sel, evals, _ = _pooled_run(
        primary_folds={False: [1.00, 1.02, 0.98], True: [0.99, 1.03, 0.97]},
        extra_folds={False: [1.00, 1.00, 1.00], True: [0.98, 0.98, 0.98]},
        mode="adaptive",
    )
    assert sel["single_split_shipped"] == "default"
    assert sel["shipped"] == "champion"
    assert sel["reason"] == "beats_default_by_1se_pooled"
    assert sel["n_paired_folds"] == 9
    assert hp != TBoost.default_hp() and "seed" in hp


def test_adaptive_finale_declines_a_decisive_single_split_gate():
    # The 2026-08-04 revert's finding, preserved: where the single-split gate is decisive on a
    # large set it was CORRECT, so no extra evaluations are spent — and no bundle is even built.
    hp, sel, evals, n_built = _pooled_run(
        primary_folds={False: [1.00, 1.02, 0.98], True: [0.90, 0.93, 0.87]},
        extra_folds={False: [1.00, 1.00, 1.00], True: [1.05, 1.05, 1.05]},
        mode="adaptive",
    )
    assert sel["rule"] == "static_ship_gate_1se"
    assert sel["pooled_finale"] == "skipped_margin_beyond_2se"
    assert sel["n_finale_splits"] == 1
    assert sel["shipped"] == "champion"
    assert n_built == 0, "a decisive cell must not materialize the extra outer-trains at all"
    assert not any((e.get("param") or "").startswith("pool_") for e in evals)


def test_always_mode_pools_even_on_a_decisive_margin():
    # Small tune sets keep hsv 16 behavior exactly: pooling is unconditional there, because on a
    # small split the whole finale can be split-specific bias end to end (freclaimdam).
    _, sel, _, n_built = _pooled_run(
        primary_folds={False: [1.00, 1.02, 0.98], True: [0.90, 0.93, 0.87]},
        extra_folds={False: [1.00, 1.00, 1.00], True: [1.05, 1.05, 1.05]},
        mode="always",
    )
    assert sel["rule"] == "static_ship_gate_1se_msplit"
    assert sel["pooled_finale"] == "always"
    assert sel["shipped"] == "default"
    assert n_built == 2


# --- the runner side: which mode each dataset size gets ------------------------------------------


def _runner_cfg(**over):
    ns = dict(
        split={"small": {"folds": 3, "repeats": 1}},
        n_configs=25,
        inner_folds=3,
        inner_repeats="auto",
        tune_split_id=0,
        time_budget_s=None,
        fold_workers=1,
        racing_se=None,
        racing_min_rows=45000,
        selection_max_rows=None,
        use_cache=False,
    )
    ns.update(over)
    return SimpleNamespace(**ns)


@pytest.mark.parametrize(
    "n_rows, expect_mode",
    [
        (2_182, "always"),        # swautoins-sized: inner_repeats 5 -> protocol v3 as at hsv 16
        (15_000, "always"),       # inner_repeats 2 -> still the measured small-set regime
        (45_238, "adaptive"),     # ausprivauto+: single pass -> the hsv 18 near-tie extension
        (183_999, "adaptive"),    # norauto, the counterexample that re-opened the extension
    ],
)
def test_runner_picks_the_pooled_finale_mode_by_tune_set_size(n_rows, expect_mode):
    captured = {}

    class _Bundle(SimpleNamespace):
        def subset(self, idx):
            return _fake_bundle(len(idx))

    bundle = _Bundle(
        task=SimpleNamespace(primary_metric="poisson_deviance", tweedie_power=None),
        X=list(range(n_rows)),
        meta={"hash": "H"},
    )
    tr = list(range(int(n_rows * 2 / 3)))
    splits = [(tr, list(range(10))) for _ in range(9)]

    def fake_select(name, otr, *a, **kw):
        captured["mode"] = kw.get("pooled_finale_mode")
        captured["n_extra"] = len(kw.get("extra_finale_otrs") or [])
        return {"budget": 1}, 0, [], [], {"shipped": "default"}

    class _Cache:
        def get(self, sig, model=None):
            return None

        def put(self, sig, payload, model=None):
            pass

    with mock.patch.object(runner, "select_config", fake_select):
        runner._tuned_once(_Cache(), _runner_cfg(), bundle, splits, "ds", "H", "t-boost", {})

    assert captured["mode"] == expect_mode
    assert captured["n_extra"] == 2, "extras are now supplied at every size; the MODE gates them"


def test_runner_supplies_no_extras_for_models_without_a_scout_regime():
    captured = {}

    class _Bundle(SimpleNamespace):
        def subset(self, idx):
            return _fake_bundle(len(idx))

    bundle = _Bundle(
        task=SimpleNamespace(primary_metric="poisson_deviance", tweedie_power=None),
        X=list(range(200_000)),
        meta={"hash": "H"},
    )
    splits = [(list(range(133_333)), list(range(10))) for _ in range(9)]

    def fake_select(name, otr, *a, **kw):
        captured["extras"] = kw.get("extra_finale_otrs")
        return {}, 0, [], [], {"shipped": "default"}

    class _Cache:
        def get(self, sig, model=None):
            return None

        def put(self, sig, payload, model=None):
            pass

    with mock.patch.object(runner, "select_config", fake_select):
        runner._tuned_once(_Cache(), _runner_cfg(), bundle, splits, "ds", "H", "lightgbm", {})

    assert captured["extras"] is None


# --- hsv 19: the volatile-set ship gate ----------------------------------------------------------


def test_ship_gate_se_mult_defaults_to_one_for_every_other_model():
    # The hook is opt-in: any adapter that does not override it keeps the historical 1-SE gate,
    # so its selections — and its cached tuned cells — are byte-untouched by hsv 19.
    for name in ("lightgbm", "xgboost", "catboost", "ebm", "glm"):
        assert model_class(name).ship_gate_se_mult(5) == 1.0
        assert hpo._gate_se_mult(name, 5) == 1.0
        assert hpo._gate_se_mult(name, 1) == 1.0


def test_triboost_gate_bar_scales_with_inner_repeats():
    # inner_repeats == clip(ceil(30000 / n_outer_train), 1, 5): the arena's own under-power flag.
    assert TBoost.ship_gate_se_mult(1) == 1.0        # large sets: unchanged, bar stays 1 SE
    assert TBoost.ship_gate_se_mult(2) == 2.0        # uslapseagent-sized
    assert TBoost.ship_gate_se_mult(5) == 5.0        # swautoins / freclaimdam-sized
    assert hpo._gate_se_mult("t-boost", 5) == 5.0
    assert hpo._gate_se_mult("t-boost", 0) == 1.0     # degenerate input floors at the 1-SE rule


def test_gate_bar_refuses_the_measured_cursed_ships_and_keeps_the_good_one():
    # The six calibration cells from the hsv 18 recompute, as (repeats, margin/SE, outer delta).
    # The bar is `repeats`; assert it splits them exactly the way the deploys did.
    cells = [
        ("freclaimdam",   5, 2.40, -0.02533),
        ("swautoins_pp",  5, 2.71, -0.01057),
        ("swautoins",     5, 1.99, -0.00632),
        ("autoclaims_us", 5, 1.00, +0.00113),
        ("uslapseagent",  2, 7.09, +0.00564),
        ("brvehins1",     1, 9.23, +0.00106),
    ]
    ships = {ds: t > TBoost.ship_gate_se_mult(rep) for ds, rep, t, _ in cells}
    assert ships == {
        "freclaimdam": False, "swautoins_pp": False, "swautoins": False,
        "autoclaims_us": False,                       # a 1.00-SE ship: the gate had no evidence
        "uslapseagent": True, "brvehins1": True,
    }
    # every cell whose champion actually damaged the board is refused...
    assert all(not ships[ds] for ds, _, _, d in cells if d < 0)
    # ...and the tightening is one-directional: it can only turn champions into the anchor.
    assert all(TBoost.ship_gate_se_mult(r) >= 1.0 for r in (1, 2, 3, 4, 5))


def test_gate_se_mult_reaches_the_static_gate_and_is_recorded(monkeypatch):
    # margin/SE == 2.0 exactly: ships at repeats=1 (bar 1.0), refused at repeats=5 (bar 5.0).
    _patched = {0: [1.00, 1.00, 1.00, 1.00], 1: [0.98, 0.98, 0.98, 0.96]}

    def fake_oof(model_name, otr, hp, k, seed, repeats=1, **_kw):
        folds = _patched[1 if "seed" in hp else 0]
        return float(np.mean(folds)), 5, list(folds), False

    monkeypatch.setattr(hpo, "_oof_score", fake_oof)
    seen = {}
    for rep, mult in ((1, 1.0), (5, 5.0)):
        monkeypatch.setattr(TBoost, "ship_gate_se_mult", staticmethod(lambda r, m=mult: m))
        _, _, _, _, sel = hpo.select_config("t-boost", _fake_bundle(), 1, 3, 123, repeats=rep)
        seen[rep] = sel
        assert sel["gate_se_mult"] == mult
    assert seen[1]["shipped"] == "champion"
    assert seen[5]["shipped"] == "default"


def test_gate_se_mult_applies_to_the_pooled_finale_too(monkeypatch):
    # The raised bar must reach the POOLED branch as well as the single-split one — freclaimdam
    # cleared a 21-fold pooled 1-SE gate and still detonated, so that branch is the one that
    # actually mattered on the recompute. Same evidence, two bars, opposite verdicts.
    folds = dict(primary_folds={False: [1.00, 1.02, 0.98], True: [0.992, 1.018, 0.972]},
                 extra_folds={False: [1.00, 1.02, 0.98], True: [0.992, 1.018, 0.972]},
                 mode="always")

    monkeypatch.setattr(TBoost, "ship_gate_se_mult", staticmethod(lambda r: 1.0))
    _, ships, _, _ = _pooled_run(**folds)
    assert ships["rule"] == "static_ship_gate_1se_msplit"
    assert ships["gate_se_mult"] == 1.0
    assert ships["shipped"] == "champion"
    assert ships["reason"] == "beats_default_by_1se_pooled"

    monkeypatch.setattr(TBoost, "ship_gate_se_mult", staticmethod(lambda r: 20.0))
    _, refused, _, _ = _pooled_run(**folds)
    assert refused["gate_se_mult"] == 20.0
    assert refused["gate_margin"] == ships["gate_margin"]      # identical evidence...
    assert refused["shipped"] == "default"                     # ...opposite decision
    assert refused["reason"] == "within_1se_of_default_pooled"


# --- regression: a shipped-default selection must not touch the deploy path ----------------------


@pytest.mark.parametrize("mode,extras", [("always", True), ("adaptive", True), (None, False)])
def test_shipped_default_deploy_is_untouched(monkeypatch, mode, extras):
    """When the gate ships the default, the deploy must be the DEFAULT deploy, bit for bit.

    Verified against the hsv 18 full recompute: all 15 datasets that shipped the default at both
    hsv 17 and hsv 18 have tuned cells identical on every split (max|diff| = 0.000e+00). This test
    locks the mechanism so a future finale/gate rework cannot quietly change it:
      * the returned hp is EXACTLY `default_hp()` — no config seed, no probe key, no finale key;
      * best_iter comes from the default anchor's own evaluation, so for an internally-early-
        stopping adapter (tri-boost: `best_iteration_` is always 0) it is 0, and `fit_one` is
        therefore called with n_estimators=None — the same pure-internal-ES path the `default`
        variant takes.
    """
    def fake_oof(model_name, otr, hp, k, seed, repeats=1, **_kw):
        # champion is strictly better on the mean, but by only ~0.5 SE of the paired diffs
        folds = [0.99, 1.03, 0.97] if "seed" in hp else [1.00, 1.02, 0.98]
        return float(np.mean(folds)), (11 if "seed" in hp else 0), list(folds), False

    monkeypatch.setattr(hpo, "_oof_score", fake_oof)
    kw = {}
    if extras:
        kw = {"extra_finale_otrs": [lambda: _fake_bundle(), lambda: _fake_bundle()],
              "pooled_finale_mode": mode}
    hp, best_iter, _, _, sel = hpo.select_config("t-boost", _fake_bundle(), _N_CAMPAIGN, 3, 123,
                                                 **kw)

    assert sel["shipped"] == "default"
    assert hp == TBoost.default_hp() == {"budget": 4000}
    assert "seed" not in hp and "max_bin" not in hp and "_tuned" not in hp
    assert best_iter == 0, "an internally-early-stopping adapter must not pin n_estimators"


def test_triboost_never_reports_a_best_iteration():
    # The invariant the regression above leans on, asserted against a real fit rather than a mock:
    # tri-boost early-stops internally, so a shipped-default selection can never pin n_estimators.
    import pandas as _pd
    from insurarena.datasets.base import DatasetBundle
    from insurarena.models.registry import build_model
    from insurarena.tasks import FREQ_SPEC

    rng = np.random.default_rng(0)
    n = 400
    X = _pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    exposure = np.ones(n)
    bundle = DatasetBundle(
        name="best_iter_probe", X=X, y=rng.poisson(0.3, n).astype(float),
        exposure=exposure, sample_weight=exposure, task=FREQ_SPEC,
    )
    m = build_model("t-boost", hp={"budget": 30, "n_bags": 1}).fit(bundle, None)
    assert int(getattr(m, "best_iteration_", 0) or 0) == 0


# --- hsv 20 (2026-08-27): the depth dimension, its constant cap, and the structural gate ---------


def test_depth_enters_as_clean_probes_and_never_as_a_grid_dimension():
    # The falsified draft: `max_depth` as a marginal grid key. The campaign-budget smoke found
    # the search DOES reach depth on fremotor_payfreq and that the shipped bundle then measured
    # -0.00103 skill over 9 paired outer splits. A structural knob offered bundled with five
    # randomized others is a knob the ship gate cannot judge, so depth is offered CLEAN.
    from insurarena.models.tboost_model import _JOINT_DIMS, _PASSTHROUGH

    assert "max_depth" in _PASSTHROUGH          # still live: the probe has to reach the estimator
    assert "max_depth" not in _STATIC_TUNE_GRID
    assert "max_depth" not in _JOINT_DIMS
    # hsv 21: three rungs. 4 was excluded at hsv 20 as UNMEASURED and the new-territory battery
    # then measured it (d4..d8 through `_evaluate_config` on 8 datasets): best rung on 3 of 8,
    # and the only non-harmful depth on prudential_life. 7 and 8 stay out — the same battery
    # measured them and they are worse than the rung below on 7 of 8 datasets.
    # hsv 21 added rung 4 on an OUTER-TEST ladder; hsv 24 (av45) retired it again — it never led
    # a recorded inner-CV selection, and outer-test rung evidence gets no exemption from the ship
    # gate. The two rungs the depth-cells battery actually won on stay, and depth 8 is still
    # reachable inside the hsv-22 joint shape.
    assert ("max_depth", 4) not in _COORDINATE_PROBES
    assert ("max_depth", 5) in _COORDINATE_PROBES
    assert ("max_depth", 6) in _COORDINATE_PROBES
    assert not any(name == "max_depth" and v == 7 for name, v in _COORDINATE_PROBES)
    assert not any(name == "max_depth" and v == 8 for name, v in _COORDINATE_PROBES)  # joint only

    # ...and the hsv-20 single-knob head keeps its nominal candidate indices 1..8 through every
    # later edit, retirements included: `coordinate_probe_hp_set`'s identity contract is that probe
    # j sits at index j+1, and hsv 24 broke it ONLY behind that head (see the adapter's hsv-24
    # note — the retirements renumber the two joint probes and the sampled tail, which is sound
    # exactly because that bump rides an adapter_version bump and no cached cell survives).
    hsv20_head = [("max_interaction_order", 1), ("max_interaction_order", 2), ("max_bin", 64),
                  ("max_bin", 128), ("learning_rate", 0.02), ("interaction_gain_hurdle", 5.0),
                  ("max_depth", 5), ("max_depth", 6)]
    assert list(_COORDINATE_PROBES)[:len(hsv20_head)] == hsv20_head
    assert list(_COORDINATE_PROBES)[len(hsv20_head):] == [
        (("max_depth", 8), ("interaction_gain_hurdle", 0.0)),
        (("max_depth", 5), ("max_interaction_order", 4)),
    ]

    # The two references the ship gate depends on stay at the library's depth 3 by NOT carrying
    # the key, so every DEFAULT-variant cell on the board is untouched.
    assert "max_depth" not in TBoost.default_hp()       # candidate 0, the gate's anchor
    assert "max_depth" not in TBoost.tuning_base_hp()   # the tuned family's base recipe

    # Each depth probe differs from the tuned base in EXACTLY the depth, so its inner-CV score is
    # readable as depth's effect and nothing else. (The generic one-knob assertion covers every
    # probe; this names the property depth is here for.)
    base = TBoost.tuning_base_hp()
    for entry, hp in zip(_COORDINATE_PROBES, TBoost.coordinate_probe_hp_set()):
        if "max_depth" not in hp:
            continue
        named = {name for name, _ in _probe_moves(entry)}
        assert {k: v for k, v in hp.items() if k not in named} \
            == {k: v for k, v in base.items() if k not in named}


def test_the_grid_is_exactly_the_hsv19_seven_dimensions():
    # The dilution cost of a new axis is paid in candidates, not in code. Routing depth (and, at
    # hsv 22, the depth x hurdle and depth x order shapes) through the probe block instead of the
    # grid is what keeps that bill at two candidates rather than a multiplied product space, and
    # this asserts the grid did not quietly grow anyway.
    assert list(_STATIC_TUNE_GRID) == [
        "interaction_gain_hurdle", "max_interaction_order", "lambda_", "colsample_bytree",
        "min_data_in_leaf", "max_bin", "learning_rate", "cat_smooth",
    ]


def test_no_candidate_ever_asks_for_an_order_deeper_than_its_tree():
    # tri-boost REFUSES `max_interaction_order > max_depth` (an oblivious tree needs one level per
    # distinct raw feature). The sampled grid's order tops out at 3 and depth bottoms out at 3, so
    # no draw is illegal — but the two dimensions cycle under independent seeds, so this must be
    # asserted, not assumed. It also covers the hsv-22 (depth 5, order 4) probe, which is legal
    # with a level of headroom and would be caught here if either coordinate were ever edited
    # alone.
    for hp in hpo.static_configs("t-boost", _N_CAMPAIGN)[1:]:
        assert hp["max_interaction_order"] <= hp.get("max_depth", 3), hp


def test_the_table_cap_is_a_constant_and_never_a_searched_dimension():
    # Ralph's readability bar is a product call, not a hyper-parameter: searching it would let the
    # selector buy skill by relaxing the bar, and two independent grid cycles would eventually
    # draw the unguarded (depth 6, no cap) shape the controls refuse.
    from insurarena.models.tboost_model import _DEPTH_LIFT_TABLE_BUDGET

    assert _DEPTH_LIFT_TABLE_BUDGET == 100
    assert "prune_table_budget" not in _STATIC_TUNE_GRID
    assert "prune_table_min_arity" not in _STATIC_TUNE_GRID
    for hp in hpo.static_configs("t-boost", _N_CAMPAIGN):
        assert "prune_table_budget" not in hp


@pytest.mark.parametrize("depth,expect_budget", [(3, None), (5, 100), (6, 100), (8, 100)])
def test_depth_gate_pins_the_table_budget_on_lifted_fits_only(depth, expect_budget):
    import pandas as _pd
    from insurarena.datasets.base import DatasetBundle
    from insurarena.models.registry import build_model
    from insurarena.tasks import FREQ_SPEC

    rng = np.random.default_rng(4)
    n = 600
    X = _pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    exposure = np.ones(n)
    bundle = DatasetBundle(
        name="depth_gate_probe", X=X, y=rng.poisson(0.4, n).astype(float),
        exposure=exposure, sample_weight=exposure, task=FREQ_SPEC,
    )
    hp = {"budget": 20, "n_bags": 1, "max_depth": depth}
    m = build_model("t-boost", hp=hp).fit(bundle, None)
    assert m.depth_gate_["fires"] is (depth > 3)
    assert m.depth_gate_["max_depth"] == depth
    assert m.depth_gate_["prune"] is True                 # a deploy fit (val is None) prunes
    assert getattr(m.model, "max_depth", 3) == depth
    assert getattr(m.model, "prune_table_budget", 0) == (expect_budget or 0)
    assert m.depth_gate_["applied"] == expect_budget

    # ...and the SCOUT regime, where `_prune_during_hpo: False` turns pruning off. The cap is a
    # deploy-time trim of the prune's keep-set and tri-boost REFUSES it on an unpruned fit, so the
    # gate must not inject it there — and must drop an explicit one rather than let the fit raise.
    scout = build_model("t-boost", hp={**hp, "_prune_during_hpo": False,
                                        "prune_table_budget": 100}).fit(bundle, bundle)
    assert scout.depth_gate_["prune"] is False
    assert scout.depth_gate_["applied"] is None
    assert scout.depth_gate_["dropped_hp_budget"] == 100
    assert getattr(scout.model, "prune_table_budget", 0) == 0


def test_gate_bar_floors_at_three_se_for_a_depth_lifted_champion():
    # hsv 19's bar is a VOLATILITY premium and it is backwards for a structural knob: it resolves
    # to 1.0 on all three depth beneficiaries (fremotor_payfreq/prem, ~52k rows => repeats 1) and
    # to 5.0 on both refusing controls (swautoins 2.2k, sgautonb 7.5k => repeats 5).
    #
    # 3.0, not the 2.0 this floor was first drafted at. The campaign-budget smoke priced that
    # draft: fremotor_payfreq's depth-lifted champion cleared 2.0 at margin/SE = 2.27 and its 9
    # deploy cells then measured -0.00103 skill. A bar that admits the one harmful structural
    # ship anyone has observed end to end is not a bar.
    deep, flat = {"max_depth": 6}, {"max_depth": 3}

    assert TBoost.ship_gate_se_mult(1, deep) == 3.0        # the beneficiaries' bar triples...
    assert TBoost.ship_gate_se_mult(1, flat) == 1.0        # ...and a depth-3 champion is untouched
    assert TBoost.ship_gate_se_mult(1, None) == 1.0        # no champion resolved => hsv-19 bar
    assert TBoost.ship_gate_se_mult(1, {}) == 1.0          # absent key means the library's 3
    assert TBoost.ship_gate_se_mult(1, {"max_depth": None}) == 1.0   # ...and so does an explicit null
    assert TBoost.ship_gate_se_mult(5, deep) == 5.0        # a FLOOR: the volatility premium wins
    assert TBoost.ship_gate_se_mult(5, flat) == 5.0
    assert TBoost.ship_gate_se_mult(2, deep) == 3.0        # ...and the floor wins below it
    # The measured counterexample, asserted as a number: 2.27 SE must NOT be enough.
    assert 2.27 < TBoost.ship_gate_se_mult(1, deep)

    # Strict tightening: the bar can only rise, so it can never create a new harmful ship.
    for r in (1, 2, 3, 4, 5):
        assert TBoost.ship_gate_se_mult(r, deep) >= TBoost.ship_gate_se_mult(r, None)


def test_hsv19_one_argument_hooks_keep_their_exact_bar():
    # `_gate_se_mult` passes the champion through only when the hook accepts it. An adapter still
    # carrying the hsv-19 signature must keep its hsv-19 bar, NOT fall back to the 1-SE default.
    assert hpo._gate_se_mult("t-boost", 5, {"max_depth": 6}) == 5.0
    assert hpo._gate_se_mult("t-boost", 1, {"max_depth": 6}) == 3.0
    assert hpo._gate_se_mult("t-boost", 1, {"max_depth": 3}) == 1.0
    assert hpo._gate_se_mult("t-boost", 1) == 1.0

    class _LegacyAdapter:
        @staticmethod
        def ship_gate_se_mult(inner_repeats):        # hsv-19 signature, one argument
            return 3.0

    with mock.patch.object(hpo, "model_class", lambda _n: _LegacyAdapter):
        assert hpo._gate_se_mult("legacy", 5, {"max_depth": 6}) == 3.0
        assert hpo._gate_se_mult("legacy", 1, None) == 3.0

    # ...and every non-overriding adapter is still byte-untouched.
    for name in ("lightgbm", "xgboost", "catboost", "ebm", "glm"):
        assert hpo._gate_se_mult(name, 5, {"max_depth": 6}) == 1.0


def test_the_depth_gate_reaches_the_real_selection(monkeypatch):
    # End to end: identical evidence, one champion depth-lifted and one not, opposite verdicts.
    # The paired diffs are (-0.001547, 0.010, 0.021547): margin/SE == 1.50 exactly, which clears
    # the hsv-19 bar of 1.0 and misses the hsv-20 depth floor of 3.0.
    def fake_oof(model_name, otr, hp, k, seed, repeats=1, **_kw):
        folds = [1.001547, 1.010, 0.958453] if "seed" in hp else [1.00, 1.02, 0.98]
        return float(np.mean(folds)), 0, list(folds), False

    monkeypatch.setattr(hpo, "_oof_score", fake_oof)
    seen = {}
    for depth in (3, 6):
        base = dict(TBoost.tuning_base_hp())
        monkeypatch.setattr(
            TBoost, "sample_hp_set",
            staticmethod(lambda n, s, d=depth: [{**base, "max_depth": d} for _ in range(n)]),
        )
        monkeypatch.setattr(TBoost, "coordinate_probe_hp_set", staticmethod(lambda: []))
        _, _, _, _, sel = hpo.select_config("t-boost", _fake_bundle(), 3, 3, 123, repeats=1)
        seen[depth] = sel
    assert seen[3]["gate_se_mult"] == 1.0
    assert seen[6]["gate_se_mult"] == 3.0
    assert seen[3]["gate_margin"] == pytest.approx(seen[6]["gate_margin"])   # same evidence...
    assert seen[3]["gate_se"] == pytest.approx(seen[6]["gate_se"])
    assert seen[3]["gate_margin"] / seen[3]["gate_se"] == pytest.approx(1.5, abs=1e-3)
    assert seen[3]["shipped"] == "champion"
    assert seen[6]["shipped"] == "default"                                    # ...opposite call


# --- hsv 20: the MULTICLF-only cell_refit_base dimension -----------------------------------------


def test_cell_refit_base_is_no_longer_searched_at_all_and_the_anchor_is_untouched():
    # hsv 20 offered 20000.0 as a probe; hsv 24 (av45) RETIRED it, because the test below proves
    # the probe cannot express its own hypothesis where it is scored — every scout eval fits
    # n_bags=1 and the fit-time gate drops the key, so at scout fidelity the probe IS the tuned
    # base on all 40 datasets. Retiring it costs no measured evidence and buys a candidate slot
    # back toward the comparators' budget.
    #
    # What must NOT move: the §G1 refit itself. The av39 fit-time gate still injects 4000.0 on
    # MULTICLF, and the knob is still reachable from a config `hp:` block for an ablation.
    from insurarena.models.tboost_model import _CELL_REFIT_BASE, _PASSTHROUGH

    assert "cell_refit_base" in _PASSTHROUGH
    assert "cell_refit_base" not in _STATIC_TUNE_GRID
    assert not any("cell_refit_base" in dict(_probe_moves(e)) for e in _COORDINATE_PROBES)
    assert _CELL_REFIT_BASE == 4000.0
    # The gate's anchor never carries the key; the av39 fit-time gate injects 4000.0 there.
    assert "cell_refit_base" not in TBoost.default_hp()
    assert "cell_refit_base" not in TBoost.tuning_base_hp()


def test_random_strength_is_reachable_but_no_longer_searched():
    # hsv 23 offered 1.0 and 4.0; hsv 24 retired both. The knob stays LIVE (a config `hp:` block
    # can still set it, and it lands in the cell signature) — what goes is the two fixed rungs,
    # which cannot cover a perturbation whose scale is data-dependent by construction.
    from insurarena.models.tboost_model import _PASSTHROUGH

    assert "random_strength" in _PASSTHROUGH
    assert "random_strength" not in _STATIC_TUNE_GRID
    assert not any("random_strength" in dict(_probe_moves(e)) for e in _COORDINATE_PROBES)
    assert "random_strength" not in TBoost.default_hp()
    assert "random_strength" not in TBoost.tuning_base_hp()
    # ...and no candidate of the campaign basis carries it, so every av45 cell scores at 0.0.
    assert all("random_strength" not in hp for hp in hpo.static_configs("t-boost", _N_CAMPAIGN))


def test_tri_races_the_same_budget_as_every_comparator():
    # THE FAIRNESS INVARIANT (av45). tri-boost had drifted to 29 candidates against every other
    # tunable model's 25 — a tuning-budget advantage on the one axis a benchmark most has to keep
    # level. `static_configs` holds len == n_configs + 1 for every model, probes or not, so the
    # max-of-N the ship gate defends against is now identical across the five tunable models.
    assert _N_CAMPAIGN == _N_COMPARATOR
    sizes = {name: len(hpo.static_configs(name, _N_COMPARATOR))
             for name in ("t-boost", "lightgbm", "xgboost", "catboost", "ebm")}
    assert set(sizes.values()) == {_N_COMPARATOR + 1}, sizes


def test_the_cell_refit_probe_is_inert_at_scout_fidelity_and_says_so():
    # The limit this probe has to be read with, asserted rather than only documented: the §G1
    # refit needs an out-of-bag complement, every scout eval fits n_bags=1, so the fit-time gate
    # DROPS the key on the evaluation the probe actually gets. At scout fidelity the probe is the
    # tuned base — on all 40 datasets, MULTICLF included. It can only ever differ at the
    # ship-fidelity finale, and only if it wins a race in which it is indistinguishable.
    import pandas as _pd
    from insurarena.datasets.base import DatasetBundle
    from insurarena.models.registry import build_model
    from insurarena.tasks import FREQ_SPEC

    rng = np.random.default_rng(9)
    n = 400
    X = _pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    exposure = np.ones(n)
    bundle = DatasetBundle(
        name="cell_refit_scout_probe", X=X, y=rng.poisson(0.4, n).astype(float),
        exposure=exposure, sample_weight=exposure, task=FREQ_SPEC,
    )
    scout_overlay = TBoost.selection_hp()
    assert scout_overlay["n_bags"] == 1, "the premise of this test"
    # The retired hsv-20 probe, reconstructed: base + the one knob. This is the measurement that
    # RETIRED it at hsv 24 — a candidate that is bit-identical to the base wherever it is ranked.
    probe = {**TBoost.tuning_base_hp(), "cell_refit_base": 20000.0}
    m = build_model("t-boost", hp={**probe, **scout_overlay, "budget": 20}).fit(bundle, bundle)
    assert m.cell_refit_gate_["fires"] is False
    assert m.cell_refit_gate_["dropped_hp_base"] == 20000.0
    assert getattr(m.model, "cell_refit_base", None) is None


@pytest.mark.parametrize("n_bags,expect_kept", [(1, False), (2, True)])
def test_cell_refit_base_is_dropped_wherever_the_fit_cannot_honor_it(n_bags, expect_kept):
    # The space is task-blind, so every candidate carries the key — including on the 35
    # non-MULTICLF datasets and on every 1-bag scout eval, where tri-boost RAISES rather than
    # silently no-opping. The gate is therefore a FILTER as well as an injector.
    import pandas as _pd
    from insurarena.datasets.base import DatasetBundle
    from insurarena.models.registry import build_model
    from insurarena.tasks import FREQ_SPEC

    rng = np.random.default_rng(5)
    n = 500
    X = _pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    exposure = np.ones(n)
    bundle = DatasetBundle(
        name="cell_refit_drop_probe", X=X, y=rng.poisson(0.4, n).astype(float),
        exposure=exposure, sample_weight=exposure, task=FREQ_SPEC,
    )
    # A POISSON bundle: never MULTICLF, so the gate must drop the key at either bag count.
    hp = {"budget": 20, "n_bags": n_bags, "cell_refit_base": 20000.0}
    m = build_model("t-boost", hp=hp).fit(bundle, None)
    assert m.cell_refit_gate_["fires"] is False
    assert m.cell_refit_gate_["dropped_hp_base"] == 20000.0
    assert getattr(m.model, "cell_refit_base", None) is None
    del expect_kept          # both bag counts drop it on a non-MULTICLF task


def test_hsv_and_adapter_versions_say_what_recomputes():
    # hsv enters the `tuned` proto and BOTH hpo protos; `av` enters every proto. The rule the two
    # counters encode is unchanged: bump `hpo_search_version` when only the tuned search moves and
    # the `default` cells stay cache-valid, bump `adapter_version` when a fit path moves and EVERY
    # cell recomputes, defaults included.
    #
    # Both were reset to "1" on 2026-09-09 when the arena was rebuilt from scratch and the cell
    # cache was discarded (the pre-reset history is in git). This assertion is the reminder that
    # the next bump of either is a deliberate edit with a recompute attached, not an accident.
    assert TBoost.hpo_search_version == "1"
    assert TBoost.adapter_version == "1"


# --- hsv 21 (2026-08-31): the pooling band, the third depth rung, and the regime recheck ---------


def test_pool_band_tracks_the_bar_in_force():
    # hsv 18 set the band at 2.0 when every bar was 1.0. hsv 19/20 raised the bar and left the
    # band behind, so a depth champion at 2.4 SE was "decided" for the trigger and REFUSED by its
    # own 3-SE gate — the one decision on the board that could not be made without more evidence
    # was the one the trigger declined to buy it for.
    assert hpo._ADAPTIVE_POOL_SE_MULT == 2.0                 # still the FLOOR
    assert hpo._adaptive_pool_band(1.0) == 2.0               # 1-SE bar: hsv-18 band, unchanged
    assert hpo._adaptive_pool_band(2.0) == 2.0
    assert hpo._adaptive_pool_band(3.0) == 3.0               # depth floor
    assert hpo._adaptive_pool_band(5.0) == 5.0               # hsv-19 volatility premium
    # a hook that returns nonsense can only ever fall back to the historical band
    for bad in (None, 0.0, -1.0, float("nan"), float("inf"), "x"):
        assert hpo._adaptive_pool_band(bad) == 2.0
    assert hpo._adaptive_pool_band() == 2.0                  # default arg == the 1-SE bar


def test_pooled_finale_wanted_scales_with_the_bar():
    def gate(margin, se):
        return {"gate_margin": margin, "gate_se": se}

    # freMTPL2freq's real single-split numbers: 2.379 SE against a 3.0 bar.
    fre = gate(0.00154468, 0.000649)
    assert hpo._pooled_finale_wanted(fre, "adaptive", 1.0) is False   # hsv 20: declined
    assert hpo._pooled_finale_wanted(fre, "adaptive", 3.0) is True    # hsv 21: bought
    # fremotor_payfreq's cautionary champion (2.27 SE, the ship that measured -0.00103) is pulled
    # INTO the band by the same rule — the point being that pooling is EVIDENCE, not a shortcut.
    assert hpo._pooled_finale_wanted(gate(2.27, 1.0), "adaptive", 3.0) is True
    # the band is two-sided, and a 3-SE bar still calls 3.01 SE decided
    assert hpo._pooled_finale_wanted(gate(-2.99, 1.0), "adaptive", 3.0) is True
    assert hpo._pooled_finale_wanted(gate(3.01, 1.0), "adaptive", 3.0) is False
    # every 1-SE-bar cell keeps its hsv-18 decision exactly (the default arg is the 1-SE bar)
    for m in (0.5, 1.5, 1.99, 2.01, -2.01, 5.0):
        assert hpo._pooled_finale_wanted(gate(m, 1.0), "adaptive") is (abs(m) <= 2.0)
    assert hpo._pooled_finale_wanted(gate(10.0, 0.001), "always", 3.0) is True   # small sets: always


def _depth_pooled_run(depth, primary_folds, extra_folds):
    """Drive the static search with a single DEPTH-LIFTED (or not) probe as the only candidate."""
    built = []

    def fake_oof(model_name, otr, hp, k, seed, repeats=1, **_kw):
        champion = "max_depth" in hp
        folds = (extra_folds if getattr(otr, "_extra", False) else primary_folds)[bool(champion)]
        return float(np.mean(folds)), 9 if champion else 7, list(folds), False

    def make_extra(tag):
        def thunk():
            b = _fake_bundle()
            b._extra = True
            built.append(tag)
            return b
        return thunk

    base = dict(TBoost.tuning_base_hp())
    with mock.patch.object(hpo, "_oof_score", fake_oof), \
         mock.patch.object(TBoost, "coordinate_probe_hp_set",
                           staticmethod(lambda: [{**base, "max_depth": depth}])):
        hp, _, _, evals, sel = hpo.select_config(
            "t-boost", _fake_bundle(), 1, 3, 123,
            extra_finale_otrs=[make_extra("a"), make_extra("b")], pooled_finale_mode="adaptive",
        )
    return hp, sel, evals, len(built)


def test_the_band_buys_evidence_for_a_depth_champion_the_hsv20_trigger_declined():
    # freMTPL2freq's shape, end to end. Single split: margin/SE = 2.4 — beyond hsv 20's band of 2
    # and short of the 3-SE structural floor, i.e. "ship the default, and don't even look".
    # Pooled over 3 splits the same protocol reads far past 3 SE and ships. (The real cell:
    # 2.379 -> 3.146, and the deploy then measures +0.001408 +/- 0.000217, t=+6.5, 9-0.)
    m = 0.0013856      # diffs [m-0.001, m, m+0.001] -> sd 0.001, se 0.000577, ratio 2.4
    primary = {False: [1.000, 1.020, 0.980],
               True: [1.000 - (m - 0.001), 1.020 - m, 0.980 - (m + 0.001)]}
    extra = {False: [1.000, 1.000, 1.000], True: [1.000 - m] * 3}

    hp, sel, evals, n_built = _depth_pooled_run(6, primary, extra)
    assert sel["single_split_gate_margin"] / sel["single_split_gate_se"] == pytest.approx(2.4, abs=1e-2)
    assert sel["single_split_shipped"] == "default"          # 2.4 SE does not clear the 3-SE floor
    assert sel["gate_se_mult"] == 3.0
    assert sel["pooled_finale"] == "adaptive"
    assert sel["pooled_finale_band"] == 3.0
    assert sel["n_paired_folds"] == 9 and n_built == 2
    assert sel["shipped"] == "champion" and sel["reason"] == "beats_default_by_1se_pooled"
    assert hp["max_depth"] == 6
    # and the recheck did not fire: the only regime-faithful candidate is the anchor, which the
    # gate already compares against — so no evaluation was spent on it.
    assert "regime_recheck" not in sel
    assert not any(e.get("param") == "regime_recheck" for e in evals)

    # SAME EVIDENCE, depth 3: bar 1.0, band 2.0, 2.4 SE is decided -> no pooling, ships on one
    # split exactly as it did at hsv 20. The band moved only where the bar moved.
    hp3, sel3, _, n3 = _depth_pooled_run(3, primary, extra)
    assert sel3["gate_se_mult"] == 1.0
    assert sel3["pooled_finale"] == "skipped_margin_beyond_2se"
    assert sel3["pooled_finale_band"] == 2.0
    assert sel3["n_finale_splits"] == 1 and n3 == 0
    assert sel3["shipped"] == "champion" and sel3["reason"] == "beats_default_by_1se"


def _regime_run(scout, finale, depth=6, disable_hook=False):
    """Three candidates (anchor + depth probe + a regime-faithful probe) whose scores depend on
    the REGIME of the evaluation: `scout`/`finale` map a candidate tag -> mean fold score."""
    base = dict(TBoost.tuning_base_hp())
    deep = {**base, "max_depth": depth}
    flat = {**base, "learning_rate": 0.02}

    def tag_of(hp):
        return "deep" if "max_depth" in hp else ("flat" if hp.get("learning_rate") == 0.02 else "anchor")

    def fake_oof(model_name, otr, hp, k, seed, repeats=1, **kw):
        pruned = bool((kw.get("selection_hp") or {}).get("_prune_during_hpo"))
        mean = (finale if pruned else scout)[tag_of(hp)]
        folds = [mean - 0.001, mean, mean + 0.001]
        return float(np.mean(folds)), 5, list(folds), False

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(hpo, "_oof_score", fake_oof))
        stack.enter_context(mock.patch.object(TBoost, "coordinate_probe_hp_set",
                                              staticmethod(lambda: [deep, flat])))
        if disable_hook:      # the hsv-20 counterfactual: no hook, no recheck
            stack.enter_context(mock.patch.object(TBoost, "selection_regime_recheck",
                                                  staticmethod(lambda hp: False)))
        hp, _, _, evals, sel = hpo.select_config("t-boost", _fake_bundle(), 2, 3, 123)
    return hp, sel, evals


def test_the_regime_recheck_makes_the_uncapped_depth_trap_impossible():
    # allstate_sev's shipped shape, synthetic. The depth probe is ranked by the scout with the
    # prune (and therefore the >=3-way TABLE cap) OFF and wins the argmax outright; every fit
    # that would ship it prunes and takes the cap, where it is the WORST of the three. Measured
    # on the real cell: uncapped rank t = +6.3, capped deploy -0.0209 skill.
    scout = {"anchor": 1.000, "deep": 0.990, "flat": 0.995}     # depth wins the scout by a mile
    finale = {"anchor": 1.000, "deep": 1.010, "flat": 0.996}    # ...and loses badly once capped

    hp, sel, evals = _regime_run(scout, finale)
    rc = sel["regime_recheck"]
    assert rc["demoted"] is True
    assert rc["faithful_finale_score"] < rc["scout_champion_finale_score"]
    assert "max_depth" not in hp and hp["learning_rate"] == 0.02   # the faithful runner-up ships
    assert sel["shipped"] == "champion"
    assert sel["gate_se_mult"] == 1.0            # ...and is judged as the ordinary knob it is
    # exactly ONE extra finale evaluation: anchor + scout champion + the faithful contender
    finale_evals = [e for e in evals if e.get("stage") == "finale"]
    assert [e["param"] for e in finale_evals] == ["default_anchor", "champion", "regime_recheck"]

    # WITHOUT the hook (hsv 20): the depth probe holds the top slot, the gate refuses it — and
    # the candidate that would have shipped is never evaluated at all. That is the trap: not a
    # harmful ship, but a cell that ships the anchor while a better champion sits unexamined.
    hp20, sel20, evals20 = _regime_run(scout, finale, disable_hook=True)
    assert "regime_recheck" not in sel20
    assert sel20["shipped"] == "default" and hp20 == TBoost.default_hp()
    assert [e["param"] for e in evals20 if e.get("stage") == "finale"] == ["default_anchor", "champion"]


def test_the_regime_recheck_keeps_a_depth_champion_that_survives_its_own_cap():
    # The other half: where the cap is not binding (measured BIT-IDENTICAL on eusavingul and
    # beMTPL16, whose shipped depth champions keep 25 and 84 tables against a budget of 100) the
    # depth candidate wins the recheck and ships — with the 3-SE structural floor over it.
    scout = {"anchor": 1.000, "deep": 0.990, "flat": 0.995}
    finale = {"anchor": 1.000, "deep": 0.990, "flat": 0.996}

    hp, sel, evals = _regime_run(scout, finale)
    assert sel["regime_recheck"]["demoted"] is False
    assert hp["max_depth"] == 6
    assert sel["gate_se_mult"] == 3.0
    assert sel["shipped"] == "champion"


def test_the_regime_recheck_is_declared_by_the_adapter_and_nobody_else():
    assert TBoost.selection_regime_recheck({"max_depth": 6}) is True
    assert TBoost.selection_regime_recheck({"max_depth": 4}) is True
    assert TBoost.selection_regime_recheck({"max_depth": 3}) is False
    assert TBoost.selection_regime_recheck({"max_depth": None}) is False   # explicit null
    assert TBoost.selection_regime_recheck({}) is False                    # every basis draw
    assert TBoost.selection_regime_recheck(None) is False
    assert hpo._regime_recheck("t-boost", {"max_depth": 6}) is True
    for name in ("lightgbm", "xgboost", "catboost", "ebm", "glm"):
        assert hpo._regime_recheck(name, {"max_depth": 6}) is False          # byte-untouched

    class _BadAdapter:
        @staticmethod
        def selection_regime_recheck(hp):
            raise RuntimeError("boom")

    # a raising hook fails to NO recheck, i.e. the historical path — never a weakened gate
    with mock.patch.object(hpo, "model_class", lambda _n: _BadAdapter):
        assert hpo._regime_recheck("bad", {"max_depth": 6}) is False


# --- hsv 22: the MULTI-KNOB probes ---------------------------------------------------------------


def test_probe_moves_normalizes_both_spellings_and_refuses_malformed_entries():
    # The one piece of new machinery. Single-knob and multi-knob entries are disambiguated on the
    # TYPE of the first element and nothing else, so the hsv 18-21 spelling keeps working verbatim.
    assert _probe_moves(("max_depth", 5)) == (("max_depth", 5),)
    assert _probe_moves((("max_depth", 8), ("interaction_gain_hurdle", 0.0))) == (
        ("max_depth", 8), ("interaction_gain_hurdle", 0.0))
    assert _probe_moves({"max_depth": 5, "max_interaction_order": 4}) == (
        ("max_depth", 5), ("max_interaction_order", 4))          # order preserved

    # a value that is itself a 2-tuple must not be mistaken for a pair list
    assert _probe_moves(("some_pair_knob", (1, 2))) == (("some_pair_knob", (1, 2)),)

    with pytest.raises(ValueError):
        _probe_moves(())                                          # moves nothing == the base
    with pytest.raises(ValueError):
        _probe_moves((("max_depth", 5), ("max_depth", 8)))        # second write silently wins


def test_the_two_joint_probes_are_exactly_the_measured_shapes():
    # The depth-conversion forensic's two >= +0.002, 9-0 arms, verbatim. Nothing interpolated.
    assert (("max_depth", 8), ("interaction_gain_hurdle", 0.0)) in _COORDINATE_PROBES
    assert (("max_depth", 5), ("max_interaction_order", 4)) in _COORDINATE_PROBES

    # NOT the third arm: (depth 8, order 4) is freMTPL2freq's, and freMTPL2freq is conceded at
    # -0.000185 — a null. A probe that buys nothing still costs a draw in the max-of-N.
    assert (("max_depth", 8), ("max_interaction_order", 4)) not in _COORDINATE_PROBES
    # ...and nothing relaxes the hurdle AND the order together: measured as a fit-time explosion
    # on both sets (37-55 min against 20-100 s controls).
    for entry in _COORDINATE_PROBES:
        names = {name for name, _ in _probe_moves(entry)}
        assert not {"interaction_gain_hurdle", "max_interaction_order"} <= names, entry

    # Both probes are built off the tuned base and differ from it in EXACTLY their named knobs.
    base = TBoost.tuning_base_hp()
    probes = dict(zip(_COORDINATE_PROBES, TBoost.coordinate_probe_hp_set()))
    d8h0 = probes[(("max_depth", 8), ("interaction_gain_hurdle", 0.0))]
    d5o4 = probes[(("max_depth", 5), ("max_interaction_order", 4))]
    assert d8h0 == {**base, "max_depth": 8, "interaction_gain_hurdle": 0.0}
    assert d5o4 == {**base, "max_depth": 5, "max_interaction_order": 4}
    # the base's own values, i.e. what each probe is a move AWAY from
    assert base["interaction_gain_hurdle"] == 2.0 and base["max_interaction_order"] == 3
    assert "max_depth" not in base


def test_the_probe_block_layout_after_the_hsv24_retirements():
    # The identity contract is that probe j sits at nominal candidate index j+1, so its cached
    # evaluation stays reusable. hsv 24 (av45) BROKE it once, deliberately and behind the
    # single-knob head: retiring `cell_refit_base` 20000.0 and `max_depth` 4 pulls the two joint
    # probes from candidate index 11/12 to 9/10, and retiring the two `random_strength` probes
    # shortens the block to 10. Sound only because the bump rides an `adapter_version` bump —
    # nothing cached survives to be mis-reused — and this test is what pins the new layout so the
    # contract runs again from here.
    single_knob = [("max_interaction_order", 1), ("max_interaction_order", 2), ("max_bin", 64),
                   ("max_bin", 128), ("learning_rate", 0.02), ("interaction_gain_hurdle", 5.0),
                   ("max_depth", 5), ("max_depth", 6)]
    assert list(_COORDINATE_PROBES)[:8] == single_knob
    assert len(_COORDINATE_PROBES) == 10

    configs = hpo.static_configs("t-boost", _N_CAMPAIGN)
    base = TBoost.tuning_base_hp()
    for j, (name, value) in enumerate(single_knob, start=1):
        assert configs[j] == {**base, name: value, "seed": j}
    assert configs[9] == {**base, "max_depth": 8, "interaction_gain_hurdle": 0.0, "seed": 9}
    assert configs[10] == {**base, "max_depth": 5, "max_interaction_order": 4, "seed": 10}


def test_the_sampled_tail_is_bit_identical_to_hsv21s_only_renumbered():
    # The tail has been the SAME 15 hp dicts since hsv 21 — `sample_hp_set`'s candidate j depends
    # only on j, and every budget move since has been chosen to keep the tail at exactly 15. Only
    # the nominal index, and therefore the per-candidate `seed`, moves: +2 at hsv 22, +2 at hsv 23,
    # and -4 at hsv 24 when the four probes retired, which puts the seeds back where hsv 21 had
    # them.
    raw = TBoost.sample_hp_set(15, hpo._CONFIG_SEED)
    tail = hpo.static_configs("t-boost", _N_CAMPAIGN)[1 + len(_COORDINATE_PROBES):]
    assert len(tail) == len(raw) == 15
    for j, (got, want) in enumerate(zip(tail, raw), start=1 + len(_COORDINATE_PROBES)):
        assert {k: v for k, v in got.items() if k != "seed"} == want
        assert got["seed"] == j


def test_both_joint_probes_are_structural_and_regime_mismatched():
    # Each of the three depth mechanisms must catch BOTH new probes, or one of them ships on
    # ordinary evidence in a regime it was not ranked in.
    for hp in (dict(TBoost.coordinate_probe_hp_set()[8]),
               dict(TBoost.coordinate_probe_hp_set()[9])):
        assert hp["max_depth"] > 3
        assert TBoost.ship_gate_se_mult(1, hp) == 3.0        # (a) the 3-SE structural floor
        assert TBoost.selection_regime_recheck(hp) is True   # (b) the hsv-21 regime recheck
        assert hpo._regime_recheck("t-boost", hp) is True
    # (c) the >=3-way TABLE cap: `prune_table_budget` is never a searched key, and the fit-time
    # depth gate arms it on max_depth > 3 (asserted end-to-end at depths 5/6/8 in the gate test).
    for hp in hpo.static_configs("t-boost", _N_CAMPAIGN):
        assert "prune_table_budget" not in hp


def test_order_lift_is_structural_on_its_own_axis():
    # hsv 22 widens the 3-SE floor's TEST from "depth-lifted" to STRUCTURAL. Nothing on the board
    # changes today — the one order-4 candidate is depth-lifted too and the grid tops out at
    # order 3 — but the floor's argument ("a wrong yes changes the model class of every deploy
    # split") is as true of an order-4 bank as of a depth-6 one, and this pins it so a future
    # order-only probe cannot slip in under a 1-SE bar.
    assert _is_structural_hp({"max_interaction_order": 4}) is True     # order alone, depth 3
    assert TBoost.ship_gate_se_mult(1, {"max_interaction_order": 4}) == 3.0
    assert _is_structural_hp({"max_depth": 5}) is True
    assert _is_structural_hp({"max_depth": 5, "max_interaction_order": 4}) is True

    # ...and the negative space, unchanged: the anchor, the base, and every sampled draw.
    assert _is_structural_hp(None) is False
    assert _is_structural_hp({}) is False
    assert _is_structural_hp(TBoost.default_hp()) is False
    assert _is_structural_hp(TBoost.tuning_base_hp()) is False
    assert _is_structural_hp({"max_depth": None, "max_interaction_order": None}) is False
    assert _is_structural_hp({"max_interaction_order": 3}) is False
    for hp in hpo.static_configs("t-boost", _N_CAMPAIGN)[1 + len(_COORDINATE_PROBES):]:
        assert _is_structural_hp(hp) is False, hp

    # The recheck hook is deliberately NOT widened: it asks a different question (was this ranked
    # in a regime it will not deploy in?), and the regime difference is the depth-armed cap.
    assert TBoost.selection_regime_recheck({"max_interaction_order": 4}) is False


def test_the_campaign_configs_carry_the_budget_the_tail_needs():
    # The tail constraint is only honored if the CONFIGS move too — `n_configs` lives in the yaml,
    # and a probe block that outgrows it silently deletes joint (order, hurdle) combinations.
    import yaml

    from insurarena.results.canonical import campaign_configs

    root = pathlib.Path(__file__).resolve().parents[1]
    # av45: ONE tri-boost config resolves the board, like every comparator. The per-dataset `ffull`
    # (5 sub-2500 sets) and eusavingul (a FULL candidate pin, seed included) configs are DELETED —
    # their behavior lives in the adapter's size gate and in the general config's own probe block +
    # finale re-race. A second t-boost config would re-open exactly the asymmetry av45 closed, so
    # this asserts the COUNT the leaderboard extraction uses, not just the contents of one file.
    # (`configs/` now holds exactly one campaign config per model, at `configs/<model>.yaml`.)
    tri = [c for m, c in campaign_configs(root) if m == "t-boost"]
    assert tri == ["configs/t-boost.yaml"], tri
    for rel in tri:
        cfg = yaml.safe_load((root / rel).read_text())
        n = int(cfg["hpo"]["n_configs"])
        assert n == _N_CAMPAIGN == _N_COMPARATOR, rel
        assert n - len(_COORDINATE_PROBES) >= 15, f"{rel}: sampled tail below joint coverage"
