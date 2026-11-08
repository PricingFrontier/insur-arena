"""Tuning-speed levers (tri-boost design/tuning-speed-roadmap.md, verified 2026-07-15):
A1 fold-parallel candidate evals, A2 fold-1 racing, B3 selection-only hp overlay."""

import numpy as np
import pytest

from insurarena.datasets.registry import build_loader
from insurarena.eval import hpo


def _bundle(n=1200):
    return build_loader("synthetic_freq", subsample=n).load()


# --- A1: fold-parallel is bit-identical to serial ------------------------------------------------


def test_fold_parallel_is_bit_identical_to_serial():
    # tri-boost is bit-deterministic across thread counts BY DESIGN, which makes it the
    # right oracle for this identity check — lightgbm's histogram sums are timing-sensitive
    # under machine load unless deterministic flags are set, and flaked here once.
    pytest.importorskip("t_boost")
    otr = _bundle()
    hp = {"budget": 20, "n_bags": 1, "seed": 3}
    serial = hpo._oof_score("t-boost", otr, dict(hp), 3, 42, fold_workers=1)
    threaded = hpo._oof_score("t-boost", otr, dict(hp), 3, 42, fold_workers=3)
    assert serial == threaded
    score, best_iter, folds, raced = serial
    assert len(folds) == 3 and not raced and np.isfinite(score)


def test_fold_parallel_repeats_keep_fold_order():
    pytest.importorskip("t_boost")
    otr = _bundle(600)
    hp = {"budget": 15, "n_bags": 1, "seed": 3}
    serial = hpo._oof_score("t-boost", otr, dict(hp), 2, 42, repeats=2, fold_workers=1)
    threaded = hpo._oof_score("t-boost", otr, dict(hp), 2, 42, repeats=2, fold_workers=4)
    assert serial == threaded
    assert len(serial[2]) == 4  # (repeat, fold) order preserved


# --- A2: racing ------------------------------------------------------------------------------------


def test_racing_discards_only_clear_losers_and_never_the_winner():
    # Deterministic replay-style check through the real static loop with a faked _oof_score.
    fold_map = {
        0: [1.00, 1.01, 0.99],   # candidate 0 (anchor/incumbent)
        1: [0.99, 1.00, 0.98],   # improves -> incumbent
        2: [1.00, 1.01, 1.00],   # completes (pool warms up)
        3: [3.00, 3.00, 3.00],   # clear loser -> must be raced after warmup
        4: [0.98, 0.99, 0.97],   # the true winner -> must NOT be raced
    }
    calls = {"racing_seen": 0}

    def fake_oof(model_name, otr, hp, k, seed, repeats=1, racing=None, **_kw):
        folds = fold_map[int(hp["seed"])]
        if racing is not None:
            calls["racing_seen"] += 1
            deficit = folds[0] - racing["inc_fold1"]
            if deficit > racing["mult"] * racing["sd"]:
                return None, 0, [folds[0]], True
        return float(np.mean(folds)), 5, list(folds), False

    def fake_configs(model_name, n, hp_override=None):
        return [{"seed": i} for i in range(5)]

    import unittest.mock as mock

    with mock.patch.object(hpo, "_oof_score", fake_oof), \
         mock.patch.object(hpo, "static_configs", fake_configs), \
         mock.patch.object(hpo, "_model_selection_hp", lambda name: {}):
        otr = _bundle(400)
        hp, _, _, evals, sel = hpo._select_config_static(
            "lightgbm", otr, 5, 3, 123, racing_se=2.0
        )
    raced = [e["config_idx"] for e in evals if e.get("raced")]
    assert raced == [3], f"expected exactly candidate 3 raced, got {raced}"
    assert hp == {"seed": 4}                     # true winner selected
    assert sel["n_raced"] == 1 and sel["racing_se"] == 2.0
    assert evals[3]["score"] is None and len(evals[3]["fold_scores"]) == 1


def test_racing_off_by_default_and_raced_payloads_never_cached():
    class SpyCache:
        def __init__(self):
            self.put_sigs = []

        def get(self, sig, model=None):
            return None

        def put(self, sig, payload, model=None):
            self.put_sigs.append(payload.get("raced", False))

    def fake_oof(model_name, otr, hp, k, seed, repeats=1, racing=None, **_kw):
        if racing is not None:
            return None, 0, [9.9], True
        return 1.0, 5, [1.0, 1.0, 1.0], False

    import unittest.mock as mock

    cache = SpyCache()
    with mock.patch.object(hpo, "_oof_score", fake_oof):
        payload, _ = hpo._evaluate_config(
            "lightgbm", _bundle(400), {"seed": 0}, 1, 3, 7, "poisson_deviance", "lower",
            True, cache, "hash", {}, racing={"inc_fold1": 0.0, "sd": 1.0, "mult": 2.0,
                                             "lower_better": True},
        )
    assert payload["raced"] is True
    assert cache.put_sigs == []                  # raced -> not cached


# --- B3: selection-only hp overlay -----------------------------------------------------------------


def test_triboost_selection_hp_applies_to_selection_fits_only():
    pytest.importorskip("t_boost")
    from insurarena.models.registry import build_model, model_class

    assert model_class("t-boost").selection_hp() == {
        "_selection_es_adaptive": 1.2, "n_bags": 1, "_prune_during_hpo": False,
    }
    assert model_class("t-boost").selection_finale_hp() == {
        "_selection_es_adaptive": 1.2, "_prune_during_hpo": True,
    }

    otr = _bundle(800)
    fold = np.random.default_rng(0).permutation(len(otr.X)) % 4
    tr, va = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    hp = {"budget": 25, "n_bags": 1, "_selection_es_adaptive": 1.2, "seed": 0}

    m_sel = build_model("t-boost", hp=hp).fit(otr.subset(tr), otr.subset(va))
    assert m_sel.model.get_params()["early_stopping_adaptive"] == 1.2

    m_dep = build_model("t-boost", hp=hp).fit(otr.subset(tr), None)
    assert m_dep.model.get_params()["early_stopping_adaptive"] == 1.5  # library default


def test_oof_score_merges_selection_hp_without_leaking_it():
    captured = {}

    class FakeModel:
        best_iteration_ = 0

        def __init__(self, hp):
            captured.setdefault("hps", []).append(dict(hp))

        def fit(self, tr, va):
            return self

        def predict(self, X, exposure=None):
            return np.full(len(X), 0.1)

    import unittest.mock as mock

    with mock.patch.object(hpo, "build_model", lambda name, hp: FakeModel(hp)):
        hp = {"budget": 10}
        hpo._oof_score("t-boost", _bundle(400), hp, 2, 7,
                       selection_hp={"_selection_es_adaptive": 1.2})
    assert all(h.get("_selection_es_adaptive") == 1.2 for h in captured["hps"])
    assert "_selection_es_adaptive" not in hp    # caller's hp untouched


# --- Scout walk + ship-fidelity finale (Ralph 2026-07-16, hpo_search_version 11) -----------------


def _fake_scout_walk(monkeypatch_ctx, finale_overlay):
    """Drive _select_config_coordinate with a 1-stage fake plan and a faked _oof_score.

    Returns (result_tuple, calls) where calls records (hp, selection_hp) per eval in order.
    Scores: candidate hurdle=0.5 clearly beats base; default is worst. Finale evals (marked by
    the overlay) return different folds and best_iter so their payloads are distinguishable.
    """
    import unittest.mock as mock

    calls = []
    SCOUT = {"_es": 1.2, "n_bags": 1}

    def fake_oof(model_name, otr, hp, k, seed, repeats=1, fold_workers=1,
                 selection_hp=None, racing=None):
        calls.append((dict(hp), dict(selection_hp or {})))
        finale = selection_hp == finale_overlay and finale_overlay != SCOUT
        base_folds = {
            ("default",): [1.05, 1.06, 1.04],
            ("base",): [1.00, 1.01, 0.99],
            ("cand",): [0.90, 0.91, 0.89],
        }
        key = ("cand",) if hp.get("hurdle") == 0.5 else ("base",) if "hurdle" in hp else ("default",)
        folds = [f - (0.01 if finale else 0.0) for f in base_folds[key]]
        return float(np.mean(folds)), (7 if finale else 3), folds, False

    plan = {"base": {"hurdle": 2.0}, "stages": [{"name": "hurdle", "values": [0.5, 2.0]}]}
    ctx = mock.patch.multiple(
        hpo,
        coordinate_tuning_plan=lambda name, hp_override=None: plan,
        static_configs=lambda name, n, hp_override=None: [{"budget": 1}],
        _model_selection_hp=lambda name: dict(SCOUT),
        _model_selection_finale_hp=lambda name: dict(finale_overlay),
        _oof_score=fake_oof,
    )
    with ctx:
        result = hpo.select_config("fake", _bundle(400), 1, 3, 42)
    return result, calls


def test_scout_walk_gates_on_ship_fidelity_finale():
    finale_overlay = {"_es": 1.2}
    (hp, best_iter, traj, evaluations, selection), calls = _fake_scout_walk(None, finale_overlay)

    # Walk: default anchor, base, one stage candidate — then finale default + finale incumbent.
    assert [e.get("stage") for e in evaluations] == ["default", "base", 1, "finale", "finale"]
    assert [e.get("param") for e in evaluations[-2:]] == ["default_anchor", "incumbent"]
    # Scout evals carried the scout overlay; finale evals the finale overlay.
    assert [c[1] for c in calls[:3]] == [{"_es": 1.2, "n_bags": 1}] * 3
    assert [c[1] for c in calls[3:]] == [finale_overlay] * 2
    # Gate compares the two FINALE payloads (indices 3, 4), not the scout anchors.
    assert selection["champion_idx"] == 4 and selection["shipped"] == "champion"
    assert selection["default_score"] == evaluations[3]["score"]
    assert selection["champion_score"] == evaluations[4]["score"]
    # Winner + deployed iteration count come from the finale incumbent eval.
    assert hp == {"hurdle": 0.5} and best_iter == 7
    # The scout overlay never leaks into the returned hp.
    assert "n_bags" not in hp and "_es" not in hp


def test_finale_skipped_when_adapter_declares_none():
    # finale overlay == selection overlay -> nothing to re-evaluate; legacy shape preserved.
    (hp, best_iter, traj, evaluations, selection), calls = _fake_scout_walk(
        None, {"_es": 1.2, "n_bags": 1})
    assert [e.get("stage") for e in evaluations] == ["default", "base", 1]
    assert selection["champion_idx"] == 2 and best_iter == 3
    assert hp == {"hurdle": 0.5}


def test_triboost_static_candidates_leave_n_bags_to_recipe():
    # Post-migration (2026-07-20) tri-boost tunes on the static basis, not a coordinate walk:
    # n_bags is not a tuned dimension. Since hsv 15 (2026-07-22) it is also NOT pinned in the
    # candidate hp — the recipe's bag count is objective-aware (gamma/tweedie 16, else 8) and an
    # hp pin would override it on champion deploys while the {budget}-only anchor picked up the
    # recipe value. The scout overlay still drops every candidate to 1 bag during OOF scoring.
    from insurarena.models.registry import model_class

    assert model_class("t-boost").coordinate_tuning_stages() == []
    assert hpo.effective_n_configs("t-boost", 99) == 99
    cands = hpo.static_configs("t-boost", 25)
    assert all("n_bags" not in c for c in cands)
