"""Selection protocol v2: size-adaptive inner repeats + the static default ship-gate."""
from types import SimpleNamespace

import pytest

from insurarena.config import _parse_inner_repeats
from insurarena.eval import hpo, runner
from insurarena.models.gbm import LightGBM


def _fake_bundle():
    return SimpleNamespace(task=SimpleNamespace(primary_metric="poisson_deviance", tweedie_power=None))


def _fake_cfg(**over):
    ns = dict(
        split={"small": {"folds": 3, "repeats": 1}},
        n_configs=2,
        inner_folds=3,
        inner_repeats="auto",
        tune_split_id=0,
        time_budget_s=None,
    )
    ns.update(over)
    return SimpleNamespace(**ns)


def test_resolve_inner_repeats_auto_is_size_adaptive():
    assert hpo.resolve_inner_repeats("auto", 1455) == 5      # swautoins-sized: capped at 5
    assert hpo.resolve_inner_repeats("auto", 4989) == 5      # sgautonb-sized: still capped
    assert hpo.resolve_inner_repeats("auto", 15_000) == 2
    assert hpo.resolve_inner_repeats("auto", 30_000) == 1
    assert hpo.resolve_inner_repeats("AUTO", 45_238) == 1    # ausprivauto+: single pass
    assert hpo.resolve_inner_repeats(3, 10) == 3             # explicit int wins regardless of size
    assert hpo.resolve_inner_repeats(0, 10) == 1
    assert hpo.resolve_inner_repeats(None, 10) == 1
    assert hpo.resolve_inner_repeats("auto", 0) == 1
    with pytest.raises(ValueError):
        hpo.resolve_inner_repeats("always", 10)


def test_parse_inner_repeats_config():
    assert _parse_inner_repeats("auto") == "auto"
    assert _parse_inner_repeats(None) == "auto"
    assert _parse_inner_repeats(4) == 4
    assert _parse_inner_repeats(-1) == 1
    with pytest.raises(ValueError):
        _parse_inner_repeats("sometimes")


def _patched_oof(monkeypatch, fold_map):
    """fold_map: candidate hp seed (0 = default) -> per-fold scores."""
    def fake_oof(model_name, otr, hp, k, seed, repeats=1, **_speed_levers):
        folds = fold_map[int(hp.get("seed", 0))]
        return float(sum(folds) / len(folds)), 7 if "seed" not in hp else 9, list(folds), False

    monkeypatch.setattr(hpo, "_oof_score", fake_oof)


def test_ship_gate_keeps_default_on_within_noise_win(monkeypatch):
    # Champion (candidate 1) is better pooled, but the paired fold diffs are noise: mean 0.0033
    # vs SE 0.0067 -> the default ships, with the champion's evidence recorded for audit.
    _patched_oof(monkeypatch, {0: [1.00, 1.02, 0.98], 1: [0.99, 1.03, 0.97]})

    hp, best_iter, traj, evals, sel = hpo.select_config("lightgbm", _fake_bundle(), 1, 3, 123)

    assert sel["shipped"] == "default"
    assert sel["reason"] == "within_1se_of_default"
    assert sel["champion_idx"] == 1
    assert sel["n_paired_folds"] == 3
    assert 0 < sel["gate_margin"] < sel["gate_se"]
    assert hp == LightGBM.default_hp()
    assert best_iter == 7                        # the default candidate's OOF iteration estimate
    assert evals[1]["fold_scores"] == [0.99, 1.03, 0.97]


def test_ship_gate_ships_champion_on_clear_win(monkeypatch):
    _patched_oof(monkeypatch, {0: [1.00, 1.02, 0.98], 1: [0.90, 0.93, 0.87]})

    hp, best_iter, _, _, sel = hpo.select_config("lightgbm", _fake_bundle(), 1, 3, 123)

    assert sel["shipped"] == "champion"
    assert sel["reason"] == "beats_default_by_1se"
    assert sel["gate_margin"] > sel["gate_se"] > 0
    assert hp["seed"] == 1
    assert best_iter == 9


def test_ship_gate_ships_default_when_default_eval_fails(monkeypatch):
    # 2026-07-22 (hsv 15): brautocoll deployed an UNGATED champion 14% behind the default
    # because the default anchor's eval crashed on a single-class inner-fold slice. A champion
    # that cannot be gate-validated must not ship: the gate now ships the default, whose deploy
    # either works (the crash was an inner-fold artifact) or fails as a VISIBLE failed cell.
    def fake_oof(model_name, otr, hp, k, seed, repeats=1, **_speed_levers):
        if "seed" not in hp:                       # the default anchor
            raise ValueError("logistic: no positive (or no negative) class under w")
        return 0.90, 9, [0.90, 0.93, 0.87], False

    monkeypatch.setattr(hpo, "_oof_score", fake_oof)
    hp, best_iter, _, evals, sel = hpo.select_config("lightgbm", _fake_bundle(), 1, 3, 123)

    assert sel["shipped"] == "default"
    assert sel["reason"] == "default_failed_ship_default"
    assert hp == LightGBM.default_hp()
    assert best_iter == 0                          # a failed eval has no OOF iteration estimate
    assert evals[0]["failed"]


def test_ship_gate_trivial_when_default_wins_search(monkeypatch):
    _patched_oof(monkeypatch, {0: [0.90, 0.92, 0.88], 1: [1.00, 1.02, 0.98]})

    hp, _, _, _, sel = hpo.select_config("lightgbm", _fake_bundle(), 1, 3, 123)

    assert sel["shipped"] == "champion" and sel["reason"] == "champion_is_default"
    assert hp == LightGBM.default_hp()


def test_pooled_finale_reverts_split_specific_champion(monkeypatch):
    # Protocol v3 (hsv 16): a champion that wins the PRIMARY tune split but not the pooled
    # finale across extra tune splits must ship the default. Scout-regime only (t-boost);
    # models whose finale == selection overlay never reach the pooled path.
    primary = _fake_bundle()
    extra1, extra2 = _fake_bundle(), _fake_bundle()
    extra_ids = {id(extra1), id(extra2)}

    def fake_oof(model_name, otr, hp, k, seed, repeats=1, **levers):
        champion = "seed" in hp                    # static candidates carry a config seed; the anchor doesn't
        if id(otr) in extra_ids:                   # extra finale splits: champion's edge evaporates
            folds = [1.05, 1.05, 1.05] if champion else [1.00, 1.00, 1.00]
        else:
            folds = [0.90, 0.90, 0.90] if champion else [1.00, 1.00, 1.00]
        return float(sum(folds) / len(folds)), 5, list(folds), False

    monkeypatch.setattr(hpo, "_oof_score", fake_oof)

    hp, best_iter, _, evals, sel = hpo.select_config(
        "t-boost", primary, 1, 3, 123, extra_finale_otrs=[extra1, extra2])
    assert sel["rule"] == "static_ship_gate_1se_msplit"
    assert sel["n_finale_splits"] == 3
    assert sel["shipped"] == "default"
    assert sel["reason"] == "within_1se_of_default_pooled"
    assert hp.get("budget")                       # the t-boost default anchor
    assert sum(1 for e in evals if (e.get("param") or "").startswith("pool_")) == 4

    # regression: without extra splits the single-split gate ships the champion as before.
    hp2, _, _, _, sel2 = hpo.select_config("t-boost", primary, 1, 3, 123)
    assert sel2["rule"] == "static_ship_gate_1se"
    assert sel2["shipped"] == "champion"
    assert "seed" in hp2


def test_repeats_reach_candidate_payloads(monkeypatch):
    seen = {}

    def fake_oof(model_name, otr, hp, k, seed, repeats=1, **_speed_levers):
        seen["repeats"] = repeats
        return 1.0, 5, [1.0] * (k * repeats), False

    monkeypatch.setattr(hpo, "_oof_score", fake_oof)
    _, _, _, evals, sel = hpo.select_config("lightgbm", _fake_bundle(), 1, 3, 123, repeats=4)

    assert seen["repeats"] == 4
    assert sel["inner_repeats"] == 4
    assert all(e["inner_repeats"] == 4 and len(e["fold_scores"]) == 12 for e in evals)


def test_selection_v2_keys_tuned_protos_only():
    cfg, bundle = _fake_cfg(), _fake_bundle()

    tuned = runner._proto(cfg, bundle, "lightgbm", "tuned", {})
    assert tuned["hpo_selection"] == "v2"
    assert tuned["inner_repeats"] == "auto"

    default = runner._proto(cfg, bundle, "lightgbm", "default", {})
    assert "hpo_selection" not in default          # default cells keep their signatures
    assert "inner_repeats" not in default

    cand = runner._hpo_config_proto(cfg, bundle, "lightgbm", {}, inner_repeats=4)
    assert cand["inner_repeats"] == 4              # resolved count, not the "auto" spec


def test_all_evals_failing_is_refused_not_shipped_as_default(monkeypatch):
    # 2026-08-30 (a CatBoost-family recompute): the coordinator could not `import catboost`,
    # so all 26 candidate evaluations raised ModuleNotFoundError. A failed eval scores +inf and
    # never displaces the incumbent, so `best_idx` stayed 0 and the gate reported
    # "champion_is_default" — publishing the DEFAULT recipe as the TUNED hp on 22 datasets
    # (ohlsson_pp shipped at mean skill -0.965). A search in which nothing scored is not a
    # selection: it must raise, so the cells fail VISIBLY and are never cached.
    def fake_oof(model_name, otr, hp, k, seed, repeats=1, **_speed_levers):
        raise ModuleNotFoundError("No module named 'catboost'")

    monkeypatch.setattr(hpo, "_oof_score", fake_oof)
    with pytest.raises(RuntimeError, match="no scored candidate"):
        hpo.select_config("lightgbm", _fake_bundle(), 1, 3, 123)


def test_ship_gate_reports_failed_anchor_even_when_it_is_the_champion():
    # Gate-level guard for the same defect: champion_idx == default_idx must not short-circuit
    # to "champion_is_default" when the anchor's own evaluation failed.
    evals = [{"config_idx": 0, "score": None, "failed": "ModuleNotFoundError: No module named 'x'",
              "fold_scores": []}]
    gate = hpo._ship_gate(evals, 0, lower_better=True)
    assert gate["shipped"] == "default"
    assert gate["reason"] == "default_failed_ship_default"


def test_failed_candidate_evaluations_are_not_cached(tmp_path, monkeypatch):
    # An environment failure must not be cached as "this config scored nothing" — otherwise every
    # later selection cache-HITs the breakage and reproduces the bad cell forever.
    from insurarena.results.cache import ResultCache

    cache = ResultCache(str(tmp_path))
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise ModuleNotFoundError("No module named 'catboost'")

    monkeypatch.setattr(hpo, "_oof_score", boom)
    for _ in range(2):
        payload, hit = hpo._evaluate_config(
            "lightgbm", _fake_bundle(), {"seed": 1}, 1, 3, 123, "poisson_deviance", "lower",
            True, cache, "deadbeef", {"proto": 1})
        assert payload["failed"] and not hit
    assert calls["n"] == 2                      # recomputed, not served from cache
    assert not list((tmp_path / "cells").glob("*.json"))

    def fine(*a, **k):
        calls["n"] += 1
        return 0.5, 9, [0.5, 0.5, 0.5], False

    monkeypatch.setattr(hpo, "_oof_score", fine)
    hpo._evaluate_config("lightgbm", _fake_bundle(), {"seed": 1}, 1, 3, 123, "poisson_deviance",
                         "lower", True, cache, "deadbeef", {"proto": 1})
    _, hit = hpo._evaluate_config("lightgbm", _fake_bundle(), {"seed": 1}, 1, 3, 123,
                                  "poisson_deviance", "lower", True, cache, "deadbeef", {"proto": 1})
    assert hit                                  # a SUCCESSFUL evaluation still caches as before
