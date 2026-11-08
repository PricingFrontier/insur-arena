from types import SimpleNamespace

import pytest

from insurarena.eval import hpo
from insurarena.eval import runner
from insurarena.models.tboost_model import TBoost, _STATIC_TUNE_GRID

# tri-boost migrated to the static 25-candidate basis on 2026-07-20. The coordinate WALK MACHINERY
# still lives in hpo.py, so the tests below exercise it by restoring tri-boost's historical
# coordinate stages (identical grids — sample_hp now draws from this same _STATIC_TUNE_GRID).
# Two DIMENSIONS joined the static grid after the walk was retired and so were never walk stages:
# cat_smooth (hsv 15, 2026-07-22) and max_bin (hsv 18, 2026-08-23). The restored historical stages
# exclude both. Contrast learning_rate 0.02 (hsv 17): a new VALUE inside a stage the walk DID have,
# which legitimately lengthened the restored walk, so there the hardcoded counts below were bumped.
# hsv 20 added no grid dimension at all — `max_depth` and `cell_refit_base` enter as COORDINATE
# PROBES, which the walk never had either — so this list is unchanged from hsv 19.
_ADDED_AFTER_WALK_RETIRED = ("cat_smooth", "max_bin")
_COORD_STAGES = [{"name": n, "values": list(v)} for n, v in _STATIC_TUNE_GRID.items()
                 if n not in _ADDED_AFTER_WALK_RETIRED]


@pytest.fixture
def triboost_coordinate(monkeypatch):
    monkeypatch.setattr(
        TBoost, "coordinate_tuning_stages",
        staticmethod(lambda: [dict(s) for s in _COORD_STAGES]),
    )


def _fake_bundle():
    return SimpleNamespace(task=SimpleNamespace(primary_metric="poisson_deviance", tweedie_power=None))


def _fake_cfg():
    return SimpleNamespace(
        split={"small": {"folds": 3, "repeats": 1}},
        n_configs=12,
        inner_folds=2,
        tune_split_id=0,
        time_budget_s=None,
    )


def test_triboost_dispatches_to_static_after_migration():
    # Regression guard for the 2026-07-20 coordinate->static migration: tri-boost declares no
    # coordinate plan, so the HPO dispatch uses the static 25-candidate search (uncapped n_configs),
    # like every other tunable model.
    assert hpo.coordinate_tuning_plan("t-boost") is None
    assert hpo.effective_n_configs("t-boost", 50) == 50


def test_triboost_coordinate_plan_order_and_carry_forward(triboost_coordinate, monkeypatch):
    def fake_eval(
        model_name,
        otr,
        hp,
        config_idx,
        k,
        seed,
        metric,
        direction,
        lower_better,
        cache,
        dataset_hash,
        candidate_proto,
        metadata=None,
        repeats=1,
        **_speed_levers,
    ):
        if "interaction_gain_hurdle" not in hp:
            score = 100.0                       # the default anchor: bad, so the gate ships the walk
        else:
            score = (
                {0.0: 6.0, 0.5: 4.5, 2.0: 3.0, 5.0: 5.0, 30.0: 0.0}[
                    float(hp["interaction_gain_hurdle"])
                ]
                + {1: 1.0, 2: 0.0, 3: 3.0}[int(hp["max_interaction_order"])]
                + {0.1: 0.0, 1.0: 1.0, 10.0: 2.0}[float(hp["lambda_"])]
                # 0.6 and 0.45 joined _STATIC_TUNE_GRID at hsv 23 (av44), and 5/20 joined
                # min_data_in_leaf in the same bump. Both entries score ABOVE the walk's incumbent
                # (0.8 and 50 stay at 0.0), so the walk's chosen values — and therefore every
                # assertion below — are unmoved; the entries only keep the lookup total.
                + {1.0: 1.0, 0.8: 0.0, 0.6: 0.5, 0.45: 1.5}[float(hp["colsample_bytree"])]
                + {4: 1.0, 8: 0.0}[int(hp.get("n_bags", 8))]   # unpinned since hsv 15 (recipe-owned)
                + {0: 1.0, 5: 0.5, 20: 0.25, 50: 0.0, 200: 2.0}[int(hp["min_data_in_leaf"])]
                # 0.02 joined _STATIC_TUNE_GRID at hsv 17 (av34). It scores BETWEEN 0.1 and 0.01
                # and above the base 0.05, so the walk's chosen rate is unchanged and every
                # assertion below is unmoved — the entry only keeps the lookup total.
                + {0.1: 1.0, 0.05: 0.0, 0.02: 1.5, 0.01: 3.0}[float(hp["learning_rate"])]
            )
        payload = {
            "kind": "tuned_hpo_config",
            "config_idx": int(config_idx),
            "hp": dict(hp),
            "metric": metric,
            "direction": direction,
            "inner_folds": int(k),
            "seed": int(seed),
            "score": float(score),
            "best_iter": int(config_idx + 1),
            "failed": "",
            "eval_time_s": 0.0,
        }
        if metadata:
            payload.update(metadata)
        return payload, False

    monkeypatch.setattr(hpo, "_evaluate_config", fake_eval)

    hp, best_iter, traj, evals, selection = hpo.select_config("t-boost", _fake_bundle(), 50, 2, 123)

    assert selection["rule"] == "coordinate_ship_gate_1se"
    assert hp["interaction_gain_hurdle"] == 30.0
    assert hp["max_interaction_order"] == 2
    assert hp["lambda_"] == 0.1
    assert hp["colsample_bytree"] == 0.8
    assert "n_bags" not in hp                    # unpinned since hsv 15: the recipe owns the bag
                                                 # count (objective-aware); no n_bags stage (v11)
    assert hp["min_data_in_leaf"] == 50
    assert hp["learning_rate"] == 0.05
    # v11 scout walk: the deployed iteration count comes from the ship-fidelity FINALE re-eval of
    # the incumbent — the last evaluation (config_idx 21 -> fake best_iter 22). Four evals longer
    # than at hsv 22: hsv 23 (av44) widened `colsample_bytree` by two values and filled
    # `min_data_in_leaf` with two more, and both are restored walk stages. The walk's CHOSEN
    # values are unchanged (asserted above) — only the number of candidates it steps through.
    assert best_iter == 22
    assert len(traj) == len(evals) == 22
    assert [e["param"] for e in evals] == [
        None,
        None,
        "interaction_gain_hurdle",
        "interaction_gain_hurdle",
        "interaction_gain_hurdle",
        "interaction_gain_hurdle",
        "max_interaction_order",
        "max_interaction_order",
        "lambda_",
        "lambda_",
        "colsample_bytree",
        "colsample_bytree",
        "colsample_bytree",  # three non-base fractions since hsv 23 added 0.6/0.45 (av44)
        "min_data_in_leaf",
        "min_data_in_leaf",
        "min_data_in_leaf",
        "min_data_in_leaf",  # four non-base sizes since hsv 23 filled 5/20 (av44)
        "learning_rate",
        "learning_rate",
        "learning_rate",     # three non-base rates since hsv 17 added 0.02 (av34)
        "default_anchor",
        "incumbent",
    ]
    assert [e["stage"] for e in evals[-2:]] == ["finale", "finale"]


def test_triboost_coordinate_plan_respects_budget_cap(triboost_coordinate, monkeypatch):
    def fake_eval(*args, metadata=None, **kwargs):
        hp = dict(args[2])
        config_idx = int(args[3])
        score = {0.0: 4.0, 0.5: 3.0, 2.0: 0.0, 5.0: 5.0, 30.0: 1.0}[
            float(hp.get("interaction_gain_hurdle", 5.0))   # default anchor scores like a bad config
        ]
        payload = {
            "kind": "tuned_hpo_config",
            "config_idx": config_idx,
            "hp": hp,
            "metric": args[6],
            "direction": args[7],
            "inner_folds": int(args[4]),
            "seed": int(args[5]),
            "score": float(score),
            "best_iter": 0,
            "failed": "",
            "eval_time_s": 0.0,
        }
        if metadata:
            payload.update(metadata)
        return payload, False

    monkeypatch.setattr(hpo, "_evaluate_config", fake_eval)

    hp, _, _, evals, _ = hpo.select_config("t-boost", _fake_bundle(), 2, 2, 123)

    # n_configs=2 stage evals + the default anchor + the base + the 2 finale re-evals (protocol
    # overhead, like the anchors) = 6 evaluations total
    assert len(evals) == 6
    assert [e["config_idx"] for e in evals] == [0, 1, 2, 3, 4, 5]
    assert [e.get("stage") for e in evals[-2:]] == ["finale", "finale"]
    assert hp["interaction_gain_hurdle"] == 2.0
    assert hp["max_interaction_order"] == 3


def test_stage_prefer_rule_defends_incumbent_on_near_tie(monkeypatch):
    # The prefer/tolerance near-tie rule is generic walk machinery (t-boost's v11 plan no longer
    # declares a preferring stage, but the code path stays). With the base already at the preferred
    # value the rule must act as a DEFENSE: the 4-bag candidate below scores 0.5% better, which is
    # within the declared tolerance, so the preferred incumbent keeps the stage.
    fake_plan = {
        "base": {"_tuned": True, "n_bags": 8},
        "stages": [{"name": "n_bags", "values": [4, 8], "prefer": "larger",
                    "relative_tolerance": 0.01}],
    }
    monkeypatch.setattr(hpo, "coordinate_tuning_plan", lambda name, hp_override=None: fake_plan)

    def fake_eval(
        model_name,
        otr,
        hp,
        config_idx,
        k,
        seed,
        metric,
        direction,
        lower_better,
        cache,
        dataset_hash,
        candidate_proto,
        metadata=None,
        repeats=1,
        **_speed_levers,
    ):
        score = 1.0 + (0.005 if int(hp.get("n_bags", 4)) == 8 else 0.0)
        payload = {
            "kind": "tuned_hpo_config",
            "config_idx": int(config_idx),
            "hp": dict(hp),
            "metric": metric,
            "direction": direction,
            "inner_folds": int(k),
            "seed": int(seed),
            "score": float(score),
            "best_iter": 0,
            "failed": "",
            "eval_time_s": 0.0,
        }
        if metadata:
            payload.update(metadata)
        return payload, False

    monkeypatch.setattr(hpo, "_evaluate_config", fake_eval)

    hp, _, _, evals, _ = hpo.select_config("t-boost", _fake_bundle(), 50, 2, 123)

    assert hp["n_bags"] == 8
    stage_evals = [e for e in evals if e.get("param") == "n_bags"]
    assert [e["value"] for e in stage_evals] == [4]
    # The rejected candidate genuinely scored better (lower deviance) — that is the point: it was
    # within the declared tolerance, so the variance-reducing 8-bag incumbent won the near-tie.
    assert stage_evals[0]["score"] < 1.005


def test_coordinate_plan_normalizes_oversized_budget_for_signatures(triboost_coordinate):
    # 18 = one candidate per non-base value across the restored stages: v11 dropped the n_bags
    # stage (13), hsv 17 (av34) added learning_rate 0.02 (14), and hsv 23 (av44) added
    # colsample_bytree 0.6/0.45 and min_data_in_leaf 5/20 (18).
    assert hpo.effective_n_configs("t-boost", 50) == 18
    assert hpo.config_set_digest("t-boost", 18) == hpo.config_set_digest("t-boost", 50)
    assert hpo.effective_n_configs("ebm", 50) == 50        # coordinate walk retired -> static search
    assert hpo.effective_n_configs("glm", 50) == 50


def test_hpo_search_version_is_in_tuned_hpo_signatures(monkeypatch):
    monkeypatch.setattr(TBoost, "hpo_search_version", "search-test")

    cfg = _fake_cfg()
    bundle = _fake_bundle()

    assert "hpo_search_version" not in runner._proto(cfg, bundle, "t-boost", "default", {})
    assert runner._proto(cfg, bundle, "t-boost", "tuned", {})["hpo_search_version"] == "search-test"
    assert runner._hpo_proto(cfg, bundle, "t-boost", {})["hpo_search_version"] == "search-test"
    assert runner._hpo_config_proto(cfg, bundle, "t-boost", {})["hpo_search_version"] == "search-test"


def _folded_eval(fold_map):
    """fake _evaluate_config keyed on (metadata stage/param, hurdle value) -> fold scores."""
    def fake_eval(model_name, otr, hp, config_idx, k, seed, metric, direction, lower_better,
                  cache, dataset_hash, candidate_proto, metadata=None, repeats=1, **_speed_levers):
        meta = metadata or {}
        if meta.get("stage") == "finale":
            # v11 ship-fidelity finale: the default anchor re-scores like the default; the
            # incumbent re-scores by its own hurdle value (base folds when it never moved).
            if meta.get("param") == "default_anchor":
                folds = fold_map["default"]
            else:
                folds = fold_map.get(float(hp.get("interaction_gain_hurdle", -1.0)),
                                     fold_map["base"])
        elif meta.get("stage") == "default":
            folds = fold_map["default"]
        elif meta.get("stage") == "base":
            folds = fold_map["base"]
        elif meta.get("param") == "interaction_gain_hurdle":
            folds = fold_map.get(float(meta["value"]), fold_map["reject"])
        else:
            folds = fold_map["reject"]
        payload = {
            "kind": "tuned_hpo_config", "config_idx": int(config_idx), "hp": dict(hp),
            "metric": metric, "direction": direction, "inner_folds": int(k),
            "inner_repeats": int(repeats), "seed": int(seed),
            "score": float(sum(folds) / len(folds)), "best_iter": 0,
            "fold_scores": list(folds), "failed": "", "eval_time_s": 0.0,
        }
        payload.update(meta)
        return payload, False
    return fake_eval


def test_coordinate_stage_move_needs_one_se(triboost_coordinate, monkeypatch):
    # Hurdle 0.0 beats the base pooled (0.99933 < 1.0) — the OLD rule would take it — but its
    # paired margin (0.00067) is under 1 SE (0.00088), so it must be rejected. Hurdle 30.0 wins
    # every fold and must be accepted. The default anchor is far worse, so the walk ships.
    fold_map = {
        "default": [1.50, 1.52, 1.48],
        "base":    [1.00, 1.02, 0.98],
        0.0:       [0.999, 1.021, 0.978],
        30.0:      [0.95, 0.97, 0.93],
        "reject":  [1.20, 1.22, 1.18],
    }
    monkeypatch.setattr(hpo, "_evaluate_config", _folded_eval(fold_map))

    hp, _, _, evals, sel = hpo.select_config("t-boost", _fake_bundle(), 50, 3, 123)

    assert hp["interaction_gain_hurdle"] == 30.0
    assert sel["shipped"] == "champion"
    assert sel["reason"] == "beats_default_by_1se"
    assert sel["base_is_default"] is False
    assert evals[0]["stage"] == "default"


def test_coordinate_gate_ships_default_when_walk_is_noise(triboost_coordinate, monkeypatch):
    # Base and every stage move are within noise of the default -> the tuned variant must ship
    # the true default hp, not the tuned-family base.
    fold_map = {
        "default": [1.00, 1.02, 0.98],
        "base":    [0.999, 1.021, 0.979],
        "reject":  [1.20, 1.22, 1.18],
    }
    monkeypatch.setattr(hpo, "_evaluate_config", _folded_eval(fold_map))

    hp, _, _, _, sel = hpo.select_config("t-boost", _fake_bundle(), 50, 3, 123)

    assert sel["shipped"] == "default"
    assert sel["reason"] == "within_1se_of_default"
    assert hp == TBoost.default_hp()
    assert "_tuned" not in hp
