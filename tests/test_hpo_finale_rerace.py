"""hsv 23 (2026-09-01) — the TOP-K FINALE RE-RACE, and the fence around it.

THE RULE. t-boost ranks candidates at SCOUT fidelity (its `selection_hp` overlay pins
`n_bags: 1` and turns the prune off) and hands the argmax to the ship gate; the FINALE — full
deploy fidelity — is spent only on the argmax, the default anchor and the hsv-21 regime-recheck
candidate. The re-race admits every basis candidate within `band` paired-fold SEs of the argmax
(at most `k` finale contenders in total, argmax always included), re-scores them at finale
fidelity, and hands the ship gate the winner of THAT.

THE EVIDENCE these tests pin (the av44 top-K re-race battery and selection forensic):
  * 10 datasets, 219 deploy fits + 57 finale evaluations, every candidate deployed on all 9 outer
    splits exactly as `runner.run_benchmark` deploys the champion.
  * At band 1.0 SE / k=4: 1 WIN (eusavingul +0.0049, t=15.5) / 9 null / 0 HARM, which is 86% of
    all the headroom the deployed candidate sets held.
  * k IS THE BINDING GUARD and the band is not. Over band in {0.25 .. 3.0} the tally is identical
    at k <= 4; at k >= 5 pg16_payfreq's pick turns -0.0009 and the net falls to +0.0040.

The house bar these tests defend, in one line: the re-race may correct a coin flip, and it may
never become an unguarded max-of-N over the whole basis.
"""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from insurarena.eval import hpo, runner
from insurarena.models.base import ModelAdapter
from insurarena.models.tboost_model import TBoost


def _fake_bundle(n_rows: int = 200_000, extra: bool = False):
    b = SimpleNamespace(
        task=SimpleNamespace(primary_metric="poisson_deviance", tweedie_power=None),
        X=list(range(n_rows)),
    )
    b._extra = extra
    return b


# A deterministic paired-fold fixture. `_ARG_FOLDS` is the argmax; `_behind(t)` returns a
# candidate whose PAIRED per-fold difference against it has mean `t * SE` and SD `_SD`, i.e. one
# that sits exactly `t` paired-fold SEs behind the argmax. Deterministic rather than sampled so the
# band arithmetic these tests assert on is exact and cannot drift with a seed.
_ARG_FOLDS = [1.0000, 1.0009, 0.9994, 1.0006, 0.9997, 1.0011, 0.9990, 1.0003, 1.0000]
_Z = [1, -1, 1, -1, 1, -1, 1, -1, 0]        # mean 0, sample SD (ddof=1) exactly 1
_SD = 0.0018                                # paired-difference SD -> SE = 0.0006 over 9 folds


def _behind(t: float) -> list[float]:
    se = _SD / 3.0                          # SD / sqrt(9)
    return [a + t * se + _SD * z for a, z in zip(_ARG_FOLDS, _Z)]


# The same construction at the 3-fold shape the end-to-end runs use (inner_folds=3, repeats=1).
_ARG3 = [1.0000, 1.0020, 0.9980]
_Z3 = [1, -1, 0]                            # mean 0, sample SD (ddof=1) exactly 1


def _behind3(t: float, sd: float = _SD) -> list[float]:
    se = sd / np.sqrt(3.0)
    return [a + t * se + sd * z for a, z in zip(_ARG3, _Z3)]


def _ev(config_idx, score, folds, hp=None, **kw):
    """One basis evaluation as `_select_config_static` records it."""
    payload = {"config_idx": config_idx, "score": score, "fold_scores": list(folds),
               # a distinct knob per candidate: `_rerace_key` ignores the draw seed on purpose,
               # so hp that differs ONLY in "seed" is one candidate by construction
               "hp": dict(hp or {"seed": config_idx, "lambda_": float(config_idx)}),
               "failed": "", "raced": False}
    payload.update(kw)
    return payload


# --- the tuned settings themselves ---------------------------------------------------------------


def test_tboost_opts_in_at_the_measured_settings_and_nothing_else_does():
    assert TBoost.finale_rerace_spec() == {"k": 4, "band": 1.0}
    assert hpo._model_rerace_spec("t-boost") == (4, 1.0)
    assert (hpo._RERACE_K, hpo._RERACE_BAND) == (4, 1.0)

    # OPT-IN is the guard that keeps a shared selection function from re-keying four other models.
    assert ModelAdapter.finale_rerace_spec() is None
    for name in ("lightgbm", "xgboost", "catboost", "ebm", "glm"):
        assert hpo._model_rerace_spec(name) is None, name


def test_a_broken_or_absent_hook_fails_open_to_the_pre_hsv23_protocol():
    # Failing open matters: the re-race can only ADD evaluations and MOVE a champion, so falling
    # back to "no re-race" is the historical behavior, never a weakened gate.
    for spec in (None, {}, {"k": 1}, {"k": 4, "band": float("nan")}, {"k": 4, "band": -1.0},
                 {"k": "four"}):
        with mock.patch.object(TBoost, "finale_rerace_spec", staticmethod(lambda s=spec: s)):
            assert hpo._model_rerace_spec("t-boost") is None, spec

    def boom():
        raise RuntimeError("adapter hook exploded")

    with mock.patch.object(TBoost, "finale_rerace_spec", staticmethod(boom)):
        assert hpo._model_rerace_spec("t-boost") is None


# --- k = 4 is the measured boundary --------------------------------------------------------------


def _pg16_evaluations():
    """pg16_payfreq's shape, from the battery's own per-set table (rerace_report.txt).

        rank   SE behind argmax   finale score   deployed outer skill
           1               0.00       1.021582                 0.0030   <- scout argmax
           2               0.21       1.022290                 0.0004
           3               0.34       1.021962                 0.0033   <- the oracle
           4               0.50       1.021781                 0.0017
           5               0.67       1.020248                 0.0021   <- the TRAP

    Rank 5 has the BEST finale score of the five (lower is better) and deploys 0.0009 BELOW the
    argmax. It is what turns the board-wide net from +0.0049 to +0.0040 the moment k reaches 5.
    Note where it sits: 0.67 SE behind, i.e. comfortably inside every band the sweep tried
    (0.25 through 3.0). Only the RANK cap excludes it.

    The SE-behind column is reproduced exactly here; the finale and deploy columns are the measured
    numbers and are quoted, not recomputed.
    """
    ladder = [0.00, 0.21, 0.34, 0.50, 0.67]
    evals = []
    for i, t in enumerate(ladder):
        folds = _ARG_FOLDS if i == 0 else _behind(t)
        evals.append(_ev(i + 1, float(np.mean(folds)), folds,
                         hp={"seed": i + 1, "lambda_": float(i + 1)}))
    return evals


def test_k_is_four_and_pg16_payfreq_is_the_counterexample():
    evals = _pg16_evaluations()
    argmax_idx = 1

    at_4 = hpo._rerace_contenders(evals, argmax_idx, [evals[0]["hp"]], {"budget": 4000},
                                  n_basis=99, rr_k=4, band=1.0, lower_better=True)
    admitted_4 = [idx for _, idx, _ in at_4]
    assert admitted_4 == [2, 3, 4], "k=4 = the argmax plus at most THREE challengers"
    assert 5 not in admitted_4, (
        "scout rank 5 is pg16_payfreq's harmful pick: it wins the finale (1.020248) and deploys "
        "at 0.0021 against the argmax's 0.0030. k=4 is the only thing that excludes it — it is "
        "just 0.67 SE behind, so no band in the swept range 0.25..3.0 refuses it."
    )

    at_5 = hpo._rerace_contenders(evals, argmax_idx, [evals[0]["hp"]], {"budget": 4000},
                                  n_basis=99, rr_k=5, band=1.0, lower_better=True)
    assert [idx for _, idx, _ in at_5] == [2, 3, 4, 5], (
        "k=5 admits it — this is the measured max-of-N boundary, not a style preference: "
        "1 win / 9 null / 0 harm at k<=4 becomes net +0.0040 at k>=5."
    )


def test_the_band_is_a_rail_and_admission_is_rank_contiguous():
    # Admission walks scout-rank order and STOPS at the first refusal, so the contender set is
    # always ranks 1..m. That is what makes k a rank truncation and therefore robust to the SE
    # scale — see `hpo._RERACE_K` on why the in-engine SE is not the offline sweep's gate_se.
    evals = [
        _ev(1, float(np.mean(_ARG_FOLDS)), _ARG_FOLDS, hp={"seed": 1, "lambda_": 1.0}),
        _ev(2, float(np.mean(_behind(0.5))), _behind(0.5), hp={"seed": 2, "lambda_": 2.0}),
        _ev(3, float(np.mean(_behind(4.0))), _behind(4.0), hp={"seed": 3, "lambda_": 3.0}),
        # ALSO a coin flip with the argmax, but it sits BEHIND the refusal in rank order
        _ev(4, float(np.mean(_behind(0.9))), _behind(0.9), hp={"seed": 4, "lambda_": 4.0}),
    ]
    # rank order is by SCORE, so the 0.9-SE candidate is rank 3 and the 4.0-SE one is rank 4 —
    # rebuild the list the way the engine sees it (sorted on score) to make the point about
    # contiguity rather than about list order.
    got = hpo._rerace_contenders(evals, 1, [evals[0]["hp"]], {"budget": 4000},
                                 n_basis=99, rr_k=4, band=1.0, lower_better=True)
    assert [idx for _, idx, _ in got] == [2, 4], "both coin flips in, the 4-SE candidate out"
    assert got[0][2] == pytest.approx(0.5, abs=1e-6)       # se_behind is recorded, in SEs
    assert got[1][2] == pytest.approx(0.9, abs=1e-6)

    # ...and a refusal that comes FIRST in rank order stops admission dead.
    blocked = [
        evals[0],
        _ev(5, float(np.mean(_behind(4.0))), _behind(4.0), hp={"seed": 5, "lambda_": 5.0}),
        _ev(6, float(np.mean(_behind(5.0))), _behind(5.0), hp={"seed": 6, "lambda_": 6.0}),
    ]
    assert hpo._rerace_contenders(blocked, 1, [blocked[0]["hp"]], {"budget": 4000},
                                  n_basis=99, rr_k=4, band=1.0, lower_better=True) == []


def test_the_default_anchor_is_never_a_contender():
    # The anchor is the ship gate's REFERENCE and is already evaluated at finale fidelity.
    # Re-racing it would enter it into its own comparison.
    anchor = {"budget": 4000, "seed": 0}
    rng = np.random.default_rng(11)
    folds = lambda base: [base + float(v) for v in rng.normal(0, 0.001, size=9)]
    evals = [
        _ev(1, 1.000, folds(1.000), hp={"seed": 1, "max_depth": 6}),
        _ev(0, 1.0004, folds(1.0004), hp=dict(anchor)),
        _ev(2, 1.0005, folds(1.0005), hp={"seed": 2, "max_bin": 64}),
    ]
    got = hpo._rerace_contenders(evals, 1, [evals[0]["hp"]], anchor,
                                 n_basis=99, rr_k=4, band=1.0, lower_better=True)
    assert [idx for _, idx, _ in got] == [2]


def test_duplicate_draws_are_deduped_on_knob_identity_not_on_the_draw_seed():
    # `static_configs` stamps hp["seed"] = i on every candidate, so two draws that are the SAME
    # model differ in exactly that key. Re-racing both spends a finale evaluation comparing a
    # config to itself.
    twin = {"max_bin": 64, "learning_rate": 0.05}
    evals = [
        _ev(1, float(np.mean(_ARG_FOLDS)), _ARG_FOLDS, hp={"seed": 1, "max_depth": 6}),
        _ev(2, float(np.mean(_behind(0.2))), _behind(0.2), hp={"seed": 2, **twin}),
        _ev(3, float(np.mean(_behind(0.3))), _behind(0.3), hp={"seed": 3, **twin}),
        _ev(4, float(np.mean(_behind(0.4))), _behind(0.4),
            hp={"seed": 4, "min_data_in_leaf": 20}),
    ]
    got = hpo._rerace_contenders(evals, 1, [evals[0]["hp"]], {"budget": 4000},
                                 n_basis=99, rr_k=4, band=1.0, lower_better=True)
    assert [idx for _, idx, _ in got] == [2, 4]


def test_raced_and_failed_candidates_are_ineligible():
    # A raced payload carries ONE fold and no score. There is nothing to band against, and the
    # re-race cannot recover it — the same honest limit the hsv-21 regime recheck records.
    evals = [
        _ev(1, float(np.mean(_ARG_FOLDS)), _ARG_FOLDS, hp={"seed": 1, "lambda_": 1.0}),
        _ev(2, np.inf, [1.0002], hp={"seed": 2, "lambda_": 2.0}, raced=True),
        _ev(3, None, [], hp={"seed": 3, "lambda_": 3.0}, failed="boom"),
        _ev(4, float(np.mean(_behind(0.5))), _behind(0.5), hp={"seed": 4, "lambda_": 4.0}),
    ]
    got = hpo._rerace_contenders(evals, 1, [evals[0]["hp"]], {"budget": 4000},
                                 n_basis=99, rr_k=4, band=1.0, lower_better=True)
    assert [idx for _, idx, _ in got] == [4]


def test_finale_re_evaluations_are_never_ranked_against_basis_scores():
    # Mixing the two scales is the artifact `cand_arm.basis_candidates` documents: finale scores
    # come from a different fidelity overlay (and, on the pooled splits, a different fold count),
    # and on eusavingul the mixed list reads the argmax's 0.19-SE lead as a 12-SE one.
    better = [f - 0.05 for f in _ARG_FOLDS]
    evals = [
        _ev(1, float(np.mean(_ARG_FOLDS)), _ARG_FOLDS, hp={"seed": 1, "lambda_": 1.0}),
        _ev(2, float(np.mean(_behind(0.4))), _behind(0.4), hp={"seed": 2, "lambda_": 2.0}),
        _ev(9, float(np.mean(better)), better, hp={"seed": 0, "budget": 4000},
            metadata={"stage": "finale", "param": "default_anchor"}),
    ]
    got = hpo._rerace_contenders(evals, 1, [evals[0]["hp"]], {"budget": 4000},
                                 n_basis=5, rr_k=4, band=1.0, lower_better=True)
    assert [idx for _, idx, _ in got] == [2], "config_idx 9 is past the basis and must be ignored"


# --- end to end through `_select_config_static` ---------------------------------------------------


def _drive(scout, finale, pooled=None, n_configs=3, extras=0, mode="always"):
    """Run t-boost's static search with faked fold vectors.

    `scout` / `finale` / `pooled` map candidate index -> fold vector. `finale` is keyed on the
    candidate index too; the FINALE anchor re-evaluation of candidate 0 reads `finale[0]`.
    """
    def fake_oof(model_name, otr, hp, k, seed, repeats=1, selection_hp=None, **_kw):
        idx = int(hp.get("seed", 0))
        table = scout
        if not (selection_hp or {}).get("n_bags"):            # the finale overlay drops n_bags=1
            table = pooled if getattr(otr, "_extra", False) else finale
        folds = list(table[idx])
        return float(np.mean(folds)), 40 + idx, folds, False

    def thunk():
        return _fake_bundle(extra=True)

    with mock.patch.object(hpo, "_oof_score", fake_oof):
        hp, best_iter, _, evals, sel = hpo.select_config(
            "t-boost", _fake_bundle(), n_configs, 3, 123,
            extra_finale_otrs=[thunk for _ in range(extras)] or None,
            pooled_finale_mode=mode,
        )
    return hp, best_iter, evals, sel


def test_the_rerace_flips_a_coin_flip_champion_and_records_why():
    # eusavingul's shape. Candidate 1 wins the scout by a hair (the real cell: 0.19 SE, a coin
    # flip); candidate 2 is the one that wins once the finale's bags are switched on. The toggle
    # experiment measures exactly that flip: +0.0000837 at 1 bag (t=+0.06), -0.0012251 with
    # bagging on (t=-3.54, 8 of 9 folds), and the candidate it promotes is worth +0.00435 outer
    # skill on 18 of 18 deploys.
    scout = {0: _behind3(20.0), 1: _ARG3, 2: _behind3(0.19), 3: _behind3(9.0)}
    finale = {0: _behind3(20.0), 1: _ARG3, 2: _behind3(-6.0), 3: _behind3(9.0)}
    hp, best_iter, evals, sel = _drive(scout, finale)

    rr = sel["finale_rerace"]
    assert rr["k"] == 4 and rr["band"] == 1.0
    assert rr["scout_champion_idx"] == 1
    assert [a["scout_idx"] for a in rr["admitted"]] == [2]
    assert rr["admitted"][0]["se_behind"] == pytest.approx(0.19, abs=1e-6)
    assert rr["single_split_changed"] is True
    assert sel["shipped"] == "champion"
    assert hp["seed"] == 2, "the re-race winner, not the scout argmax, is what ships"
    assert best_iter == 42, "the deployed iteration count comes from the WINNER's finale eval"
    tags = [e.get("param") for e in evals if e.get("stage") == "finale"]
    assert tags.count("rerace") == 1, "exactly one extra finale evaluation was spent"


def test_the_rerace_holds_the_slot_on_a_tie_and_stays_silent_when_nothing_is_admitted():
    # A tie at the finale must NOT move the cell: the re-race is a correction mechanism, not a
    # second coin flip.
    scout = {0: _behind3(20.0), 1: _ARG3, 2: _behind3(0.19), 3: _behind3(9.0)}
    tie = {0: _behind3(20.0), 1: _ARG3, 2: list(_ARG3), 3: _behind3(9.0)}
    hp, _, _, sel = _drive(scout, tie)
    assert sel["finale_rerace"]["single_split_changed"] is False
    assert hp["seed"] == 1

    # And where the scout is decisive there is nothing to re-race at all: 10 of the 40 cached
    # hsv-22 cells admit no contender, and those cells cost exactly what they cost today.
    decisive = {0: _behind3(20.0), 1: _ARG3, 2: _behind3(9.0), 3: _behind3(12.0)}
    hp, _, evals, sel = _drive(decisive, decisive)
    assert "finale_rerace" not in sel
    assert [e.get("param") for e in evals if e.get("stage") == "finale"].count("rerace") == 0


def test_the_pooled_finale_decides_the_rerace_where_the_cell_buys_it():
    # The ohlsson counterexample, in shape: on the PRIMARY split the runner-up wins the finale,
    # and on the pooled evidence the scout argmax is right after all. Replaying the battery with
    # the pick made on split 0 alone mis-picks exactly this cell (its scout rank 3 wins split 0
    # and deploys at 0.0045 against the argmax's 0.0047), which is why the re-race is settled at
    # pooled fidelity rather than on one split.
    scout = {0: _behind3(20.0), 1: _ARG3, 2: _behind3(0.19), 3: _behind3(9.0)}
    finale = {0: _behind3(20.0), 1: _ARG3, 2: _behind3(-2.0), 3: _behind3(9.0)}
    pooled = {0: _behind3(20.0), 1: _behind3(-9.0), 2: _ARG3, 3: _behind3(9.0)}
    hp, best_iter, evals, sel = _drive(scout, finale, pooled=pooled, extras=2, mode="always")

    rr = sel["finale_rerace"]
    assert rr["single_split_pick_idx"] != rr["pooled_pick_idx"], "the pooled evidence overruled"
    assert rr["pooled_changed"] is True
    assert hp["seed"] == 1, "the scout argmax is restored on the pooled evidence"
    assert best_iter == 41
    assert sel["n_finale_splits"] == 3
    # every contender is carried onto the extra splits, not just the single-split winner
    pool_evals = [e for e in evals if str(e.get("param") or "").startswith("pool_")]
    assert len(pool_evals) == 2 * 3, "2 extra splits x (anchor + 2 finale contenders)"
    assert sel["champion_idx"] == rr["pooled_pick_idx"]


def test_the_regime_recheck_still_fires_and_counts_against_k():
    # The hsv-21 demotion mechanism is unchanged: a REGIME-MISMATCHED argmax is re-raced against
    # the best regime-FAITHFUL candidate whether or not the band admits it (allstate_sev's
    # demotion depends on that). It is one more draw in the max-of-N, so it consumes one of k's
    # four slots rather than sitting outside the guard.
    scout = {0: _behind3(20.0), 1: _ARG3, 2: _behind3(0.19), 3: _behind3(0.4)}
    finale = {0: _behind3(20.0), 1: _ARG3, 2: _behind3(2.0), 3: _behind3(3.0)}

    def recheck(hp):
        return int((hp or {}).get("seed", -1)) == 1        # candidate 1 is the "depth" candidate

    with mock.patch.object(TBoost, "selection_regime_recheck", staticmethod(recheck)):
        hp, _, evals, sel = _drive(scout, finale)
    assert "regime_recheck" in sel
    tags = [e.get("param") for e in evals if e.get("stage") == "finale"]
    assert tags.count("regime_recheck") == 1
    # champion + faithful + re-race contenders, all inside the one guard
    assert sel["finale_rerace"]["n_contenders"] <= 4
    assert len(tags) - tags.count("default_anchor") == sel["finale_rerace"]["n_contenders"]


# --- the fence: no other model may re-key ---------------------------------------------------------

# Re-captured 2026-09-09, when every adapter_version and hpo_search_version was reset to "1" and the
# whole cell cache was discarded. These are the version-1 baseline keys, with `_fake_cfg` below.
# Every one of them must now survive untouched: `hpo._select_config_static` is shared by all five
# tunable models, and a selection-protocol change that moved another model's BEHAVIOR without moving
# its KEY is the one failure a version-keyed cache cannot detect.
_HEAD_SIGNATURES = {
    "lightgbm": ("c34f41dfa1d66336705ea604", "bcd104f8601842d72d2ae6f7",
                 "b652a8c0f073f771e69277c9", "818a24fb86039d19"),
    "xgboost": ("cb038c16d2fb360a22090007", "cc007f5a7dc859c4ba9efe1b",
                "1eac3a06d7319f181b7277cb", "2abba36ec4187357"),
    "catboost": ("0aaeca1fce01325cf3b4dc6a", "cbfc7c732be8abc750741f8d",
                 "f6326d54a2e93521f5782cdb", "f706fe51427f1d03"),
    "ebm": ("74c233aaedeb17ffb32107a2", "01f2cfbcd43d76827f9109d7",
            "58a2a3e40c4918bb7778f53e", "a4fa961b1d8e06b0"),
}


def _sig_cfg():
    return SimpleNamespace(
        split={"small": {"folds": 3, "repeats": 1}}, n_configs=27, inner_folds=3,
        tune_split_id=0, time_budget_s=None, inner_repeats="auto",
        racing_se=2.0, racing_min_rows=45000, selection_max_rows=200000,
    )


@pytest.mark.parametrize("name", sorted(_HEAD_SIGNATURES))
def test_other_models_signatures_are_byte_identical_across_hsv23(name):
    from insurarena.results.cache import cell_signature

    cfg, bundle = _sig_cfg(), _fake_bundle()
    tuned, default, hpo_sig, digest = _HEAD_SIGNATURES[name]
    assert cell_signature("HASH", name, "tuned", 0,
                          runner._proto(cfg, bundle, name, "tuned", {})) == tuned
    assert cell_signature("HASH", name, "default", 0,
                          runner._proto(cfg, bundle, name, "default", {})) == default
    assert cell_signature("HASH", name, "tuned__hpo", 0,
                          runner._hpo_proto(cfg, bundle, name, {})) == hpo_sig
    assert hpo.config_set_digest(name, 27, None) == digest


def test_other_models_never_enter_the_finale_block_at_all():
    # Belt and braces behind the opt-in: the re-race lives inside the ship-fidelity finale, and
    # that block is entered only when an adapter declares a `selection_finale_hp` different from
    # its `selection_hp`. Only t-boost does.
    for name in ("lightgbm", "xgboost", "catboost", "ebm"):
        assert hpo._model_selection_finale_hp(name) == hpo._model_selection_hp(name), name
    assert hpo._model_selection_finale_hp("t-boost") != hpo._model_selection_hp("t-boost")


def test_tboost_cells_key_from_the_version_1_baseline():
    # The other side of the fence. On 2026-09-09 the arena was rebuilt from scratch: every
    # adapter_version and hpo_search_version went back to "1" and the cell cache was discarded, so
    # there is no longer any history to re-key away from — these are the keys the first run of the
    # new era writes. Both counters are pinned at "1" so that the next bump is a deliberate edit,
    # and the default signature is pinned so a future edit cannot quietly move the default fit path
    # without moving the key.
    from insurarena.results.cache import cell_signature

    cfg, bundle = _sig_cfg(), _fake_bundle()
    assert TBoost.adapter_version == "1"
    assert TBoost.hpo_search_version == "1"
    got = cell_signature("HASH", "t-boost", "default", 0,
                         runner._proto(cfg, bundle, "t-boost", "default", {}))
    # If this constant ever moves without `adapter_version` moving, a default fit path changed
    # while its key did not — the one failure a version-keyed cache cannot detect.
    assert got == "b468b6ffdd068078a4207890"
    # At "1" both counters are omitted from the proto entirely (`runner._proto` writes them only
    # when they differ from the default), which is what keeps the version-1 keys clean.
    assert "hpo_search_version" not in runner._proto(cfg, bundle, "t-boost", "tuned", {})
    assert "av" not in runner._proto(cfg, bundle, "t-boost", "default", {})
