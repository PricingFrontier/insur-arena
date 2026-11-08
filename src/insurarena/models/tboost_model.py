"""t-boost — a depth-3 oblivious gradient boosting machine that is interpretable (it decomposes to
a functional-ANOVA table) yet insurance-native. A single, explainable fitted artifact.

Configured via the library's OWN recipe (``t_boost.recommended_recipe``), which the author ships so
an external harness "does not systematically under-report t-boost": the load-bearing knobs are
``early_stopping_rounds=500`` (with adaptive patience), ``leaf_refine_steps``, and exactness-
preserving in-model bagging (``n_bags=8``, now the constructor default in BOTH regimes); the tuned
regime adds reanchoring and leakage-free k-fold categorical target encoding. That bagging is *within*
the single interpretable model (the bagged fit still collapses to one ANOVA table), so it stays a
single explainable artifact — consistent with this benchmark's single-model rule. Regime mapping:
benchmark ``default`` -> recipe ``tuned=False``; benchmark ``tuned`` -> recipe ``tuned=True`` (HPO
searches the tuned recipe family).

Note on `ccb` below: the design notes in this file cite margins against "ccb" — constrained
CatBoost, a CatBoost variant pinned to t-boost's structural budget (``max_ctr_complexity=1``,
depth free) that was carried as a measurement instrument while this adapter was developed. It is
no longer part of the benchmark and ships no adapter, config or cells; the figures are retained
because they are the evidence for choices this file still makes.

Wired like the other GBMs for an apples-to-apples comparison: the target lives on the rate scale with
``sample_weight=exposure`` (NOT t-boost's native exposure-offset arg), predictions are on the rate
scale. Categoricals are handled natively (robust to unseen). Task -> objective: freq=poisson,
sev=gamma, pp=tweedie, clf=logistic. Lazy import; optional dependency ``t-boost``.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import os
import warnings

import numpy as np
import pandas as pd

# t-boost warns once per fit that it casts features to float32; expected + benign, so silence it
# (otherwise it floods the benchmark log with one line per model fit).
warnings.filterwarnings("ignore", message=".*converts input features to float32.*")

from insurarena.datasets.base import DatasetBundle
from insurarena.models.base import Explanation, ModelAdapter
from insurarena.scoring import task_targets
from insurarena.tasks import TaskType

# task -> t-boost objective string
_OBJ = {
    TaskType.FREQ: "poisson",
    TaskType.SEV: "gamma",
    TaskType.PP: "tweedie",
    TaskType.CLF: "logistic",
    TaskType.MULTICLF: "logistic",
}

# hp keys forwarded to the recipe as overrides (soft knobs the tuned search may vary); the recipe's
# load-bearing structure (early stopping, leaf-refine, bagging, cat encoding) is left to the recipe.
_PASSTHROUGH = (
    # hp keys forwarded to the recipe as overrides. AUDITED 2026-09-04 (see
    # scratchpad/explain_prune/CONSOLIDATION.md): 20 entries were removed because NOTHING set
    # them — not `_STATIC_TUNE_GRID`, not `_COORDINATE_PROBES`, not `tuning_base_hp` /
    # `default_hp` / `selection_hp`, not a fit-time gate, and not a config `hp:` block (no config
    # sets any t-boost hp at all since the av45 fairness cycle deleted the second campaign
    # config). This tuple enters NO cache proto, so the removal is byte-identical and costs no
    # recompute — verified by replaying cached cells at delta_vs_cache == 0.
    #
    # THE RULE FOR ADDING OR REMOVING HERE: a key removed while something still sets it would
    # change every fit while leaving the cell signature IDENTICAL — the cache would then serve
    # stale results silently. That is the one edit in this file that can corrupt rather than
    # invalidate. `prune_fold_fidelity` and `prune_keep_budget` look dead to a naive grep for
    # `overrides["..."] =` and are NOT: the av40 size gate writes them through a loop variable
    # (see `fit`). Check for loop- and dict-writes, not just literal assignment.
    "learning_rate", "lambda_", "colsample_bytree",
    "min_data_in_leaf", "max_bin", "max_interaction_order",
    # THE SURVIVING ABLATION KNOBS. Unexercised by any config, and kept deliberately rather than
    # by oversight: `cat_direct_max_levels` and `reanchor` are MEASURED LIVE on the artifact (7/7
    # and 6/6 datasets), so they are a real lever; `random_strength` was retired from the SEARCH
    # at hsv 24 on its own measured record while staying reachable from an `hp:` block, which
    # `tests/test_hpo_probes_and_pooled_finale.py::test_random_strength_is_reachable_but_no_longer
    # _searched` asserts explicitly. Removing any of these three is a POLICY change about whether
    # the arena keeps an ablation surface at all — not a liveness cleanup.
    "reanchor", "cat_direct_max_levels", "random_strength",
    "cat_smooth",   # fixed TS pseudo-count override of Auto-m (tuned grid dim since hsv 15)
    "n_bags",
    "interaction_gain_hurdle", "interaction_gain_hurdle_mode",
    "prune_fold_fidelity",  # fold-fidelity-v2 (2026-08-31): pin each fold model's bank to the
                          # candidate set so every table gets paired evidence (was: 88% absence-
                          # dropped on credit_g). Default off/byte-identical; armed by the av40
                          # size gate below 2,500 rows (with unbounded keep budget).
    "prune_keep_budget",  # av37 evidence-gate admission cap (default 32). budget=max(32, legacy)
                          # is a TOTAL no-op wherever the legacy keep-set exceeds 32. Written by
                          # the same size gate as `prune_fold_fidelity` — and NOTE that the pair
                          # is not independent: an unbounded keep budget makes
                          # `prune_min_stability` and `prune_min_mean_gain` provably INERT (the
                          # av37 evidence path re-admits everything the stability gate refused).
    "cat_min_data_per_group",           # rare-level pooling floor — the av35 rare-floor gate's knob
    # av39 K>=3 cell refit: the gate below arms `cell_refit_base` on MULTICLF only. Explicit hp
    # outranks the gate (pass 0 or null to ablate it back to the av38 fit).
    "graduate",             # 2026-09-06 ablation knob: honest-gated graduation on/off (see av41)
    "ref_measure",          # 2026-09-06 ablation knob: purification measure (exposure vs product_marginals)
    "measure_floor",        # 2026-09-06 ablation knob: positivity floor of the exposure measure
    "cell_refit_base",
    # av40 DEPTH DIMENSION (hsv 20). `max_depth` is the t-boost depth-cells lift's own knob
    # (3..8; the library default and this benchmark's every prior cell is 3). It is a COORDINATE
    # PROBE, not a `_STATIC_TUNE_GRID` dimension (`tests/test_hpo_probes_and_pooled_finale.py`
    # asserts that), and `prune_table_budget` is the readability contract that rides with it:
    # NOT searched, set to a constant by the depth gate in `fit`. Listed here because that gate
    # writes it through `overrides`.
    "max_depth", "prune_table_budget",
)

# Harness-only hp keys that deliberately never reach the estimator, so the closure guard in
# `fit` must not raise on them. `budget`/`n_estimators` are consumed into `budget` by hand;
# `seed` is passed positionally to the recipe; the leading-underscore keys are protocol flags
# read by the adapter itself (`_tuned` selects the tuned recipe regime, `_prune_during_hpo`
# forces the prune on inside the search, `_selection_es_adaptive` is the scout overlay's
# patience). Anything NOT here and not in `_PASSTHROUGH` is a silent drop and now raises.
_HP_NOT_FORWARDED = frozenset({
    "budget", "n_estimators", "seed", "n_jobs",
    "_tuned", "_prune_during_hpo", "_selection_es_adaptive",
    "frequency_formulation",
})

_DEFAULT_BUDGET = 4000   # n_trees CAP; early stopping (rounds=500 in the recipe) decides real length

# ---------------------------------------------------------------------------------------------
# The RARE-FLOOR GATE (adapter-side).
#
# t-boost's categorical encoder pools every level holding less than `cat_min_data_per_group`
# into a shared "rare" bucket before it computes a target statistic. The library default, 10.0,
# is the right call on ordinary rating factors: a level seen by one policy has no credible mean,
# and pooling it is what stops the encoder from memorizing. But the floor is stated in units of
# POOLING WEIGHT, and this adapter passes `sample_weight=exposure` on the rate scale, so on a
# portfolio of short exposures the floor stops being a rarity test and starts being an exposure
# test: pg17's `pol_insee_code` (French INSEE commune) has ~48% of its total exposure sitting in
# communes below the floor, so nearly half the portfolio's geography collapses into one bucket
# and the feature is destroyed rather than regularized.
#
# So the gate: measure the damage directly on the slice about to be fit, and open the floor to
# 1.0 only where it is severe. The statistic is exactly what `collapse_rare_levels` thresholds —
# the per-level sum of `CatRowTerm.denom` — and because this adapter leaves t-boost's native
# exposure-offset arg unused, that denom IS the `sample_weight` passed below, per row. The share
# is therefore WEIGHTED, not a row count, which is the whole point: it is what separates pg17
# (48% weighted) from brvehins1's `VehModel` (0.7% weighted but 4.2% of ROWS — a row-count gate
# would have fired there and moved 20 cells for nothing).
#
# Deterministic and per-fit: a pure function of the fit slice's X, cat_features and weights, so
# every cell's decision is reproducible from its own inputs, re-derived on each fit (selection
# folds included) rather than carried in from a dataset-level table.
_RARE_FLOOR = 10.0            # the library/recipe `cat_min_data_per_group` the gate replaces
_RARE_SHARE_THRESHOLD = 0.25  # fire when some categorical's weighted sub-floor share exceeds this
_RARE_SMALL_THRESHOLD = 2500  # ... and the slice is not "small" by config `split.small_threshold`
_RARE_OPEN_FLOOR = 1.0        # what the gate passes when it fires

# SIZE-GATED BAGGING (av 38): recipe n_bags 8 -> 32 on datasets below the arena's own small
# threshold. DATASET-level n, deliberately not fit-slice rows: the ship-fidelity finale
# re-scores on inner-CV slices (~4n/9) while the deploy sees 2n/3, so a slice rule would bag
# the finale and the deploy differently for 3000 <= n < 4500 — freclaimdam (n=3948) lands
# exactly there and read null-to-negative under b32, while the dataset rule leaves it
# byte-identical. Battery (2026-08-25, 30 splits x pinned-3-seed + native-vs-shipped arms,
# both replicated): swautoins +0.0102, swautoins_pp +0.0073 (closes its ccb margin to
# -0.0002), credit_g +0.0012, brautocoll/autobi_us clean nulls preserving their wins; no
# catalogue set falls in 2500 <= n < 3000 so this boundary and "fit rows < 2000" fire on the
# same five sets. Cost is free where it fires (deploys 0.17-0.36 s at 8 bags). av31's
# blanket-bump revert stands for n >= 2500 (large controls moved <= +0.0005). Known effect
# to report, not fix: richer OOB evidence keeps slightly more tables on some sets
# (autobi_us +4.4 deployed mean, +20%; brautocoll +0.4; accuracy flat). An explicit
# `n_bags` in hp outranks the gate; the HPO scout overlay (n_bags=1) still wins at scout
# fidelity, so selection cost is unchanged and production is bit-for-bit the measured arm.
_BAGS_SMALL_THRESHOLD = 2500
_BAGS_SMALL_BAGS = 32

# SIZE-GATED FOLD FIDELITY + UNBOUNDED KEEP BUDGET (av 40, the av45 FAIRNESS cycle).
#
# WHY THIS IS A GATE AND NOT A CONFIG. Until av44 this pair of knobs reached the five sub-2500
# datasets through a SECOND campaign config (`configs/tuned_tboost_small_ffull.yaml`), i.e.
# t-boost was the only model on the board resolving from more than one config while every
# comparator ran exactly one. That is a benchmarking asymmetry, not a modelling one: the knobs
# are chosen by DATASET SIZE, which the adapter can read for itself. Moving them into the adapter
# makes the small-set behavior part of the MODEL — one config, same as lgb/xgb/catboost/ebm — and
# deletes the per-dataset config. No fit changes shape: the gate fires on exactly the same five
# sets the deleted config listed, with the same two values.
#
# The rule is the av38 bags gate's, verbatim: DATASET-level n from the bundle meta (survives
# `.subset()`, so the ship-fidelity finale on its ~4n/9 slice and the deploy fit on its 2n/3
# slice reach the SAME verdict), threshold `_BAGS_SMALL_THRESHOLD` = the arena's own
# `split.small_threshold`, and an explicit `hp` value outranks the gate so an ablation is one
# config line. No catalogue set sits in 2500 <= n < 3000, so this boundary and "n < 2500" fire on
# credit_g (1.0k), brautocoll (1.3k), autobi_us (1.3k), swautoins (2.2k) and swautoins_pp (2.2k)
# and nothing else.
#
# WHAT THE KNOBS DO, and the evidence for every set they touch (fold-fidelity battery; ffunl
# confirm battery @750f7bd2, 5 sets x 30 splits x 2 seeds, paired on deviance):
#   * `prune_fold_fidelity=True` pins each prune fold-model's bank to the shared candidate set, so
#     every candidate table gets PAIRED evidence instead of being absence-dropped (88% of
#     credit_g's order-<=3 candidates had no paired z under the default); and
#   * `prune_keep_budget=1_000_000` lifts the av37 admission cap, which is the half that makes the
#     first one pay — fidelity ALONE is negative (credit_g -0.0038, t=-3.4) because a sharper jury
#     spends the 32-table budget on fewer, safer tables. Measured together (`ffunl`):
#         credit_g     +0.00296  t=+5.17  46/60 wins, deployed bank 99 -> 186 tables
#         autobi_us    +0.00201  t=+0.99  (noise-positive, kept for coherence of the size rule)
#         brautocoll   +0.00003  t=+0.19  a null
#         swautoins    +0.00000  60/60 fits BYTE-IDENTICAL
#         swautoins_pp +0.00001  56/60 fits byte-identical
#     Gate on the confirm battery: credit_g's +0.0030 stands and no other set regresses at t>=2.
#     Cost 1.0-1.6x deploy wall on datasets that fit in under a second.
# The readability price is stated, not buried: credit_g deploys ~186 tables instead of ~99, all of
# them order-<=2 (the >=3-way census is 0.0 on all five sets, before and after).
#
# MULTICLF IS EXCLUDED. t-boost REFUSES `prune_fold_fidelity` on the K>=3 prune path (it selects
# on one train/select split and has no per-fold refits to pin a bank into) and that path takes no
# keep-budget argument at all, so the gate would be handing a fit knobs it cannot honor — the same
# filter discipline the cell-refit and depth gates keep. Inert today: the five gated sets are
# CLF/SEV/FREQ/PP.
_FIDELITY_KEEP_BUDGET = 1_000_000

# K>=3 CELL REFIT (av 39). t-boost's §G1 out-of-bag cell refit re-solves the realized
# order-<=2 cell values against the bags' OWN out-of-bag rows after the soup is built — free
# honest evidence the greedy in-fit budget never spent. It shipped for the single-output path
# long ago; the mc-cellrefit branch extends it to K>=3 as K decoupled diagonal-Hessian solves
# (the softmax row Hessian is exactly weakly diagonally dominant, so the decoupled direction is
# 0.97-0.997 cosine-aligned with the joint Newton one and errs only by OVERSHOOTING) followed by
# ONE joint backtrack that picks a single lambda minimising the TRUE multinomial deviance of the
# out-of-bag slice with all K corrections applied together.
#
# That backtrack is why this is armed by default rather than tuned. lambda=0 is a reachable,
# honest verdict — it drops the whole correction on all K classes — so the feature cannot spend
# skill it has not first measured on held-out rows. The battery says exactly that: prudential_life
# (the set with real weak-pair structure) takes +0.00159 at t=9.69, while on fremotor_payfreq and
# pg16_payfreq the guard simply declines — worst paired case -0.00004, and 14 of 18 declines are
# BYTE-IDENTICAL to the av38 fit. The cost is one extra OOB scoring pass and K cell solves
# (~1.15-1.3x deploy wall on the K>=3 sets, which are the small end of the catalog).
#
# MULTICLF only, deliberately: K<=2 and regression already run this on the single-output path
# under their own recipe defaults, and nothing here should move them. Needs a bag partition
# (n_bags >= 2), which every deploy fit has; the HPO scout overlay pins n_bags=1, so SELECTION
# fits are untouched and selection cost is unchanged. An explicit `cell_refit_base` in hp
# outranks the gate (both knobs are in _PASSTHROUGH), so the ablation is one config line.
#
# `base` is the ridge the per-cell solve starts from, shaped down per-support by `gamma`; 4000.0
# is the value the battery measured (its 200.0 companion arm read the same within noise).
_CELL_REFIT_BASE = 4000.0


def _subfloor_shares(X: pd.DataFrame, cat_cols: list[str], w: np.ndarray) -> dict[str, float]:
    """Per-categorical weighted share of the fit slice sitting in sub-floor levels."""
    tot = w.sum()
    out: dict[str, float] = {}
    for c in cat_cols:
        # Same nan-normalization `_prep_X` applies, so the levels counted here are the levels the
        # encoder will actually see.
        s = X[c].astype("string").fillna("nan").to_numpy()
        denom = pd.Series(w).groupby(s).sum()
        sub = pd.Series(s).map(denom).to_numpy(dtype=float) < _RARE_FLOOR
        out[c] = float(w[sub].sum() / tot) if tot > 0 else 0.0
    return out


def _rare_floor_gate(X: pd.DataFrame, cat_cols: list[str], w: np.ndarray) -> dict:
    """Decide the rare-floor gate for one fit slice: `fires`, plus what drove it.

    The SIZE exclusion is the arena's own `split.small_threshold`. Below it the sub-floor share is
    both routinely huge and meaningless — brautocoll (n=876) runs ~0.68 on `Hour` purely because
    every level is thin at that sample size — and opening the floor on a slice that small trades
    regularization the fit genuinely needs for credibility it cannot have.
    """
    n = int(len(X))
    shares = _subfloor_shares(X, cat_cols, w) if cat_cols else {}
    col, mx = (max(shares.items(), key=lambda kv: kv[1]) if shares else ("", 0.0))
    return {
        "fires": bool(n >= _RARE_SMALL_THRESHOLD and mx > _RARE_SHARE_THRESHOLD),
        "max_share": mx,
        "col": col,
        "n_train": n,
    }

# The tuned candidate space. Migrated 2026-07-20 from the greedy coordinate walk to the static
# 25-candidate random-search basis, for cross-model methodology parity with lgb/xgb/catboost/ebm
# (SPEC §5.5). The candidate SPACE is populated from the SAME knobs + validated value grids the
# retired coordinate walk swept — not a fresh space — so t-boost's domain knowledge (e.g. the
# interaction_gain_hurdle grid that steps over the mid-γ valley onto the γ≥1.5 plateau, and order-1
# for the additive sweet spot) carries over. n_bags has no entry: scout evals fit 1 bag regardless
# (selection_hp), deploys keep the recipe's objective-aware bag count (gamma/tweedie 16 since
# 2026-07-22, else 8 — see recommended_recipe and the tuning_base_hp note). `sample_hp_set` (hpo_search_version 13)
# draws the whole n-candidate set jointly space-filling over this grid; `sample_hp` (one independent
# per-candidate draw; the coordinate walk swept these one at a time) is kept as the primitive
# `static_configs` falls back to for models without a `sample_hp_set`.
_STATIC_TUNE_GRID = {
    "interaction_gain_hurdle": [0.0, 0.5, 2.0, 5.0, 30.0],
    "max_interaction_order": [1, 2, 3],
    "lambda_": [0.1, 1.0, 10.0],
    # Column subsampling. Widened 1.0/0.8 -> 1.0/0.8/0.6/0.45 at hsv 23 (2026-09-01) by the
    # LightGBM gap forensic (lever 2). Every LightGBM cell t-boost
    # loses to was reproduced BIT-EXACTLY and its champion read off: the winners on exactly the
    # sets in question sit at feature_fraction 0.43-0.61 — pg16_payfreq 0.43, credit_g 0.49,
    # swautoins 0.50, fremotor_prem 0.53, prudential_life 0.61 — i.e. the ENTIRE region the
    # champions live in was below t-boost's floor and unreachable. Priced inside the contract, not
    # inferred: on credit_g's depth-3 arm (30 splits, paired, order<=3 throughout) forcing
    # feature_fraction back to 1.0 costs -0.0047. 0.45 rather than 0.4 so the low rung sits inside
    # the measured band instead of just under it; two new values rather than one so the dimension
    # stays balanced over the sampled tail (4 values x 15 slots via `sample_hp_set`'s cycle).
    "colsample_bytree": [1.0, 0.8, 0.6, 0.45],
    # Leaf resolution. FILLED at hsv 23 (2026-09-01), same forensic, lever 3. t-boost's grid jumped
    # 0 -> 50 -> 200 while every one of the 15 reproduced LightGBM champions lands between 1 and
    # 46 (1, 2, 4, 5, 12, 13, 17, 18, 20, 27, 46) — a real interior optimum the search could not
    # address. Measured on credit_g's depth-3 arm against its champion's min_data_in_leaf=2:
    # moving to 20 costs -0.0064 and to 1 costs -0.0028, so the curve is sharp AND the optimum is
    # interior. 5 and 20 bracket it; 0 (the library's own floor) and the 50/200 rungs stay, so no
    # previously reachable value is withdrawn.
    "min_data_in_leaf": [0, 5, 20, 50, 200],
    # Quantile-histogram resolution. Added 2026-08-23 (hpo_search_version 18): `max_bin` has been
    # in _PASSTHROUGH since the adapter was written but was never in this grid — a LIVE dimension
    # the tuned search could not reach, pinned at the recipe's 254 on every cell. The edge
    # forensics measured it as a real, dataset-specific, SIGN-VARYING axis:
    #     euhealth   max_bin 16/32/48/64/128/192 vs 254 (9 splits x 3 seeds, paired):
    #                -0.000438 / +0.000184 / +0.000204 / +0.000206 / +0.000085 / +0.000062
    #                — a clean unimodal curve, interior optimum ~48-64, t=3.34 at 64;
    #     euhealth_pp (SAME 176k rows, SAME 4 predictors, tweedie instead of poisson):
    #                max_bin=64 is -0.000097 +/- 0.000033 (t=-2.95, 0/6 splits) — HARMFUL;
    #     pg17_payfreq s0: -0.000098 (same sign as euhealth_pp);
    #     norauto:   BIT-IDENTICAL (its 4 predictors are two binaries and two small ordinals,
    #                so there are fewer than 32 candidate cuts to lose).
    # A knob that is worth +0.000206 on one dataset and -0.000097 on its own sibling is exactly
    # the profile of something that must be SEARCHED per cell, never defaulted globally — the
    # forensics' one act-on-it-independently finding. The mechanism is bias in the SPLIT SEARCH,
    # not roughness of the deployed table (forcing Whittaker-Henderson graduation_alpha to 3000
    # recovers at most +0.00006 of it), so no post-hoc smoothing substitutes.
    # 254 leads the list so it is the value the tuned-family BASE keeps: `tuning_base_hp` does not
    # pin max_bin and the recipe pins 254, so a candidate drawing 254 is byte-identical to hsv 17's
    # candidate at that slot on this dimension. The default anchor (`default_hp` = {budget}) never
    # carried max_bin and still does not — the ship-gate reference is untouched.
    "max_bin": [254, 128, 64, 32],
    # 0.02 added 2026-08-21 (hpo_search_version 17): the enriched-space scan found the
    # 0.05 -> 0.01 hole is real — tic2000's honest out-of-grid lr gained +0.0008 skill over the
    # best in-grid candidate, i.e. the basis was stepping OVER the optimum, not past it. A 4th
    # value keeps the marginal balanced at n=25 (7/6/6/6 via sample_hp_set's per-dim cycle).
    "learning_rate": [0.1, 0.05, 0.02, 0.01],
    # Categorical TS shrinkage: None = the library's credible-set Auto-m; 1000 pins a strong
    # fixed pseudo-count. Added 2026-07-22 (hpo_search_version 15) after the brvehins1_pp
    # probe series: Auto-m is fold-unstable on zero-inflated Tweedie with sparse high-card
    # cats (VehModel, 3831 levels) — on the 6 blowup splits a weak auto-m exploded held-out
    # deviance while cat_smooth=1000 repaired exactly those splits and beat constrained
    # catboost 9/9 (29.40 vs 30.35, −14.3% vs the shipped cells; freq sibling −0.77%). The
    # None arm is byte-identical to not passing the knob, so half the basis is unchanged.
    "cat_smooth": [None, 1000.0],
}

# The load-bearing pair for t-boost's tuned search: interaction admission gates on the hurdle,
# and the hurdle's effect is order-dependent (a strict hurdle matters more at order 3 than order 1).
#
# DO NOT ADD A THIRD NAME. `sample_hp_set` uses this tuple ONLY as an exclusion filter for the
# marginal cycles; the joint product itself is built from these two names literally and unpacks a
# 2-tuple. A third name would be excluded from the marginals, never enter the joint product, and
# therefore never be set on any candidate — silently, with no error. hsv 20's `max_depth` is a
# MARGINAL dimension for exactly this reason.
_JOINT_DIMS = ("max_interaction_order", "interaction_gain_hurdle")

# The library's (and every pre-av40 cell's) tree depth. Mirrors `t_boost.sklearn._LEGACY_MAX_DEPTH`.
_LEGACY_MAX_DEPTH = 3

# The library's pre-lift fixed interaction order, and the value `tuning_base_hp` pins. Mirrors
# `t_boost.sklearn._LEGACY_MAX_ORDER`. Order ABOVE this is structural (see `_is_structural_hp`);
# the engine's own ceiling for a schema-4-readable bank is 4, which is also the only lifted value
# any probe names. (Was written as `_ORDER_LIFT_MAX_ORDER` — a symbol that exists in neither repo.)
_LEGACY_MAX_ORDER = 3

# The >=3-way TABLE cap applied to every DEPTH-LIFTED fit — a CONSTANT, never a searched value.
#
# Ralph moved the explainability bar on 2026-08-27: cells within a table are free ("perfectly
# happy with as many cells within tables to capture the correct resolution"), the count of
# multi-way TABLES is the scarce thing ("100 3-way tables is not explainable"). A depth-6 tree
# reuses features harder and therefore proposes far more >=3-way supports than a depth-3 one, so
# lifting depth without capping table count would buy skill by spending the readability budget.
#
# 100 is the value the depth-cells battery actually won at: fremotor_payfreq depth 6 + budget 100
# shipped 236 >=3-way tables against the depth-3 control's 381 — deeper AND smaller. It is pinned
# rather than searched because a readability contract is a product call, not a hyper-parameter:
# searching it would let the selector buy skill by relaxing the bar.
_DEPTH_LIFT_TABLE_BUDGET = 100

# Ship-gate floor for a DEPTH-LIFTED champion, in paired-fold SEs. See `ship_gate_se_mult`.
#
# 3, not 2. The first draft of this floor was 2.0 and the campaign-budget smoke measured it as
# too low ON ITS FIRST TRY: `fremotor_payfreq`'s depth-lifted champion cleared 2.0 at margin/SE
# = 2.27, shipped, and then measured -0.00103 skill over 9 paired outer splits. A bar that
# admits the one bad structural ship we have observed is not a bar. 3.0 would have refused it.
_DEPTH_LIFT_SE_MULT_FLOOR = 3.0

# COORDINATE PROBES (hpo_search_version 18, 2026-08-23) — the allocation fix.
#
# The problem the headroom scan measured, not a hypothesis: a winning single-knob value that is
# ALREADY IN _STATIC_TUNE_GRID gets buried, because `sample_hp_set` only ever offers it JOINTLY
# with five other randomized knobs. Its single-knob ablation table (6 datasets, one knob moved off
# the tuned base at a time, inner CV):
#     tic2000  max_interaction_order=2   +0.567%   <- the single largest one-knob move on the board
#     tic2000  learning_rate=0.03        +0.319%
#     ohlsson  cat_smooth=100/1000       +0.241%
# and yet on tic2000 NO candidate of the 25-draw basis carrying order=2 ever reached the gate as a
# clean order=2 signal: the six draws that had it also had a random lambda_/colsample/min_data/lr
# attached, and the basis champion (draw39, a joint draw) was measured HARMFUL out of sample
# (-0.00166 skill, and the gate shipped it). Meanwhile the scan's own explicit single-knob arm
# `probe_lr003` was the best honest arm on tic2000 (+0.00081, t=1.77) — a coordinate probe would
# have been seen.
#
# So: a deterministic block of one-knob-off-BASE candidates, evaluated as ordinary candidates
# alongside the random basis (nominal indices 1..len, i.e. immediately after the default anchor).
# "Off-base" means off `tuning_base_hp` — the tuned-family recipe every sampled candidate is also
# built from — so each probe differs from that reference in EXACTLY ONE coordinate and its
# inner-CV score is readable as that knob's effect.
#
# The curated subset is small on purpose. The same scan's other measured result is that naive
# space enrichment BACKFIRES through the winner's curse (its gate fired 2 harmful ships on
# enriched champions; mean honest edge of 48 stage-1-selected arms was -0.019% against a mean
# apparent edge of +0.105%, a -0.18x shrinkage). Every probe added is one more draw in the max-of-N
# that the gate has to defend against, so only directions with campaign evidence behind them are
# admitted, and the block REPLACES basis draws rather than extending the candidate count
# (see hpo.static_configs) — the max-of-N stays at n_configs.
#
#   max_interaction_order 1, 2  — the additive / pairwise sweet spots. Order 2 is the tic2000
#       finding above; order<=2 is also the norauto finding (its own inner CV ranked order<=2
#       above the anchor by ~0.00034 skill, and the order-1 arm measured +0.000126 +/- 0.000050
#       on OUTER test, halving the cell's seed SD).
#   max_bin 64, 128            — the euhealth ladder's interior optimum and its shoulder
#       (+0.000206 t=3.34 and +0.000085). New dimension this version; a coordinate probe is how a
#       new dimension gets seen cleanly instead of only through joint draws (the headroom scan's
#       explicit warning about enrichment).
#   learning_rate 0.02         — the hsv-17 grid addition, whose whole justification was an
#       out-of-grid single-knob measurement (tic2000 +0.0008 skill over the best in-grid draw).
#   interaction_gain_hurdle 5.0 — the top of the gamma>=1.5 admission plateau the retired
#       coordinate walk mapped; +0.054% tic2000 / +0.042% euhealth / +0.017% beMTPL97 single-knob.
#
# NOT probed: `lambda_`, `colsample_bytree`, `min_data_in_leaf`, `cat_smooth`. Their single-knob
# ablations are within noise on 5 of 6 datasets, and cat_smooth's one real result (ohlsson +0.24%)
# is a two-value dimension the basis already balances 13/12 at n=25.
#
# The tuned-family BASE itself is deliberately NOT a probe. It would cost an eval that no decision
# consults: the ship gate's reference is the true default anchor (`default_hp`, candidate 0), which
# is already evaluated twice — once in the basis and again at finale fidelity.
#
# --- hsv 20 (2026-08-27): the DEPTH probes, and why depth is a probe and not a grid key -------
#
# `max_depth` was drafted as a marginal grid dimension and the campaign-budget selection smoke
# falsified that draft on the one dataset the whole depth program was built on. Both halves of
# that measurement matter:
#
#   * the SEARCH found depth. On `fremotor_payfreq` at n_configs=25 the best candidate eval on
#     the board was a depth-6 draw and the champion was depth-lifted;
#   * the SHIP lost. Its 9 deploy cells came in at -0.00103 skill against the av39 tuned cell
#     (SE 0.00154, t=-0.67, W-L 3-6) at 1.70x the fit time.
#
# It is not a depth failure. The same build, the same split, the same cap, at depth 6 on the
# ANCHOR recipe is +0.00550. What lost was the BUNDLE a random basis necessarily hands you:
# depth 6 arriving with learning_rate 0.01, max_bin 128, min_data_in_leaf 50, cat_smooth 1000 and
# lambda_ 0.1 attached. That is verbatim the risk the depth-cells handover listed under "what
# this battery does NOT cover" — "a tuned champion combines depth with learning_rate, max_bin,
# min_data_in_leaf and the hurdle; nothing here says the depth gain survives alongside a tuned
# learning_rate" — and it does not.
#
# A joint basis cannot fix this by being bigger. A structural knob has to be offered CLEAN or the
# gate is judging a bundle, which is the exact pathology the probe block was invented for at
# hsv 18 (tic2000's order-2). So depth enters as two one-knob probes off the tuned base, and the
# grid keeps the seven dimensions it had at hsv 19 — which also disposes of the dilution cost of
# spending 25 fixed candidates on a new axis.
#
#   max_depth 4, 5, 6 — RUNG 4 RETIRED at hsv 24: it never led a recorded inner-CV selection (see
#       the hsv-24 note). 5 and 6 stay — the two rungs the depth-cells battery won on (fremotor_prem
#       d5 +0.00465 6-0; fremotor_payfreq d6 +0.00479 t=41.4 3-0 at 0.62x the shipped >=3-way
#       table count). Each probe rides the `prune_table_budget = 100` constant that the fit-time
#       depth gate wires in, so a depth probe is only ever offered WITH the readability contract
#       — the sampler can never present the unguarded (depth 6, no cap) shape the controls refuse.
#
#       4 WAS MEASURED at hsv 21 (2026-08-31) and added. The new-territory battery ran the full
#       rung ladder d4..d8 through `_evaluate_config` on the eight datasets where ccb v2's freed
#       depth opened a new gap, and d4 — the one rung the hsv-20 block did not offer — is the
#       BEST rung on three of the eight (prudential_life, allstate_sev +0.00075/t=+4.62,
#       pg17_coverage) and positive on five. It is also the cheapest and the most readable rung,
#       and on prudential_life it is the only depth the search should be allowed to reach
#       (d5/d6/d7/d8 all measure NEGATIVE there, -0.00141 to -0.00752). Offering 5 and 6 without
#       4 was an interpolation gap in a monotone-looking dimension, which is the same shape of
#       mistake the retired `cell_refit_base` 12000.0 arm was deleted for — except in the other
#       direction: an UNMEASURED value was excluded, and then measured to be the best one.
#
#       APPENDED, not inserted before 5 and 6, on purpose: `coordinate_probe_hp_set`'s identity
#       contract is that probe j keeps nominal candidate index j+1 forever. Sorting the block by
#       depth would renumber the depth-5, depth-6 and cell_refit probes for no gain.
#
#   cell_refit_base 20000.0 — RETIRED at hsv 24 (inert at scout fidelity, see the hsv-24 note; the
#       fit-time gate's 4000.0 anchor is untouched). The MULTICLF §G1 ridge base, MEASURED
#       (5x av39's 4000.0: pg16
#       +0.00025 t=1.24, fremotor +0.00004 t=1.48 — positive on both panels, harmful on
#       neither). 4000.0 stays the anchor via the av39 fit-time gate, and the interpolated
#       12000.0 is gone: only measured values ship. The fit-time gate DROPS this key wherever a
#       fit cannot honor it, which makes the probe MULTICLF-only without the space having to
#       know the task — `coordinate_probes` is dataset-blind by construction.
#       READ THE LIMIT recorded on `cell_refit_gate_` below before reading anything into this
#       probe's score: the §G1 refit needs an out-of-bag complement and every scout eval fits
#       n_bags=1, so the key is dropped on the eval this probe actually gets. At scout fidelity
#       the probe IS the tuned base, on all 40 datasets.
#
# --- hsv 22 (2026-08-31): the JOINT probes — depth carried WITH the knob it needs -------------
#
# The depth-conversion forensic measured where
# constrained-catboost's depth term actually lives and how much of it t-boost can reach. Its
# structural finding is about THIS FILE, not about the engine:
#
#     `max_depth` is a COORDINATE PROBE (base hp + ONE knob), while `interaction_gain_hurdle` and
#     `max_interaction_order` are the `_JOINT_DIMS` cycle, which always runs at the tuning BASE
#     depth of 3 — `tuning_base_hp()` pins hurdle 2.0 / order 3. So no candidate in ANY basis, at
#     any hsv, has ever carried a lifted depth TOGETHER with a moved hurdle or a moved order.
#     Depth has only ever been judged at hurdle 2.0 / order 3, which is the exact cell where it
#     converts worst. Measured directly on beMTPL16, one knob at a time (splits 0-2):
#         depth 3->6 at hurdle 0.0 : +0.002460     depth 3->6 at hurdle 2.0 : +0.001341
#     i.e. the depth x hurdle interaction is +0.0011 and the search could not express it.
#
# Two shapes were measured over 9 outer splits, paired on deviance, and both are added VERBATIM —
# no interpolation, no third combination:
#
#   (max_depth 8, interaction_gain_hurdle 0.0)  — eusavingul +0.003844 (t=+6.51, 9-0): 89% of its
#       whole t-boost<->ccb2 gap, residual vs the ccb2 cell -0.000492 (t=-0.95, W-L 3-6) = a TIE.
#       Nothing about the bank moves: it stays at 25 tables = the COMPLETE order-<=3 lattice on 5
#       features (5 + 10 + 10), with ZERO order-4 tables. The order axis is inert here — order
#       3->4 and 3->5 are BIT-IDENTICAL on 2 of 3 splits and emit no order-4 table. What depth
#       buys on this set really is resolution; the hurdle was refusing the order-2 -> order-3
#       transitions that carry it.
#
#   (max_depth 5, max_interaction_order 4)      — beMTPL16 +0.004715 (t=+13.01, 9-0): 89% of its
#       gap, residual -0.000600 (W-L 5-4) = a TIE. Depth 5 ALONE is only +0.002202 (41% of the
#       gap); the order relaxation is where the other half is. The bank emits 6-10 GENUINE order-4
#       tables out of 123-155 — minimal and evidence-gated, which is what Ralph's 2026-08-25
#       order-4 authorization asks for. Readability price, stated rather than buried: the deployed
#       bank grows ~99 -> ~145 tables (order-3 count 26 -> 63) and the >=3-way census reaches ~72
#       against the budget of 100. The cap stays inert but the MARGIN IS NOW THIN — which is
#       exactly why `prune_table_budget = 100` rides this probe UNCHANGED (max_depth 5 > 3, so the
#       fit-time depth gate arms it as it does for every other depth probe) instead of being
#       relaxed to fit the wider bank.
#
# NOT ADDED, deliberately:
#   * (max_depth 8, max_interaction_order 4) — the forensic's third arm. It is freMTPL2freq's, and
#     freMTPL2freq is CONCEDED: -0.000185, a null. A safe null is not evidence, and every probe
#     costs one more draw in the max-of-N the ship gate has to defend against (the headroom scan's
#     enrichment result, above).
#   * anything relaxing the hurdle AND the order together. Measured as a FIT-TIME EXPLOSION on
#     both sets — eusavingul order-5 x hurdle-0 took 37-47 min against a 20 s control; beMTPL16
#     depth-6 x order-5 took 45-55 min against 100 s. A cost result is still a result.
#   * promoting `max_depth` into `_JOINT_DIMS`. That tuple is unpacked as a 2-tuple and carries a
#     DO NOT ADD A THIRD NAME warning for a reason. Widening it would also put depth on all 15
#     joint slots — i.e. bundled, the offering hsv 20 falsified.
#
# READ THE CAVEAT BEFORE THE NUMBERS: every arm above is scored on OUTER TEST and chosen post-hoc
# among ~13 arms. These probes get no exemption from the ship gate, which is inner-CV. Both are
# depth-lifted, so both draw the >=3-way table cap, the 3-SE structural floor and the hsv-21
# regime recheck; the order-4 probe is ALSO structural on its own axis (see `_is_structural_hp`).
#
# --- hsv 23 (2026-09-01): the RANDOMIZED-SPLIT probes — ADDED, THEN RETIRED AT hsv 24 ---------
#
# `random_strength` 1.0 and 4.0 entered at hsv 23 as the LightGBM gap forensic's top-ranked lever
# (t-boost's analog of `extra_trees`, worth +0.0097 on credit_g and +0.0040 on ohlsson for LightGBM at
# depth 3). They are GONE at hsv 24, on the evidence the av44 selection smokes recorded before the
# campaign ran (`configs/t-boost.yaml`, the hsv-23 block): the probes are live and
# non-inert — they move the score — but on inner CV they rank 14th and 9th of 30 on credit_g, 16th
# and 6th on swautoins, 15th and 7th on fremotor_payfreq, 11th and 14th on freMTPL2freq, and no
# smoked cell ships one. (Only eusavingul ranked them 1-2, and that was UNDER the av44 full pin,
# which fixed every other coordinate — the pin the av45 cycle also removes.) Neither value
# dominates the other, which is what the mechanism predicts: t-boost's `random_strength` is an ABSOLUTE
# additive perturbation of the per-level Newton gain (`score = max(gain,0)*penalty +
# U(-s,s)/sqrt(round+1)`, engine/split.rs), so it is scale-dependent by construction and
# t-boost's own research note says a fixed sigma is "either a no-op or destabilizing"
# (research/06-techniques/5-losses-sampling-objectives.md). Two fixed rungs cannot cover a
# data-dependent scale; the campaign read the no-op half. Retired rather than rescaled — a
# gain-relative sigma would be a t-boost engine change, not a search-space one.
#
# --- hsv 24 (2026-09-02): the FAIRNESS cycle — FOUR PROBES RETIRED, budget back to 25 ----------
#
# The block only ever grew. At hsv 21 it hit the joint-coverage floor (a 15-draw sampled tail is
# what `sample_hp_set` needs to see all 15 (order, hurdle) combinations), and hsv 22 and hsv 23
# then bought their new probes by RAISING `n_configs` — 25 -> 27 -> 29 — instead of paying the
# floor. Each step was individually defensible and the sum was not: t-boost ended up racing 29
# candidates while lightgbm, xgboost, catboost and ebm each race 25. That is a TUNING BUDGET
# asymmetry in t-boost's favour on the one axis a benchmark most has to keep level, so the budget goes
# back to 25 and the block is cut to 10 probes — 25 - 10 = 15, the tail is exactly the floor
# again, and t-boost races the same 25 candidates as every comparator.
#
# WHICH FOUR, and why these and not others — each was retired on ITS OWN measured record, not to
# hit a number:
#   * `random_strength` 1.0 and 4.0 (hsv 23) — no smoked cell ships one, and no value covers the
#     scale-dependence the mechanism has (above).
#   * `cell_refit_base` 20000.0 (hsv 20) — PROVABLY INERT where it is scored. The §G1 refit needs
#     an out-of-bag complement and every scout eval fits n_bags=1, so the fit-time gate DROPS the
#     key on the evaluation this probe actually gets: at scout fidelity the probe IS the tuned
#     base, on all 40 datasets (asserted in tests). It could only ever differ at the ship-fidelity
#     finale, and only by winning a race in which it is indistinguishable from the base — a
#     candidate slot that cannot express its own hypothesis where it is scored is the cheapest one
#     to cut. The fit-time gate's 4000.0 anchor is untouched, so nothing about the §G1 refit moves.
#   * `max_depth` 4 (hsv 21) — added on an OUTER-TEST rung ladder (best rung on 3 of 8 sets there,
#     chosen post-hoc among the rungs) and never seen winning an INNER-CV cell since: it leads no
#     recorded av43 or av44 selection smoke, and the four cells those smokes resolved ship depth 6,
#     a depth-3 basis draw, (depth 8, hurdle 0) and the anchor. Outer-test rung evidence is exactly
#     what the probe block's own caveat says gets no exemption from the ship gate. Depth keeps its
#     measured rungs — 5 and 6, the two the depth-cells battery won on, plus 8 inside the
#     eusavingul joint shape; the interpolation rung is what goes.
# KEPT, all of them measured and all of them with campaign wins behind them: the order 1/2, max_bin
# 64/128, learning_rate 0.02 and hurdle 5.0 block, `max_depth` 5 and 6, and the two hsv-22 joint
# shapes — (d8, h0) and (d5, o4).
#
# THE INDEX-STABILITY CONTRACT, and the one time it is broken. `coordinate_probe_hp_set` promises
# probe j sits at nominal candidate index j+1 forever, so a cached evaluation stays reusable.
# Retiring probes renumbers everything behind them: the two joint probes move from candidate 11
# and 12 to 9 and 10, and the 15 sampled draws move up by four, which changes the per-candidate
# `seed` (`static_configs` stamps the NOMINAL index) and therefore their fits. That is a real
# break and it is taken knowingly, ONCE: hsv 24 is an `hpo_search_version` bump riding an
# `adapter_version` bump, so every t-boost cell recomputes anyway and there is nothing cached
# for the contract to protect. The contract resumes from this block.
_COORDINATE_PROBES = (
    ("max_interaction_order", 1),
    ("max_interaction_order", 2),
    ("max_bin", 64),
    ("max_bin", 128),
    ("learning_rate", 0.02),
    ("interaction_gain_hurdle", 5.0),
    ("max_depth", 5),
    ("max_depth", 6),
    # hsv 22 — the two MULTI-KNOB probes. Appended at hsv 22 behind probes 1..10; two of those ten
    # were RETIRED at hsv 24, so these now sit at nominal candidate indices 9 and 10 (see the
    # hsv-24 note above for why the identity contract was broken deliberately, once).
    (("max_depth", 8), ("interaction_gain_hurdle", 0.0)),      # eusavingul's measured shape
    (("max_depth", 5), ("max_interaction_order", 4)),          # beMTPL16's measured shape
)


def _probe_moves(entry) -> tuple[tuple[str, object], ...]:
    """Normalize one `_COORDINATE_PROBES` entry to its ordered ((knob, value), ...) moves.

    Two spellings, disambiguated on the TYPE of the first element and nothing else:
        ("max_depth", 5)                                   -> one knob  (hsv 18-21)
        (("max_depth", 5), ("max_interaction_order", 4))   -> two knobs (hsv 22)
    A mapping is accepted too, for a caller that finds it clearer; iteration order is preserved in
    every case, because `coordinate_probe_hp_set`'s duplicate check reports the first offender and
    a stable order keeps that message stable.

    Refuses an empty entry (a probe that moves nothing IS the base) and an entry that names the
    same knob twice (the second write would silently win). Both are construction errors that would
    otherwise reach the cache as a wasted evaluation.
    """
    if isinstance(entry, dict):
        moves = tuple(entry.items())
    elif len(entry) == 2 and isinstance(entry[0], str):
        moves = ((entry[0], entry[1]),)
    else:
        moves = tuple((str(name), value) for name, value in entry)
    if not moves:
        raise ValueError("a coordinate probe must move at least one knob")
    names = [name for name, _ in moves]
    if len(set(names)) != len(names):
        raise ValueError(f"coordinate probe names a knob twice: {names}")
    return moves


def _is_structural_hp(hp: dict | None) -> bool:
    """Does this candidate change the MODEL CLASS rather than perturb the depth-3/order-3 one?

    Two axes, and a candidate is structural if it lifts either:
      * `max_depth` above the library's 3 — more realized cells per tree, a different table
        census, a different readability profile (which is why `_DEPTH_LIFT_TABLE_BUDGET` rides
        with it), and it is not reversible per split: every deploy fit takes it.
      * `max_interaction_order` above the legacy 3 — the deployed bank gains a table shape a
        filing has never had to print before, and mandatory heredity drags in the whole subset
        lattice of every surviving order-4 support.

    Used by `ship_gate_se_mult` for the 3-SE floor. Order was added at hsv 22 with the
    (depth 5, order 4) probe: that probe is depth-lifted too, so it draws the floor either way and
    NOTHING on the board changes behavior today — but the floor's justification is "a wrong yes
    changes the model class of every deploy split", and that is as true of order as of depth. The
    grid tops out at order 3, so no sampled draw can reach this branch.

    `or`, not a dict default: a config `hp:` block may name either key with an explicit null,
    meaning "leave it to the library". int(None) would raise, and a raising hook is swallowed back
    to the 1-SE rule by `hpo._gate_se_mult` — i.e. an exception here would SILENTLY WEAKEN the
    gate, the one direction it must never move in.
    """
    if not hp:
        return False
    return (int(hp.get("max_depth") or _LEGACY_MAX_DEPTH) > _LEGACY_MAX_DEPTH
            or int(hp.get("max_interaction_order") or _LEGACY_MAX_ORDER) > _LEGACY_MAX_ORDER)


def _dim_seed(seed: int, key: str) -> int:
    """Stable per-dimension seed, independent of every other dimension and of n (see sample_hp_set)."""
    token = f"{int(seed)}:{key}"
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")


def _dim_sequence(values: list, n: int, seed: int, key: str) -> list:
    """`n` draws of `values` as an INDEPENDENTLY RESHUFFLED ROUND-ROBIN.

    Each successive block of `len(values)` draws is a fresh permutation from this dimension's own
    stream, so the sequence keeps the marginal balance a round-robin gives (every value appears
    floor/ceil(n / len(values)) times) WITHOUT being a fixed function of the index — which is what
    decorrelates it from every other dimension.

    THE BUG THIS REPLACES (hsv 25, measured 2026-09-04). The previous form was
    `cycle[i % len(cycle)]` with one shuffle per dimension: the per-dimension seed permuted each
    cycle's ORDER but every dimension was indexed by the SAME `i`, so its PHASE was shared. Two
    dimensions of equal length were therefore locked in a fixed bijection, and dimensions whose
    lengths share factors were partially confounded. At the shipped `n_configs: 25` the sampled
    tail is 15 draws and the damage was:
        colsample_bytree x max_bin x learning_rate x cat_smooth (4 x 4 x 4 x 2 = 128 nominal)
            -> FOUR distinct combinations, each repeated 3-4 times
        (lambda_, order, hurdle)          -> 15 of a nominal 45
        (min_data_in_leaf, order, hurdle) -> 15 of a nominal 75
    Every per-dimension MARGINAL was balanced, which is why it survived review — the docstring's
    marginal claim was true and its "without correlating it to the others" was not. Concretely,
    `colsample_bytree = 0.45` (added at hsv 23 to close a measured LightGBM gap) could only ever
    be drawn alongside `lr=0.1, max_bin=32, cat_smooth=None`, so it could never be scored on its
    own merits — exactly the confounding the coordinate-probe block exists to avoid.

    PREFIX STABILITY IS PRESERVED, and it is load-bearing (`hpo.static_configs` documents that
    this sampler's guarantees are properties of a contiguous prefix, so growing `n_configs`
    appends candidates instead of reshuffling them). Blocks are drawn in order from one stream,
    so `_dim_sequence(v, n, ...)` is a prefix of `_dim_sequence(v, n', ...)` for every n' > n.
    """
    rng = np.random.default_rng(_dim_seed(seed, key))
    out: list = []
    while len(out) < n:
        block = list(values)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


class TBoost(ModelAdapter):
    name = "t-boost"
    family = "glassbox-gbm"                 # boosted AND interpretable — its own point on the tradeoff
    adapter_version = "1"                   # reset 2026-09-09 (arena rebuilt from scratch); prior history in git
    hpo_search_version = "1"                # reset 2026-09-09 (arena rebuilt from scratch); prior history in git
    needs_es_probe = False                  # early-stops internally (recipe validation_fraction); skip probe
    supported_tasks = (TaskType.FREQ, TaskType.SEV, TaskType.PP, TaskType.CLF, TaskType.MULTICLF)

    # The count/rarity channel is now a NATIVE t-boost feature (cat_channels=["mean","count"],
    # enabled in the recommended_recipe call below since av 32) — the adapter's earlier __share
    # column shim is retired. The library adds a target-free count axis to every high-card
    # categorical (cat_count_min_levels=20, matching the retired shim's gate) and COLLAPSES it
    # losslessly with the mean-TS channel into one rating table per categorical. So _prep_X is
    # back to pure dtype coercion — no derived columns.
    def _prep_X(self, X: pd.DataFrame) -> pd.DataFrame:
        Xp = X.loc[:, self._columns].copy(deep=False)
        for c in self._columns:
            s = Xp[c]
            if c in self._cat:
                if s.isna().any() or not pd.api.types.is_object_dtype(s):
                    Xp = Xp.copy(deep=False)
                    Xp[c] = s.astype("string").fillna("nan").astype(object)
            elif (
                pd.api.types.is_extension_array_dtype(s.dtype)
                or not pd.api.types.is_numeric_dtype(s)
                or s.isna().any()
            ):
                Xp = Xp.copy(deep=False)
                Xp[c] = pd.to_numeric(s, errors="coerce").astype(float)
        return Xp

    def fit(self, train: DatasetBundle, val: DatasetBundle | None = None) -> "TBoost":
        from t_boost import recommended_recipe

        self._columns = list(train.X.columns)
        self._cat = [c for c in train.cat_features if c in self._columns]
        self._task = train.task.task
        self._is_multi = self._task == TaskType.MULTICLF
        self._is_clf = self._task in (TaskType.CLF, TaskType.MULTICLF)
        freq_form = self.hp.get("frequency_formulation", "rate_weight")
        if freq_form not in ("rate_weight", "count_offset"):
            raise ValueError(
                f"unsupported frequency_formulation {freq_form!r}, must be 'rate_weight' or 'count_offset'"
            )
        if freq_form == "count_offset" and self._task != TaskType.FREQ:
            raise ValueError(
                f"frequency_formulation='count_offset' is only valid for FREQ tasks, got task {self._task.value}"
            )
        self.frequency_formulation = freq_form

        fit_exposure = None
        if freq_form == "count_offset":
            from insurarena.scoring import validate_bundle
            validate_bundle(train)
            if train.exposure is None:
                raise ValueError("frequency_formulation='count_offset' requires bundle.exposure")
            fit_exposure = np.asarray(train.exposure, dtype=float)
            y = np.asarray(train.y, dtype=float)
            w = np.ones(len(y), dtype=float)
        else:
            y, w, _ = task_targets(train)          # rate + weight=exposure (native offset arg intentionally unused)
            if self._is_clf:
                y = np.asarray(y, dtype=int)        # classifier path needs discrete labels; K>=3 uses softmax
            else:
                y = np.asarray(y, dtype=float)

        # n_estimators (set by the harness from OOF selection) overrides the n_trees cap.
        budget = int(self.hp.get("n_estimators", self.hp.get("budget", _DEFAULT_BUDGET)))
        # THE CLOSURE GUARD (2026-09-04). This filter is the ONLY route from `self.hp` to the
        # estimator, and it used to drop an unlisted key SILENTLY. That is the one failure a
        # version-keyed cache cannot detect: a grid or probe dimension not named in
        # `_PASSTHROUGH` still enters the hp dict, still changes `hpo_config_set`, still forces a
        # full recompute — and never reaches the fit, so every candidate scores as if the
        # dimension did not exist. `max_bin` sat in exactly that state for the adapter's whole
        # life until hsv 18. Raising here makes that state unreachable rather than merely
        # documented, and `tests/test_passthrough_closure.py` asserts the closure holds over
        # every hp producer so the failure cannot be reintroduced.
        #
        # NOTE the scope, which is narrower than it first looks: keys written by a fit-time GATE
        # go straight into `overrides` below and never pass through here, so a gate-written key
        # need not be listed for the FIT to be correct — it is listed so that an explicit `hp`
        # value can outrank the gate. Only keys reaching the fit SOLELY via `self.hp` (the tune
        # grid, the coordinate probes, `tuning_base_hp`) are exposed to the silent-drop mode.
        _unlisted = sorted(k for k in self.hp
                           if k in _own_hp_keys()
                           and k not in _PASSTHROUGH and k not in _HP_NOT_FORWARDED)
        if _unlisted:
            raise ValueError(
                f"hp keys {_unlisted} are produced by this adapter's own search but are not in "
                f"_PASSTHROUGH, so they would be dropped silently and never reach the estimator. "
                f"Add them to _PASSTHROUGH (and bump `hpo_search_version` if a candidate carries "
                f"them), or to `_HP_NOT_FORWARDED` if they are harness-only control keys."
            )
        overrides = {c: self.hp[c] for c in _PASSTHROUGH if c in self.hp and c != "frequency_formulation"}
        if self._task == TaskType.PP:
            overrides["tweedie_rho"] = float(train.task.tweedie_power or 1.5)
        # t-boost table pruning: deploy the CV-pruned tables-only model (val is None). Skipped
        # during the tuned OOF search (val given) since pruning is a structural refinement, not a tuned
        # hyper-parameter — keeps config search tractable. Prune sub-params stay at recipe defaults.
        # Both branches are EXPLICIT since t-boost 2026-07-15 made prune=True the library default
        # (benchmark parity — the constructor now matches what this adapter always deployed): the
        # else-pin preserves this adapter's original prune-free OOF-search fits, keeping every cached
        # cell's behavior identical across that library change (no adapter_version bump needed).
        if val is None or bool(self.hp.get("_prune_during_hpo", False)):
            overrides["prune"] = True
        else:
            overrides["prune"] = False
        # Selection-only ES patience (roadmap B3, verified: identical fold scores at 1.21x less
        # wall on brvehins1). Injected by hpo._oof_score via selection_hp() below — present only
        # on selection fits (val given); deploy fits keep the library default (1.5).
        sel_es = self.hp.get("_selection_es_adaptive")
        if val is not None and sel_es is not None:
            overrides["early_stopping_adaptive"] = float(sel_es)

        _t = os.environ.get("OMP_NUM_THREADS")             # respect the parallel-worker thread budget
        n_jobs = self.hp.get("n_jobs") or (int(_t) if _t else None)
        # Native count/rarity channel (av 32): enable the library's second target-free categorical
        # axis on high-card cats (>=20 post-pooling levels — matches the retired adapter shim), which
        # t-boost collapses losslessly with the mean-TS channel into one rating table per cat.
        # Inert without categoricals; passed only when the dataset has any.
        #
        # MULTICLF (av 34): swap the mean channel for the P3 per-class frequency channels. The
        # multiclass mean-TS is E[class_index | level] — an ORDINAL read of nominal classes — so
        # "class_freq" replaces it with one channel per class, E[1[y==k] | level] (>=3 post-pooling
        # levels; the K channels still collapse into ONE rating table per categorical). This is an
        # ENGINE DEFAULT for the benchmark, not a tuned knob: default and tuned variants both get
        # it, exactly like the count channel above. K=2/regression are unaffected (their mean-TS
        # already IS the class-1 frequency), so only the MULTICLF branch changes.
        cat_channels = ["count", "class_freq"] if self._is_multi else ["mean", "count"]
        cat_channel_kw = ({"cat_channels": cat_channels, "cat_count_min_levels": 20}
                          if self._cat else {})
        # RARE-FLOOR GATE (av 35): open `cat_min_data_per_group` to 1.0 on the slices where the
        # library's weighted 10.0 floor is demolishing a categorical rather than regularizing it
        # (see the module-level note). Measured on the fit slice about to be used, so selection
        # folds and the deploy fit each decide for themselves. An explicit `hp` value outranks the
        # gate — a config that names the floor (ablation, or a future tuned dimension) gets what it
        # asked for. `self.rare_floor_gate_` keeps the decision inspectable after the fit.
        w_gate = np.asarray(fit_exposure if fit_exposure is not None else w, dtype=float)
        self.rare_floor_gate_ = _rare_floor_gate(train.X, self._cat, w_gate)
        if self.rare_floor_gate_["fires"] and "cat_min_data_per_group" not in self.hp:
            overrides["cat_min_data_per_group"] = _RARE_OPEN_FLOOR
        # SIZE-GATED BAGGING (av 38): see the module-level note by _BAGS_SMALL_THRESHOLD.
        # Dataset-level n via bundle meta (survives .subset(), so selection folds and the
        # deploy fit reach the same verdict); explicit hp outranks the gate.
        _n_ds = int(train.meta.get("n_rows", len(train.X)))
        self.bags_gate_ = {"fires": bool(_n_ds < _BAGS_SMALL_THRESHOLD), "n_rows": _n_ds,
                           "n_bags": _BAGS_SMALL_BAGS}
        if self.bags_gate_["fires"] and "n_bags" not in self.hp:
            overrides["n_bags"] = _BAGS_SMALL_BAGS
        # SIZE-GATED FOLD FIDELITY + KEEP BUDGET (av 40): see the module-level note by
        # `_FIDELITY_KEEP_BUDGET`. Same dataset-level n, same threshold and same explicit-hp
        # precedence as the bags gate above — this is the av45 fairness cycle folding the deleted
        # `tuned_tboost_small_ffull.yaml` into the adapter, so t-boost resolves from ONE
        # config like every comparator. Applied on selection fits too (the deleted config's `hp:`
        # block reached every candidate), which is what keeps the ranked fit and the shipped fit
        # in the same regime; the knobs are inert where the fit does not prune.
        # NOT ON MULTICLF, and not silently: the K>=3 prune path selects on ONE train/select split
        # (`fit_multiclass_pruned`) and has no per-fold refits to pin a candidate bank into, so
        # t-boost REFUSES `prune_fold_fidelity` there rather than no-opping, and takes no
        # `keep_budget` argument on that path at all. Same filter discipline as the cell-refit and
        # depth gates: a gate never hands a fit a knob the fit cannot honor. Inert on the live
        # catalog — the five gated sets are CLF/SEV/FREQ/PP and none is MULTICLF — but a small
        # K>=3 dataset joining the catalog must not crash the campaign.
        self.fidelity_gate_ = {"fires": bool(_n_ds < _BAGS_SMALL_THRESHOLD and not self._is_multi),
                               "n_rows": _n_ds, "task": self._task.value,
                               "prune_fold_fidelity": True,
                               "prune_keep_budget": _FIDELITY_KEEP_BUDGET, "applied": {}}
        if self.fidelity_gate_["fires"]:
            for _k, _v in (("prune_fold_fidelity", True),
                           ("prune_keep_budget", _FIDELITY_KEEP_BUDGET)):
                if _k not in self.hp:                    # explicit hp outranks the gate
                    overrides[_k] = _v
                    self.fidelity_gate_["applied"][_k] = _v
        # K>=3 CELL REFIT (av 39): see the module-level note by _CELL_REFIT_BASE. Armed on
        # MULTICLF targets only; explicit hp outranks the gate. The fit-time verdict (accepted
        # lambda, reachable support coverage, whether the joint backtrack declined) is filled in
        # after the fit below — the gate dict itself is written here so the DECISION is
        # inspectable even on a fit that raises.
        # The refit needs an out-of-bag complement, and t-boost REFUSES (rather than silently
        # no-opping) at n_bags < 2 — so the HPO scout overlay's `n_bags: 1` must not be handed a
        # cell_refit_base. Only an explicit hp `n_bags` can be below 2; absent one the recipe
        # bags >= 4 (>= 32 under the av38 size gate), so the deploy fits all qualify.
        _bags = self.hp.get("n_bags", overrides.get("n_bags"))
        _bags_ok = _bags is None or int(_bags) >= 2
        self.cell_refit_gate_ = {"fires": bool(self._is_multi and _bags_ok),
                                 "task": self._task.value, "n_bags": _bags,
                                 "base": _CELL_REFIT_BASE, "report": None}
        if self.cell_refit_gate_["fires"] and "cell_refit_base" not in self.hp:
            overrides["cell_refit_base"] = _CELL_REFIT_BASE
        elif not self.cell_refit_gate_["fires"]:
            # FILTER half of the gate. NOTE (audited 2026-09-04): this branch is currently
            # UNREACHABLE-WITH-EFFECT. The comment it replaced claimed "`cell_refit_base` is now a
            # tuned GRID dimension ... every candidate carries it" — true at hsv 20, FALSE since
            # hsv 24 retired that probe, so nothing sets the key and the pop can only ever return
            # None. Kept (not deleted) because it is the filter half of a gate whose injector half
            # is live, and because a future probe or `hp:` block re-arms it immediately: a scout
            # eval at n_bags=1 handed a refit base makes t-boost RAISE. Dropped, not zeroed, so
            # the recipe sees exactly what a candidate without the key would have handed it.
            #
            # This is the ONLY behavior change at fixed hp in av40, and it cannot touch a cached
            # cell: no DEFAULT-variant hp dict has ever carried `cell_refit_base` (`default_hp` is
            # {budget}), no shipped config names it in an `hp:` block, and every TUNED cell
            # recomputes under hsv 20 regardless. `adapter_version` therefore stays 39.
            _dropped = overrides.pop("cell_refit_base", None)
            self.cell_refit_gate_["dropped_hp_base"] = _dropped
        # DEPTH LIFT + TABLE BUDGET (av 40, hsv 20): see `_STATIC_TUNE_GRID["max_depth"]` and
        # `_DEPTH_LIFT_TABLE_BUDGET`. Depth is SEARCHED; the readability contract that rides with
        # it is a CONSTANT — a lifted fit gets the >=3-way table cap whether or not it asked, and
        # the search can never buy skill by relaxing the bar. Explicit hp still outranks the gate
        # so an ablation is reachable. Inert at max_depth=3, which is every default cell and every
        # candidate that draws the incumbent rung, so those fits are byte-identical to hsv 19.
        #
        # The cap is a DEPLOY-time trim of the prune's keep-set, and t-boost REFUSES it (rather
        # than silently no-opping) on an unpruned fit — there is no keep-set to trim. The HPO
        # scout overlay sets `_prune_during_hpo: False`, so `overrides["prune"]` is the condition,
        # not the task or the bag count. Same filter discipline as the cell-refit gate above: drop
        # an explicit hp value the fit cannot honor rather than let it raise.
        # `or` rather than a dict default: a config `hp:` block may name the key with a null,
        # which means "leave it to the library", i.e. the legacy depth. int(None) would raise.
        _depth = int(overrides.get("max_depth") or _LEGACY_MAX_DEPTH)
        _pruning = bool(overrides.get("prune", False))
        self.depth_gate_ = {"fires": bool(_depth > _LEGACY_MAX_DEPTH), "max_depth": _depth,
                            "prune": _pruning, "table_budget": _DEPTH_LIFT_TABLE_BUDGET,
                            "applied": None}
        if not _pruning:
            _dropped_tb = overrides.pop("prune_table_budget", None)
            self.depth_gate_["dropped_hp_budget"] = _dropped_tb
        elif self.depth_gate_["fires"] and "prune_table_budget" not in self.hp:
            overrides["prune_table_budget"] = _DEPTH_LIFT_TABLE_BUDGET
        self.depth_gate_["applied"] = overrides.get("prune_table_budget")
        self.model = recommended_recipe(
            _OBJ[self._task],
            budget=budget,
            tuned=bool(self.hp.get("_tuned", False)),      # default variant -> False, tuned variant -> True
            seed=int(self.hp.get("seed", 0)),
            n_jobs=n_jobs,
            categorical_features=self._cat or None,
            **cat_channel_kw,
            **overrides,
        )
        fit_kwargs = {}
        if fit_exposure is not None:
            fit_kwargs["exposure"] = fit_exposure
        if train.group_ids is not None:
            # Group-honest internal carves (t-boost 2026-07-13, extended to every objective
            # 2026-07-20): on grouped panels the row-level ES/prune-select carves leak
            # near-duplicate rows into validation and defeat early stopping (fits ran to the
            # 4000-tree cap — measured on the payfreq multiclass panels AND on fremotor_prem's
            # gamma severity fit). Passing the bundle's groups aligns the INTERNAL carves with
            # the benchmark's group-aware outer splits. Honored for every task (regression,
            # binary, and multiclass) since adapter_version 28 (previously multiclass-only,
            # the estimator rejected groups for K=2).
            fit_kwargs["groups"] = np.asarray(train.group_ids)
        self.model.fit(self._prep_X(train.X), y, sample_weight=np.asarray(w, dtype=float), **fit_kwargs)
        # K>=3 cell-refit verdict for THIS fit (av 39). `cell_refit_report_` is the library's
        # runtime-only report on the DEPLOYED fit: `lambda` (the joint backtrack's accepted step;
        # 0.0 = declined outright), `declined`, `n_supports`/`n_blocked`/`reachable_coverage`
        # (how much of the realized order-<=2 structure the refit could reach — supports touching
        # a P1 multi-channel categorical axis have no per-bin cell map and are excluded, which is
        # the honest denominator for reading a lambda==0 verdict on a categorical-heavy set),
        # `n_rejected`, `guard_rows`. Stays None on the selection fits (n_bags=1 has no out-of-bag
        # complement) and on every non-MULTICLF task, so the gate dict alone says which it was.
        self.cell_refit_gate_["report"] = getattr(self.model, "cell_refit_report_", None)
        # DEPLOYED >=3-WAY TABLE CENSUS (av 40). The budget caps SUPPORTS — distinct interactions
        # in the shared keep-set — but a filing reader faces one printed table per support PER
        # CLASS, so on a K-class book the census is the number the explainability bar is really
        # about and it is ~K x the cap. Both are recorded here, unreduced, because which of the
        # two Ralph's "100 3-way tables" line means is HIS call and this adapter must not decide
        # it by only reporting the flattering one. `table_budget` is the library's own report
        # (absent unless a budget was armed); `census_by_arity` counts the deployed bank across
        # every class bank. None on a fit with no prune report (an unpruned scout eval).
        self.table_census_ = self._deployed_census()
        self.best_iteration_ = 0               # early stopping is internal; no external probe needed
        self._fitted = True
        return self

    def _deployed_census(self) -> dict | None:
        """Deployed tables BY ARITY over every class bank, plus the library's budget report.

        The budget counts supports; this counts the tables a filing actually prints. On a K-class
        fit the two differ by roughly K, which is the open product question (`_DEPTH_LIFT_TABLE_
        BUDGET`) — so both numbers are reported and neither is reduced to the other.
        """
        rep = getattr(self.model, "pruning_report_", None)
        if not rep:
            return None
        out: dict = {"kept_supports": len(rep.get("kept") or []),
                     "table_budget": rep.get("table_budget"),
                     # 2026-09-26: the keep-set selector (ranked path by default) and the banding
                     # summary (per class bank on a K-class fit) — a fit-path change made visible.
                     "selector": rep.get("selector")}
        band = rep.get("banding")
        if band is not None:
            keys = ("tolerance", "budget", "combined_mse", "calibration_rounds", "cells_before",
                    "cells_after", "largest_after", "skipped")
            out["banding"] = ([{k: b.get(k) for k in keys if k in b} for b in band]
                              if isinstance(band, list) else {k: band.get(k) for k in keys if k in band})
        # GRADUATION VERDICT (2026-09-20). The library reports, per fit, how many rows the
        # deployed model actually trained on and whether the smoothed bank was adopted. This
        # is surfaced because it was NOT: between 2026-09-09 and 2026-09-20 graduation carved
        # `prune_validation_fraction` (15%) of every single-output fit's rows and never gave
        # them back, and 930 cells went onto the board with no record of it — the regression
        # was only found by diffing two boards. A fit-path change that silently removes
        # training data must be visible in the evidence, not inferable from the scores.
        grad = rep.get("graduation_validation")
        if grad:
            out["graduation"] = {k: grad.get(k) for k in
                                 ("training_rows", "holdout_rows", "adopted", "n_smoothed",
                                  "evidence", "skipped") if grad.get(k) is not None}
        # Single-output estimators keep the deployed tables on `_model`, K-class ones on
        # `_multi_model` (before 2026-09-28 only `_model` was read, so multiclass cells silently
        # recorded no arity census).
        tm = getattr(self.model, "_model", None) or getattr(self.model, "_multi_model", None)
        # t-boost >= 2026-09-27 counts natively (identical numbers); serializing catelematic13's
        # banded bank to JSON cost ~50 s and a ~15 GB transient per fit.
        native = getattr(tm, "deployed_census", None)
        if native is not None:
            try:
                c = native()
                out["census_by_arity"] = {str(k): v for k, v in sorted(c["tables"].items())}
                out["cells_by_arity"] = {str(k): v for k, v in sorted(c["cells"].items())}
                out["census_ge3"] = sum(v for k, v in c["tables"].items() if k >= 3)
                out["n_class_banks"] = int(c["banks"])
                return out
            except Exception:  # noqa: BLE001 - fall back to the JSON walk below
                pass
        banks = []
        try:
            doc = json.loads(tm.to_json())["model"]
            banks = [c["bank"] for c in doc["classes"]] if "classes" in doc else [doc["bank"]]
        except Exception:      # noqa: BLE001 - a census is a diagnostic; never fail a fit for it
            return out
        by_arity: dict[int, int] = {}
        cells_by_arity: dict[int, int] = {}
        for bank in banks:
            for t in (bank.get("tables") or []) + (bank.get("factored") or []):
                k = len(t.get("u") or [])
                by_arity[k] = by_arity.get(k, 0) + 1
                shape = (t.get("values") or {}).get("shape")
                if shape:
                    cells_by_arity[k] = cells_by_arity.get(k, 0) + int(np.prod(shape))
        out["census_by_arity"] = {str(k): v for k, v in sorted(by_arity.items())}
        out["cells_by_arity"] = {str(k): v for k, v in sorted(cells_by_arity.items())}
        out["census_ge3"] = sum(v for k, v in by_arity.items() if k >= 3)
        out["n_class_banks"] = len(banks)
        return out

    def predict(self, X: pd.DataFrame, exposure: np.ndarray | None = None) -> np.ndarray:
        Xp = self._prep_X(X)
        if self._is_multi:
            proba = np.asarray(self.model.predict_proba(Xp), dtype=float)
            proba = np.clip(proba, 1e-15, 1.0)
            return proba / proba.sum(axis=1, keepdims=True)
        if self._is_clf:
            return np.clip(np.asarray(self.model.predict_proba(Xp)[:, 1], dtype=float), 0.0, 1.0)
        return np.clip(np.asarray(self.model.predict(Xp), dtype=float), 1e-9, None)

    def explain(self) -> Explanation:
        # t-boost decomposes to a functional-ANOVA table (interpretable), but no scored artifact here.
        return Explanation(kind="none")

    @staticmethod
    def default_hp() -> dict:
        # benchmark 'default' -> the recipe's untuned base (in-model n_bags=4 bagging, one ANOVA table).
        return {"budget": _DEFAULT_BUDGET}

    @staticmethod
    def selection_hp() -> dict:
        # Scout regime (Ralph, 2026-07-16): stage evals are RANKING evidence, so spend their
        # compute on CV splits, not in-fit variance reduction — 1 bag (bagging shrinks only the
        # model-variance slice of eval noise: paired-fold gate SEs measured just ~1.2x wider at
        # halved bags, sgautonb 2026-07-16) and no prune pipeline (B1: ~1.5x per eval; the
        # rank-what-ships concern is answered by the finale below, which re-scores the walk's
        # winner at ship fidelity before anything is decided). ES 1.2 per B3 (verified
        # truncation-identical). Deploy fits never see this overlay. hpo_search_version 11.
        return {"_selection_es_adaptive": 1.2, "n_bags": 1, "_prune_during_hpo": False}

    @staticmethod
    def selection_finale_hp() -> dict:
        # Ship-fidelity overlay for the walk's finale: the final incumbent and the default
        # anchor are re-evaluated with the full recipe (candidate n_bags, pruned evals on BOTH
        # sides — matched-fidelity gate, deploy-faithful) and only the B3 ES ratio retained.
        # The ship decision and deployed iteration count come from these evals.
        # (2026-08-04: dropping the ES ratio here was drafted and REVERTED same day — the
        # "finale mis-estimates deploys by +5%" homesite evidence compared av32 finale evals
        # against av31 deploy cells; at av32-vs-av32 the finale estimates were faithful.)
        # STILL STANDS at hsv 18. That draft bundled TWO changes and only one of them had a
        # counterexample: hsv 18 re-opens the pooled-finale extension (norauto, a measured
        # large-set case — see eval/runner._tuned_once) and deliberately leaves this overlay
        # alone, because nothing since has contradicted the av32-vs-av32 finding above. The
        # finale's FIDELITY is not the thing under test; the amount of EVIDENCE it gates on is.
        return {"_selection_es_adaptive": 1.2, "_prune_during_hpo": True}

    @staticmethod
    def tuning_base_hp() -> dict:
        # Greedy tuning starts from the tuned recipe with explicit defaults for every staged knob, so
        # each candidate changes exactly one knob at a time. The low adaptive hurdle base lets the
        # new lower-order refinement competition work on weak-signal portfolios, while the first
        # coordinate stage can still move to stricter interaction admission.
        return {
            "_tuned": True,
            "_prune_during_hpo": True,
            "budget": _DEFAULT_BUDGET,
            "interaction_gain_hurdle": 2.0,
            "interaction_gain_hurdle_mode": "adaptive",
            "max_interaction_order": 3,
            "lambda_": 1.0,
            "colsample_bytree": 0.8,
            # n_bags deliberately NOT pinned (was 8 until hsv 15): the recipe's bag count is
            # objective-aware since 2026-07-22 (gamma/tweedie 16, else 8), and an explicit pin
            # here would silently override it on every tuned champion deploy while the anchor
            # ({budget} only) picked up the recipe value — an anchor-vs-champion fidelity split.
            # Scout evals still fit 1 bag regardless (selection_hp overlay wins over base hp).
            "min_data_in_leaf": 0,
            "learning_rate": 0.05,
        }

    @staticmethod
    def coordinate_tuning_stages() -> list[dict]:
        # Retired 2026-07-20 → static 25-candidate basis (see _STATIC_TUNE_GRID / sample_hp).
        # Returning [] routes the HPO dispatch to _select_config_static, the same static random
        # search every other tunable model uses. Kept as a documented stub (not deleted) so the
        # coordinate→static migration is legible at the adapter.
        return []

    @staticmethod
    def sample_hp(rng: np.random.Generator) -> dict:
        # benchmark 'tuned' -> the recipe's TUNED family (reanchor + k-fold cat encoding + 8-bag
        # deploy, prune-on: tuning_base_hp) with the six retired-coordinate knobs drawn JOINTLY from
        # their validated grids (_STATIC_TUNE_GRID). max_interaction_order includes 1 (pure additive),
        # so tuning can find the additive sweet spot on low-signal data (as EBM can). The scout
        # overlay (selection_hp) reduces every candidate to 1 bag during OOF scoring; n_bags=8 deploys.
        # Superseded by sample_hp_set below (hpo_search_version 13, Ralph 2026-07-20): kept as the
        # single-candidate primitive `static_configs` falls back to for every model that doesn't
        # declare sample_hp_set.
        hp = TBoost.tuning_base_hp()
        for name, values in _STATIC_TUNE_GRID.items():
            hp[name] = values[int(rng.integers(len(values)))]
        return hp

    @staticmethod
    def ship_gate_se_mult(inner_repeats: int, champion_hp: dict | None = None) -> float:
        # hsv 19 (2026-08-24). The ship gate's bar scales with `inner_repeats`, which is exactly
        # `clip(ceil(30000 / n_outer_train), 1, 5)` — the arena's own statement of how far this
        # tune split falls short of the evidence the protocol wants. At repeats==1 the bar is 1.0
        # and NOTHING changes; at repeats==5 the champion must clear 5 SEs.
        #
        # WHY, measured on the hsv 18 full recompute (40 datasets, 20 champion ships, each scored
        # against the anchor deploy at the IDENTICAL per-cell seed wherever hsv 17 shipped the
        # anchor). Split the 20 ships by the repeats flag:
        #   inner_repeats > 1 : swautoins -0.00632, swautoins_pp -0.01057, freclaimdam -0.02533,
        #                       autoclaims_us +0.00113, uslapseagent +0.00564   => SUM -0.03545
        #   inner_repeats == 1: 15 ships, worst single -0.00110                 => SUM +0.00337
        # The gate is not mildly miscalibrated on volatile sets; it is inverted. Every unit of net
        # damage on the whole board came from five cells the protocol had ALREADY flagged as
        # under-powered.
        #
        # Two things are wrong at repeats>1, and one factor fixes both:
        #  1. VARIANCE — a real error. `_paired_margin` divides by sqrt(len(diffs)) with
        #     len(diffs) = k*repeats (+k per pooled extra split). But the repeats re-partition the
        #     SAME rows — that is their entire purpose — so they add no independent evidence while
        #     shrinking the reported SE by sqrt(repeats). Undoing that is worth sqrt(repeats) and
        #     is simply correct.
        #  2. LOSS ASYMMETRY — a decision. Even after (1) the tune margin is optimistic: the
        #     champion is a max over ~26 candidates, re-scored at the finale on the very rows that
        #     selected it, and the deploy on these sets is hyper-volatile (freclaimdam's own anchor
        #     moves +0.027 between two seeds at IDENTICAL hp — ~50x the margin under test). The
        #     measured payoff is brutally asymmetric: the worst wrong champion cost -0.0253, the
        #     best right one on any repeats>1 cell was worth +0.0056. A second sqrt(repeats) of
        #     premium is cheap insurance against that tail.
        # Product: repeats * SE. Counterfactual over the 20 hsv-18 ships: refuses all 3 cursed
        # ships, keeps uslapseagent (t=7.09 vs bar 2.0 — 3.5x headroom), leaves all 15 repeats==1
        # ships EXACTLY as they were (hsv 18's norauto and euhealth wins survive), flips only 4
        # cells, net recovery +0.04109 skill.
        # Rejected alternatives, scored on the same 20 ships:
        #   sqrt(2 ln m) selection-pressure (Bonferroni-flavoured): m is 19-26 on every cell
        #     because n_configs is fixed, so it is a near-constant ~2.5-SE bar. Catches 2/3 cursed
        #     and KILLS norauto + euhealth + 10 more repeats==1 ships. +0.02847.
        #   2*SE when repeats>1: catches 1/3 — freclaimdam (2.40) and swautoins_pp (2.71) both
        #     clear it. +0.00519.
        #   max of those two: same 2/3, same headline damage. +0.02847.
        #   flat 2.75*SE everywhere: catches 3/3 but costs the repeats==1 wins. +0.03910.
        #   sqrt(repeats) alone — the variance fix (1) WITHOUT the premium (2): catches 1/3,
        #     +0.00519. That is why (2) is not decoration.
        # The chosen rule dominates all of them on every axis, and is a strict TIGHTENING: it can
        # only turn a champion into the anchor, never the reverse, so it cannot create a new
        # harmful ship anywhere on the board.
        #
        # Note it does not interact with the hsv 18 adaptive pooled finale: that trigger only ever
        # runs at repeats==1 (repeats>1 pools unconditionally), where this returns 1.0.
        bar = float(max(1, int(inner_repeats)))
        # --- hsv 20 (2026-08-27): a FLOOR of 2 SE for a DEPTH-LIFTED champion ------------------
        #
        # hsv 19's bar is a VOLATILITY premium: it reads `inner_repeats`, which measures how thin
        # the tune split is, and says nothing about what is being shipped. That is the right shape
        # for a knob whose failure mode is noise. It is the WRONG shape for a STRUCTURAL knob, and
        # the depth-cells battery measured it as exactly backwards on this one. All four battery
        # datasets are large enough that `resolve_inner_repeats` returns 1 — so the hsv-19 bar is
        # 1.0 on all three depth BENEFICIARIES (fremotor_payfreq, fremotor_prem, and the second
        # fremotor split), while the two datasets that REFUSE depth (swautoins 2.2k, sgautonb
        # 7.5k) are small enough to draw the 5.0 bar. The gate is strictest precisely where the
        # answer is already "don't", and weakest precisely where a wrong yes is expensive.
        #
        # Why depth is not like the other dimensions. Every other knob in `_STATIC_TUNE_GRID`
        # perturbs a depth-3 fit; `max_depth` changes the model class. A lifted champion ships a
        # different artifact — more realized cells per tree, a different table census, a different
        # readability profile (which is why `_DEPTH_LIFT_TABLE_BUDGET` rides with it) — and it is
        # not reversible per split: the deploy fits all take it. A 1-SE bar on a max-over-26
        # champion (max-over-28 since hsv 22) is not enough evidence to change the model class of
        # a shipped cell.
        #
        # So: `max(hsv-19 bar, 3.0)` when the champion is depth-lifted. A FLOOR, not a multiplier,
        # so the volatility premium still wins wherever it is larger (the refusing controls keep
        # their 5.0), and a strict TIGHTENING everywhere — it can only turn a champion into the
        # anchor, never the reverse, so it cannot create a harmful ship.
        #
        # WHY 3 AND NOT 2, which is what this floor was first drafted at. The campaign-budget
        # selection smoke put a number on it. `fremotor_payfreq`'s depth-lifted champion cleared
        # a 2.0 bar at margin/SE = 2.27 and its 9 deploy cells then measured -0.00103 skill
        # against the av39 tuned cell (SE 0.00154, W-L 3-6) at 1.70x the fit time. That is the
        # only structural ship anyone has observed end to end, it was harmful, and a 2-SE bar
        # admitted it. 3.0 refuses it. The same asymmetric-loss argument hsv 19 made and won: the
        # downside of a wrong structural yes is an order of magnitude larger than the upside of a
        # right one, and unlike an ordinary knob a wrong yes here changes the model class on every
        # deploy split of the cell.
        #
        # `champion_hp is None` (the hook called before a champion is resolved, or by an older
        # caller) returns the hsv-19 bar unchanged.
        # `or`, not a dict default: an explicit null means the library depth. int(None) would
        # raise, and `_gate_se_mult` swallows a raising hook back to the 1-SE rule — i.e. an
        # exception here would SILENTLY WEAKEN the gate, which is the one direction it must never
        # move in.
        #
        # hsv 22 widens the TEST (not the bar) from "depth-lifted" to STRUCTURAL — depth above 3
        # OR interaction order above 3 — see `_is_structural_hp`. Behavior on the board is
        # unchanged: the one order-lifted candidate that exists is the (depth 5, order 4) probe,
        # which is depth-lifted too, and the grid tops out at order 3 so no sampled draw can be
        # order-lifted. The widening is so the floor's own argument stays true of what it tests:
        # a wrong yes here changes the MODEL CLASS on every deploy split of the cell, and an
        # order-4 bank is as much a different artifact as a depth-6 one.
        if _is_structural_hp(champion_hp):
            return max(bar, _DEPTH_LIFT_SE_MULT_FLOOR)
        return bar

    @staticmethod
    def finale_rerace_spec() -> dict:
        """TOP-K FINALE RE-RACE, ON since hpo_search_version 23 (2026-09-01).

        WHY THIS ADAPTER AND NO OTHER. t-boost is the only model on the board whose scout runs
        at a REDUCED fidelity (`selection_hp` pins `n_bags: 1` and turns the prune off), so it is
        the only model whose argmax can be decided by a statistic its deploys never see. The
        eusavingul toggle experiment isolates the component exactly: at 1 bag the depth-8 candidate
        trails by +0.0000837 (t=+0.06, a coin flip); switch bagging on and NOTHING else and the
        pair moves by -0.0012251 at t=-3.54, 8 of 9 folds, and the order flips
        (av44 selection forensic, §1). Bagging is what converts that candidate's extra
        resolution into score, and the scout is the one regime where it is switched off.

        MEASURED RECORD at these settings (10 datasets, 219 deploy fits, all 9 outer splits each,
        the av44 top-K re-race battery): 1 WIN (eusavingul +0.0049, t=15.5) / 9 null /
        0 HARM. The rule captured 86% of all the headroom the deployed candidate sets held and
        harmed nothing; where the runner-up was materially worse the finale REFUSED it
        (catelematic13 0.3182 vs 0.3589, freMTPL2freq 0.0292 vs 0.0344, ohlsson 0.0005 vs 0.0047).
        It is a safe ranker as well as a productive one.

        k=4 IS THE BOUNDARY, not a preference: at k>=5 pg16_payfreq's pick turns -0.0009 and the
        net falls to +0.0040. `hpo._RERACE_K` carries the sweep, and the counterexample is pinned
        in `tests/test_hpo_finale_rerace.py`.

        COST, measured off the 40 cached hsv-22 cells rather than estimated: at band 1.0 SE the
        rule admits 1.625 extra contenders per cell on average (10 cells admit nothing, 13 admit
        the full 3), i.e. ~65 extra primary-split finale evaluations board-wide, plus the pooled
        re-scores on the cells that pool.
        """
        return {"k": 4, "band": 1.0}

    @staticmethod
    def selection_regime_recheck(hp: dict | None) -> bool:
        """Is this candidate RANKED in a regime it will never DEPLOY in? (hsv 21, 2026-08-31)

        True for exactly one shape: a DEPTH-LIFTED candidate. `selection_hp` turns the prune
        pipeline off for every scout evaluation (B1, ~1.5x per eval), and `prune_table_budget` is
        a trim of the prune's keep-set — so the fit-time depth gate has no keep-set to cap and
        DROPS the cap on the very evaluations that rank the candidate. The finale and all nine
        deploy fits prune, and take the cap.

        On a narrow set that is a distinction without a difference: where the cap is not binding,
        capped and uncapped fits are BIT-IDENTICAL (measured on eusavingul and beMTPL16, whose
        shipped depth champions keep 25 and 84 tables against a budget of 100). On a WIDE set it
        is a different model. allstate_sev's depth-6 probe proposes 3,174-3,944 >=3-way tables
        and the cap keeps 100 of them: uncapped it ranks +6.3 t against the anchor, capped it
        deploys at -0.0209 skill — a bigger number than the entire t-boost<->ccb2 gap this program
        exists to close, and it is the same candidate under both numbers.

        Two ways to close that: rank depth WITH the cap (turn the prune on for depth probes only,
        inside the scout), or re-race a mismatched argmax at finale fidelity before the gate sees
        it. The first is cheaper to describe and worse to do: it would score depth probes on
        PRUNED evals while every other candidate is scored unpruned, i.e. it fixes one
        apples-to-oranges comparison by introducing another one across the whole basis, and it
        pays ~1.5x on three of the twenty-six candidates every cell on the board. The second
        compares like with like — both candidates at finale fidelity, both pruned, both capped —
        and costs one extra finale evaluation only where a depth probe actually took the top
        slot. `hpo._select_config_static` implements the second.

        `max_depth` absent (every default cell, every basis draw) or an explicit null (a config
        `hp:` block meaning "library depth") is depth 3: no mismatch, no recheck, and the
        adapters that never declare this hook are byte-untouched.

        hsv 22: BOTH new joint probes carry `max_depth` > 3 (8 and 5), so both are recheck-eligible
        with no change to this test. It stays keyed on DEPTH alone — deliberately, and is not
        merged into `_is_structural_hp`. The two hooks answer different questions: that one asks
        "is this a different model CLASS?" (a ship-gate question, true of order 4 at any depth),
        while this one asks "did the SCOUT rank it in a regime it will not deploy in?", and the
        regime difference is specifically the `prune_table_budget` cap, which the fit-time depth
        gate arms on DEPTH. An order-4, depth-3 candidate — which nothing on the board can produce
        — would deploy in the regime it was ranked in, so making it recheck-eligible would buy a
        finale evaluation for a mismatch that does not exist.
        """
        return bool(hp) and int(hp.get("max_depth") or _LEGACY_MAX_DEPTH) > _LEGACY_MAX_DEPTH

    @staticmethod
    def coordinate_probe_hp_set() -> list[dict]:
        # hpo_search_version 18 (2026-08-23), widened to MULTI-KNOB probes at hsv 22 (2026-08-31).
        # See the `_COORDINATE_PROBES` note above for WHY these directions and no others.
        #
        # Identity contract (what `hpo.static_configs` and the cache rely on):
        #   * the returned list is a pure function of `_COORDINATE_PROBES` and `tuning_base_hp` —
        #     no seed, no rng, no dependence on n_configs;
        #   * probe j occupies nominal candidate index j+1 forever, so its cached evaluation is
        #     reusable and growing/shrinking n_configs never reshuffles it;
        #   * every probe asserts-by-construction that EVERY knob it names is at a DIFFERENT value
        #     in the base (checked below), so no probe — of one knob or of two — can be a silent
        #     duplicate of the base recipe, and no coordinate of a joint probe can be a silent
        #     no-op that quietly reduces it to a single-knob probe already in the block.
        #
        # WHY MULTI-KNOB AT ALL, given the block's founding argument was "offer a knob CLEAN".
        # The two are not in tension: "clean" means the probe's score is READABLE, i.e. attributable
        # to a named, campaign-measured direction rather than to a random bundle. A joint probe
        # whose shape was measured as a joint shape (see the hsv-22 note) is exactly as readable —
        # it just names a direction in two coordinates instead of one. What the block still refuses
        # is a direction NOBODY measured, which is why `_probe_moves` enumerates rather than
        # products: there is no combinatorial expansion here, only the shapes the forensic ran.
        base = TBoost.tuning_base_hp()
        probes = []
        for entry in _COORDINATE_PROBES:
            hp = dict(base)
            for name, value in _probe_moves(entry):
                if name in base and base[name] == value:
                    raise ValueError(
                        f"coordinate probe {name}={value!r} is already the tuning base")
                hp[name] = value
            probes.append(hp)
        return probes

    @staticmethod
    def sample_hp_set(n: int, seed: int) -> list[dict]:
        # Winner's-curse fix, part 2 (hpo_search_version 13, Ralph 2026-07-20): sample_hp's pure
        # per-candidate independent draw left the static-25 basis lumpy on a real campaign (colsample
        # 18/7, order-2 landing only 5/25) and gave essentially no joint coverage of the load-bearing
        # max_interaction_order x interaction_gain_hurdle pair (interaction admission gates on the
        # hurdle, and the hurdle's effect is order-dependent). This instead assigns each of the n
        # candidates a slot in a fixed CYCLE per dimension:
        #  - the joint pair cycles through a seeded permutation of all 15 (order, hurdle)
        #    combinations, so n>=15 covers every combination at least once, and any further
        #    candidates repeat the same permuted cycle (balanced round-robin, not a second random
        #    draw that could re-favor a combination already seen);
        #  - every OTHER knob in _STATIC_TUNE_GRID (lambda_, colsample_bytree, min_data_in_leaf,
        #    learning_rate, cat_smooth, and max_bin since hsv 18) each cycles through its OWN
        #    seeded permutation independently — a Latin-hypercube-style design that keeps every
        #    knob's marginal balanced AND (since hsv 25) decorrelated from the others — see
    #    `_dim_sequence`; before hsv 25 every dimension shared the index `i`, so the marginals
    #    were balanced but the JOINT was degenerate. A new dimension is
        #    therefore additive: it gets its own `_dim_seed` stream and leaves every other
        #    dimension's cycle byte-identical, so candidate i's OTHER knobs do not move when one
        #    is added (only the new key appears). max_bin's 4 values balance 5/5/5/4 over the 19
        #    sampled slots hsv 18 leaves after the coordinate-probe block.
        # Each dimension's permutation comes from its own seed (`_dim_seed`), not a single shared
        # `rng` stream, so — like `_config_seed` for the other models — candidate i's hp depends only
        # on i, not on n: growing n_configs appends new candidates instead of reshuffling old ones.
        n = int(n)
        base = TBoost.tuning_base_hp()

        joint_values = [
            (order, hurdle)
            for order in _STATIC_TUNE_GRID["max_interaction_order"]
            for hurdle in _STATIC_TUNE_GRID["interaction_gain_hurdle"]
        ]
        # One INDEPENDENT sequence per dimension (see `_dim_sequence` for the phase-collapse bug
        # this replaces at hsv 25). The joint pair keeps its own stream, so at n >= 15 the first
        # block is still a permutation of all 15 (order, hurdle) combinations — the coverage
        # guarantee `hpo.static_configs` and the `n_configs` note both depend on.
        joint_seq = _dim_sequence(joint_values, n, seed, "joint")
        marginal_seqs = {
            name: _dim_sequence(list(values), n, seed, name)
            for name, values in _STATIC_TUNE_GRID.items()
            if name not in _JOINT_DIMS
        }

        configs = []
        for i in range(n):
            hp = dict(base)
            order, hurdle = joint_seq[i]
            hp["max_interaction_order"] = order
            hp["interaction_gain_hurdle"] = hurdle
            for name, seq in marginal_seqs.items():
                hp[name] = seq[i]
            configs.append(hp)
        return configs

    @staticmethod
    def hp_cost_hint(hp: dict) -> float:
        # A4-static (hpo_search_version 14, Ralph 2026-07-21): relative fit cost of a candidate,
        # used ONLY to order candidate EVALUATION cheap-first so racing (A2) has a strong incumbent
        # before the expensive candidates run. It never changes which candidates exist, their
        # indices, their seeds, or their cached scores — see `_select_config_static`.
        #
        # learning_rate dominates: early stopping decides the real tree count, so halving the rate
        # roughly doubles the rounds. Measured on the scout path (1 bag, prune-free), trees at
        # lr 0.1 -> 0.05 -> 0.01:
        #     uslapseagent (CLF 19.5k)   87 ->  208 -> 1191   (0.68s -> 1.42s -> 5.58s)
        #     eusavingul   (CLF 14.6k)  247 ->  420 -> 2256   (1.54s -> 2.13s -> 6.31s)
        #     fremotor_prem(SEV 34.6k)  283 ->  232 -> 1852   (1.63s -> 1.32s -> 5.80s)
        # i.e. cost tracks ~1/lr, sub-linearly (ES truncates before the full 10x). The 8 lr=0.01
        # candidates of a 25-config basis burned 46.8% of uslapseagent's whole search.
        # max_interaction_order is a weak second term (order 3 admits more candidate splits per
        # node than order 1); it only breaks ties among equal-lr candidates.
        # `max_bin` (a tuned dimension since hsv 18) is deliberately NOT a term here: a coarser
        # histogram is genuinely cheaper, but the hint's job is to order EVALUATION for racing,
        # and adding an unmeasured term would move racing discards on every dataset for a
        # second-order effect. The hint is a cost model, not a completeness exercise.
        lr = float(hp.get("learning_rate", 0.05)) or 0.05
        order = float(hp.get("max_interaction_order", 3))
        return (0.05 / lr) ** 0.75 * (1.0 + 0.1 * order)


@lru_cache(maxsize=1)
def _own_hp_keys() -> frozenset:
    """Every hp key THIS adapter's own machinery can emit.

    The closure guard in `fit` is scoped to these. A key outside the set did not come from
    t-boost's search and dropping it is correct — `tests/test_adapters.py`'s conformance sweep
    deliberately hands every adapter one shared hp dict containing other models' knobs
    (`outer_bags`, `smoothing_rounds`, ...), and an adapter is supposed to ignore those. What the
    guard must catch is the opposite case: a dimension t-boost itself searches that was never
    wired through `_PASSTHROUGH`, which re-keys every tuned cell and then never reaches the fit.
    `max_bin` sat in exactly that state until hsv 18.
    """
    keys = set(_STATIC_TUNE_GRID)
    for entry in _COORDINATE_PROBES:
        keys |= {k for k, _v in _probe_moves(entry)}
    for producer in (TBoost.default_hp, TBoost.tuning_base_hp, TBoost.selection_hp):
        try:
            keys |= set(producer())
        except Exception:                     # a producer that needs context is not a source here
            pass
    return frozenset(keys)
