"""av35 rare-floor gate: open t-boost's `cat_min_data_per_group` where the library's weighted
10.0 floor is destroying a categorical rather than regularizing it.

The floor is stated in units of POOLING WEIGHT and this adapter passes `sample_weight=exposure`,
so on short-exposure portfolios it degenerates from a rarity test into an exposure test. The gate
measures the damage on the fit slice — the weighted share of the slice sitting in sub-floor levels,
per categorical — and opens the floor to 1.0 when the worst categorical exceeds 25% AND the slice
is not "small" by config `split.small_threshold`.

Two design decisions carry the whole gate and each gets a test of its own:

  * the share is WEIGHTED, not a row count. brvehins1's `VehModel` sits at 0.7% of exposure but
    4.2% of ROWS; a row-count gate fires there and moves ~20 cells for nothing.
  * the SIZE exclusion is load-bearing. brautocoll (n=876) runs ~0.68 on `Hour` purely because
    every level is thin at that sample size — opening the floor there trades regularization the
    fit genuinely needs for credibility it cannot have.

`_DECISIONS` is the nine-dataset rare-floor probe battery (103 measured splits) reduced to
the per-dataset envelope: the measured `n_train` and the min/max
weighted share ACROSS that dataset's splits, so pinning both ends pins every split. It is carried
inline rather than as a fixture file because it is a small, frozen evidence record, and the
dataset-level assertions run against synthetic frames that reproduce the measured share — the real
loaders need network, and these tests must run in CI.
"""
import numpy as np
import pandas as pd
import pytest

from insurarena.datasets.base import DatasetBundle
from insurarena.models.tboost_model import (
    _RARE_FLOOR,
    _RARE_OPEN_FLOOR,
    _RARE_SHARE_THRESHOLD,
    _RARE_SMALL_THRESHOLD,
    TBoost,
    _rare_floor_gate,
    _subfloor_shares,
)
from insurarena.tasks import FREQ_SPEC

# dataset -> (n_train, min share over splits, max share over splits, expected `fires`)
_DECISIONS = {
    "pg17_coverage":  (66666, 0.47257, 0.47828, True),
    "pg17_payfreq":   (66666, 0.47229, 0.47815, True),
    "beMTPL16":       (47194, 0.08834, 0.09175, False),
    "brvehins1":     (255080, 0.00638, 0.00668, False),
    "brvehins1_pp":  (255080, 0.00646, 0.00665, False),
    "brautocoll":       (876, 0.64155, 0.73288, False),   # excluded by size, not by share
    "tic2000":         (6548, 0.00260, 0.00351, False),
    "allstate_sev":  (125545, 0.00234, 0.00276, False),
    "pg15training":   (66680, 0.00000, 0.00000, False),
}


def _frame(n: int, share: float, *, sub_weight: float = 1.0, fat_weight: float = 1.0, seed: int = 0):
    """A frame whose single categorical puts exactly `share` of ROWS in sub-floor levels.

    Sub-floor rows each get their own level (pooled weight = `sub_weight` < the 10.0 floor); the
    rest are packed into fat levels of >= 40 rows (pooled weight >= 40 * `fat_weight` >= the
    floor). Any remainder joins the LAST fat level rather than opening a short one — a short tail
    level would itself fall below the floor and quietly inflate the share. With equal weights the
    WEIGHTED share equals the row share, which is what lets the dataset-level cases below
    reproduce a measured share exactly.
    """
    rng = np.random.default_rng(seed)
    k = int(round(share * n))
    n_fat = n - k
    assert sub_weight < _RARE_FLOOR <= 40 * fat_weight and n_fat >= 40
    last = n_fat // 40 - 1
    levels = [f"rare_{i}" for i in range(k)] + [f"fat_{min(i // 40, last)}" for i in range(n_fat)]
    w = np.concatenate([np.full(k, sub_weight), np.full(n_fat, fat_weight)])
    X = pd.DataFrame({"x1": rng.normal(size=n), "cat": levels})
    return X, w


def _bundle(n: int, share: float, *, sub_weight: float = 1.0, fat_weight: float = 1.0, seed: int = 0):
    """The same frame as a FREQ bundle, where `task_targets` hands the adapter weight == exposure."""
    X, w = _frame(n, share, sub_weight=sub_weight, fat_weight=fat_weight, seed=seed)
    rng = np.random.default_rng(seed + 1)
    counts = rng.poisson(0.2 * w)
    return DatasetBundle(
        name="rare_floor_probe", X=X, y=counts.astype(float), exposure=w, sample_weight=w,
        task=FREQ_SPEC, cat_features=["cat"],
    )


# --------------------------------------------------------------------------- share computation

def test_subfloor_share_is_the_weighted_share_of_sub_floor_levels():
    X, w = _frame(4000, 0.30)
    assert _subfloor_shares(X, ["cat"], w)["cat"] == pytest.approx(0.30, abs=1e-9)


def test_share_pools_a_level_across_its_rows_before_testing_the_floor():
    # 12 rows of one level at weight 1.0 pool to 12.0 -> ABOVE the 10.0 floor, so the share is 0,
    # even though no single row carries anywhere near the floor. The floor is per LEVEL, not per row.
    X = pd.DataFrame({"cat": ["a"] * 12})
    assert _subfloor_shares(X, ["cat"], np.ones(12))["cat"] == pytest.approx(0.0)
    # ... and 9 rows of it pool to 9.0, which is below: the whole level is sub-floor.
    X = pd.DataFrame({"cat": ["a"] * 9})
    assert _subfloor_shares(X, ["cat"], np.ones(9))["cat"] == pytest.approx(1.0)


def test_missing_levels_are_normalized_to_one_nan_level_like_prep_x():
    X = pd.DataFrame({"cat": ["a"] * 20 + [None] * 20})
    # The 20 nulls pool into a single "nan" level of weight 20.0 -> above the floor, share 0.
    assert _subfloor_shares(X, ["cat"], np.ones(40))["cat"] == pytest.approx(0.0)


def test_share_is_taken_per_column_and_the_gate_reports_the_worst():
    rng = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame({
        "clean": [f"fat_{i // 40}" for i in range(n)],
        "dirty": [f"rare_{i}" if i < int(0.4 * n) else f"fat_{i // 40}" for i in range(n)],
        "x1": rng.normal(size=n),
    })
    w = np.ones(n)
    shares = _subfloor_shares(X, ["clean", "dirty"], w)
    assert shares["clean"] == pytest.approx(0.0)
    assert shares["dirty"] == pytest.approx(0.4, abs=1e-9)
    g = _rare_floor_gate(X, ["clean", "dirty"], w)
    assert (g["fires"], g["col"], g["n_train"]) == (True, "dirty", n)


def test_no_categoricals_means_no_share_and_no_fire():
    X = pd.DataFrame({"x1": np.arange(5000, dtype=float)})
    g = _rare_floor_gate(X, [], np.ones(5000))
    assert g == {"fires": False, "max_share": 0.0, "col": "", "n_train": 5000}


# ------------------------------------------------------------------ threshold + size exclusion

@pytest.mark.parametrize("share,fires", [(0.10, False), (0.24, False), (0.26, True), (0.60, True)])
def test_threshold_is_the_share_exceeding_the_constant(share, fires):
    X, w = _frame(4000, share)
    assert _rare_floor_gate(X, ["cat"], w)["fires"] is fires


def test_size_exclusion_holds_a_severely_collapsed_but_small_slice():
    # brautocoll's shape: a share far above threshold on a slice far below `split.small_threshold`.
    small, big = _RARE_SMALL_THRESHOLD - 100, _RARE_SMALL_THRESHOLD + 100
    for n, expected in ((small, False), (big, True)):
        X, w = _frame(n, 0.68)
        g = _rare_floor_gate(X, ["cat"], w)
        assert g["max_share"] > _RARE_SHARE_THRESHOLD      # the share alone WOULD fire
        assert g["fires"] is expected                       # only size separates the two


def test_size_exclusion_uses_the_arena_small_threshold():
    assert _RARE_SMALL_THRESHOLD == 2500                    # config `split.small_threshold`


# ---------------------------------------------------------- the weighted-vs-row-count decision

def test_gate_is_weighted_not_row_counted():
    """brvehins1 `VehModel`'s shape: 4.2% of ROWS sub-floor but 0.7% of EXPOSURE. Must not fire.

    Sub-floor rows are short-exposure ones, so they are numerous but carry almost no weight — the
    encoder pooling them costs the model nearly nothing, which is exactly why the gate declines.
    """
    n = 20000
    X, w = _frame(n, 0.42, sub_weight=0.01, fat_weight=5.0)
    row_share = float(np.mean([str(v).startswith("rare_") for v in X["cat"]]))
    g = _rare_floor_gate(X, ["cat"], w)
    assert row_share > _RARE_SHARE_THRESHOLD                # a row-count gate would fire here
    assert g["max_share"] < 0.01                            # weighted, it is negligible
    assert g["fires"] is False


def test_the_same_row_share_at_ordinary_weights_does_fire():
    """The control for the test above: identical row share, weights no longer degenerate."""
    X, w = _frame(20000, 0.42)
    assert _rare_floor_gate(X, ["cat"], w)["fires"] is True


# ------------------------------------------------------------- the nine-dataset decision battery

@pytest.mark.parametrize("dataset", sorted(_DECISIONS))
def test_measured_battery_decision_is_reproduced(dataset):
    """Every measured split of the nine probe datasets, at both ends of its share range."""
    n, lo, hi, expected = _DECISIONS[dataset]
    for share in (lo, hi):
        X, w = _frame(n, share)
        g = _rare_floor_gate(X, ["cat"], w)
        # the builder can only place a whole number of rows, so it lands within one row of target
        assert g["max_share"] == pytest.approx(share, abs=1.0 / n), dataset
        assert g["fires"] is expected, f"{dataset} @ share={share}"


def test_battery_fires_on_exactly_the_two_pg17_datasets():
    fired = {d for d, (_n, _lo, _hi, f) in _DECISIONS.items() if f}
    assert fired == {"pg17_coverage", "pg17_payfreq"}


# ------------------------------------------------------------------------------ adapter wiring

def test_firing_slice_opens_the_floor_on_the_estimator():
    m = TBoost(hp={"budget": 20, "n_bags": 1}).fit(_bundle(3000, 0.50))
    assert m.rare_floor_gate_["fires"] is True
    assert m.rare_floor_gate_["col"] == "cat"
    assert m.model.get_params()["cat_min_data_per_group"] == _RARE_OPEN_FLOOR


def test_quiet_slice_keeps_the_library_floor():
    m = TBoost(hp={"budget": 20, "n_bags": 1}).fit(_bundle(3000, 0.05))
    assert m.rare_floor_gate_["fires"] is False
    assert m.model.get_params()["cat_min_data_per_group"] == _RARE_FLOOR


def test_small_slice_keeps_the_library_floor_however_bad_the_share():
    m = TBoost(hp={"budget": 20, "n_bags": 1}).fit(_bundle(_RARE_SMALL_THRESHOLD - 100, 0.68))
    assert m.rare_floor_gate_["fires"] is False
    assert m.model.get_params()["cat_min_data_per_group"] == _RARE_FLOOR


def test_explicit_hp_outranks_the_gate():
    """A config that names the floor gets what it asked for, gate or no gate (both directions)."""
    fires = TBoost(hp={"budget": 20, "n_bags": 1, "cat_min_data_per_group": 25.0}).fit(
        _bundle(3000, 0.50))
    assert fires.rare_floor_gate_["fires"] is True           # still measured and reported ...
    assert fires.model.get_params()["cat_min_data_per_group"] == 25.0   # ... but not applied

    quiet = TBoost(hp={"budget": 20, "n_bags": 1, "cat_min_data_per_group": 2.0}).fit(
        _bundle(3000, 0.05))
    assert quiet.model.get_params()["cat_min_data_per_group"] == 2.0


def test_gate_decision_is_deterministic_across_repeat_fits():
    b = _bundle(3000, 0.50)
    a = TBoost(hp={"budget": 20, "n_bags": 1}).fit(b).rare_floor_gate_
    c = TBoost(hp={"budget": 20, "n_bags": 1}).fit(b).rare_floor_gate_
    assert a == c


def test_purely_numeric_dataset_never_fires():
    rng = np.random.default_rng(5)
    n = 4000
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    b = DatasetBundle(name="numeric_only", X=X, y=rng.poisson(0.2, n).astype(float),
                      exposure=np.ones(n), sample_weight=np.ones(n), task=FREQ_SPEC)
    m = TBoost(hp={"budget": 20, "n_bags": 1}).fit(b)
    assert m.rare_floor_gate_["fires"] is False
    assert m.model.get_params()["cat_min_data_per_group"] == _RARE_FLOOR
