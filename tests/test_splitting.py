import numpy as np
import pandas as pd
import pytest

from insurarena.datasets.base import DatasetBundle
from insurarena.eval.splitting import outer_splits
from insurarena.tasks import FREQ_SPEC


def _bundle_with_groups(n=600, g=120, seed=0):
    rng = np.random.default_rng(seed)
    gid = rng.integers(0, g, n)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.poisson(0.3, n).astype(float)
    exp = rng.uniform(0.2, 1, n)
    return DatasetBundle(
        "g", X, y, exp, exp, FREQ_SPEC, group_ids=gid,
        cat_features=[], meta={"stratify_on": "claim_occurrence"},
    )


def test_group_no_leak_across_splits():
    b = _bundle_with_groups()
    cfg = {"small_threshold": 100000, "small": {"folds": 3, "repeats": 2}, "large": {"folds": 3, "repeats": 1}}
    splits = outer_splits(b, cfg, seed=42)
    assert len(splits) == 6
    for tr, te in splits:
        assert set(b.group_ids[tr]).isdisjoint(set(b.group_ids[te]))


def test_grouped_unstratified_repeats_give_distinct_test_sets():
    b = _bundle_with_groups()
    b.meta = {}  # no stratify label: the GroupKFold branch
    cfg = {"small_threshold": 100000, "small": {"folds": 3, "repeats": 3}, "large": {"folds": 3, "repeats": 1}}
    splits = outer_splits(b, cfg, seed=42)
    assert len(splits) == 9
    tests = {tuple(np.sort(te)) for _, te in splits}
    assert len(tests) == 9
    for tr, te in splits:
        assert set(b.group_ids[tr]).isdisjoint(set(b.group_ids[te]))


def test_exposure_as_feature_is_rejected():
    n = 60
    rng = np.random.default_rng(0)
    exp = rng.uniform(0.2, 1, n)
    X = pd.DataFrame({"Exposure": exp, "a": rng.normal(size=n)})  # illegal: exposure is scale
    with pytest.raises(ValueError):
        DatasetBundle("bad", X, rng.poisson(0.3, n).astype(float), exp, exp, FREQ_SPEC)
