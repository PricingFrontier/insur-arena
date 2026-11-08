import json
from pathlib import Path

import numpy as np
import pytest

from insurarena.config import Config
from insurarena.datasets.registry import build_loader
from insurarena.eval import campaign, runner
from insurarena.eval.splitting import outer_splits
from insurarena.results.cache import ResultCache
from insurarena.seeds import seed_for


def cfg(tmp_path, **kw):
    return Config(datasets=['synthetic_freq'],
                  models=[{'name': 'glm', 'variants': ['default', 'tuned']}],
                  subsample_rows=180, n_configs=1, inner_folds=2, inner_repeats=1,
                  split={'small_threshold': 10000, 'small': {'folds': 3, 'repeats': 1},
                         'large': {'folds': 3, 'repeats': 1}},
                  cache_dir=str(tmp_path), threads_per_worker=1, **kw)


def test_each_search_is_confined_to_its_outer_train(tmp_path, monkeypatch):
    conf = cfg(tmp_path)
    bundle = build_loader('synthetic_freq', subsample=180, cache_dir=str(tmp_path)).load()
    splits = outer_splits(bundle, conf.split, seed_for('synthetic_freq'))
    seen = []
    def select(name, train, *args, **kwargs):
        seen.append((set(train.X.index), kwargs))
        assert kwargs['extra_finale_otrs'] is None
        return {}, 0, [1.0], [], {}
    monkeypatch.setattr(runner, 'select_config', select)
    # Force the branch that historically pooled other outer training partitions.
    monkeypatch.setattr(runner, '_model_selection_finale_hp', lambda _: {'n_bags': 8})
    monkeypatch.setattr(runner, '_model_selection_hp', lambda _: {'n_bags': 1})
    cache = ResultCache(str(tmp_path), enabled=False)
    for sid, (tr, te) in enumerate(splits):
        scoped, proto = campaign.partition_protocol(conf, bundle, 'glm', 'tuned', {}, sid, splits, {})
        runner._tuned_once(cache, scoped, bundle, splits, bundle.name, str(sid), 'glm', {})
        assert seen[-1][0] == set(bundle.X.iloc[tr].index)
        assert not seen[-1][0] & set(bundle.X.iloc[te].index)
        assert proto['tune_split_id'] == sid
    assert len(seen) == len(splits)


@pytest.mark.parametrize('workers', [1, 2])
def test_campaign_retains_predictions_and_fold_specific_search(tmp_path, workers):
    frame = runner.run_benchmark(cfg(tmp_path, max_workers=workers), verbose=False)
    assert len(frame) == 6
    assert not frame.failed.any(), frame.failed.tolist()
    assert all(Path(p).exists() for paths in frame.evidence_paths for p in paths.values())
    dumps = list((tmp_path / 'evidence').glob('*.npz'))
    assert len(dumps) == 6
    for p in dumps:
        with np.load(p, allow_pickle=False) as data:
            assert 'weights' in data
            assert len(data['predictions']) == 60
    selected = [json.loads(p.read_text()) for p in (tmp_path/'cells').glob('*.json')]
    assert len([d for d in selected if 'selection' in d]) == 3


def test_timing_pass_is_fresh_and_serial(tmp_path):
    conf = cfg(tmp_path, evaluation_mode='timing', timing_repeats=2, timing_batch_size=16)
    conf.models = [{'name': 'glm', 'variants': ['default']}]
    frame = runner.run_benchmark(conf, verbose=False)
    assert not frame.failed.any(), frame.failed.tolist()
    for paths in frame.evidence_paths:
        metadata = json.loads(Path(next(v for v in paths.values() if str(v).endswith('.json'))).read_text())
        assert 'serving_timing' in metadata
    conf.max_workers = 2
    with pytest.raises(ValueError, match='max_workers=1'):
        runner.run_benchmark(conf, verbose=False)


def test_cached_evidence_missing_forces_recompute(tmp_path):
    conf = cfg(tmp_path)
    conf.models = [{'name': 'glm', 'variants': ['default']}]
    first = runner.run_benchmark(conf, verbose=False)
    path = next((tmp_path/'evidence').glob('*.npz'))
    path.unlink()
    second = runner.run_benchmark(conf, verbose=False)
    assert path.exists()
    np.testing.assert_allclose(first.primary_value, second.primary_value)


def test_xgboost_early_stopping_uses_validation_weights(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from insurarena.models.xgb import XGBoost
    from insurarena.scoring import task_targets
    calls = []
    class Booster:
        def __init__(self, **kw):
            pass
        def fit(self, X, y, **kw):
            calls.append(kw)
            self.best_iteration = 1
    monkeypatch.setitem(sys.modules, 'xgboost', SimpleNamespace(XGBRegressor=Booster))
    bundle = build_loader('synthetic_freq', subsample=180, cache_dir=str(tmp_path)).load()
    train, val = bundle.subset(np.arange(100)), bundle.subset(np.arange(100, 180))
    XGBoost({'n_estimators': 2}).fit(train, val)
    from insurarena.models.base import filter_tiny_exposure_fit_rows
    _, expected, _ = task_targets(filter_tiny_exposure_fit_rows(val))
    np.testing.assert_array_equal(calls[0]['sample_weight_eval_set'][0], expected)


def test_hygiene_is_the_boards():
    # The board's data (bundles cached 2026-08-21) was built by this hygiene: flag only columns
    # that have a missing value, then drop constant and duplicate columns. A cold load must
    # reproduce it, or a rerun silently scores different data (see _hygiene's docstring).
    import pandas as pd
    from insurarena.datasets.spec import _hygiene
    X = pd.DataFrame({'a': [1., np.nan, 3.], 'b': ['x', None, 'y'], 'c': [5., 5., 5.], 'd': [1., 2., 3.]})
    out = _hygiene(X)
    # b__isna duplicates a__isna and is dropped; c is constant; d has no missing value, so no flag.
    assert list(out.columns) == ['a', 'b', 'd', 'a__isna']
    np.testing.assert_array_equal(out['a__isna'], [0., 1., 0.])


def test_partial_report_reads_only_active_campaign(tmp_path):
    from insurarena.cli import _load_cached_cells
    conf = cfg(tmp_path)
    conf.models = [{'name': 'glm', 'variants': ['default']}]
    runner.run_benchmark(conf, verbose=False)
    assert len(_load_cached_cells(str(tmp_path))) == 3
    conf.models = [{'name': 'glm', 'variants': ['tuned']}]
    runner.run_benchmark(conf, verbose=False)
    frame = _load_cached_cells(str(tmp_path))
    assert len(frame) == 3 and set(frame.variant) == {'tuned'}
