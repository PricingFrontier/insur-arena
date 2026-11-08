import json

import numpy as np

from insurarena.config import Config
from insurarena.eval.runner import run_benchmark
from insurarena.results.store import ResultStore


def _cfg(tmp_path):
    return Config(
        threads_per_worker=1,
        datasets=["synthetic_freq"],
        models=[{"name": "glm", "variants": ["default", "tuned"]}],
        subsample_rows=1500, n_configs=2, inner_folds=2,
        split={"small_threshold": 100000, "small": {"folds": 3, "repeats": 1}, "large": {"folds": 3, "repeats": 1}},
        cache_dir=str(tmp_path), store_predictions=False, use_cache=True,
    )


def test_cache_reuse_is_consistent(tmp_path):
    cfg = _cfg(tmp_path)
    df1 = run_benchmark(cfg, store=ResultStore(str(tmp_path), False), verbose=False)
    cells = list((tmp_path / "cells").glob("*.json"))
    assert len(cells) >= 6                              # 3 splits x {default,tuned} + baselines cached
    df2 = run_benchmark(cfg, store=ResultStore(str(tmp_path), False), verbose=False)
    assert len(df1) == len(df2)
    a = df1.sort_values(["model", "variant", "split_id"]).reset_index(drop=True)
    b = df2.sort_values(["model", "variant", "split_id"]).reset_index(drop=True)
    assert np.allclose(a["skill_vs_glm"], b["skill_vs_glm"], equal_nan=True)


def test_new_subsample_invalidates(tmp_path):
    df1 = run_benchmark(_cfg(tmp_path), store=ResultStore(str(tmp_path), False), verbose=False)
    n1 = len(list((tmp_path / "cells").glob("*.json")))
    cfg2 = _cfg(tmp_path)
    cfg2.subsample_rows = 1200                          # different data -> different content hash -> new cells
    run_benchmark(cfg2, store=ResultStore(str(tmp_path), False), verbose=False)
    n2 = len(list((tmp_path / "cells").glob("*.json")))
    assert n2 > n1                                      # new cells added, old retained


def _cached_hpo_config_indices(tmp_path):
    idx = []
    for path in (tmp_path / "cells").glob("*.json"):
        d = json.loads(path.read_text())
        if d.get("kind") == "tuned_hpo_config":
            idx.append(int(d["config_idx"]))
    return sorted(idx)


def test_hpo_config_cache_appends_when_budget_grows(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.n_configs = 1
    run_benchmark(cfg, store=ResultStore(str(tmp_path), False), verbose=False)
    first = _cached_hpo_config_indices(tmp_path)
    assert first == [0, 0, 0, 1, 1, 1]                              # default + one candidate independently on each of 3 outer splits

    cfg2 = _cfg(tmp_path)
    cfg2.n_configs = 3
    run_benchmark(cfg2, store=ResultStore(str(tmp_path), False), verbose=False)
    second = _cached_hpo_config_indices(tmp_path)
    assert second == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]                       # each split keeps its prefix and appends its own tail
