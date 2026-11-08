import numpy as np
import pandas as pd

from insurarena.results.stats import critical_difference, pairwise_winrate, per_dataset_significance


def _df():
    rng = np.random.default_rng(0)
    skill = {"glm/default": 0.0, "lightgbm/default": 0.05, "ebm/default": 0.03}
    prim = {"glm/default": 1.0, "lightgbm/default": 0.95, "ebm/default": 0.97}
    rows = []
    for d in range(6):
        for comp in skill:
            m, v = comp.split("/")
            for s in range(3):
                rows.append(dict(dataset=f"d{d}", task="freq", model=m, variant=v, split_id=s,
                                 skill_vs_glm=skill[comp] + rng.normal(0, 0.004),
                                 primary_value=prim[comp] + rng.normal(0, 0.008)))
    return pd.DataFrame(rows)


def test_critical_difference():
    res = critical_difference(_df())
    assert not res["table"].empty
    assert res["table"].iloc[0]["competitor"] == "lightgbm/default"   # best mean rank
    assert res["cd"] > 0


def test_pairwise_winrate():
    wr = pairwise_winrate(_df())
    assert wr.loc["lightgbm/default", "glm/default"] > 0.5


def test_per_dataset_significance():
    sig = per_dataset_significance(_df())
    assert "not_sig_worse_than_best" in sig.columns
    assert int(sig["is_best"].sum()) == 6                            # one best per dataset
