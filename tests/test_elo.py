import numpy as np
import pandas as pd

from insurarena.results.elo import elo_ratings


def test_elo_anchor_and_order():
    rng = np.random.default_rng(0)
    strengths = {"glm/default": 0.0, "glm/tuned": 0.02, "lightgbm/default": 0.05}
    rows = []
    for ds in range(10):
        for comp, s in strengths.items():
            model, variant = comp.split("/")
            rows.append(dict(dataset=f"d{ds}", task="freq", model=model, variant=variant,
                             skill_vs_glm=s + rng.normal(0, 0.004)))
    elo = elo_ratings(pd.DataFrame(rows), n_boot=25)
    assert len(elo) == 3
    # glm/default is the anchor at exactly 1000
    assert int(elo.loc[elo.competitor == "glm/default", "elo"].iloc[0]) == 1000
    # the strongest competitor tops the ladder and outranks the anchor
    assert elo.iloc[0]["competitor"] == "lightgbm/default"
    assert elo.iloc[0]["elo"] > 1000
