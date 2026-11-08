import numpy as np
import pandas as pd
import pytest

from insurarena.results.plots import elo_bar_chart


def test_elo_chart_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    skill = {"glm/default": 0.0, "glm/tuned": 0.03, "lightgbm/default": 0.04, "lightgbm/tuned": 0.06}
    rows = []
    for d in range(5):
        for comp, s in skill.items():
            m, v = comp.split("/")
            for sp in range(3):
                rows.append(dict(dataset=f"d{d}", task="freq", model=m, variant=v, split_id=sp,
                                 skill_vs_glm=s + rng.normal(0, 0.004),
                                 primary_value=1 - s + rng.normal(0, 0.005)))
    path = elo_bar_chart(pd.DataFrame(rows), str(tmp_path / "elo.png"))
    assert path is not None
    out = tmp_path / "elo.png"
    assert out.exists() and out.stat().st_size > 0
