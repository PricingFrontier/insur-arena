import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "elo_chart.py"


def _load():
    spec = importlib.util.spec_from_file_location("elo_chart", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_elo_chart_renders_from_the_ladder(tmp_path):
    pytest.importorskip("matplotlib")
    ladder = pd.DataFrame(
        [
            ("t-boost/tuned", 1446, 1352, 1561, 40),
            ("t-boost/default", 1382, 1288, 1489, 40),
            ("lightgbm/tuned", 1384, 1298, 1488, 40),
            ("tabpfn3/default", 1310, 1151, 1471, 38),  # partial coverage gets the extra key
            ("glm/default", 1000, 1000, 1000, 40),
        ],
        columns=["competitor", "elo", "elo_lo", "elo_hi", "n_datasets"],
    )
    out = [tmp_path / "elo.png", tmp_path / "elo.pdf"]
    _load().elo_chart(ladder, [str(p) for p in out])
    assert all(p.exists() and p.stat().st_size > 0 for p in out)
