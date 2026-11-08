"""Guards on the published deliverable: results/leaderboard_cells.csv.

Offline — reads the committed table only. These are the gates the export must keep passing:
one row per (model, variant, dataset, split_id), full coverage, no nulls, and a ladder that
still anchors glm/default at 1000.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from insurarena.results.canonical import COLUMNS, ladder

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "results" / "leaderboard_cells.csv"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not CELLS.exists():
        pytest.skip("results/leaderboard_cells.csv not present")
    return pd.read_csv(CELLS)


def test_schema_is_the_published_column_set(df):
    assert list(df.columns) == COLUMNS
    assert df.isna().sum().sum() == 0


def test_one_row_per_model_variant_dataset_split(df):
    key = ["model", "variant", "dataset", "split_id"]
    assert not df.duplicated(subset=key).any()


def test_every_competitor_has_full_catalog_coverage(df):
    per = df.groupby(["model", "variant"]).agg(rows=("skill_vs_glm", "size"),
                                               datasets=("dataset", "nunique"))
    n_datasets = df["dataset"].nunique()
    # 35 datasets x 9 outer splits + 5 small datasets x 30 = 465 rows per competitor
    assert (per["datasets"] == n_datasets).all(), per[per["datasets"] != n_datasets]
    assert (per["rows"] == per["rows"].iloc[0]).all(), per


def test_glm_is_the_anchor_and_has_no_tuned_variant(df):
    glm = df[df["model"] == "glm"]
    assert set(glm["variant"]) == {"default"}
    # skill is 1 - D_glm/D_glm_baseline: zero up to refit nondeterminism
    assert glm["skill_vs_glm"].abs().max() < 1e-3


def test_ladder_is_anchored_and_covers_every_competitor(df):
    """The ladder's STRUCTURE, not its exact ordering.

    This test used to pin `lightgbm/tuned > t-boost/tuned > ebm/tuned > glm/default`. The first
    of those pairs is not a fact about the models — at the av44 board the two sat 2 Elo apart
    (1451 vs 1449) with overlapping bootstrap intervals, i.e. a statistical tie that the ±0.002
    re-draw churn of any recompute can flip either way. A test that has to be edited whenever a
    coin lands the other way is not testing anything; worse, it invites editing the BOARD to keep
    the test green. So the pins here are the ones the extraction is actually responsible for: the
    anchor, full coverage, and the two orderings that would indicate a broken pipeline rather than
    a close race.
    """
    lad = ladder(df, n_boot=0)
    elo = lad.set_index("competitor")["elo"]

    assert elo["glm/default"] == 1000                     # the anchor, by construction
    assert (lad["n_datasets"] == df["dataset"].nunique()).all(), lad

    # every model on the board appears, with the variants it actually has
    competitors = set(elo.index)
    assert {f"{m}/tuned" for m in
            ("t-boost", "lightgbm", "xgboost", "catboost", "ebm")
            } <= competitors, competitors
    assert "glm/tuned" not in competitors                 # the anchor has no tuned variant

    # a tuned GBM beating the GLM anchor and its own default is the benchmark working at all; a
    # failure here is a pipeline fault, not a close race.
    for m in ("t-boost", "lightgbm", "catboost", "xgboost"):
        assert elo[f"{m}/tuned"] > elo["glm/default"], m
    assert elo["t-boost/tuned"] > elo["t-boost/default"]

    # ...and the leaders are a pack, so assert membership rather than an order within it.
    leaders = set(lad.head(5)["competitor"])
    assert {"t-boost/tuned", "lightgbm/tuned"} <= leaders, lad.head(6)
