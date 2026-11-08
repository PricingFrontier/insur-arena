"""THE canonical leaderboard extraction — one score per (model, variant, dataset, split).

Ralph, 2026-09-01: leaderboards must always be built with FULL per-model coverage, and the
published table and the Elo ladder must come from the SAME extraction so they cannot drift.
Both `scripts/elo_ladder.py` (print the ladder) and its `--export` mode (write
`results/leaderboard_cells.csv` + `results/elo_ladder.csv` + the chart) call `canonical_cells()`
here.

The protocol
------------
The board is ONE run, `runs/benchmark`, whose `cells/` holds exactly the final cell for every
(model, variant, dataset, split) and nothing else. Extraction reads every cell there and
refuses to publish unless the run is complete and
self-consistent:

  * exactly one cell per (model, variant, dataset, split_id), none failed;
  * every competitor covers every dataset with the same split count as the `glm/default`
    anchor (= 1000 Elo), which `skill_vs_glm` divides by;
  * every dataset carries ONE data generation (`source_hash`) across all of its cells, so no
    score is compared against an anchor fitted on different data.

A rerun lands in its own cache dir; it reaches the board by REPLACING the cells it supersedes in
`runs/benchmark` (superseded cells are archived under `runs/`, never pooled). Consolidated
2026-10-02: fremotor_prem and euhealth_pp from the fixed-split rerun (every model), t-boost on
the other 38 datasets from its nested rerun (engine 0.6.0), every other cell from the
2026-09-09 board rebuild. All adapters are at adapter_version 1.

Cells are computed by several campaigns, each keying its cells with its own software manifest,
so the board cannot be re-resolved by recomputing signatures from today's tree (that is what
`scripts/resolved_report.py` does, for a single live campaign). The run's manifest lists the
board's cells directly instead.

The per-cell caches are NOT in git (see `.gitignore`) — `results/leaderboard_cells.csv` is the
committed, publishable form of exactly what this module extracts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from insurarena.model_ids import current_id
from insurarena.results.elo import elo_ratings

ROOT = Path(__file__).resolve().parents[3]
BOARD = Path("runs") / "benchmark"
ANCHOR = ("glm", "default")

#: the ONE campaign config per model, `configs/<model>.yaml`
CAMPAIGN_CONFIGS = (
    ("t-boost", "configs/t-boost.yaml"),
    ("catboost", "configs/catboost.yaml"),
    ("lightgbm", "configs/lightgbm.yaml"),
    ("xgboost", "configs/xgboost.yaml"),
    ("ebm", "configs/ebm.yaml"),
    ("glm", "configs/glm.yaml"),
    ("tabpfn3", "configs/tabpfn3.yaml"),
)

#: the published column set, in order
COLUMNS = [
    "dataset", "task", "model", "variant", "split_id", "seed", "skill_vs_glm",
    "primary_metric", "primary_value", "n_train", "n_test", "source_hash",
]
KEY = ["model", "variant", "dataset", "split_id"]


def campaign_configs(root: Path = ROOT) -> list[tuple[str, str]]:
    """(model, config) pairs for the campaign configs present on disk."""
    return [(m, c) for m, c in CAMPAIGN_CONFIGS if (root / c).is_file()]


def board_cells(root: Path = ROOT) -> list[dict]:
    """The result rows of every cell in the board run.

    Reads `cells/` itself, not `active_campaign.json`: any `insurarena run` into the board's
    cache dir rewrites that manifest with only its own signatures, while the cells directory IS
    the board. Stray cells a campaign leaves behind surface as duplicate keys in
    `canonical_cells`, never as silently missing ones.
    """
    cells = root / BOARD / "cells"
    if not cells.is_dir():
        raise FileNotFoundError(
            f"no {cells} (the per-cell caches are not in git). Re-extraction is for "
            "maintainers with the board run on disk — to work from the published table instead, "
            "use `scripts/elo_ladder.py --cells results/leaderboard_cells.csv`.")
    rows = []
    for f in sorted(cells.glob("*.json")):
        payload = json.loads(f.read_text())
        # A campaign cell wraps its result row under "row"; a pre-campaign cell IS the row.
        # Tuning-evaluation payloads carry neither and are not results.
        row = payload["row"] if isinstance(payload.get("row"), dict) else payload
        if not row.get("model") or "primary_value" not in row or row["model"] == "__glm_baseline__":
            continue
        row["model"] = current_id(row["model"])   # pre-rename rows
        rows.append(row)
    return rows


def canonical_cells(root: Path = ROOT) -> pd.DataFrame:
    """The canonical per-split table: exactly `COLUMNS`, one row per (model, variant, dataset,
    split_id). Raises if the board run is incomplete or inconsistent (see module docstring)."""
    df = pd.DataFrame(board_cells(root))
    failed = df["failed"].fillna("").astype(str) != ""
    if failed.any():
        raise ValueError(f"{int(failed.sum())} failed cell(s) on the board:\n"
                         f"{df.loc[failed, KEY].to_string(index=False)}")
    dup = df.duplicated(subset=KEY, keep=False)
    if dup.any():
        raise ValueError(f"more than one cell per key:\n{df.loc[dup, KEY].to_string(index=False)}")

    gens = df.groupby("dataset").source_hash.nunique()
    if (gens > 1).any():
        raise ValueError(f"datasets span >1 data generation: {list(gens[gens > 1].index)}")

    is_anchor = (df.model == ANCHOR[0]) & (df.variant == ANCHOR[1])
    want = df[is_anchor].groupby("dataset").size()
    have = df.groupby(["model", "variant", "dataset"]).size().unstack("dataset")
    short = have.reindex(columns=want.index).fillna(0).astype(int).ne(want)
    if set(df.dataset) != set(want.index) or short.any().any():
        gaps = short.stack()
        raise ValueError("incomplete coverage vs the glm anchor:\n"
                         f"{gaps[gaps].index.to_frame(index=False).to_string(index=False)}")

    df = df[COLUMNS].sort_values(KEY).reset_index(drop=True)
    if df.isna().any().any():
        raise ValueError(f"null fields:\n{df.isna().sum()[lambda s: s > 0]}")
    return df


def ladder(df: pd.DataFrame, n_boot: int = 200) -> pd.DataFrame:
    """The Elo ladder from a canonical frame (glm/default anchored at 1000)."""
    cols = ["model", "variant", "dataset", "task", "skill_vs_glm"]
    return elo_ratings(df[cols], n_boot=n_boot).sort_values("elo", ascending=False)
