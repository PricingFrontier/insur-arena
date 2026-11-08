"""Write results/leaderboard_cells_detail.csv: secondary metrics and timings for every board cell.

One row per (model, variant, dataset, split_id), in the same order as
`results/leaderboard_cells.csv`. Like that table it is extracted from the per-cell caches, which
are not in git, so this is for maintainers with the board run on disk:

    python scripts/export_cell_details.py                # board in this checkout
    python scripts/export_cell_details.py --root <path>  # board in another checkout

It lives in scripts/ because any edit under src/insurarena re-keys every cell on the board.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurarena.results.canonical import KEY, ROOT, board_cells  # noqa: E402

ID_COLUMNS = ["dataset", "task", "model", "variant", "split_id"]
TIMING_COLUMNS = ["train_time_s", "inference_time_s", "tuning_time_s", "tuning_cache_hit"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="checkout holding runs/benchmark")
    args = ap.parse_args()

    cells = pd.DataFrame(board_cells(args.root)).sort_values(KEY).reset_index(drop=True)
    published = pd.read_csv(ROOT / "results" / "leaderboard_cells.csv")
    if not cells[KEY].equals(published[KEY]) or not np.allclose(
            cells.primary_value, published.primary_value, rtol=1e-12, atol=0):
        raise ValueError("the board cells do not match results/leaderboard_cells.csv")

    metrics = sorted(c for c in cells.columns if c.startswith("sec_"))
    out = ROOT / "results" / "leaderboard_cells_detail.csv"
    cells[ID_COLUMNS + metrics + TIMING_COLUMNS].to_csv(out, index=False)
    print(f"{len(cells)} rows, {len(metrics)} metrics -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
