"""Full-coverage Elo ladder — THE standard way to produce cross-technique leaderboards.

The extraction protocol lives in `insurarena.results.canonical` so that the printed ladder, the
committed `results/leaderboard_cells.csv`, and `results/elo_ladder.csv` cannot drift apart.

Usage:
    .venv/bin/python scripts/elo_ladder.py                 # print the ladder
    .venv/bin/python scripts/elo_ladder.py --export results  # + rewrite the committed tables/chart
    .venv/bin/python scripts/elo_ladder.py --cells results/leaderboard_cells.csv
                                                           # ladder from the committed table alone
                                                           # (no per-cell caches needed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurarena.results.canonical import ROOT, canonical_cells, ladder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=None,
                    help="read a canonical CSV instead of re-extracting from the caches")
    ap.add_argument("--export", default=None, metavar="DIR",
                    help="write leaderboard_cells.csv, elo_ladder.csv and elo_chart.png there")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    if args.cells:
        df = pd.read_csv(args.cells)
    else:
        df = canonical_cells()

    elo = ladder(df, n_boot=args.n_boot)
    print(elo.to_string(index=False))
    print(f"\n{len(df)} canonical cells | {df.dataset.nunique()} datasets | "
          f"{df.groupby(['model', 'variant']).ngroups} competitors")
    if args.export:
        out = Path(args.export)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "leaderboard_cells.csv", index=False)
        elo.to_csv(out / "elo_ladder.csv", index=False)
        print(f"\ncells   -> {out / 'leaderboard_cells.csv'}  ({len(df)} rows)")
        print(f"ladder  -> {out / 'elo_ladder.csv'}")
        from elo_chart import elo_chart  # scripts/, so the chart can change without re-keying cells

        paths = [out / "elo_chart.png", ROOT / "runs" / "benchmark" / "elo_chart.png"]
        elo_chart(elo, [str(p) for p in paths])
        for path in paths:
            print(f"chart   -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
