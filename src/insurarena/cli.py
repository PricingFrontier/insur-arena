"""CLI: `insurarena run --config <cfg>` and `insurarena report --cache-dir <dir>` (partial results)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from insurarena.config import Config
from insurarena.datasets.registry import benchmark_exclusion_reason
from insurarena.model_ids import current_id
from insurarena.eval.runner import run_benchmark
from insurarena.results.elo import elo_ratings, format_elo
from insurarena.results.leaderboard import format_table, leaderboard
from insurarena.results.stats import (
    budget_curve,
    critical_difference,
    format_budget_curve,
    format_cd,
    format_winrate,
    pairwise_winrate,
)
from insurarena.results.store import ResultStore


def _report(df: pd.DataFrame, out_dir: Path, trajectories: list | None = None) -> None:
    ok = df[df.get("failed", "") == ""] if "failed" in df.columns else df
    print("\n=== LEADERBOARD (accuracy axis: skill-vs-GLM) ===")
    print(format_table(leaderboard(ok)))
    print("\n=== ELO LADDER (cross-technique; glm/default = 1000) ===")
    print(format_elo(elo_ratings(ok)))
    print("\n=== CRITICAL DIFFERENCE (Friedman + Nemenyi, avg rank) ===")
    print(format_cd(critical_difference(ok)))
    print("\n=== PAIRWISE WIN-RATE (P(row beats col)) ===")
    print(format_winrate(pairwise_winrate(ok)))
    if trajectories:
        print("\n=== TUNING BUDGET CURVE (OOF primary vs #configs; lower better) ===")
        print(format_budget_curve(budget_curve(trajectories)))
    from insurarena.results.plots import elo_bar_chart

    chart = elo_bar_chart(ok, str(out_dir / "elo_chart.png"))
    if "failed" in df.columns:
        n_fail = int((df["failed"] != "").sum())
        if n_fail:
            print(f"\n({n_fail} failed cells recorded, excluded from the leaderboard)")
    if chart:
        print(f"chart   -> {chart}")


def _load_cached_cells(cache_dir: str) -> pd.DataFrame:
    """Assemble a results frame from the per-cell cache (for partial reports mid-run)."""
    root = Path(os.path.expanduser(cache_dir))
    manifest = root / "active_campaign.json"
    if manifest.exists():
        active = json.loads(manifest.read_text())
        files = [root / "campaign_records" / f"{sig}.json" for sig in active['signatures']]
    else:
        files = (root / 'cells').glob('*.json')
    rows = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("model") and d.get("variant") and "primary_value" in d and d.get("model") != "__glm_baseline__":
            d["model"] = current_id(d["model"])   # pre-rename rows
            d.pop("trajectory", None)
            rows.append(d)
    return pd.DataFrame(rows)


def _filter_ineligible_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if df.empty or "dataset" not in df.columns:
        return df, []
    excluded = sorted({
        str(name) for name in df["dataset"].unique()
        if benchmark_exclusion_reason(str(name)) is not None
    })
    if not excluded:
        return df, []
    return df[~df["dataset"].isin(excluded)].copy(), excluded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="insurarena")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the benchmark from a config")
    run.add_argument("--config", required=True)
    rep = sub.add_parser("report", help="build leaderboard/Elo/chart from cached cells (partial OK)")
    rep.add_argument("--cache-dir", required=True)
    rep.add_argument(
        "--include-ineligible-datasets",
        action="store_true",
        help="include undersized datasets when reproducing historical reports",
    )
    args = parser.parse_args(argv)

    if args.cmd == "run":
        cfg = Config.from_yaml(args.config)
        store = ResultStore(cfg.cache_dir, cfg.store_predictions)
        df = run_benchmark(cfg, store=store)
        out = store.save()
        _report(df, out.parent, store.trajectories)
        print(f"\nresults -> {out}")
        return 0

    if args.cmd == "report":
        df = _load_cached_cells(args.cache_dir)
        if df.empty:
            print("(no cached cells found)")
            return 1
        if not args.include_ineligible_datasets:
            df, excluded = _filter_ineligible_datasets(df)
            if excluded:
                print(f"excluded cached results for ineligible datasets: {', '.join(excluded)}")
            if df.empty:
                print("(no eligible cached cells found)")
                return 1
        print(f"loaded {len(df)} cached result cells across "
              f"{df['dataset'].nunique()} datasets, {df['model'].nunique()} techniques")
        _report(df, Path(os.path.expanduser(args.cache_dir)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
