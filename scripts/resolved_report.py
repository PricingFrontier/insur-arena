"""Signature-resolved report: the leaderboard from CURRENT-generation cells only.

`insurarena report --cache-dir` (and any glob over `cells/`) pools every generation of every
cell ever cached — after adapter/protocol bumps that mixes stale fits into the means (measured
2026-08-04: 1,580 raw t-boost-tuned cell files vs ~465 logical cells; brvehins1_pp's pooled
tuned mean read −0.054 while its current cells say +0.005). This script recomputes each cell's
CURRENT signature exactly as the runner would (same `campaign.partition_protocol`, same
adapter/hpo versions from the
live registry) and aggregates only cells that exist under it, reporting everything else as a
COVERAGE GAP instead of silently substituting old results.

Usage:
    .venv/bin/python scripts/resolved_report.py [--config configs/t-boost.yaml]
        [--models t-boost,catboost,ebm,lightgbm,xgboost]
        [--variants default,tuned] [--csv out.csv]

The config supplies the dataset list AND the protocol fields that key tuned cells (split
schedule, n_configs, inner_folds, racing, selection_max_rows) — pass the config the campaign
actually ran. Models/variants beyond it are fine: absent cells just show as gaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurarena.config import Config
from insurarena.datasets.registry import build_loader
from insurarena.eval.campaign import digest_rows, partition_protocol, software_manifest
from insurarena.eval.splitting import outer_splits
from insurarena.model_ids import current_id
from insurarena.models.registry import model_class
from insurarena.results.cache import cell_signature
from insurarena.results.elo import elo_ratings, format_elo
from insurarena.results.leaderboard import format_table, leaderboard
from insurarena.seeds import seed_for


SPELLINGS = ("configured", "resolved")


def _manifest(cache_dir: str, split_key: str, spelling: str) -> str:
    """The `_split_manifest` string as the campaign would have written it.

    It is part of the cell key, so it must match character for character — see the probe in
    `main`. `configured` is the path the config spells; `resolved` is that path made absolute,
    which is what the campaign cells on disk actually carry.
    """
    base = Path(cache_dir).expanduser()
    if spelling == "resolved":
        base = base.resolve()
    return str(base / "splits" / f"{split_key}.npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/t-boost.yaml")
    ap.add_argument("--models",
                    default="t-boost,catboost,ebm,lightgbm,xgboost,glm")
    ap.add_argument("--variants", default="default,tuned")
    ap.add_argument("--csv", default=None, help="also write the per-split frame here")
    ap.add_argument("--software", choices=("campaign", "live"), default="campaign",
                    help="whose software manifest keys the cells: the one active_campaign.json "
                         "recorded (default), or one recomputed from the tree as it is now")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    cells_dir = Path(cfg.cache_dir).expanduser() / "cells"
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    # Campaign cells are keyed by `campaign.partition_protocol`, whose fingerprint INCLUDES the
    # software manifest (digest of every harness .py + package versions). Recomputing it here
    # keys the lookup to the tree as it stands NOW, so any edit landed after the campaign — even
    # to a module the run never imported — silently misses every cell it wrote. The campaign
    # recorded the manifest it actually ran under; key with that, and say so when they diverge.
    live = software_manifest()
    software = live
    manifest_path = Path(cfg.cache_dir).expanduser() / "active_campaign.json"
    if args.software == "campaign":
        if not manifest_path.exists():
            print(f"no {manifest_path}; keying with the live tree instead", file=sys.stderr)
        elif json.loads(manifest_path.read_text()).get("software") is None:
            # the consolidated board (runs/benchmark): cells from several campaigns, no single
            # manifest — use `insurarena.results.canonical` for it, not this script
            print(f"{manifest_path} records no single software manifest (a consolidated run); "
                  "keying with the live tree instead", file=sys.stderr)
        else:
            software = json.loads(manifest_path.read_text())["software"]
            if software != live:
                drift = sorted(k for k in set(software) | set(live)
                               if software.get(k) != live.get(k))
                print(f"note: tree has moved since the campaign ({', '.join(drift)} differ); "
                      "keying cells with the RECORDED manifest", file=sys.stderr)

    rows, gaps = [], []
    chosen = None          # split-manifest spelling, fixed by the first cell that resolves
    for ds in cfg.datasets:
        try:
            bundle = build_loader(ds, subsample=cfg.subsample_rows, cache_dir=cfg.cache_dir).load()
        except Exception as e:  # noqa: BLE001
            gaps.append((ds, "*", "*", f"load failed: {e}"))
            continue
        h = str(bundle.meta.get("hash", ""))
        splits = outer_splits(bundle, cfg.split, seed_for(ds))
        n_splits = len(splits)
        # `partition_protocol` reads `cfg._split_manifest` — the path of the split archive the
        # campaign wrote (campaign.py). The name is a pure function of the bundle and the splits,
        # so recompute the path; the archive itself is not needed and nothing is written here.
        split_key = hashlib.sha256(json.dumps(
            {"data": bundle.meta.get("hash"),
             "rows": [(digest_rows(tr), digest_rows(te)) for tr, te in splits]},
            sort_keys=True).encode()).hexdigest()
        # ...but only its NAME is determined; its SPELLING is not. The campaign stored whatever
        # string its own cfg.cache_dir produced, and cells record an absolute path while the
        # configs on disk say `runs/benchmark`. Nothing in Config or the CLI absolutizes it, so
        # the spelling that ran is not recoverable from the config — probe the ones we know and
        # lock onto whichever actually resolves a cell.
        for name in models:
            cls = model_class(name)
            if bundle.task.task not in cls.supported_tasks:
                continue
            # the runner keys cells with the config's model-entry hp override (runner.py:
            # _proto(cfg, bundle, name, variant, entry.get("hp", {}))); resolving with {} here
            # silently matched STALE cells for any config that carries an hp block (found via
            # the b160/ffull small-set campaigns). Mirror the runner exactly.
            hp_override = next((e.get("hp", {}) for e in getattr(cfg, "models", [])
                                if e.get("name") == name), {})
            for variant in variants:
                if name == "glm" and variant == "tuned":
                    continue
                found = 0
                for sid in range(n_splits):
                    # Under `nested-evidence-v1` the tuning runs inside each outer split, so the
                    # protocol — `tune_split_id` with it — varies PER SPLIT. A single `_proto`
                    # hoisted out of this loop, as this script did before, could only ever match
                    # split 0, and not even that once the campaign's own keys are folded in.
                    p = None
                    for spelling in ([chosen] if chosen else SPELLINGS):
                        cfg._split_manifest = _manifest(cfg.cache_dir, split_key, spelling)
                        _, proto = partition_protocol(cfg, bundle, name, variant, hp_override,
                                                      sid, splits, software)
                        q = cells_dir / f"{cell_signature(h, name, variant, sid, proto)}.json"
                        if q.exists():
                            if chosen is None:
                                chosen = spelling
                                print(f"split-manifest spelling: {spelling}", file=sys.stderr)
                            p = q
                            break
                    if p is None:
                        continue
                    payload = json.loads(p.read_text())
                    # A campaign cell wraps its result row; a pre-campaign cell IS the row.
                    d = payload["row"] if isinstance(payload.get("row"), dict) else payload
                    d["model"] = current_id(d.get("model", name))   # pre-rename rows
                    d.pop("trajectory", None)
                    sec = d.pop("secondary", None)
                    if isinstance(sec, dict):
                        d.update({f"sec_{k}": v for k, v in sec.items()})
                    rows.append(d)
                    found += 1
                if found < n_splits:
                    gaps.append((ds, name, variant, f"{found}/{n_splits} cells at current signature"))

    df = pd.DataFrame(rows)
    if df.empty:
        print("no current-generation cells found — check --config matches the campaign that ran")
        return 1
    if "failed" in df.columns:
        df = df[(df["failed"].isna()) | (df["failed"] == "")]
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"per-split frame -> {args.csv}  ({len(df)} cells)")

    print("\n=== LEADERBOARD (current-generation cells only; skill-vs-GLM) ===")
    print(format_table(leaderboard(df)))
    print("\n=== ELO LADDER (current-generation cells only) ===")
    print(format_elo(elo_ratings(df)))
    print("\n=== COVERAGE GAPS (no/partial cells at the CURRENT protocol — rerun, don't guess) ===")
    if not gaps:
        print("(none — full coverage)")
    for ds, name, variant, note in gaps:
        print(f"  {ds:22s} {name:22s} {variant:8s} {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
