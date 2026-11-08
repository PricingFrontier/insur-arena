"""Measure how large every t-boost effect of order >= 3 would be as ONE dense table.

For each (dataset, variant), refit the deployed t-boost cell with the most boxes (same split, seed,
chosen parameters and best iteration as its evidence) and, for every high-order effect, compute
the size of that effect as a single table on its own grid: along each axis, cells that every box
puts on the same side are merged, so the grid is the coarsest one the effect's boxes are exact on.

Usage:
    python scripts/tboost_table_sizes.py OUT.csv [--workers 4] [--datasets a,b,...]

Diagnostic only: reads the board's evidence and refits in-process; writes nothing to runs/.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIELDS = ["dataset", "variant", "split_id", "class", "features", "order", "boxes", "local_shape",
          "local_cells", "global_cells"]


def heaviest_cells(only: set[str] | None) -> list[dict]:
    best: dict[tuple, dict] = {}
    for path in glob.glob(str(ROOT / "runs/benchmark/evidence/*.json")):
        d = json.loads(Path(path).read_text())
        if d.get("model") not in ("t-boost", "triboost") or d.get("failed"):
            continue
        if only and d["dataset"] not in only:
            continue
        boxes = ((d.get("triboost") or {}).get("box_counts") or {}).get("factored_boxes") or 0
        key = (d["dataset"], d["variant"])
        if key not in best or boxes > best[key]["boxes"]:
            best[key] = dict(dataset=d["dataset"], variant=d["variant"], split_id=d["split_id"],
                             seed=d["seed"], hp=d.get("chosen_parameters") or {},
                             best_iter=d.get("best_iteration") or 0, boxes=boxes)
    return sorted(best.values(), key=lambda c: -c["boxes"])


def local_cells(masks: list[list[bool]]) -> int:
    """Cells on one axis that no box distinguishes collapse into one."""
    if not masks:
        return 1
    signatures = {tuple(m[c] for m in masks) for c in range(len(masks[0]))}
    return len(signatures)


def measure(cell: dict) -> list[dict]:
    from insurarena.config import Config
    from insurarena.datasets.registry import build_loader
    from insurarena.eval.hpo import fit_one
    from insurarena.eval.splitting import outer_splits
    from insurarena.seeds import seed_for

    cfg = Config.from_yaml(str(ROOT / "configs/t-boost.yaml"))
    bundle = build_loader(cell["dataset"], subsample=cfg.subsample_rows, cache_dir=cfg.cache_dir).load()
    splits = outer_splits(bundle, cfg.split, seed_for(cell["dataset"]))
    tr, _ = splits[cell["split_id"]]
    model = fit_one("t-boost", cell["hp"], bundle.subset(tr), cell["seed"],
                    n_estimators=int(cell["best_iter"]) or None)
    doc = json.loads(model.model._model.to_json())["model"]
    banks = [c["bank"] for c in doc["classes"]] if "classes" in doc else [doc["bank"]]
    rows = []
    for k, bank in enumerate(banks):
        for eff in bank.get("factored") or []:
            order = len(eff["u"])
            shape = [local_cells([b["low"][d] for b in eff["boxes"]]) for d in range(order)]
            glob_shape = [len(eff["boxes"][0]["low"][d]) if eff["boxes"] else 0 for d in range(order)]
            rows.append(dict(dataset=cell["dataset"], variant=cell["variant"], split_id=cell["split_id"],
                             **{"class": k}, features="x".join(str(u) for u in eff["u"]), order=order,
                             boxes=len(eff["boxes"]), local_shape="x".join(map(str, shape)),
                             local_cells=int(np.prod(shape)), global_cells=int(np.prod(glob_shape))))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", default="")
    args = ap.parse_args()
    cells = heaviest_cells(set(filter(None, args.datasets.split(","))) or None)
    print(f"{len(cells)} cells to refit", flush=True)
    with open(args.out, "w", newline="") as fh, ProcessPoolExecutor(args.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        futures = {pool.submit(measure, c): c for c in cells}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:  # noqa: BLE001 - one failed refit must not stop the survey
                print(f"FAILED {c['dataset']}/{c['variant']}: {type(exc).__name__}: {exc}", flush=True)
                continue
            w.writerows(rows)
            fh.flush()
            big = max((r["local_cells"] for r in rows), default=0)
            print(f"done {c['dataset']:22s} {c['variant']:8s} split {c['split_id']}: "
                  f"{len(rows)} high-order effects, largest table {big:,} cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
