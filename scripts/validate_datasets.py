"""Fetch-test every manifest dataset through the generic loader. Prints OK/FAIL per dataset.

Usage:  PYTHONPATH=src python3 scripts/validate_datasets.py
"""
from __future__ import annotations

import time

import numpy as np

from insurarena.datasets.registry import benchmark_exclusion_reason, build_loader, list_datasets
from insurarena.scoring import task_targets


def main() -> int:
    # Validate retained historical loaders as well as the eligible benchmark subset.
    names = list_datasets(include_synthetic=False)
    ok = 0
    for n in names:
        t = time.time()
        try:
            b = build_loader(n, subsample=1500).load()
            yt, w, exp = task_targets(b)
            eligibility = "excluded" if benchmark_exclusion_reason(n) else "eligible"
            print(f"OK   {n:22s} {b.task.task.value:4s} X={str(b.X.shape):12s} "
                  f"exp={'Y' if b.exposure is not None else 'N'} cats={len(b.cat_features):2d} "
                  f"grp={b.n_groups:<5d} ymean={float(np.nanmean(yt)):.4g} {eligibility:8s} "
                  f"({time.time()-t:.0f}s)", flush=True)
            ok += 1
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"FAIL {n:22s} {type(e).__name__}: {str(e)[:140]}", flush=True)
    print(f"\n{ok}/{len(names)} datasets load OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
