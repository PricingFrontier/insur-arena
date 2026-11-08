"""Process-pool workers for cell-level parallelism (model x variant x split).

Only the heavy compute (fit -> predict -> score for ONE cell) runs in a worker; the main process keeps
all bookkeeping (cache signatures, skill-vs-GLM, ResultRow assembly) so results are identical to the
sequential path. Workers are SPAWNED, not forked: by the time the pool is created the coordinator may
already have background native threads alive (BLAS, a Rust rayon pool inside a model library, ...), and
fork()-ing a multi-threaded process only clones the calling thread — any mutex one of the other threads
held mid-critical-section is inherited already-locked, with no thread left alive to release it. The
first time a worker touches that library it deadlocks forever. Spawned children start a brand-new
interpreter with none of that inherited state, so the loaded dataset bundle + split schedule are passed
explicitly through the pool initializer instead of relying on copy-on-write.

Thread control: the initializer sets OMP/BLAS thread env BEFORE the ML libs are lazily imported inside
`fit`, so each worker's fit uses `threads` cores. Thread count is deliberately NOT part of the cache
signature (it's an execution detail, not a modelling choice); any sub-ULP differences from reduction
order are far below benchmark resolution.
"""
from __future__ import annotations

import gc
import os
import time

# ---- shared state: populated by init_worker INSIDE each spawned worker (no COW to rely on) -----------
_STATE: dict = {}


def _trim_memory() -> None:
    """Best-effort release of Python/C allocator slack after a large model cell."""
    gc.collect()
    if os.name != "posix":
        return
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def init_worker(threads: int, ds_name: str, bundle, splits) -> None:
    """Runs once at worker startup: sets thread budgets, then stashes the bundle + splits that a
    fork-based pool would otherwise have inherited via copy-on-write."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS"):
        os.environ[var] = str(max(1, int(threads)))
    _STATE["ds"] = ds_name
    _STATE["bundle"] = bundle
    _STATE["splits"] = splits


def baseline_cell(sid: int) -> tuple[int, float]:
    """Frozen task-GLM deviance for one split (the skill denominator)."""
    from insurarena.eval.hpo import fit_one
    from insurarena.models.registry import model_class
    from insurarena.scoring import primary_value
    from insurarena.seeds import seed_for

    bundle, (tr, te) = _STATE["bundle"], _STATE["splits"][sid]
    otr, ote = bundle.subset(tr), bundle.subset(te)
    try:
        glm = fit_one("glm", model_class("glm").default_hp(), otr, seed_for(_STATE["ds"], "baseline", sid))
        return sid, float(primary_value(ote, glm.predict(ote.X, ote.exposure)))
    except Exception:  # noqa: BLE001
        return sid, float("nan")


def model_cell(name: str, variant: str, sid: int, hp_override: dict,
               n_configs: int, inner_folds: int, time_budget,
               tuned_hp: dict | None = None, tuned_best_iter: int = 0) -> dict:
    """Fit + predict + score ONE (model, variant, split). Returns raw metrics; main adds skill/row."""
    import numpy as np

    from insurarena.eval.hpo import fit_one
    from insurarena.models.registry import model_class
    from insurarena.scoring import primary_value, score_bundle
    from insurarena.seeds import seed_for

    bundle, (tr, te) = _STATE["bundle"], _STATE["splits"][sid]
    n_train, n_test = len(tr), len(te)
    seed = seed_for(_STATE["ds"], name, variant, sid)
    out = {"sid": sid, "n_train": n_train, "n_test": n_test, "seed": int(seed),
           "primary_value": float("nan"), "secondary": {}, "train_time_s": float("nan"),
           "inference_time_s": float("nan"), "best_iter": 0, "trajectory": None, "failed": ""}
    otr = ote = model = pred = None
    try:
        otr = bundle.subset(tr)
        t0 = time.perf_counter()
        if variant == "tuned":
            if tuned_hp is None:
                raise ValueError("tuned model_cell requires a tuned-once hp payload")
            model = fit_one(name, tuned_hp, otr, seed, n_estimators=int(tuned_best_iter or 0) or None)
            out["best_iter"] = int(tuned_best_iter or 0)
        else:
            hp = {**model_class(name).default_hp(), **(hp_override or {})}
            model = fit_one(name, hp, otr, seed)
        out["train_time_s"] = time.perf_counter() - t0
        otr = None
        ote = bundle.subset(te)
        ti = time.perf_counter()
        pred = model.predict(ote.X, ote.exposure)
        out["inference_time_s"] = time.perf_counter() - ti
        out["primary_value"] = float(primary_value(ote, pred))
        out["secondary"] = score_bundle(ote, pred, bundle.task.secondary_metrics)
    except Exception as e:  # noqa: BLE001 - record + keep going (A4)
        out["failed"] = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        otr = ote = model = pred = None
        _trim_memory()
    return out
