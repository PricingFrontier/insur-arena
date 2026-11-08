"""Orchestrate dataset x model x variant x split. One fitted model per cell.

Per-cell results are cached (keyed on dataset content-hash + model/variant/hp + split + protocol),
so a run only computes cells it hasn't seen — adding a model/dataset, or changing the subsample /
tuning protocol, re-runs just the affected cells (everything else is reused).
"""
from __future__ import annotations

import gc
import os
import time

import numpy as np
import pandas as pd

from insurarena.datasets.registry import benchmark_exclusion_reason, build_loader
from insurarena.eval.hpo import (
    _model_selection_finale_hp,
    _model_selection_hp,
    config_set_digest,
    effective_n_configs,
    fit_one,
    resolve_inner_repeats,
    select_config,
)
from insurarena.eval.splitting import outer_splits, selection_subsample
from insurarena.metrics.deviance import skill_vs_baseline
from insurarena.models.registry import model_class
from insurarena.results.cache import ResultCache, cell_signature
from insurarena.results.schema import ResultRow
from insurarena.results.store import ResultStore
from insurarena.scoring import primary_value, score_bundle, task_targets
from insurarena.seeds import seed_for

_NAN = float("nan")


def _selected_dataset_names(cfg, verbose: bool) -> list[str]:
    if bool(getattr(cfg, "include_ineligible_datasets", False)):
        return list(cfg.datasets)
    selected = []
    for name in cfg.datasets:
        reason = benchmark_exclusion_reason(name)
        if reason is not None:
            if verbose:
                print(f"[{name}] skipped: {reason}")
            continue
        selected.append(name)
    return selected


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


def _racing_se_for(cfg, bundle) -> float | None:
    """The racing threshold that actually applies to THIS dataset, or None if racing is suppressed.

    Racing (A2) discards a candidate on its FIRST fold. That is only sound when one pass over the
    inner folds is itself a trustworthy signal — and the protocol already tells us when it is not:
    `inner_repeats: auto` repeats the inner k-fold until ~`_AUTO_REPEATS_TARGET_ROWS` (30k) rows
    have been scored, precisely because below that selection noise dominates real config
    differences. Discarding on one fold of a tune split the protocol has declared too noisy to
    score once is self-contradictory, and it bit us: on swautoins_pp (2,182 rows) racing dropped a
    champion that had cleared the ship-gate by >1 SE, and the fallback default-on-tuned-family
    scored BELOW the plain default variant (+0.0069 -> -0.0041).

    The gate is expressed in DATASET rows so that this function and `_proto` reach the same verdict
    without either having to build the outer splits (a mismatch there would key a cell as raced
    when it wasn't). For the 3-fold outer schedule the tune split trains on ~2n/3, so repeats==1
    needs 2n/3 >= 30_000, i.e. n >= 45_000 — the value `racing_min_rows: 45000` encodes.
    """
    se = getattr(cfg, "racing_se", None)
    if not se:
        return None
    floor = getattr(cfg, "racing_min_rows", None)
    if floor and len(bundle.X) < int(floor):
        return None
    return float(se)


def _proto(cfg, bundle, name: str, variant: str, hp_override: dict) -> dict:
    """Protocol fingerprint for a cell — what (besides data + model) determines the result."""
    p = {"split": cfg.split, "tweedie_power": bundle.task.tweedie_power, "hp": hp_override or {}}
    p["validation_protocol"] = "nested-outer-v1"
    p["seed_scheme"] = "cell"          # bump: fit_one now forwards the per-cell seed into hp (was cosmetic)
    cls = model_class(name)
    av = str(getattr(cls, "adapter_version", "1"))
    if av != "1":                          # omit when default so existing cells keep their signature
        p["av"] = av
    if variant == "tuned":
        hsv = str(getattr(cls, "hpo_search_version", "1"))
        if hsv != "1":
            p["hpo_search_version"] = hsv
        p["n_configs"] = effective_n_configs(name, cfg.n_configs, hp_override)
        p["inner_folds"] = cfg.inner_folds
        # Selection protocol v2: per-fold paired candidate scores, size-adaptive inner repeats,
        # and the static ship-gate (default wins unless beaten by >1 paired SE). The spec-level
        # inner_repeats ("auto" or int) is deterministic given the split schedule, so it keys the
        # cell; bumping either entry re-selects (and refits) every tuned cell, never default ones.
        p["hpo_selection"] = "v2"
        p["inner_repeats"] = getattr(cfg, "inner_repeats", "auto")
        p["hpo_config_set"] = config_set_digest(name, cfg.n_configs, hp_override)
        p["hpo_scope"] = "outer_training_partition"
        p["tune_split_id"] = int(getattr(cfg, "tune_split_id", 0))
        if cfg.time_budget_s is not None:
            p["time_budget_s"] = float(cfg.time_budget_s)
        # Racing (A2) can in principle change which config is selected, so it keys the tuned
        # selection cell; fold_workers is wall-clock-only and deliberately does NOT. Only the
        # EFFECTIVE threshold is keyed: on a dataset where `racing_min_rows` suppresses racing the
        # result is the no-racing result, so its key must be the no-racing key (and adding the gate
        # therefore re-keys exactly the small datasets whose selections it changes, leaving every
        # dataset above the floor cached).
        eff_racing = _racing_se_for(cfg, bundle)
        if eff_racing:
            p["racing_se"] = float(eff_racing)
        # A3 selection subsampling changes which rows the SEARCH scores on, so it can move the
        # selection and keys the tuned cell. Deploy fits are untouched (full outer-train), so
        # `default` cells never see it.
        if getattr(cfg, "selection_max_rows", None):
            p["selection_max_rows"] = int(cfg.selection_max_rows)
    return p


def _hpo_proto(cfg, bundle, name: str, hp_override: dict) -> dict:
    """Fingerprint for the tuned-once HPO choice shared by all outer splits."""
    p = _proto(cfg, bundle, name, "tuned", hp_override)
    p["kind"] = "tuned_once_hpo"
    return p


def _hpo_config_proto(cfg, bundle, name: str, hp_override: dict, inner_repeats: int = 1) -> dict:
    """Fingerprint for one reusable tuned candidate evaluation.

    Deliberately excludes n_configs: increasing the requested prefix should append new candidate
    evaluations while reusing earlier candidate scores. Takes the RESOLVED repeat count (not the
    "auto" spec) — a candidate's evaluation content is defined by how many folds actually scored it.
    """
    p = {
        "split": cfg.split,
        "tweedie_power": bundle.task.tweedie_power,
        "hp_override": hp_override or {},
        "inner_folds": cfg.inner_folds,
        "inner_repeats": int(inner_repeats),
        "hpo_scope": "outer_training_partition",
        "tune_split_id": int(getattr(cfg, "tune_split_id", 0)),
        "seed_scheme": "cell",         # bump: candidates now fit under per-candidate seeds (was cosmetic)
    }
    cls = model_class(name)
    av = str(getattr(cls, "adapter_version", "1"))
    if av != "1":
        p["av"] = av
    hsv = str(getattr(cls, "hpo_search_version", "1"))
    if hsv != "1":
        p["hpo_search_version"] = hsv
    # A3: candidate evaluations are scored on the capped search sample, so the cap is part of
    # what defines a candidate's score and must key the reusable candidate cell too.
    if getattr(cfg, "selection_max_rows", None):
        p["selection_max_rows"] = int(cfg.selection_max_rows)
    return p


def _tuned_once(cache, cfg, bundle, splits, ds_name: str, h: str, name: str, hp_override: dict):
    """Select tuned HP once per (dataset, model), cache it, and reuse for all outer splits."""
    tune_sid = int(getattr(cfg, "tune_split_id", 0))
    if not 0 <= tune_sid < len(splits):
        raise ValueError(f"tune_split_id={tune_sid} outside available split range 0..{len(splits)-1}")
    sig = cell_signature(h, name, "tuned__hpo", tune_sid, _hpo_proto(cfg, bundle, name, hp_override))
    cached = cache.get(sig, model=name)
    if cached is not None:
        return (
            cached["hp"],
            int(cached.get("best_iter") or 0),
            cached.get("trajectory"),
            cached.get("evaluations", []),
            True,
        )
    tr, _ = splits[tune_sid]
    seed = seed_for(ds_name, name, "tuned", "hpo", tune_sid)
    # A3 (SPEC §5.5): cap the rows the SEARCH scores on. Whole groups, stratified, deterministic —
    # see `selection_subsample`. The outer-split deploy fits below always refit on the full
    # outer-train, so only candidate ranking sees the cap. Repeats resolve on the rows actually
    # scored (a no-op in practice: "auto" already returns 1 at every size this cap applies to).
    tr = selection_subsample(bundle, tr, getattr(cfg, "selection_max_rows", None), seed)
    repeats = resolve_inner_repeats(getattr(cfg, "inner_repeats", "auto"), len(tr))
    # Protocol v3 (hsv 16, 2026-07-23): on small tune sets (the same inner_repeats>1 boundary
    # racing arms on), a scout-regime model's ship gate pools its finale over two extra tune
    # splits — a single split's finale can validate a champion whose inner-CV edge is
    # split-specific bias (freclaimdam picked the same cursed champion on 3/3 splits; the
    # anchor beat it on outer test). Extra splits are deterministic offsets across the repeat
    # structure; scout-regime detection = the adapter declares a finale overlay different from
    # its selection overlay (t-boost today), so every other model is byte-untouched.
    # NOTE (2026-08-04): an "extend pooling to every dataset size" change was drafted off
    # homesite_conv's tuned-below-default rows and REVERTED same day — those rows compared an
    # av32 champion against av31 default cells; at av32-vs-av32 the champion genuinely beats
    # the default on 9/9 deploys and the single-split gate was CORRECT. Re-extend only with
    # a real large-set counterexample (see triboost-vs-ccb memory 2026-08-04).
    #
    # RE-OPENED 2026-08-23 (hsv 18) against that condition, quoted above verbatim: "Re-extend only
    # with a real large-set counterexample." norauto is it (n=183,999, so `repeats` is 1 and hsv
    # 16's pooling never reached the cell). Its tuned search's own 28 inner-CV evaluations ranked
    # max_interaction_order<=2 above the anchor by ~0.00034 skill; `static_ship_gate_1se` then read
    # gate_margin 5.81e-5 against gate_se 6.57e-5 on n_paired_folds=3 and shipped the default as
    # "within_1se_of_default" — and the direction it discarded was REAL: the order-1 arm measures
    # +0.000126 +/- 0.000050 on OUTER test (t=2.53, 8/9 splits; 9 splits x 3 seeds) and halves the
    # cell's prediction seed SD. That is exactly the failure mode the pooled finale exists to fix —
    # a champion whose advantage is real and of precisely the size the gate cannot resolve at 3
    # paired folds — occurring on a large set.
    #
    # The 2026-08-04 finding is preserved by making the extension ADAPTIVE, not unconditional:
    # extras are supplied at every size, but `hpo._pooled_finale_wanted` spends them only where the
    # single-split gate lands within +/-2 SE. homesite_conv's champion (decisive, 9/9 at matched
    # adapter versions) never reaches the pooled path; norauto's 0.88-SE margin does. Extras are
    # passed as THUNKS, so a decisive large-set cell never materializes the two extra outer-trains.
    extra_finale_otrs, pooled_finale_mode = [], "always"
    if not getattr(cfg, "_nested_campaign", False) and _model_selection_finale_hp(name) != _model_selection_hp(name):
        # "always" keeps protocol v3 byte-identical on the small tune sets it was measured on.
        pooled_finale_mode = "always" if repeats > 1 else "adaptive"
        step = max(1, len(splits) // 3)
        for j in (1, 2):
            sid = (tune_sid + j * step) % len(splits)
            if sid == tune_sid:
                continue

            def _extra_otr(sid=sid, j=j):
                etr, _ = splits[sid]
                etr = selection_subsample(bundle, etr,
                                          getattr(cfg, "selection_max_rows", None), seed + j)
                return bundle.subset(etr)

            extra_finale_otrs.append(_extra_otr)
    selection_started = time.perf_counter()
    hp, best_iter, traj, evaluations, selection = select_config(
        name, bundle.subset(tr), cfg.n_configs, cfg.inner_folds, seed, cfg.time_budget_s,
        hp_override=hp_override, cache=cache, dataset_hash=h,
        candidate_proto=_hpo_config_proto(cfg, bundle, name, hp_override, inner_repeats=repeats),
        repeats=repeats,
        fold_workers=int(getattr(cfg, "fold_workers", 1) or 1),
        racing_se=_racing_se_for(cfg, bundle),   # None on small/noisy sets — see the helper's doc
        extra_finale_otrs=extra_finale_otrs or None,
        pooled_finale_mode=pooled_finale_mode,
    )
    payload = {
        "dataset": ds_name,
        "model": name,
        "hp": hp,
        "best_iter": int(best_iter or 0),
        "trajectory": [None if t != t else float(t) for t in traj],
        "evaluations": evaluations,
        "selection": selection,
        "tuning_seconds": time.perf_counter() - selection_started,
    }
    cache.put(sig, payload, model=name)
    return payload["hp"], payload["best_iter"], payload["trajectory"], payload["evaluations"], False


def _baseline_deviances(bundle, splits, ds_name, cache, h, split_cfg):
    """Frozen task-GLM (default) deviance per split — cached; the skill denominator."""
    dev = []
    glm_default = model_class("glm").default_hp()
    proto = {"split": split_cfg, "tweedie_power": bundle.task.tweedie_power}  # split schedule determines folds
    for sid, (tr, te) in enumerate(splits):
        sig = cell_signature(h, "__glm_baseline__", "default", sid, proto)
        c = cache.get(sig)
        if c is not None:
            dev.append(c["baseline_dev"])
            continue
        otr, ote = bundle.subset(tr), bundle.subset(te)
        try:
            glm = fit_one("glm", glm_default, otr, seed_for(ds_name, "baseline", sid))
            d = primary_value(ote, glm.predict(ote.X, ote.exposure))
        except Exception:  # noqa: BLE001
            d = _NAN
        dev.append(d)
        cache.put(sig, {"baseline_dev": d})
    return dev


def run_benchmark(cfg, store: ResultStore | None = None, verbose: bool = True) -> pd.DataFrame:
    """Run the nested protocol, including evidence in both execution modes."""
    from insurarena.eval.campaign import run_campaign
    return run_campaign(cfg, store=store, verbose=verbose)


def _legacy_run_benchmark(cfg, store: ResultStore | None = None, verbose: bool = True) -> pd.DataFrame:
    if int(getattr(cfg, "max_workers", 1)) > 1:
        return _run_benchmark_parallel(cfg, store, verbose)
    store = store or ResultStore(cfg.cache_dir, cfg.store_predictions)
    cache = ResultCache(cfg.cache_dir, enabled=getattr(cfg, "use_cache", True))
    hits = misses = 0

    for ds_name in _selected_dataset_names(cfg, verbose):
        bundle = build_loader(ds_name, subsample=cfg.subsample_rows, cache_dir=cfg.cache_dir).load()
        h = str(bundle.meta.get("hash", ""))
        splits = outer_splits(bundle, cfg.split, seed_for(ds_name))
        if verbose:
            print(f"[{ds_name}] n={len(bundle.X)} groups={bundle.n_groups} splits={len(splits)} hash={h}")
        baseline = _baseline_deviances(bundle, splits, ds_name, cache, h, cfg.split)

        for entry in cfg.models:
            name = entry["name"]
            if bundle.task.task not in model_class(name).supported_tasks:
                if verbose:
                    print(f"  (skip {name}: does not support task {bundle.task.task.value})")
                continue
            for variant in entry.get("variants", ["default"]):
                proto = _proto(cfg, bundle, name, variant, entry.get("hp", {}))
                n_fail = n_cached = 0
                tuned_ready = False
                tuned_hp, tuned_best_iter = None, 0
                for sid, (tr, te) in enumerate(splits):
                    sig = cell_signature(h, name, variant, sid, proto)
                    cached = cache.get(sig, model=name)
                    if cached is not None:
                        traj = cached.pop("trajectory", None)
                        store.add(cached)
                        if traj:
                            store.write_trajectory(ds_name, name, sid, traj)
                        hits += 1
                        n_cached += 1
                        continue

                    n_train, n_test = len(tr), len(te)
                    seed = seed_for(ds_name, name, variant, sid)
                    pv = skill = inf_t = train_t = _NAN
                    sec, failed, traj = {}, "", None
                    otr = ote = model = pred = None
                    try:
                        otr = bundle.subset(tr)
                        t0 = time.perf_counter()
                        if variant == "tuned":
                            if not tuned_ready:
                                tuned_hp, tuned_best_iter, tuned_traj, tuned_evals, _ = _tuned_once(
                                    cache, cfg, bundle, splits, ds_name, h, name, entry.get("hp", {}))
                                if tuned_traj:
                                    store.write_trajectory(
                                        ds_name, name, int(getattr(cfg, "tune_split_id", 0)), tuned_traj)
                                if tuned_evals:
                                    store.write_hpo_evaluations(
                                        ds_name, name, int(getattr(cfg, "tune_split_id", 0)), tuned_evals)
                                tuned_ready = True
                            hp, best_iter, traj = tuned_hp, int(tuned_best_iter or 0), None
                            model = fit_one(name, hp, otr, seed, n_estimators=best_iter or None)
                        else:
                            hp = {**model_class(name).default_hp(), **entry.get("hp", {})}
                            model = fit_one(name, hp, otr, seed)
                        train_t = time.perf_counter() - t0
                        otr = None
                        ote = bundle.subset(te)
                        ti = time.perf_counter()
                        pred = model.predict(ote.X, ote.exposure)
                        inf_t = time.perf_counter() - ti
                        pv = primary_value(ote, pred)
                        skill = skill_vs_baseline(pv, baseline[sid])
                        sec = score_bundle(ote, pred, bundle.task.secondary_metrics)
                    except Exception as e:  # noqa: BLE001 - record + keep going (A4)
                        failed = f"{type(e).__name__}: {str(e)[:120]}"
                        n_fail += 1

                    row = ResultRow(
                        dataset=ds_name, task=bundle.task.task.value, source_hash=h,
                        model=name, variant=variant, family=model_class(name).family,
                        split_id=sid, n_train=n_train, n_test=n_test,
                        primary_metric=bundle.task.primary_metric, primary_value=pv, skill_vs_glm=skill,
                        train_time_s=train_t, seed=int(seed), inference_time_s=inf_t, failed=failed, secondary=sec,
                    )
                    d = row.to_dict()
                    store.write(row)
                    payload = dict(d)
                    if traj is not None:
                        traj = [None if t != t else float(t) for t in traj]  # NaN -> None for json
                        store.write_trajectory(ds_name, name, sid, traj)
                        payload["trajectory"] = traj
                    if not failed:                          # never cache a failure (transient)
                        cache.put(sig, payload, model=name)
                        yt, w, exp = task_targets(ote)
                        store.write_predictions(f"{ds_name}__{name}__{variant}__s{sid}", yt, pred, exp)
                    misses += 1
                    otr = ote = model = pred = None
                    _trim_memory()
                if verbose:
                    sub = [r for r in store.rows if r["model"] == name and r["variant"] == variant
                           and r["dataset"] == ds_name]
                    ms = float(np.nanmean([r["skill_vs_glm"] for r in sub])) if sub else _NAN
                    tag = f" [{n_cached} cached]" if n_cached else ""
                    tag += f" ({n_fail} failed)" if n_fail else ""
                    print(f"  {name}/{variant:<8s} mean skill-vs-GLM = {ms:+.4f}{tag}")

    if verbose:
        print(f"cache: {hits} reused, {misses} computed")
    return store.to_frame()


def _print_ds_summary(store: ResultStore, ds_name: str, cfg, verbose: bool) -> None:
    """Per-(model,variant) mean skill-vs-GLM for a finished dataset (mirrors the sequential print)."""
    if not verbose:
        return
    for entry in cfg.models:
        name = entry["name"]
        for variant in entry.get("variants", ["default"]):
            sub = [r for r in store.rows if r["dataset"] == ds_name and r["model"] == name
                   and r["variant"] == variant]
            if not sub:
                continue
            ms = float(np.nanmean([r["skill_vs_glm"] for r in sub]))
            nf = sum(1 for r in sub if r.get("failed"))
            tag = f" ({nf} failed)" if nf else ""
            print(f"  {name}/{variant:<8s} mean skill-vs-GLM = {ms:+.4f}{tag}")


def _run_benchmark_parallel(cfg, store: ResultStore | None, verbose: bool) -> pd.DataFrame:
    """Cell-level parallel driver: one process pool per dataset, cells (model x variant x split) fanned
    out to workers. Main keeps all bookkeeping (cache, skill, rows) so output matches the sequential
    path. Prediction dumps are skipped here (store_predictions is off for the heavy full-data runs)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from insurarena.eval import parallel as P

    store = store or ResultStore(cfg.cache_dir, cfg.store_predictions)
    cache = ResultCache(cfg.cache_dir, enabled=getattr(cfg, "use_cache", True))
    hits = misses = 0
    n_workers = int(cfg.max_workers)
    tpw = int(getattr(cfg, "threads_per_worker", 0)) or max(1, (os.cpu_count() or 2) // n_workers)
    # Publish the per-fit thread budget into the PARENT environment BEFORE spawning workers, so each
    # spawned worker inherits it at exec time. init_worker also sets these, but that runs AFTER the
    # worker interpreter has started -- too late if the worker's first rayon/BLAS use has already
    # built a global pool sized to all logical cores. Those idle pool threads spin-wait, so with N
    # workers you get N x (cores) spinning threads oversubscribing the box (load storm, ~10x slower
    # on wide-feature fits). Setting it here, pre-exec, makes the child build a correctly-sized pool.
    for _tvar in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_tvar] = str(tpw)
    # spawn, not fork: by this point the coordinator may already have background native threads alive
    # (BLAS, a rayon pool inside a model library, ...) from loading a prior dataset or fitting a
    # baseline; fork()-ing a threaded process can inherit one of their mutexes already locked, with
    # no thread left in the child to release it -> the first worker to touch that library hangs
    # forever. Spawn starts each worker as a fresh interpreter, so the bundle + splits are passed
    # explicitly through the pool initializer (see init_worker) instead of relying on copy-on-write.
    ctx = mp.get_context("spawn")
    if verbose:
        print(f"[parallel] {n_workers} workers x {tpw} threads/fit")

    for ds_name in _selected_dataset_names(cfg, verbose):
        bundle = build_loader(ds_name, subsample=cfg.subsample_rows, cache_dir=cfg.cache_dir).load()
        h = str(bundle.meta.get("hash", ""))
        splits = outer_splits(bundle, cfg.split, seed_for(ds_name))
        n_splits = len(splits)
        if verbose:
            print(f"[{ds_name}] n={len(bundle.X)} groups={bundle.n_groups} splits={n_splits} hash={h}")

        base_proto = {"split": cfg.split, "tweedie_power": bundle.task.tweedie_power}
        baseline: list = [_NAN] * n_splits
        base_todo = []
        for sid in range(n_splits):
            c = cache.get(cell_signature(h, "__glm_baseline__", "default", sid, base_proto))
            if c is not None:
                baseline[sid] = c["baseline_dev"]
            else:
                base_todo.append(sid)

        model_tasks = []                                   # (name, variant, sid, sig, hp_override, tuned_hp, best_iter)
        for entry in cfg.models:
            name = entry["name"]
            if bundle.task.task not in model_class(name).supported_tasks:
                if verbose:
                    print(f"  (skip {name}: does not support task {bundle.task.task.value})")
                continue
            for variant in entry.get("variants", ["default"]):
                hp_override = entry.get("hp", {})
                proto = _proto(cfg, bundle, name, variant, hp_override)
                missing = []
                for sid in range(n_splits):
                    sig = cell_signature(h, name, variant, sid, proto)
                    cached = cache.get(sig, model=name)
                    if cached is not None:
                        traj = cached.pop("trajectory", None)
                        store.add(cached)
                        if traj:
                            store.write_trajectory(ds_name, name, sid, traj)
                        hits += 1
                        continue
                    missing.append((sid, sig))
                tuned_hp, tuned_best_iter = None, 0
                if missing and variant == "tuned":
                    # The selection runs HERE, in the coordinator, outside any worker's try/except.
                    # `select_config` now REFUSES a search in which nothing scored (see
                    # `hpo._require_scored_evaluations`) rather than quietly shipping the default
                    # anchor, so that refusal has to land as failed cells for this dataset — not as
                    # a traceback that kills a multi-hour campaign at dataset 3 of 40. Leaving
                    # `tuned_hp` None makes `parallel.model_cell` raise per split, which is the
                    # sequential path's behavior: visible failed cells, nothing cached, re-done next run.
                    try:
                        tuned_hp, tuned_best_iter, tuned_traj, tuned_evals, _ = _tuned_once(
                            cache, cfg, bundle, splits, ds_name, h, name, hp_override)
                    except Exception as e:  # noqa: BLE001
                        tuned_hp, tuned_best_iter, tuned_traj, tuned_evals = None, 0, None, None
                        if verbose:
                            print(f"  !! {name}/tuned selection failed on {ds_name}: "
                                  f"{type(e).__name__}: {str(e)[:160]}")
                    if tuned_traj:
                        store.write_trajectory(ds_name, name, int(getattr(cfg, "tune_split_id", 0)), tuned_traj)
                    if tuned_evals:
                        store.write_hpo_evaluations(
                            ds_name, name, int(getattr(cfg, "tune_split_id", 0)), tuned_evals)
                for sid, sig in missing:
                    model_tasks.append((name, variant, sid, sig, hp_override, tuned_hp, tuned_best_iter))

        if base_todo or model_tasks:
            def _commit(tag, res):
                """Commit ONE finished model cell (row + cache) — durably, as soon as it lands."""
                nonlocal misses
                _, name, variant, sid, sig = tag
                failed = res["failed"]
                pv = res["primary_value"]
                skill = skill_vs_baseline(pv, baseline[sid]) if not failed else _NAN
                row = ResultRow(
                    dataset=ds_name, task=bundle.task.task.value, source_hash=h,
                    model=name, variant=variant, family=model_class(name).family,
                    split_id=sid, n_train=res["n_train"], n_test=res["n_test"],
                    primary_metric=bundle.task.primary_metric, primary_value=pv, skill_vs_glm=skill,
                    train_time_s=res["train_time_s"], seed=int(res["seed"]),
                    inference_time_s=res["inference_time_s"], failed=failed, secondary=res["secondary"],
                )
                store.write(row)
                payload = dict(row.to_dict())
                traj = res["trajectory"]
                if traj is not None:
                    traj = [None if t != t else float(t) for t in traj]
                    store.write_trajectory(ds_name, name, sid, traj)
                    payload["trajectory"] = traj
                if not failed:
                    cache.put(sig, payload, model=name)
                misses += 1

            # A model cell's skill needs its split's GLM baseline, but futures complete in any
            # order — commit immediately once the baseline is known, else park until it lands.
            have_base = {sid for sid in range(n_splits) if sid not in base_todo}
            parked: dict[int, list] = {}
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=P.init_worker,
                                     initargs=(tpw, ds_name, bundle, splits)) as ex:
                futs = {ex.submit(P.baseline_cell, sid): ("__base__", sid) for sid in base_todo}
                for (name, variant, sid, sig, hp_ov, tuned_hp, tuned_best_iter) in model_tasks:
                    f = ex.submit(P.model_cell, name, variant, sid, hp_ov,
                                  cfg.n_configs, cfg.inner_folds, cfg.time_budget_s,
                                  tuned_hp, int(tuned_best_iter or 0))
                    futs[f] = ("__cell__", name, variant, sid, sig)
                for fut in as_completed(futs):
                    tag = futs[fut]
                    res = fut.result()
                    if tag[0] == "__base__":
                        sid = tag[1]
                        baseline[sid] = res[1]
                        cache.put(cell_signature(h, "__glm_baseline__", "default", sid, base_proto),
                                  {"baseline_dev": res[1]})
                        have_base.add(sid)
                        for t, r in parked.pop(sid, []):
                            _commit(t, r)
                    elif tag[3] in have_base:
                        _commit(tag, res)
                    else:
                        parked.setdefault(tag[3], []).append((tag, res))
            for sid in sorted(parked):                     # baseline future failed -> commit with NaN skill
                for t, r in parked[sid]:
                    _commit(t, r)

        _print_ds_summary(store, ds_name, cfg, verbose)

    if verbose:
        print(f"cache: {hits} reused, {misses} computed")
    return store.to_frame()
