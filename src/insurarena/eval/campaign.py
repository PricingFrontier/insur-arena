"""Nested campaign with identical sequential/process execution and retained evidence."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from threadpoolctl import threadpool_limits

from insurarena.datasets.registry import build_loader
from insurarena.eval import parallel as P
from insurarena.eval.hpo import fit_one
from insurarena.eval.splitting import outer_splits
from insurarena.metrics.deviance import skill_vs_baseline
from insurarena.model_ids import current_id
from insurarena.models.registry import model_class
from insurarena.results.cache import ResultCache, cell_signature
from insurarena.results.schema import ResultRow
from insurarena.results.store import ResultStore
from insurarena.scoring import primary_value, score_bundle
from insurarena.seeds import seed_for

PROTOCOL = 'nested-evidence-v1'


def digest_rows(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def software_manifest():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    versions = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'catboost',
                 'interpret', 'glum', 'rustystats', 'tabpfn', 'tabicl', 'tabdpt'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    from insurarena.results.engine_fingerprint import engine_fingerprint
    tb_hash = hashlib.sha256()
    try:
        import t_boost
        for path in sorted(Path(t_boost.__file__).parent.rglob('*.py')):
            tb_hash.update(path.name.encode())
            tb_hash.update(path.read_bytes())
        tb_source = tb_hash.hexdigest()
    except ImportError:
        tb_source = None
    return {'harness_sha256': h.hexdigest(), 'packages': versions,
            # The two key names are FROZEN: this dict is part of every model's cell proto, so
            # renaming a key would re-key the whole board. See insurarena.model_ids.
            'triboost_engine': engine_fingerprint('t-boost'), 'triboost_python': tb_source,
            'python': platform.python_version()}


def partition_protocol(cfg, bundle, name, variant, hp, sid, splits, software):
    from insurarena.eval.runner import _proto
    scoped = replace(cfg, tune_split_id=sid)
    scoped._nested_campaign = True
    proto = _proto(scoped, bundle, name, variant, hp)
    tr, te = splits[sid]
    proto.update(protocol=PROTOCOL, train_rows_sha256=digest_rows(tr),
                 test_rows_sha256=digest_rows(te), software=software,
                 threads_per_worker=cfg.threads_per_worker,
                 segment_columns=cfg.segment_columns, predictions=cfg.store_predictions,
                 evaluation_mode=cfg.evaluation_mode,
                 split_manifest=getattr(cfg, '_split_manifest', None),
                 timing_repeats=cfg.timing_repeats, timing_batch_size=cfg.timing_batch_size)
    if cfg.evaluation_mode == 'timing':
        proto['timing_run_id'] = getattr(cfg, '_timing_run_id', None)
    return scoped, proto


def _cell(cfg, name, variant, sid, hp_override, signature, software):
    """Worker owns model and writes artifacts; no model/prediction IPC."""
    from insurarena.eval.evidence import (collect_evidence, prediction_payload,
                                          measure_prediction, write_evidence)
    from insurarena.eval.runner import _tuned_once
    from insurarena.scoring import target_contract
    bundle, splits = P._STATE['bundle'], P._STATE['splits']
    ds = P._STATE['ds']
    tr, te = splits[sid]
    scoped, proto = partition_protocol(cfg, bundle, name, variant, hp_override, sid, splits, software)
    cache = ResultCache(cfg.cache_dir, enabled=cfg.use_cache)
    # Candidate cache identity includes exact outer-training rows and software.
    selection_identity = hashlib.sha256(json.dumps({
        'data': bundle.meta.get('hash'), 'train': digest_rows(tr), 'software': software,
        'protocol': PROTOCOL}, sort_keys=True).encode()).hexdigest()
    seed = seed_for(ds, name, variant, sid)
    result = dict(trajectory=None, evaluations=[], cached=False)
    times = {'tuning_time_s': 0.0, 'tuning_cache_hit': False, 'scoring_time_s': None,
             'inspection_time_s': None}
    pv = baseline = train_time = predict_time = float('nan')
    secondary, failed, metadata, paths = {}, '', {}, {}
    try:
        otr, ote = bundle.subset(tr), bundle.subset(te)
        target_contract(otr)
        target_contract(ote)
        hp = {**model_class(name).default_hp(), **hp_override}
        best_iter = 0
        if variant == 'tuned':
            start = time.perf_counter()
            hp, best_iter, trajectory, evaluations, hit = _tuned_once(
                cache, scoped, bundle, splits, ds, selection_identity, name, hp_override)
            times.update(tuning_time_s=time.perf_counter() - start, tuning_cache_hit=hit)
            result.update(trajectory=trajectory, evaluations=evaluations)
        # Baseline is separate from competitor fitting and timing.
        base_sig = cell_signature(selection_identity, '__glm_baseline__', 'default', sid,
                                  {'protocol': PROTOCOL, 'test': digest_rows(te)})
        base = cache.get(base_sig)
        if base is None:
            glm = fit_one('glm', model_class('glm').default_hp(), otr, seed_for(ds, 'baseline', sid))
            baseline = float(primary_value(ote, glm.predict(ote.X, ote.exposure)))
            if np.isfinite(baseline):
                cache.put(base_sig, {'baseline_dev': baseline})
            del glm
        else:
            baseline = base['baseline_dev']
        start = time.perf_counter()
        model = fit_one(name, hp, otr, seed, n_estimators=int(best_iter or 0) or None)
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(ote.X, ote.exposure)
        predict_time = time.perf_counter() - start
        start = time.perf_counter()
        pv = float(primary_value(ote, pred))
        secondary = score_bundle(ote, pred, bundle.task.secondary_metrics)
        times['scoring_time_s'] = time.perf_counter() - start
        start = time.perf_counter()
        metadata = collect_evidence(model, otr, ote, pred, tr, te, cfg.segment_columns)
        times['inspection_time_s'] = time.perf_counter() - start
        if cfg.evaluation_mode == 'timing':
            metadata['serving_timing'] = measure_prediction(
                model, ote, repeats=cfg.timing_repeats, batch_size=cfg.timing_batch_size)
        metadata.update(chosen_parameters=hp, best_iteration=int(best_iter or 0),
                        target_contract=target_contract(ote), timings=times,
                        fit_time_includes='preprocessing, early-stopping probe/refit and any native table compilation',
                        timing_scope=('serial benchmark process; external host contention not controlled'
                                      if cfg.evaluation_mode == 'timing' else 'accuracy workload; not isolated latency'),
                        protocol=proto, signature=signature, dataset=ds, model=name,
                        variant=variant, split_id=sid, seed=int(seed))
        predictions = prediction_payload(ote, pred, te, cfg.segment_columns) if cfg.store_predictions else None
        paths = write_evidence(Path(cfg.cache_dir).expanduser() / 'evidence', signature, metadata, predictions)
    except Exception as exc:
        failed = f'{type(exc).__name__}: {str(exc)[:500]}'
        paths = write_evidence(Path(cfg.cache_dir).expanduser() / 'evidence', signature,
                               {'failed': failed, 'protocol': proto, 'dataset': ds, 'model': name,
                                'variant': variant, 'split_id': sid, 'timings': times})
    paths = {k: str(v) for k, v in paths._asdict().items() if v is not None}
    row = ResultRow(dataset=ds, task=bundle.task.task.value, source_hash=str(bundle.meta.get('hash', '')),
                    model=name, variant=variant, family=model_class(name).family, split_id=sid,
                    n_train=len(tr), n_test=len(te), primary_metric=bundle.task.primary_metric,
                    primary_value=pv, skill_vs_glm=skill_vs_baseline(pv, baseline),
                    train_time_s=train_time, inference_time_s=predict_time, seed=int(seed),
                    failed=failed, secondary=secondary).to_dict()
    row.update(times, evaluation_mode=cfg.evaluation_mode, evidence_paths=paths,
               train_rows_sha256=digest_rows(tr), test_rows_sha256=digest_rows(te), signature=signature)
    result['row'] = row
    if not failed and cfg.evaluation_mode != 'timing':
        cache.put(signature, result, model=name)
    P._trim_memory()
    return result


def _cached(cache, sig, name, cfg):
    if cfg.evaluation_mode == 'timing':
        return None  # timings always measured afresh
    result = cache.get(sig, model=name)
    if result is None or 'row' not in result:
        return None
    result['row']['model'] = current_id(result['row'].get('model', name))  # pre-rename rows
    paths = result['row'].get('evidence_paths', {})
    # Cached metrics cannot substitute for absent retained evidence.
    if not paths or not all(Path(p).exists() for p in paths.values()):
        return None
    return result


def run_campaign(cfg, store=None, verbose=True):
    from insurarena.eval.runner import _selected_dataset_names
    if cfg.evaluation_mode not in ('accuracy', 'timing'):
        raise ValueError('evaluation_mode must be accuracy or timing')
    if cfg.evaluation_mode == 'timing' and cfg.max_workers != 1:
        raise ValueError('timing mode requires max_workers=1')
    if cfg.timing_repeats < 1 or cfg.timing_batch_size < 1 or cfg.max_workers < 1:
        raise ValueError('worker count and timing sizes must be positive')
    threads = cfg.threads_per_worker or max(1, (os.cpu_count() or 1) // cfg.max_workers)
    cfg = replace(cfg, threads_per_worker=threads, cache_dir=str(Path(cfg.cache_dir).expanduser().resolve()))
    if cfg.evaluation_mode == 'timing':
        cfg._timing_run_id = uuid.uuid4().hex
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
        os.environ[var] = str(threads)
    software = software_manifest()
    store = store or ResultStore(cfg.cache_dir, cfg.store_predictions)
    cache = ResultCache(cfg.cache_dir, enabled=cfg.use_cache)
    records_dir = Path(cfg.cache_dir) / 'campaign_records'
    records_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol': PROTOCOL, 'evaluation_mode': cfg.evaluation_mode,
                'software': software, 'signatures': []}
    manifest_path = Path(cfg.cache_dir) / 'active_campaign.json'
    def persist_manifest():
        tmp = manifest_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, manifest_path)
    persist_manifest()
    def commit(result):
        row = result['row']
        store.add(row)
        path = records_dir / f"{row['signature']}.json"
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(row, default=str))
        os.replace(tmp, path)
        if result.get('trajectory'):
            store.write_trajectory(row['dataset'], row['model'], row['split_id'], result['trajectory'])
        if result.get('evaluations'):
            store.write_hpo_evaluations(row['dataset'], row['model'], row['split_id'], result['evaluations'])
        if verbose:
            print(f"  {row['model']}/{row['variant']} split={row['split_id']} "
                  f"{row['failed'] or row['primary_value']}", flush=True)
    for ds in _selected_dataset_names(cfg, verbose):
        bundle = build_loader(ds, subsample=cfg.subsample_rows, cache_dir=cfg.cache_dir).load()
        missing = set(cfg.segment_columns) - set(bundle.X.columns)
        if missing:
            raise ValueError(f'unknown segment columns for {ds}: {sorted(missing)}')
        splits = outer_splits(bundle, cfg.split, seed_for(ds))
        split_dir = Path(cfg.cache_dir) / 'splits'
        split_dir.mkdir(parents=True, exist_ok=True)
        split_key = hashlib.sha256(json.dumps({'data': bundle.meta.get('hash'),
            'rows': [(digest_rows(tr), digest_rows(te)) for tr, te in splits]}, sort_keys=True).encode()).hexdigest()
        split_path = split_dir / f'{split_key}.npz'
        np.savez_compressed(split_path, **{f'{side}_{sid}': rows for sid, pair in enumerate(splits)
                                          for side, rows in zip(('train', 'test'), pair)})
        cfg._split_manifest = str(split_path)
        tasks = []
        for entry in cfg.models:
            name = entry['name']
            if bundle.task.task not in model_class(name).supported_tasks:
                continue
            for variant in entry.get('variants', ['default']):
                if variant not in ('default', 'tuned'):
                    raise ValueError(f'unsupported variant {variant}')
                for sid in range(len(splits)):
                    _, proto = partition_protocol(cfg, bundle, name, variant, entry.get('hp', {}), sid, splits, software)
                    sig = cell_signature(str(bundle.meta.get('hash', '')), name, variant, sid, proto)
                    manifest['signatures'].append(sig)
                    persist_manifest()
                    cached = _cached(cache, sig, name, cfg)
                    if cached is not None:
                        commit(cached)
                    else:
                        tasks.append((cfg, name, variant, sid, entry.get('hp', {}), sig, software))
        if cfg.max_workers == 1:
            P.init_worker(threads, ds, bundle, splits)
            with threadpool_limits(limits=threads):
                for task in tasks:
                    commit(_cell(*task))
        else:
            with ProcessPoolExecutor(max_workers=cfg.max_workers, mp_context=mp.get_context('spawn'),
                                     initializer=P.init_worker, initargs=(threads, ds, bundle, splits)) as pool:
                futures = [pool.submit(_cell, *task) for task in tasks]
                for future in as_completed(futures):
                    commit(future.result())
    store.rows.sort(key=lambda row: (row["dataset"], row["model"], row["variant"], row["split_id"]))
    return store.to_frame()
