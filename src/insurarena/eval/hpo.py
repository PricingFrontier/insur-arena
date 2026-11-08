"""HPO: a static seeded config set, k-fold out-of-fold selection, single refit (no bag).

The campaign calls this independently within each outer training partition, then refits
one model on that partition with its selected config. The k folds only *score* configs; they are not bagged.

Selection protocol v2: every candidate records per-fold OOF scores on shared folds (paired across
candidates), small tune splits repeat the inner CV with fresh fold seeds (`resolve_inner_repeats`),
and the static search ships its champion only if it beats the default candidate by more than one
standard error of the paired per-fold differences — otherwise the default is shipped.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from insurarena.datasets.base import DatasetBundle
from insurarena.eval.splitting import inner_holdout, kfold
from insurarena.metrics.registry import METRICS
from insurarena.model_ids import hash_key
from insurarena.models.registry import build_model, model_class
from insurarena.results.cache import ResultCache, cell_signature
from insurarena.scoring import primary_value

_CONFIG_SEED = 20240601  # fixed -> the sampled config set is identical across datasets

_AUTO_REPEATS_TARGET_ROWS = 30_000  # "auto" repeats until ~this many rows have been OOF-scored
_AUTO_REPEATS_CAP = 5
_GATE_SE_MULT = 1.0  # static ship-gate: champion must beat default by > this many paired-fold SEs
_ADAPTIVE_POOL_SE_MULT = 2.0  # adaptive pooled finale: FLOOR of the pooling band, in paired-fold SEs
                              # (hsv 21: the band is max(this, the ship bar in force) — see
                              # `_adaptive_pool_band`)


def resolve_inner_repeats(spec, n_rows: int) -> int:
    """Inner-CV repeat count for tuned selection on a tune split with `n_rows` outer-train rows.

    "auto" repeats the k-fold until roughly `_AUTO_REPEATS_TARGET_ROWS` rows have been scored
    (capped), so tiny portfolios — where selection noise dwarfs real config differences but fits
    cost fractions of a second — get a lower-variance signal, while large datasets keep one pass.
    """
    if spec is None:
        return 1
    if isinstance(spec, str):
        if spec.strip().lower() != "auto":
            raise ValueError(f"inner_repeats must be a positive int or 'auto', got {spec!r}")
        if n_rows <= 0:
            return 1
        return int(np.clip(int(np.ceil(_AUTO_REPEATS_TARGET_ROWS / n_rows)), 1, _AUTO_REPEATS_CAP))
    return max(1, int(spec))


def _json_default(v):
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


def _config_seed(model_name: str, config_idx: int) -> int:
    """Stable, per-index seed so increasing n_configs appends instead of shifting old configs."""
    key = f"{_CONFIG_SEED}:{hash_key(model_name)}:{int(config_idx)}"  # legacy key: see model_ids
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def _same_value(a, b) -> bool:
    try:
        if isinstance(a, float) or isinstance(b, float):
            return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-12))
    except Exception:
        pass
    return a == b


def coordinate_probes(model_name: str, n_configs: int) -> list[dict]:
    """The adapter's deterministic one-knob-off-base probe block, truncated to the budget.

    An adapter may declare `coordinate_probe_hp_set() -> list[dict]` (t-boost only, hsv 18,
    2026-08-23): a fixed, seedless list of candidates that each move EXACTLY ONE coordinate off
    the tuned-family base. They take the nominal indices immediately after the default anchor and
    REPLACE that many draws of the random basis (see `static_configs`), so the candidate count —
    and therefore the max-of-N the ship gate has to defend against — is unchanged.

    Motivation (headroom scan, 2026-08-21): a winning single-knob value that is already IN the
    search grid can still be invisible to the gate, because a jointly-randomized basis only ever
    offers it bundled with five other randomized knobs. Measured: tic2000's
    `max_interaction_order=2` is the largest single-knob move on a 6-dataset ablation (+0.567%
    inner CV) and no draw of the 25-candidate basis ever presented it cleanly, while the basis
    champion that DID reach the gate there was harmful out of sample (-0.00166 skill).

    Models that don't declare the hook get `[]`, so their candidate set and `config_set_digest`
    stay byte-identical.
    """
    fn = getattr(model_class(model_name), "coordinate_probe_hp_set", None)
    if not callable(fn):
        return []
    return [dict(hp) for hp in (fn() or [])][: max(0, int(n_configs))]


def static_configs(model_name: str, n_configs: int, hp_override: dict | None = None) -> list[dict]:
    """Fixed, seeded config set shared across datasets (default first, then n_configs candidates).

    Layout, and the identity contract each part keeps:

        index 0        the default anchor (`default_hp`) — the ship-gate reference.
        index 1..P     the adapter's coordinate probes (`coordinate_probes`), P >= 0. A pure
                       function of the adapter, so probe j sits at index j forever.
        index P+1..n   the sampled basis.

    `len(result) == n_configs + 1` at every n, probes or not, so the probe block DISPLACES basis
    draws instead of enlarging the candidate set.

    An adapter may declare `sample_hp_set(n, seed) -> list[dict]` (t-boost only, 2026-07-20) to
    draw the whole candidate set jointly space-filling instead of one independent `sample_hp` draw
    per index. That sampler's balance guarantees are properties of a CONTIGUOUS PREFIX (its joint
    (order, hurdle) cycle covers all 15 combinations once n >= 15), so it is asked for the reduced
    count and its candidate j lands at nominal index P+j — the sub-basis is exactly the previous
    version's first `n - P` candidates. The per-index `sample_hp` fallback instead keeps keying on
    the NOMINAL index, which is what `_config_seed` documents and what leaves every model without
    probes byte-identical.
    """
    cls = model_class(model_name)
    configs = [dict(cls.default_hp())]
    probes = coordinate_probes(model_name, n_configs)
    configs.extend(dict(hp) for hp in probes)
    n_sampled = max(0, int(n_configs) - len(probes))
    sample_set = getattr(cls, "sample_hp_set", None)
    if callable(sample_set):
        configs.extend(dict(hp) for hp in sample_set(n_sampled, _CONFIG_SEED))
    else:
        for i in range(len(probes) + 1, len(probes) + n_sampled + 1):
            rng = np.random.default_rng(_config_seed(model_name, i))
            configs.append(cls.sample_hp(rng))
    for i, hp in enumerate(configs):
        if i:
            hp["seed"] = i                       # candidate i is seeded by its NOMINAL index
    if hp_override:
        configs = [{**hp, **hp_override} for hp in configs]
    return configs


def coordinate_tuning_plan(model_name: str, hp_override: dict | None = None) -> dict | None:
    """Return a normalized greedy coordinate-tuning plan for models that declare one."""
    cls = model_class(model_name)
    raw_stages = list(cls.coordinate_tuning_stages())
    if not raw_stages:
        return None

    fixed = dict(hp_override or {})
    base = cls.tuning_base_hp()
    base = dict(cls.default_hp() if base is None else base)
    base.update(fixed)

    stages = []
    for stage in raw_stages:
        name = stage.get("name")
        values = list(stage.get("values", []))
        if not name or name in fixed or not values:
            continue
        normalized = {"name": str(name), "values": values}
        if stage.get("prefer") is not None:
            normalized["prefer"] = str(stage["prefer"])
        if stage.get("relative_tolerance") is not None:
            normalized["relative_tolerance"] = float(stage["relative_tolerance"])
        stages.append(normalized)
    return {"strategy": "coordinate", "base": base, "stages": stages}


def effective_n_configs(model_name: str, n_configs: int, hp_override: dict | None = None) -> int:
    """Requested n_configs capped to the actual coordinate-plan length when applicable."""
    plan = coordinate_tuning_plan(model_name, hp_override=hp_override)
    if plan is None:
        return int(n_configs)
    extra = 0
    base = plan["base"]
    for stage in plan["stages"]:
        param = stage["name"]
        extra += sum(0 if _same_value(base.get(param), v) else 1 for v in stage["values"])
    return min(int(n_configs), int(extra))


def config_set_digest(model_name: str, n_configs: int, hp_override: dict | None = None) -> str:
    """Fingerprint the exact static candidate prefix used by a tuned run."""
    plan = coordinate_tuning_plan(model_name, hp_override=hp_override)
    if plan is not None:
        payload = {"n_configs": effective_n_configs(model_name, n_configs, hp_override), **plan}
    else:
        payload = static_configs(model_name, n_configs, hp_override=hp_override)
    raw = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _oof_score(
    model_name: str,
    otr: DatasetBundle,
    hp: dict,
    k: int,
    seed: int,
    repeats: int = 1,
    fold_workers: int = 1,
    selection_hp: dict | None = None,
    racing: dict | None = None,
):
    """k-fold OOF primary-metric score for one config, optionally over repeated fold assignments.

    Returns (score, mean_best_iter, fold_scores, raced): score is the pooled OOF value averaged
    over repeats; fold_scores holds one per-fold value per (repeat, fold) in a deterministic
    order. All candidates share the fold seeds (`seed + 1000*r`), so fold_scores pair
    positionally across candidates — the paired differences are what the ship-gate's SE is
    computed from.

    Tuning-speed levers (design/tuning-speed-roadmap.md, verified 2026-07-15):
    - `fold_workers > 1` (A1) runs the independent (repeat, fold) fits in a thread pool —
      per-fit results are deterministic given their seeds, so scores are bit-identical to the
      serial order; wall-clock only. Fits release the GIL (t-boost/lightgbm/xgboost).
    - `selection_hp` (B3) is the adapter's declared selection-only hp overlay (see
      `ModelAdapter.selection_hp`), merged into the fit hp here — and ONLY here, so deploy
      fits and the returned/cached candidate hp never carry it.
    - `racing` (A2) is `{"inc_fold1": float, "sd": float, "mult": float, "lower_better": bool}`:
      fold 1 of repeat 0 is evaluated first, and if its paired deficit vs the incumbent's
      fold 1 exceeds `mult * sd` the remaining folds are skipped — returns
      `(None, 0, [fold1], True)`. Replayed on the 18 logged campaign searches: at mult=2.0
      this saves 24–27% of eval time with zero selection changes.
    """
    hp = dict(hp)
    hp.setdefault("seed", int(seed))             # candidate hp may pin its own seed; else use the search seed
    fit_hp = {**hp, **(selection_hp or {})}      # selection-only overlay never leaves this function

    tasks = []                                    # (repeat, tr_idx, va_idx) in deterministic order
    for r in range(max(1, int(repeats))):
        for tr_idx, va_idx in kfold(otr, k, seed + 1000 * r):
            tasks.append((r, tr_idx, va_idx))

    def run_fold(task):
        r, tr_idx, va_idx = task
        m = build_model(model_name, hp=fit_hp).fit(otr.subset(tr_idx), otr.subset(va_idx))
        vab = otr.subset(va_idx)
        p = np.asarray(m.predict(vab.X, vab.exposure), dtype=float)
        return r, va_idx, p, float(primary_value(vab, p)), int(getattr(m, "best_iteration_", 0) or 0)

    results: list = [None] * len(tasks)
    start = 0
    if racing is not None and np.isfinite(racing.get("sd", np.nan)) and racing["sd"] > 0.0:
        results[0] = run_fold(tasks[0])
        start = 1
        fold1 = results[0][3]
        deficit = fold1 - racing["inc_fold1"] if racing["lower_better"] else racing["inc_fold1"] - fold1
        if deficit > racing["mult"] * racing["sd"]:
            return None, 0, [fold1], True

    remaining = list(range(start, len(tasks)))
    if fold_workers > 1 and len(remaining) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(int(fold_workers), len(remaining))) as pool:
            for i, res in zip(remaining, pool.map(run_fold, [tasks[i] for i in remaining])):
                results[i] = res
    else:
        for i in remaining:
            results[i] = run_fold(tasks[i])

    scores, fold_scores, iters = [], [], []
    oof_by_repeat: dict[int, np.ndarray] = {}
    for r, va_idx, p, fs, bi in results:
        if r not in oof_by_repeat:
            oof_by_repeat[r] = np.empty((len(otr.X), *p.shape[1:]), dtype=float)
        oof_by_repeat[r][va_idx] = p
        fold_scores.append(fs)
        if bi:
            iters.append(bi)
    for r in sorted(oof_by_repeat):
        scores.append(float(primary_value(otr, oof_by_repeat[r])))
    return float(np.mean(scores)), (int(np.mean(iters)) if iters else 0), fold_scores, False


def _config_eval_signature(
    dataset_hash: str,
    model_name: str,
    config_idx: int,
    hp: dict,
    candidate_proto: dict | None,
) -> str:
    proto = dict(candidate_proto or {})
    proto.update({"kind": "tuned_hpo_config", "config_idx": int(config_idx), "hp": hp})
    return cell_signature(dataset_hash, model_name, "tuned__hpo_config", int(config_idx), proto)


def _score_from_payload(payload: dict, lower_better: bool) -> float:
    score = payload.get("score")
    if score is None or payload.get("failed"):
        return np.inf if lower_better else -np.inf
    return float(score)


def _evaluate_config(
    model_name: str,
    otr: DatasetBundle,
    hp: dict,
    config_idx: int,
    k: int,
    seed: int,
    metric: str,
    direction: str,
    lower_better: bool,
    cache: ResultCache | None,
    dataset_hash: str | None,
    candidate_proto: dict | None,
    metadata: dict | None = None,
    repeats: int = 1,
    fold_workers: int = 1,
    selection_hp: dict | None = None,
    racing: dict | None = None,
) -> tuple[dict, bool]:
    """Evaluate or load one HPO candidate. Returns (payload, cache_hit).

    A raced candidate (see `_oof_score`) yields a score-less payload with `raced: True` and a
    single fold score; it is deliberately NOT cached — a full evaluation under any racing
    setting is identical to the non-racing one (same folds, same seeds), so completed payloads
    are shared across racing settings, while raced ones are cheap to re-derive.
    """
    sig = None
    if cache is not None and dataset_hash:
        sig = _config_eval_signature(dataset_hash, model_name, config_idx, hp, candidate_proto)
        cached = cache.get(sig, model=model_name)
        if cached is not None:
            return cached, True

    t0 = time.perf_counter()
    failed = ""
    raced = False
    fold_scores: list[float] = []
    try:
        score, best_iter, fold_scores, raced = _oof_score(
            model_name, otr, hp, k, seed, repeats=repeats,
            fold_workers=fold_workers, selection_hp=selection_hp, racing=racing,
        )
        if raced:
            score, best_iter = (np.inf if lower_better else -np.inf), 0
        else:
            score = float(score)
            best_iter = int(best_iter or 0)
            if not np.isfinite(score):
                failed = "non-finite score"
    except Exception as e:  # noqa: BLE001 - a bad config must not abort the search
        score, best_iter, fold_scores = (np.inf if lower_better else -np.inf), 0, []
        failed = f"{type(e).__name__}: {str(e)[:160]}"

    payload = {
        "kind": "tuned_hpo_config",
        "config_idx": int(config_idx),
        "hp": hp,
        "metric": metric,
        "direction": direction,
        "inner_folds": int(k),
        "inner_repeats": int(max(1, repeats)),
        "seed": int(seed),
        "score": float(score) if np.isfinite(score) and not failed and not raced else None,
        "best_iter": int(best_iter or 0),
        "fold_scores": [float(s) for s in fold_scores] if not failed else [],
        "failed": failed,
        "eval_time_s": float(time.perf_counter() - t0),
    }
    if raced:
        payload["raced"] = True
    if metadata:
        payload.update(metadata)
    # A FAILURE IS NOT A MEASUREMENT. The deploy path has always refused to cache one
    # ("never cache a failure (transient)", `runner.run_benchmark`); the candidate cache used to
    # persist them, so a fit that failed for an environment reason — an import that was not there
    # yet, an OOM, a killed worker — became a permanent "this config scored nothing" verdict that
    # every later selection cache-HIT. That is how a 2026-08-30 CatBoost-family recompute would
    # have re-shipped its own broken selections even after catboost was installed. Raced
    # payloads stay uncached for their own (cheap-to-re-derive) reason, unchanged.
    if cache is not None and sig is not None and not raced and not failed:
        cache.put(sig, payload, model=model_name)
    return payload, False


def _evaluation_order(model_name: str, configs: list[dict]) -> list[tuple[int, dict]]:
    """`(nominal_idx, hp)` pairs in the order candidates should be EVALUATED (roadmap A4-static).

    Racing (A2) discards a candidate whose fold-1 paired deficit vs the running incumbent exceeds
    the pooled-SD threshold, so it saves the most when a *strong* incumbent exists early — and it
    saves the most wall-clock when the candidates it discards are the *expensive* ones. Evaluating
    cheap-first delivers both: more candidates land per unit time up front, and the costly ones meet
    a well-established incumbent. Measured motivation (uslapseagent, 25-config basis): the 8
    `learning_rate=0.01` candidates were 46.8% of total search time and none of them won.

    Only the *order* changes. Each candidate keeps its nominal index — which is what seeds it and
    keys its cached evaluation — so caches stay shared with any other ordering, and the config set
    itself is untouched. The default anchor (index 0) always goes first: it is the ship-gate
    reference and the natural first incumbent, and `_select_config_static` never races it.

    Models that do not declare `hp_cost_hint` keep plain nominal order, so every other model's
    search is unchanged.
    """
    pairs = list(enumerate(configs))
    cost = getattr(model_class(model_name), "hp_cost_hint", None)
    if not callable(cost) or len(pairs) < 3:
        return pairs
    head, tail = pairs[:1], pairs[1:]
    try:
        keyed = sorted(tail, key=lambda p: float(cost(p[1])))   # stable: ties keep nominal order
    except Exception:                                          # a bad hint must never break search
        return pairs
    return head + keyed


def _nominal_trajectory(evaluations: list[dict], n: int, lower_better: bool) -> list[float]:
    """Cumulative-best over NOMINAL config index, so the budget curve means "the first N configs
    of the fixed basis" regardless of the order they were evaluated in (see `_evaluation_order`).

    Raced candidates carry no score and simply carry the running best forward — correct by
    construction, since racing only discards candidates already shown worse than the incumbent.
    """
    scored = sorted(
        ((int(e.get("config_idx", i)), e.get("score")) for i, e in enumerate(evaluations)),
        key=lambda p: p[0],
    )
    out: list[float] = []
    best: float | None = None
    for _, score in scored:
        if score is not None and np.isfinite(score) and _better(float(score), best, lower_better):
            best = float(score)
        out.append(float(best) if best is not None else float("nan"))
    return out[:n] if len(out) >= n else out + [out[-1] if out else float("nan")] * (n - len(out))


def _better(score: float, best_score: float | None, lower_better: bool) -> bool:
    return np.isfinite(score) and (
        best_score is None or (score < best_score if lower_better else score > best_score)
    )


def _stage_preferred(
    stage: dict,
    candidate_value,
    incumbent_value,
    score: float,
    incumbent_score: float | None,
    lower_better: bool,
) -> bool:
    """Optional near-tie rule for structural variance-reduction stages.

    Used sparingly: the candidate still has to be within a declared relative tolerance of the
    incumbent score, and the stage decides whether larger or smaller values win near-ties.
    """
    prefer = stage.get("prefer")
    tol = float(stage.get("relative_tolerance") or 0.0)
    if not prefer or tol <= 0.0 or incumbent_score is None or not np.isfinite(score):
        return False
    if lower_better:
        if score > float(incumbent_score) * (1.0 + tol):
            return False
    elif score < float(incumbent_score) * (1.0 - tol):
        return False
    try:
        c = float(candidate_value)
        i = float(incumbent_value)
    except Exception:
        c, i = candidate_value, incumbent_value
    if prefer == "larger":
        return c > i
    if prefer == "smaller":
        return c < i
    return False


def _stage_defended(
    stage: dict,
    candidate_value,
    incumbent_value,
    score: float,
    incumbent_score: float | None,
    lower_better: bool,
) -> bool:
    """Mirror of `_stage_preferred`: when the INCUMBENT is the stage-preferred value, a candidate
    must beat it by more than the declared tolerance to displace it — otherwise a preferred base
    (e.g. the 8-bag default) would lose every within-tolerance coin flip it was meant to win."""
    prefer = stage.get("prefer")
    tol = float(stage.get("relative_tolerance") or 0.0)
    if not prefer or tol <= 0.0 or incumbent_score is None or not np.isfinite(score):
        return False
    try:
        c = float(candidate_value)
        i = float(incumbent_value)
    except Exception:  # noqa: BLE001
        c, i = candidate_value, incumbent_value
    incumbent_is_preferred = (i > c) if prefer == "larger" else (i < c) if prefer == "smaller" else False
    if not incumbent_is_preferred:
        return False
    if lower_better:
        return score > float(incumbent_score) * (1.0 - tol)
    return score < float(incumbent_score) * (1.0 + tol)


def _paired_margin(inc_folds, cand_folds, lower_better: bool):
    """(mean, SE) of the paired per-fold improvement of candidate over incumbent, or (None, None)
    when the two evaluations don't share a usable fold grid (legacy/failed payloads)."""
    f0, fc = inc_folds or [], cand_folds or []
    if len(f0) != len(fc) or len(f0) < 2:
        return None, None
    diffs = np.asarray(f0, dtype=float) - np.asarray(fc, dtype=float)
    if not lower_better:
        diffs = -diffs                           # normalize: positive = candidate better
    return float(diffs.mean()), float(diffs.std(ddof=1) / np.sqrt(len(diffs)))


def _evaluation_by_config_idx(evaluations: list[dict], idx: int) -> dict:
    """The evaluation carrying CONFIG index `idx`.

    `evaluations` is appended in EVALUATION order, which stopped being config order once
    candidates are ordered cost-first (`_evaluation_order`) — so gate lookups must match on the
    recorded `config_idx` rather than index positionally. Positional fallback keeps legacy payloads
    (and the coordinate walk, which numbers its evaluations sequentially) behaving exactly as before.
    """
    for e in evaluations:
        if int(e.get("config_idx", -1)) == int(idx):
            return e
    return evaluations[idx] if 0 <= idx < len(evaluations) else {}


def _gate_se_mult(model_name: str, repeats: int, champion_hp: dict | None = None) -> float:
    """The adapter's ship-gate bar in paired-fold SEs (hsv 19; champion-aware since hsv 20).

    1.0 for every adapter that does not override `ship_gate_se_mult`, so their gates — and their
    cached tuned cells — are byte-untouched.

    `champion_hp` is the hp of the candidate the gate is about to judge. It is passed through to
    the hook when the hook accepts it and DROPPED SILENTLY when it does not, so an adapter still
    carrying the hsv-19 one-argument signature keeps its exact hsv-19 bar rather than falling all
    the way back to `_GATE_SE_MULT` through the except branch.
    """
    try:
        fn = getattr(model_class(model_name), "ship_gate_se_mult", None)
        if not callable(fn):
            return _GATE_SE_MULT
        try:
            v = float(fn(int(max(1, repeats)), champion_hp))
        except TypeError:      # hsv-19 one-argument hook
            v = float(fn(int(max(1, repeats))))
    except Exception:      # noqa: BLE001 - an unregistered name or a bad hook falls back to 1 SE,
        return _GATE_SE_MULT   # i.e. the historical gate; never break selection over the bar
    return v if np.isfinite(v) and v > 0.0 else _GATE_SE_MULT


def _regime_recheck(model_name: str, hp: dict | None) -> bool:
    """Does this candidate's SCOUT ranking come from a different regime than its DEPLOY? (hsv 21)

    An adapter may declare `selection_regime_recheck(hp) -> bool`. False (and absent) means the
    scout overlay ranks the candidate in the same regime it deploys in, which is the case for
    every knob on the board except one: t-boost's `max_depth`.

    The trap this exists for, measured on allstate_sev (new-territory battery §H). The scout
    overlay sets `_prune_during_hpo: False`, so a depth-lifted candidate is RANKED with no prune
    and therefore no `prune_table_budget` — there is no keep-set to cap. Every fit that ships it
    (finale, and all 9 deploys) prunes and takes the cap. On a wide set the two regimes are not
    the same model: the depth-6 probe ranks +6.3 t uncapped, and the cap then costs -0.0209 skill
    — more than the whole t-boost<->ccb2 gap the depth program is trying to close. A candidate ranked
    in a regime it will never deploy in can win the argmax on evidence that does not survive
    contact with the shipped shape, and it displaces the runner-up while doing it.

    Adapters that don't declare the hook are byte-untouched: no recheck, no extra evaluation.
    A raising hook is treated as "no recheck" — the recheck can only ever ADD an evaluation and
    demote a champion, so failing it open is the historical behavior, not a weakened gate.
    """
    try:
        fn = getattr(model_class(model_name), "selection_regime_recheck", None)
        return bool(fn(hp)) if callable(fn) else False
    except Exception:      # noqa: BLE001 - an unregistered name or a bad hook: no recheck
        return False


def _ship_gate(evaluations: list[dict], champion_idx: int, lower_better: bool,
               rule: str = "static_ship_gate_1se", default_idx: int = 0,
               se_mult: float = _GATE_SE_MULT) -> dict:
    """Decide whether the champion ships or the default (candidate `default_idx`) does.

    The champion must beat the default by more than `se_mult` standard errors of the paired
    per-fold score differences (folds are shared across candidates, so the comparison is matched).
    Anything less is within selection noise: shipping the default caps the downside of tuning at
    zero instead of paying the winner's curse on small/sparse portfolios.

    `se_mult` defaults to the historical `_GATE_SE_MULT` (1.0) and is raised only by adapters that
    declare `ship_gate_se_mult` — see that hook and t-boost's override (hsv 19).

    `default_idx` points at the default-anchor evaluation to gate against — 0 (the walk's opening
    anchor) unless a ship-fidelity finale re-evaluated the anchor (then the finale's index).
    """
    gate = {"rule": rule, "champion_idx": int(champion_idx), "shipped": "champion"}
    d0 = _evaluation_by_config_idx(evaluations, default_idx)
    if champion_idx == default_idx:
        # The anchor won the search outright. That is only a RESULT when the anchor was actually
        # scored. A failed evaluation scores -/+inf, so it can never beat the running incumbent —
        # which means "the anchor is the champion" is ALSO what you get when every candidate
        # (anchor included) blew up, and the old short-circuit reported that as a tuning verdict.
        # 2026-08-30: it shipped a CatBoost-family depth-6 default recipe as the TUNED hp on
        # 22 datasets after the search process could not `import catboost` (every one of the 26
        # evaluations carried "ModuleNotFoundError: No module named 'catboost'"; ohlsson_pp then
        # boarded at tuned skill -0.965). Ship the default here too, so the decision is recorded
        # as the failure it is — and `select_config` refuses the cell outright when NOTHING scored.
        if d0.get("failed") or d0.get("score") is None:
            gate.update(shipped="default", reason="default_failed_ship_default",
                        default_score=d0.get("score"))
            return gate
        gate["reason"] = "champion_is_default"
        return gate
    ch = _evaluation_by_config_idx(evaluations, champion_idx)
    gate["default_score"], gate["champion_score"] = d0.get("score"), ch.get("score")
    if d0.get("failed") or d0.get("score") is None:
        # A champion cannot be gate-validated without a working default eval, and shipping it
        # ungated is how brautocoll deployed a winner's-curse champion 14% behind the default
        # (2026-07-22: the default anchor's eval crashed on a single-class inner-fold slice —
        # t-boost "logistic: no positive class under w" — while its full-train deploy fits
        # fine). Conservative rule: ship the default. Its deploy either works, or fails as a
        # VISIBLE failed cell — never a silent regression.
        gate.update(shipped="default", reason="default_failed_ship_default")
        return gate
    margin, se = _paired_margin(d0.get("fold_scores"), ch.get("fold_scores"), lower_better)
    if margin is None:
        gate["reason"] = "gate_unavailable"      # legacy payloads without fold scores: ship as before
        return gate
    gate.update(gate_margin=margin, gate_se=se, gate_se_mult=float(se_mult),
                n_paired_folds=int(len(d0.get("fold_scores") or [])))
    if np.isfinite(margin) and margin > se_mult * se:
        gate["reason"] = "beats_default_by_1se"
    else:
        gate.update(shipped="default", reason="within_1se_of_default")
    return gate


def _adaptive_pool_band(se_mult: float = _GATE_SE_MULT) -> float:
    """The adaptive pooled finale's near-tie band, in paired-fold SEs (hsv 21, 2026-08-31).

    `max(_ADAPTIVE_POOL_SE_MULT, se_mult)` — the historical 2 SE, RAISED to the ship bar whenever
    that bar is higher than 2. The window is "how close is this decision to the line?", so the
    line it is measured against has to be the line that will actually judge the champion.

    hsv 18 set the band at a constant 2.0 when every bar on the board was 1.0 (2x the bar: one SE
    of slack either side of the decision point). hsv 19 raised the bar to `max(1, inner_repeats)`
    and hsv 20 gave a DEPTH-LIFTED champion a 3.0 floor, and neither rescaled the band — so a
    depth champion sitting at 2.4 SE was "decided" for pooling purposes while its own gate needed
    3.0 SE and therefore refused it. The trigger declined to buy evidence for the one decision on
    the board that could not be made without it. Measured on freMTPL2freq: the depth-6 champion's
    single tune split reads margin/SE = 2.379 against a 3.0 bar (ships the default, no pooling);
    the SAME protocol's pooled evidence over 3 tune splits reads +0.00100453 / 0.000319 = 3.146
    over 9 paired folds and ships it — and the shipped cell then measures +0.001408 +/- 0.000217
    skill (t = +6.5, 9-0) over the 9 outer splits it deploys on.

    A FLOOR, not a replacement: at the 1-SE bar the band stays 2.0, so every cell whose champion
    is not depth-lifted keeps its hsv-18/19/20 pooling decision exactly (a board-wide replay of
    the campaign's own selection records flips exactly one cell, freMTPL2freq). It only ever
    WIDENS the band, i.e. it can only buy more evidence, never less; and the fremotor_payfreq
    cautionary champion (2.27 SE, the ship that measured -0.00103) is pulled INTO the band by
    this and still refused on the pooled evidence.
    """
    try:
        v = float(se_mult)
    except (TypeError, ValueError):
        return _ADAPTIVE_POOL_SE_MULT
    if not (np.isfinite(v) and v > 0.0):
        return _ADAPTIVE_POOL_SE_MULT
    return max(_ADAPTIVE_POOL_SE_MULT, v)


def _pooled_finale_wanted(gate: dict, mode: str, se_mult: float = _GATE_SE_MULT) -> bool:
    """Should the protocol-v3 pooled finale actually be SPENT on this cell? (hsv 18, 2026-08-23)

    `mode="always"` is protocol v3 as shipped at hsv 16: small tune sets (inner_repeats>1) pool
    unconditionally, because there the whole tune split is small enough that a single split's
    finale can be split-specific bias end to end (freclaimdam picked the same cursed champion on
    3/3 tune splits; the anchor beat it on outer test by 1.8%).

    `mode="adaptive"` is the large-set extension. The 2026-08-04 draft that extended pooling to
    every dataset size unconditionally was reverted the same day, and the note it left at
    `eval/runner._tuned_once` set the re-open condition verbatim:

        "Re-extend only with a real large-set counterexample."

    norauto is that counterexample, measured end to end (edge forensics, 9 splits x 3 seeds):
    its inner CV ranked max_interaction_order<=2 above the anchor by ~0.00014 deviance
    (~0.00034 skill) over 28 evaluations; `static_ship_gate_1se` then read gate_margin 5.81e-5
    against gate_se 6.57e-5 on n_paired_folds=3 and shipped the default as "within_1se_of_default"
    — and the ORDER the gate discarded was real: the order-1 arm scores +0.000126 +/- 0.000050
    on OUTER test (t=2.53, 8/9 splits) and halves the cell's prediction seed SD. n=183,999 rows,
    so `resolve_inner_repeats("auto")` returns 1 and hsv 16's pooling never applied. The gate was
    honest and symmetric; it simply had no resolution at 3 paired folds.

    But the 2026-08-04 revert is also still correct about what it actually measured: on large sets
    where the single-split gate is DECISIVE, it was right (homesite_conv's champion beats the
    default 9/9 at matched adapter versions). So the extension is spent only where the decision is
    genuinely close — |margin| within `_adaptive_pool_band(se_mult)` SEs of the BAR THAT WILL
    JUDGE THIS CHAMPION (hsv 21; `_ADAPTIVE_POOL_SE_MULT` = 2.0 unless the bar is higher),
    two-sided:

      * margin just under +1 SE  -> the norauto case: ships default today, may be a real champion;
      * margin just over  +1 SE  -> ships champion today on one split's evidence, which is the
                                    winner's-curse direction the pooled evidence exists to check;
      * |margin| well beyond     -> decided; pooling would only spend 4 finale evaluations to
                                    confirm what one split already resolved.

    "Well beyond" is relative to the bar: a 2.4-SE margin is decided under a 1-SE gate and a
    coin-flip under the 3-SE structural floor a depth-lifted champion draws. `se_mult` is the
    bar `_ship_gate` is about to be handed, so the band tracks it automatically as the gate
    evolves — see `_adaptive_pool_band` for the freMTPL2freq measurement that forced this.

    Not resolvable at all (no margin/SE — legacy or failed payloads) pools, which is the
    conservative side: more evidence for a gate that currently has none.
    """
    if str(mode) != "adaptive":
        return True
    margin, se = gate.get("gate_margin"), gate.get("gate_se")
    if margin is None or se is None:
        return True
    margin, se = float(margin), float(se)
    if not (np.isfinite(margin) and np.isfinite(se)):
        return True
    return abs(margin) <= _adaptive_pool_band(se_mult) * se


# TOP-K FINALE RE-RACE (hpo_search_version 23, 2026-09-01) — the two tuned constants.
#
# Measured, not chosen — the av44 top-K re-race battery.
# 10 datasets, 219 deploy fits + 57 finale evaluations, every candidate deployed on all 9 outer
# splits exactly as `runner.run_benchmark` deploys the champion, then band x K swept as a pure
# function of those numbers.
#
#   K = 4 IS A HARD BOUNDARY, and it is the only guard that binds.
#       K <= 4 -> 1 WIN (eusavingul +0.0049, t=15.5) / 9 null / 0 HARM, net +0.0049 at EVERY band.
#       K >= 5 -> pg16_payfreq's pick turns -0.0009 and the net falls to +0.0040; unguarded
#                 (K = inf) the eusavingul win itself dissolves into the noise.
#       This is the documented max-of-N / winner's-curse boundary and it sits at 4. The
#       counterexample is pinned in `tests/test_hpo_finale_rerace.py`.
#
#   THE BAND IS IRRELEVANT and is spent only as a cheap sanity rail. Every candidate that ever
#       changes the pick sits within 0.75 SE of the argmax anyway, so 0.25 and 3.0 give identical
#       results at K <= 4. 1.0 SE is a nominal value that costs nothing.
#
# WHAT THE SE IS, stated because it is NOT the same statistic the offline sweep used. The sweep
# scored `se_behind` against the cached cell's `gate_se` (a pooled default-vs-champion SE that
# does not exist yet at the point the re-race runs). In-engine the band is measured with
# `_paired_margin` between the argmax's and the candidate's own SCOUT fold vectors — the
# coin-flip statistic the selection census reports (|t| < 1 SE on 30 of 40 hsv-22 cells). The
# substitution is safe for one structural reason, not by assumption: admission walks candidates in
# SCOUT-RANK order and STOPS at the first refusal, so the admitted set is always ranks 1..m, and
# `K` truncates that same rank order. Whatever the SE scale, the contender set is a subset of the
# top-K scout ranks — which is exactly the set the sweep found K=4 safe on.
_RERACE_K = 4
_RERACE_BAND = 1.0


def _model_rerace_spec(model_name: str) -> tuple[int, float] | None:
    """The adapter's top-K finale re-race settings, or None for "do not re-race" (the default).

    OPT-IN, and that is load-bearing rather than stylistic. `_select_config_static` is shared by
    lightgbm / xgboost / catboost / ebm as well as t-boost, and the re-race is model-generic BY
    DESIGN — any adapter that declares a scout/finale fidelity split can turn it on. But turning it
    on for a model whose `hpo_search_version` is not bumped in the same breath would leave that
    model's cached tuned cells keyed identically to cells computed under a DIFFERENT selection
    protocol, which is the one failure a version-keyed cache cannot detect. So an adapter opts in
    explicitly and bumps its own hsv, and every adapter that does not is byte-untouched.

    (Belt and braces: the four other tunable adapters declare no `selection_finale_hp`, so
    `finale_hp == sel_hp` and they never enter the finale block at all. The hook makes the
    intention auditable instead of relying on that coincidence holding forever.)

    A missing, empty or raising hook means no re-race — failing OPEN to the pre-hsv-23 behavior,
    never into an unmeasured one.
    """
    fn = getattr(model_class(model_name), "finale_rerace_spec", None)
    if not callable(fn):
        return None
    try:
        spec = fn() or {}
    except Exception:
        return None
    if not spec:
        return None
    try:
        rr_k = int(spec.get("k", _RERACE_K))
        band = float(spec.get("band", _RERACE_BAND))
    except (TypeError, ValueError):
        return None
    if rr_k <= 1 or not np.isfinite(band) or band < 0.0:
        return None
    return rr_k, band


def _rerace_key(hp: dict | None) -> tuple:
    """Candidate identity for the re-race, ignoring the nominal DRAW index.

    `static_configs` stamps `hp["seed"] = i` on every candidate, so two draws that are the same
    model differ in that one key. Re-racing both would spend a finale evaluation to compare a
    config against itself, so the dedupe ignores it — the same knob-space identity the offline
    battery used (`cand_arm.basis_candidates`).
    """
    return tuple(sorted((str(name), repr(value))
                        for name, value in (hp or {}).items() if name != "seed"))


def _rerace_contenders(evaluations, best_idx, taken_hp, default_hp, n_basis, rr_k, band,
                       lower_better):
    """The scout runners-up the finale should re-race, in scout-rank order.

    Returns `[(hp, scout_config_idx, se_behind), ...]`, at most `rr_k - len(taken_hp)` entries.

    RULES, each one deliberate:
      * BASIS ONLY. Finale re-evaluations already in `evaluations` are scored under a different
        fidelity overlay and (on the pooled splits) a different fold count; mixing them into the
        ranking is the artifact `cand_arm.basis_candidates` documents — on eusavingul it reads the
        argmax's 0.19-SE lead as a 12-SE one.
      * RANK-CONTIGUOUS. Candidates are walked best-score-first and admission STOPS at the first
        refusal, so the contender set is always scout ranks 1..m. See `_RERACE_K` for why that,
        and not the SE scale, is what makes K the measured guard.
      * THE ANCHOR IS NEVER A CONTENDER. `configs[0]` is the ship gate's reference and is already
        evaluated at finale fidelity; re-racing it would enter it into its own comparison.
      * K COUNTS EVERY FINALE CONTENDER, the hsv-21 regime-recheck candidate included. K is a bound
        on the max-of-N the ship gate has to defend against, and a demotion contender is one more
        draw in exactly that sense. It is also why `taken_hp` is passed in rather than assumed.
      * RACED AND FAILED CANDIDATES ARE INELIGIBLE. A raced payload holds one fold and no score;
        there is nothing to band against, and this cannot recover it (the same honest limit the
        regime recheck records).
    """
    room = int(rr_k) - len(taken_hp)
    if room <= 0:
        return []
    arg = _evaluation_by_config_idx(evaluations, best_idx)
    arg_folds = list(arg.get("fold_scores") or [])
    if len(arg_folds) < 2:
        return []
    seen = {_rerace_key(hp) for hp in taken_hp}
    seen.add(_rerace_key(default_hp))
    basis = [
        e for e in evaluations
        if 0 <= int(e.get("config_idx", -1)) < int(n_basis)
        and not e.get("failed") and not e.get("raced")
        and e.get("score") is not None and np.isfinite(float(e.get("score")))
        and len(e.get("fold_scores") or []) == len(arg_folds)
    ]
    basis.sort(key=lambda e: float(e["score"]), reverse=not lower_better)
    out = []
    for e in basis:
        if len(out) >= room:
            break
        key = _rerace_key(e.get("hp"))
        if key in seen:
            continue                        # the argmax itself, the anchor, or a duplicate draw
        margin, se = _paired_margin(arg_folds, e.get("fold_scores"), lower_better)
        if margin is None or se is None:
            break
        deficit = -float(margin)                  # >= 0: how far this candidate trails the argmax
        if se > 0.0:
            if deficit > float(band) * se:
                break                             # rank-contiguous: first refusal ends admission
            se_behind = deficit / se
        else:
            if deficit > 0.0:
                break                             # zero spread and strictly worse -> not a tie
            se_behind = 0.0
        seen.add(key)
        out.append((dict(e["hp"]), int(e["config_idx"]), float(se_behind)))
    return out


def _resolve_finale_otr(entry):
    """Extra finale tune-split bundles may arrive as callables so the adaptive path can decline to
    materialize them (a large set's outer-train is not free). Plain bundles still work unchanged."""
    return entry() if callable(entry) else entry


class _RacingPool:
    """A2 racing bookkeeping: pooled SD of per-fold paired differences between completed
    candidates and the incumbent they were compared against. Verified by replay on the 18
    logged campaign searches (design/tuning-speed-roadmap.md): at mult=2.0, 24–27% of eval
    time saved with zero selection changes and the true winner never discarded."""

    WARMUP_DIFFS = 6  # trust the SD only once two full 3-fold candidates have paired

    def __init__(self, mult: float | None, lower_better: bool):
        self.mult = None if not mult else float(mult)
        self.lower_better = lower_better
        self.diffs: list[float] = []

    def context(self, inc_folds) -> dict | None:
        """The racing dict for `_oof_score`, or None while unarmed/warming up."""
        if self.mult is None or not inc_folds or len(self.diffs) < self.WARMUP_DIFFS:
            return None
        sd = float(np.std(self.diffs, ddof=1))
        if not np.isfinite(sd) or sd <= 0.0:
            return None
        return {"inc_fold1": float(inc_folds[0]), "sd": sd,
                "mult": self.mult, "lower_better": self.lower_better}

    def add(self, inc_folds, payload: dict) -> None:
        """Pool a COMPLETED candidate's paired per-fold diffs vs the incumbent it raced."""
        folds = payload.get("fold_scores") or []
        if self.mult is None or payload.get("raced") or payload.get("failed"):
            return
        if inc_folds and len(folds) == len(inc_folds):
            d = np.asarray(folds, dtype=float) - np.asarray(inc_folds, dtype=float)
            if not self.lower_better:
                d = -d
            self.diffs.extend(float(v) for v in d)


def _model_selection_hp(model_name: str) -> dict:
    """The adapter's selection-only hp overlay (B3); {} for adapters that don't declare one."""
    fn = getattr(model_class(model_name), "selection_hp", None)
    return dict(fn() or {}) if callable(fn) else {}


def _model_selection_finale_hp(model_name: str) -> dict:
    """The adapter's SHIP-FIDELITY overlay for the coordinate walk's finale re-evaluations.

    When an adapter's `selection_hp` reduces fit fidelity (fewer bags, no pruning — the
    scout regime), it declares `selection_finale_hp` with only the fidelity-NEUTRAL keys
    (e.g. the B3 ES ratio): the walk then re-evaluates the final incumbent and the default
    anchor under this overlay, and the ship gate + deployed iteration count come from those
    matched full-fidelity evals rather than the scout scores. Adapters that don't declare it
    get `selection_hp` back, which disables the finale (nothing to re-evaluate)."""
    fn = getattr(model_class(model_name), "selection_finale_hp", None)
    if callable(fn):
        return dict(fn() or {})
    return _model_selection_hp(model_name)


def _select_config_static(
    model_name,
    otr: DatasetBundle,
    n_configs,
    k,
    seed,
    time_budget=None,
    hp_override=None,
    cache: ResultCache | None = None,
    dataset_hash: str | None = None,
    candidate_proto: dict | None = None,
    repeats: int = 1,
    fold_workers: int = 1,
    racing_se: float | None = None,
    extra_finale_otrs: list | None = None,
    pooled_finale_mode: str = "always",
):
    direction = METRICS[otr.task.primary_metric].direction
    lower_better = direction != "higher"
    configs = static_configs(model_name, n_configs, hp_override=hp_override)
    # Coordinate probes (hsv 18) occupy nominal indices 1..n_probes. They are exempt from A2
    # racing exactly as the anchor is: the block exists so that each curated (dim, value) gets a
    # CLEAN look at the gate, and a fold-1 racing discard would deny it that look for the same
    # reason the joint basis did — noise attached to a single evaluation. They are also the cheap
    # candidates (5 of t-boost's 6 keep the base learning rate), so the exemption costs little,
    # and it can only move the selection TOWARD the no-racing reference behavior.
    n_probes = len(coordinate_probes(model_name, n_configs))
    sel_hp = _model_selection_hp(model_name)
    racing_pool = _RacingPool(racing_se, lower_better)
    best_hp, best_iter, best_score, best_idx = configs[0], 0, None, 0
    best_folds = None
    # The best candidate whose scout ranking is regime-FAITHFUL (hsv 21). Only tracked so that a
    # regime-mismatched argmax can be re-raced against it at finale fidelity; `None` and the
    # recheck is skipped. The anchor qualifies, and starts as the running answer, so the field is
    # never empty on a search where anything scored.
    alt_hp, alt_score, alt_idx = None, None, None
    traj, evaluations = [], []
    t0 = time.perf_counter()
    for config_idx, hp in _evaluation_order(model_name, configs):
        payload, cached = _evaluate_config(
            model_name, otr, hp, config_idx, k, seed, otr.task.primary_metric, direction,
            lower_better, cache, dataset_hash, candidate_proto, repeats=repeats,
            fold_workers=fold_workers, selection_hp=sel_hp,
            racing=racing_pool.context(best_folds) if config_idx > n_probes else None,
        )
        score = _score_from_payload(payload, lower_better)
        it = int(payload.get("best_iter") or 0)
        racing_pool.add(best_folds, payload)
        if _better(score, best_score, lower_better):
            best_hp, best_iter, best_score, best_idx = hp, it, score, config_idx
            best_folds = payload.get("fold_scores")
        elif best_folds is None and not payload.get("raced") and not payload.get("failed"):
            best_folds = payload.get("fold_scores")
        if not payload.get("failed") and not _regime_recheck(model_name, hp) \
                and _better(score, alt_score, lower_better):
            alt_hp, alt_score, alt_idx = hp, score, config_idx
        traj.append(float(best_score) if best_score is not None else float("nan"))
        evaluations.append({**payload, "cached": bool(cached), "best_so_far": traj[-1]})
        if time_budget and (time.perf_counter() - t0) > time_budget:
            break
    # The budget curve answers "what would the first N configs of the fixed basis have given?", so
    # it is rebuilt over NOMINAL config order — invariant to the evaluation order above (and to
    # racing, which only ever discards candidates already known worse than the running incumbent).
    # Identical to the old in-place `traj` whenever evaluation order is nominal and nothing raced.
    traj = _nominal_trajectory(evaluations, len(traj), lower_better)

    # Ship-fidelity finale (mirrors `_select_config_coordinate`): when the adapter's selection
    # overlay reduces fit fidelity (t-boost's 1-bag prune-free scout), the scout scores above are
    # ranking evidence only. Re-evaluate the scout champion and the default anchor under the
    # adapter's finale overlay (matched fidelity on both sides) and take the ship gate + deployed
    # iteration count from those instead. A no-op for every other model: `finale_hp == sel_hp` skips
    # this block entirely, so their scores/evaluations/selection are unchanged.
    finale_hp = _model_selection_finale_hp(model_name)
    champion_idx, default_gate_idx = best_idx, 0
    recheck_note = None
    rerace_note = None
    scored: list = []
    if finale_hp != sel_hp:
        next_idx = len(configs)
        payload, cached = _evaluate_config(
            model_name, otr, configs[0], next_idx, k, seed, otr.task.primary_metric, direction,
            lower_better, cache, dataset_hash, candidate_proto,
            metadata={"stage": "finale", "param": "default_anchor", "value": None},
            repeats=repeats, fold_workers=fold_workers, selection_hp=finale_hp,
        )
        d_score = _score_from_payload(payload, lower_better)
        if _better(d_score, best_score, lower_better):
            best_score = d_score
        traj.append(float(best_score) if best_score is not None else float("nan"))
        evaluations.append({**payload, "cached": bool(cached), "best_so_far": traj[-1]})
        default_gate_idx = next_idx
        next_idx += 1
        if best_hp == configs[0]:
            champion_idx = default_gate_idx      # scout picked the default: one eval serves both
            best_iter = int(payload.get("best_iter") or 0)
        else:
            # REGIME RECHECK (hsv 21, 2026-08-31). The scout's argmax may have been ranked in a
            # regime it will never deploy in — for t-boost, a depth-lifted candidate ranked with
            # the prune (and therefore the >=3-way TABLE cap) switched OFF. See `_regime_recheck`
            # for the allstate_sev measurement: uncapped rank t = +6.3, capped deploy -0.0209.
            #
            # The fix is to make the argmax itself regime-honest, not merely to let the gate
            # refuse afterwards: a mismatched candidate that wins the top slot and then loses the
            # gate takes the runner-up's shot with it, and the cell ships the anchor rather than
            # the best candidate that was actually judged in its own regime. So the mismatched
            # champion and the best regime-FAITHFUL candidate are re-raced HERE, both at finale
            # fidelity (identical overlay on both sides, so the two scores are comparable), and
            # the winner is the champion the ship gate is handed.
            #
            # COST: exactly ONE extra finale evaluation, and only on a cell whose scout argmax is
            # regime-mismatched (on the av42 board: only where a depth probe takes the top slot).
            # It is skipped when the faithful runner-up IS the default anchor, because the gate
            # already runs that comparison — with a bar on top of it — from the anchor's own
            # finale evaluation.
            #
            # WHAT IT DOES NOT FIX, stated rather than hidden: only the BEST regime-faithful
            # candidate is re-raced, and it is chosen on the scout's own (uncapped-incumbent)
            # ranking. Where A2 racing is armed (n >= racing_min_rows) a faithful candidate can
            # in principle have been discarded on fold 1 against a mismatched incumbent, and this
            # cannot recover it. Fixing that would mean either racing everything to completion or
            # ranking depth capped inside the scout, both of which cost the whole board; this
            # costs one evaluation on the cells that have the problem.
            contenders = [(best_hp, "champion")]
            if (
                _regime_recheck(model_name, best_hp)
                and alt_hp is not None
                and alt_hp != best_hp
                and alt_hp != configs[0]
            ):
                contenders.append((alt_hp, "regime_recheck"))
            # TOP-K FINALE RE-RACE (hsv 23, 2026-09-01). The scout ranks candidates at REDUCED
            # fidelity — for t-boost the selection overlay pins n_bags=1 — and on 30 of the 40
            # hsv-22 cells the gap between scout rank 1 and rank 2 is under one paired-fold SE.
            # On 12 of those the coin-flip winner went on to SHIP, and only that one candidate was
            # ever measured at the fidelity it deploys in. eusavingul is the hard case: rank 2
            # (max_depth 8, hurdle 0.0) trails the argmax by 0.19 SE at scout fidelity, beats it by
            # t=3.4 the moment bagging is switched on, and beats it by +0.00435 skill on 18 of 18
            # outer deploys (av44 selection forensic, §1).
            #
            # So the runners-up the scout could not separate from the argmax are re-scored at
            # FINALE fidelity too, and the winner of that — not the scout's argmax — is the
            # candidate handed to the ship gate. `_RERACE_K` / `_rerace_contenders` carry the
            # measured record and the admission rules; `_model_rerace_spec` is why no other model
            # moves.
            #
            # It composes with the hsv-21 regime recheck rather than replacing it: the recheck's
            # faithful contender is admitted on its own (mismatch) grounds even when the band
            # refuses it, so allstate_sev's demotion still happens, and it counts against K.
            rerace_spec = _model_rerace_spec(model_name)
            rerace_admitted = []
            rr_k = rr_band = None
            if rerace_spec is not None:
                rr_k, rr_band = rerace_spec
                rerace_admitted = _rerace_contenders(
                    evaluations, best_idx, [hp for hp, _ in contenders], configs[0],
                    len(configs), rr_k, rr_band, lower_better,
                )
                contenders.extend((hp, "rerace") for hp, _, _ in rerace_admitted)
            scored = []
            for cfg_hp, tag in contenders:
                payload, cached = _evaluate_config(
                    model_name, otr, cfg_hp, next_idx, k, seed, otr.task.primary_metric, direction,
                    lower_better, cache, dataset_hash, candidate_proto,
                    metadata={"stage": "finale", "param": tag, "value": None},
                    repeats=repeats, fold_workers=fold_workers, selection_hp=finale_hp,
                )
                c_score = _score_from_payload(payload, lower_better)
                if _better(c_score, best_score, lower_better):
                    best_score = c_score
                traj.append(float(best_score) if best_score is not None else float("nan"))
                evaluations.append({**payload, "cached": bool(cached), "best_so_far": traj[-1]})
                scored.append((c_score, next_idx, cfg_hp, tag, payload))
                next_idx += 1
            # The scout champion holds the slot on a tie: the recheck is a demotion mechanism and
            # must not move a cell on noise.
            win = scored[0]
            for cand in scored[1:]:
                if _better(cand[0], win[0], lower_better):
                    win = cand
            champion_idx, best_hp = win[1], win[2]
            best_iter = int(win[4].get("best_iter") or 0)
            faithful = next((c for c in scored if c[3] == "regime_recheck"), None)
            if faithful is not None:
                recheck_note = {
                    "scout_champion_idx": int(best_idx),
                    "scout_champion_finale_score": scored[0][0],
                    "faithful_idx": int(alt_idx),
                    "faithful_finale_score": faithful[0],
                    "demoted": bool(win[3] == "regime_recheck"),
                }
            if rerace_admitted:
                # On the record so a cell can say whether the re-race earned its evaluations —
                # the same audit contract `regime_recheck` keeps. `pooled_*` is filled in below
                # wherever the cell actually buys pooled evidence.
                rerace_note = {
                    "k": int(rr_k),
                    "band": float(rr_band),
                    "scout_champion_idx": int(best_idx),
                    "n_contenders": len(scored),
                    "admitted": [{"scout_idx": int(idx), "se_behind": float(sb)}
                                 for _, idx, sb in rerace_admitted],
                    "single_split_pick_idx": int(win[1]),
                    "single_split_changed": bool(win[3] == "rerace"),
                }

    # Protocol v3 pooled finale (hsv 16, 2026-07-23): a single tune split's finale can validate a
    # champion whose inner-CV advantage is split-specific bias, not signal — measured on the
    # small-N curse sets (freclaimdam picked the SAME cursed champion on 3/3 tune splits; the
    # anchor beat it on outer test by 1.8%; credit_g drew a third-generation cursed champion).
    # When the caller supplies extra tune-split bundles, the default anchor and the champion are
    # re-evaluated on each at finale fidelity (repeats=1 — pooling supplies the replication), and
    # the ship gate runs on the POOLED paired fold vectors. Fold seeds are shared per split-eval,
    # so champion/default pairs stay positionally aligned within each split and concatenation
    # preserves the pairing. Since hsv 23 the pooled slots also carry the top-K re-race's other
    # finale contenders, so the pooled evidence DECIDES the re-race as well as the gate — see the
    # block itself.
    #
    # hsv 18 (2026-08-23) adds the ADAPTIVE mode for large sets: the single-split gate is computed
    # FIRST and the pooled evidence is bought only when that decision lands within the near-tie
    # band — +/-2 SE at hsv 18, and since hsv 21 +/- max(2, the bar in force) SE, so that a
    # champion facing the 3-SE structural floor is measured against the line it will be held to
    # rather than against a line no longer on the board (`_adaptive_pool_band`). See
    # `_pooled_finale_wanted` for the norauto counterexample that re-opened the extension and for
    # why the 2026-08-04 revert stays respected outside the near-tie band. Computing the gate
    # before pooling is byte-neutral for `mode="always"`: `_ship_gate` reads only evaluations that
    # already exist at this point and consumes no candidate index, so every small-set cell keeps
    # the indices, seeds and payloads it had at hsv 16.
    # Ship-gate bar in paired-fold SEs (hsv 19; champion-aware since hsv 20). 1.0 unless the
    # adapter declares otherwise, so every model that doesn't override `ship_gate_se_mult` keeps
    # its exact historical gate. Resolved HERE rather than before the loop because the bar may
    # depend on the candidate under test (`best_hp` — the scout champion, which is what both the
    # single-split gate below and the pooled branch further down judge). Moving the call is
    # byte-neutral for every hsv-19 adapter: the hook is a pure function and nothing between the
    # old call site and this one reads `se_mult`.
    se_mult = _gate_se_mult(model_name, repeats, best_hp)
    selection = _ship_gate(evaluations, champion_idx, lower_better, default_idx=default_gate_idx,
                           se_mult=se_mult)
    if recheck_note is not None:
        # On the record so a cell can say whether the recheck changed the champion — the only way
        # to audit whether the extra finale evaluation is earning itself.
        selection["regime_recheck"] = recheck_note
    if rerace_note is not None:
        selection["finale_rerace"] = rerace_note
    pooled_note = None
    if (
        extra_finale_otrs
        and finale_hp != sel_hp
        and champion_idx != default_gate_idx
        and _pooled_finale_wanted(selection, pooled_finale_mode, se_mult)
    ):
        # hsv 23: where the cell buys pooled evidence, the pooled finale also DECIDES the top-K
        # re-race — every contender is carried onto the extra tune splits, not just the
        # single-split winner. That is the fidelity the rule was measured at (the offline battery
        # scored candidates on tune splits 0/3/6 and weighted by fold count), and it is the
        # methodologically consistent one: a max-of-N over four candidates settled on ONE split's
        # finale is precisely the split-specific-bias shape protocol v3 exists to defend against.
        # It is not a hypothetical — replaying the battery with the pick made on split 0 alone
        # mis-picks ohlsson (scout rank 3 wins that split; pooled correctly keeps rank 1), which
        # deploys 0.0045 against the argmax's 0.0047.
        #
        # HONEST GAP: on a cell where `_pooled_finale_wanted` declines the extras — adaptive mode,
        # decisive single-split gate — the re-race is decided on the primary split alone. Buying
        # pooled evidence there anyway would spend it on exactly the decisions one split already
        # resolved, which is the trade `_adaptive_pool_band` is tuned for.
        pool_slots = [(configs[0], "pool_default", default_gate_idx)]
        if rerace_note is not None and scored:
            pool_slots.extend(
                (c_hp, "pool_champion" if c_idx == champion_idx else "pool_rerace", c_idx)
                for _, c_idx, c_hp, _, _ in scored
            )
        else:
            pool_slots.append((best_hp, "pool_champion", champion_idx))
        pooled = {slot_idx: list(_evaluation_by_config_idx(evaluations, slot_idx)
                                 .get("fold_scores") or [])
                  for _, _, slot_idx in pool_slots}
        extra_failed = any(not folds for folds in pooled.values())
        n_extra = 0
        for j, eotr_entry in enumerate(extra_finale_otrs):
            eotr = _resolve_finale_otr(eotr_entry)
            got = []
            for cfg_hp, tag, slot_idx in pool_slots:
                payload, cached = _evaluate_config(
                    model_name, eotr, cfg_hp, next_idx, k, seed + 7001 * (j + 1),
                    otr.task.primary_metric, direction, lower_better, cache, dataset_hash,
                    candidate_proto,
                    metadata={"stage": "finale", "param": tag, "value": j},
                    repeats=1, fold_workers=fold_workers, selection_hp=finale_hp,
                )
                traj.append(float(best_score) if best_score is not None else float("nan"))
                evaluations.append({**payload, "cached": bool(cached), "best_so_far": traj[-1]})
                got.append((slot_idx, payload))
                next_idx += 1
            if any(p.get("failed") or p.get("score") is None for _, p in got):
                extra_failed = True
            else:
                for slot_idx, payload in got:
                    pooled[slot_idx].extend(payload.get("fold_scores") or [])
                n_extra += 1
        if rerace_note is not None and len(pool_slots) > 2 and not extra_failed:
            # Pooled score = the MEAN of the concatenated fold vector, which reproduces the
            # battery's fold-count weighting exactly (the primary split contributes
            # inner_folds x inner_repeats folds, each extra split inner_folds).
            # The single-split winner holds the slot on a tie, for the same reason the scout
            # champion does above: a re-race is a correction mechanism, not a coin flip.
            pooled_pick = champion_idx
            pooled_best = float(np.mean(pooled[champion_idx]))
            for _, _, slot_idx in pool_slots[1:]:
                if slot_idx == champion_idx:
                    continue
                cand_mean = float(np.mean(pooled[slot_idx]))
                if _better(cand_mean, pooled_best, lower_better):
                    pooled_pick, pooled_best = slot_idx, cand_mean
            # The anchor's slot is recorded too (it is `pool_slots[0]`, and `default_gate_idx`
            # names it) so a cell's record shows what the gate is about to compare against, but it
            # is NOT a candidate in this pick — the anchor is judged by the ship gate, with a bar,
            # a few lines below. Where the hsv-21 recheck also fired its faithful contender is in
            # `scored` and therefore pooled as well, so a demotion is decided on the same pooled
            # evidence as everything else rather than on one split.
            rerace_note["pooled_means"] = {str(slot_idx): float(np.mean(pooled[slot_idx]))
                                           for _, _, slot_idx in pool_slots}
            rerace_note["pooled_pick_idx"] = int(pooled_pick)
            rerace_note["pooled_changed"] = bool(pooled_pick != champion_idx)
            if pooled_pick != champion_idx:
                pooled_win = next(c for c in scored if c[1] == pooled_pick)
                champion_idx, best_hp = pooled_pick, pooled_win[2]
                best_iter = int(pooled_win[4].get("best_iter") or 0)
                selection["champion_idx"] = int(champion_idx)
                selection["champion_score"] = pooled_win[0]
        pooled_note = {"extra_failed": extra_failed, "n_extra": n_extra,
                       "pooled_d": pooled[default_gate_idx], "pooled_c": pooled[champion_idx]}

    if pooled_note is not None:
        selection["rule"] = "static_ship_gate_1se_msplit"
        selection["n_finale_splits"] = 1 + pooled_note["n_extra"]
        selection["pooled_finale"] = str(pooled_finale_mode)
        selection["pooled_finale_band"] = float(_adaptive_pool_band(se_mult))
        # Keep the single-split verdict the pooled one supersedes — without it a cell's record
        # cannot say whether pooling CHANGED the decision, which is the only way to audit whether
        # the adaptive extension is earning its evaluations.
        selection["single_split_gate_margin"] = selection.get("gate_margin")
        selection["single_split_gate_se"] = selection.get("gate_se")
        selection["single_split_shipped"] = selection.get("shipped")
        if pooled_note["extra_failed"] or selection.get("reason") == "default_failed_ship_default":
            # any missing side of the pooled evidence -> conservative: ship the default (its
            # deploy either works or fails as a visible cell; never an ungated champion).
            selection.update(shipped="default", reason="default_failed_ship_default")
        else:
            margin, se = _paired_margin(pooled_note["pooled_d"], pooled_note["pooled_c"],
                                        lower_better)
            if margin is not None:
                selection.update(gate_margin=margin, gate_se=se, gate_se_mult=float(se_mult),
                                 n_paired_folds=int(len(pooled_note["pooled_d"])))
                if np.isfinite(margin) and margin > se_mult * se:
                    selection.update(shipped="champion", reason="beats_default_by_1se_pooled")
                else:
                    selection.update(shipped="default", reason="within_1se_of_default_pooled")
    elif extra_finale_otrs and finale_hp != sel_hp and champion_idx != default_gate_idx:
        # Extras were available but the adaptive trigger declined them: record why, so a cell that
        # kept the single-split rule is distinguishable from one that never had extras at all.
        # The band is written into the string (hsv 21): at the 1-SE bar it still reads
        # "skipped_margin_beyond_2se", exactly as every pre-hsv-21 record does.
        band = _adaptive_pool_band(se_mult)
        selection["pooled_finale"] = f"skipped_margin_beyond_{band:g}se"
        selection["pooled_finale_band"] = float(band)
        selection["n_finale_splits"] = 1
    selection["inner_repeats"] = int(max(1, repeats))
    if racing_se:
        selection["racing_se"] = float(racing_se)
        selection["n_raced"] = sum(1 for e in evaluations if e.get("raced"))
    if selection["shipped"] == "default":
        best_hp = configs[0]
        best_iter = int(_evaluation_by_config_idx(evaluations, default_gate_idx).get("best_iter") or 0)
    return best_hp, best_iter, traj, evaluations, selection


def _select_config_coordinate(
    model_name,
    otr: DatasetBundle,
    n_configs,
    k,
    seed,
    time_budget=None,
    hp_override=None,
    cache: ResultCache | None = None,
    dataset_hash: str | None = None,
    candidate_proto: dict | None = None,
    repeats: int = 1,
    fold_workers: int = 1,
    racing_se: float | None = None,
):
    direction = METRICS[otr.task.primary_metric].direction
    lower_better = direction != "higher"
    plan = coordinate_tuning_plan(model_name, hp_override=hp_override)
    if plan is None:
        raise ValueError(f"{model_name} does not declare a coordinate tuning plan")

    sel_hp = _model_selection_hp(model_name)
    racing_pool = _RacingPool(racing_se, lower_better)
    selection = {"rule": "coordinate_ship_gate_1se", "inner_repeats": int(max(1, repeats))}
    # Budget: the default anchor and the tuned-family base are protocol overhead, not part of the
    # stage budget — n_configs still counts stage evaluations only (full t-boost plan = 13).
    max_evals = max(2, int(n_configs) + 2)
    t0 = time.perf_counter()
    evaluations, traj, config_idx = [], [], 0

    def _eval(hp, meta, racing_ctx=None, sel=None):
        nonlocal config_idx
        payload, cached = _evaluate_config(
            model_name, otr, hp, config_idx, k, seed, otr.task.primary_metric, direction,
            lower_better, cache, dataset_hash, candidate_proto, metadata=meta, repeats=repeats,
            fold_workers=fold_workers, selection_hp=sel_hp if sel is None else sel,
            racing=racing_ctx,
        )
        idx = config_idx
        config_idx += 1
        return payload, cached, idx

    def _track(payload, cached, best_so_far):
        traj.append(float(best_so_far) if np.isfinite(best_so_far) else float("nan"))
        evaluations.append({**payload, "cached": bool(cached), "best_so_far": traj[-1]})

    # Candidate 0 is ALWAYS the true default — the anchor the final ship-gate compares against.
    default_hp_cfg = static_configs(model_name, 0, hp_override=hp_override)[0]
    base_hp = dict(plan["base"])
    base_is_default = default_hp_cfg == base_hp
    selection["base_is_default"] = bool(base_is_default)
    meta0 = {"stage": "base" if base_is_default else "default", "param": None, "value": None}
    payload, cached, idx0 = _eval(default_hp_cfg, meta0)
    best_score = _score_from_payload(payload, lower_better)
    _track(payload, cached, best_score)
    incumbent_hp, incumbent_iter, incumbent_idx = dict(default_hp_cfg), int(payload.get("best_iter") or 0), idx0
    incumbent_score, incumbent_folds = best_score, payload.get("fold_scores")

    if not base_is_default:
        # The greedy walk starts from the tuned-family base regardless of how it scored — the plan's
        # stages describe the base recipe's neighborhood; the final gate settles base-family vs default.
        payload, cached, idx1 = _eval(base_hp, {"stage": "base", "param": None, "value": None})
        b_score = _score_from_payload(payload, lower_better)
        if _better(b_score, best_score, lower_better):
            best_score = b_score
        _track(payload, cached, best_score)
        incumbent_hp, incumbent_iter, incumbent_idx = base_hp, int(payload.get("best_iter") or 0), idx1
        incumbent_score, incumbent_folds = b_score, payload.get("fold_scores")

    for stage_idx, stage in enumerate(plan["stages"], start=1):
        if config_idx >= max_evals:
            break
        param = stage["name"]
        stage_hp, stage_iter, stage_inc_idx = dict(incumbent_hp), incumbent_iter, incumbent_idx
        stage_score, stage_folds = incumbent_score, incumbent_folds
        timed_out = False
        for value in stage["values"]:
            if config_idx >= max_evals:
                break
            if _same_value(incumbent_hp.get(param), value):
                continue
            hp = dict(incumbent_hp)
            hp[param] = value
            payload, cached, idx = _eval(
                hp,
                {"stage": stage_idx, "param": param, "value": value},
                racing_ctx=racing_pool.context(incumbent_folds),
            )
            racing_pool.add(incumbent_folds, payload)
            score = _score_from_payload(payload, lower_better)
            it = int(payload.get("best_iter") or 0)
            defended = _stage_defended(
                stage, value, stage_hp.get(param), score, stage_score, lower_better
            )
            # Noise-aware acceptance: a move must beat the stage incumbent by >1 paired-fold SE
            # (falling back to the raw comparison only when fold grids don't pair, e.g. a failed
            # eval). The declared prefer/defended near-tie rules keep their existing authority.
            margin, se = _paired_margin(stage_folds, payload.get("fold_scores"), lower_better)
            if margin is not None:
                improves = np.isfinite(score) and margin > _GATE_SE_MULT * se
            else:
                improves = _better(score, stage_score, lower_better)
            if (improves and not defended) or _stage_preferred(
                stage, value, stage_hp.get(param), score, stage_score, lower_better
            ):
                stage_hp, stage_iter, stage_score = hp, it, score
                stage_folds, stage_inc_idx = payload.get("fold_scores"), idx
            if _better(score, best_score, lower_better):
                best_score = score
            _track(payload, cached, best_score)
            if time_budget and (time.perf_counter() - t0) > time_budget:
                timed_out = True
                break
        incumbent_hp, incumbent_iter, incumbent_idx = stage_hp, stage_iter, stage_inc_idx
        incumbent_score, incumbent_folds = stage_score, stage_folds
        if timed_out or (time_budget and (time.perf_counter() - t0) > time_budget):
            break

    # Ship-fidelity finale: when the adapter's selection overlay reduces fit fidelity (the
    # scout regime — Ralph 2026-07-16: 1-bag, prune-free stage evals), the walk's scores are
    # ranking evidence only. Re-evaluate the final incumbent and the default anchor under the
    # adapter's finale overlay (full recipe, matched fidelity on BOTH sides) and take the ship
    # gate and the deployed iteration count from those. Like the two anchors, these are
    # protocol overhead, not stage budget (`max_evals` deliberately not consulted).
    finale_hp = _model_selection_finale_hp(model_name)
    default_gate_idx = 0
    if finale_hp != sel_hp:
        payload, cached, idx_d = _eval(
            default_hp_cfg, {"stage": "finale", "param": "default_anchor", "value": None},
            sel=finale_hp,
        )
        d_score = _score_from_payload(payload, lower_better)
        if _better(d_score, best_score, lower_better):
            best_score = d_score
        _track(payload, cached, best_score)
        default_gate_idx = idx_d
        if incumbent_hp == default_hp_cfg:
            incumbent_idx = idx_d               # walk ended on the default: one eval serves both
            incumbent_iter = int(payload.get("best_iter") or 0)
        else:
            payload, cached, idx_i = _eval(
                incumbent_hp, {"stage": "finale", "param": "incumbent", "value": None},
                sel=finale_hp,
            )
            i_score = _score_from_payload(payload, lower_better)
            if _better(i_score, best_score, lower_better):
                best_score = i_score
            _track(payload, cached, best_score)
            incumbent_idx = idx_i
            incumbent_iter = int(payload.get("best_iter") or 0)

    selection.update(_ship_gate(evaluations, incumbent_idx, lower_better,
                                rule="coordinate_ship_gate_1se",
                                default_idx=default_gate_idx,
                                se_mult=_gate_se_mult(model_name, repeats)))
    if racing_se:
        selection["racing_se"] = float(racing_se)
        selection["n_raced"] = sum(1 for e in evaluations if e.get("raced"))
    if selection["shipped"] == "default":
        incumbent_hp = dict(default_hp_cfg)
        incumbent_iter = int(evaluations[default_gate_idx].get("best_iter") or 0)
    return incumbent_hp, incumbent_iter, traj, evaluations, selection


def _require_scored_evaluations(model_name: str, evaluations: list[dict]) -> None:
    """A selection is only a selection if at least one candidate actually scored.

    When every evaluation fails, the walk keeps its opening anchor as `best_idx` (a failed
    candidate scores -/+inf and never displaces the incumbent) and the ship gate reports the
    anchor as champion. Silently caching that as a tuned choice publishes the DEFAULT recipe
    under the "tuned" label with no search behind it. Raising instead surfaces the cell the way
    every other infrastructure failure surfaces: a visible failed cell that is never cached, so
    the next run re-does the work.
    """
    if any(e.get("score") is not None and not e.get("failed") for e in evaluations):
        return
    reasons = sorted({str(e.get("failed") or ("raced" if e.get("raced") else "unscored"))
                      for e in evaluations})
    raise RuntimeError(
        f"{model_name}: HPO selection produced no scored candidate over {len(evaluations)} "
        f"evaluations ({'; '.join(reasons[:3])})"
    )


def select_config(
    model_name,
    otr: DatasetBundle,
    n_configs,
    k,
    seed,
    time_budget=None,
    hp_override=None,
    cache: ResultCache | None = None,
    dataset_hash: str | None = None,
    candidate_proto: dict | None = None,
    repeats: int = 1,
    fold_workers: int = 1,
    racing_se: float | None = None,
    extra_finale_otrs: list | None = None,
    pooled_finale_mode: str = "always",
):
    """Pick the best config by k-fold OOF score (repeated `repeats` times on small tune splits).

    Models may declare a greedy coordinate plan; otherwise the static seeded random candidate
    prefix is used, guarded by the default ship-gate. Returns (hp, best_iter, trajectory,
    evaluations, selection) where selection records the shipping decision and its evidence.

    `fold_workers` (A1) parallelizes each candidate's inner folds — wall-clock only, scores
    bit-identical. `racing_se` (A2) arms fold-1 discard at that many pooled paired-diff SDs;
    None disables. Both from design/tuning-speed-roadmap.md, verified 2026-07-15.

    `extra_finale_otrs` supplies the protocol-v3 pooled finale's extra tune splits (bundles, or
    zero-arg callables returning one so a large set's outer-train is materialized only if used).
    `pooled_finale_mode` is "always" (hsv 16 small-set behavior) or "adaptive" (hsv 18: spend them
    only on a near-tie single-split gate) — see `_pooled_finale_wanted`.
    """
    if coordinate_tuning_plan(model_name, hp_override=hp_override) is not None:
        out = _select_config_coordinate(
            model_name, otr, n_configs, k, seed, time_budget=time_budget, hp_override=hp_override,
            cache=cache, dataset_hash=dataset_hash, candidate_proto=candidate_proto, repeats=repeats,
            fold_workers=fold_workers, racing_se=racing_se,
        )
    else:
        out = _select_config_static(
            model_name, otr, n_configs, k, seed, time_budget=time_budget, hp_override=hp_override,
            cache=cache, dataset_hash=dataset_hash, candidate_proto=candidate_proto, repeats=repeats,
            fold_workers=fold_workers, racing_se=racing_se, extra_finale_otrs=extra_finale_otrs,
            pooled_finale_mode=pooled_finale_mode,
        )
    _require_scored_evaluations(model_name, out[3])
    return out


def fit_one(model_name, hp, otr: DatasetBundle, seed: int, n_estimators: int | None = None):
    """Refit ONE model on the full outer-train (train+val). No bagging.

    If n_estimators is given (from OOF selection) it is used directly; otherwise an inner holdout
    is used as an early-stopping probe to fix the iteration count, then the model is refit.
    """
    final_hp = dict(hp)
    final_hp.setdefault("seed", int(seed))       # per-cell seed; an explicit config-level hp seed wins
    if n_estimators:
        final_hp["n_estimators"] = int(n_estimators)
        return build_model(model_name, hp=final_hp).fit(otr, None)
    if not model_class(model_name).needs_es_probe:      # early-stops internally -> refit directly, no probe
        return build_model(model_name, hp=final_hp).fit(otr, None)
    tr_idx, va_idx = inner_holdout(otr, frac=0.125, seed=seed)
    probe = build_model(model_name, hp=final_hp).fit(otr.subset(tr_idx), otr.subset(va_idx))
    bi = int(getattr(probe, "best_iteration_", 0) or 0)
    if bi:
        final_hp["n_estimators"] = int(bi)
    return build_model(model_name, hp=final_hp).fit(otr, None)
