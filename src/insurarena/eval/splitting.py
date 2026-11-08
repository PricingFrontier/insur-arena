"""Nested CV (SPEC §5.1–5.3): size-dependent outer repeats, a single grouped/stratified
inner holdout (no 8-fold bag), group- and stratify-aware."""
from __future__ import annotations

import numpy as np

from insurarena.datasets.base import DatasetBundle


def _stratify_label(bundle: DatasetBundle):
    if bundle.meta.get("stratify_on") == "class_label":
        return np.asarray(bundle.y).astype(int)          # multiclass: stratify on the class itself
    if bundle.meta.get("stratify_on") == "claim_occurrence":
        return (np.asarray(bundle.y) > 0).astype(int)
    return None


def selection_subsample(bundle: DatasetBundle, idx, max_rows: int | None, seed: int):
    """Row subset of `idx` for HPO SEARCH fits only (SPEC §5.5, roadmap lever A3).

    Selection needs a *ranking* of configs, not deployment fidelity; on very large portfolios the
    paired per-fold SEs that drive the 1-SE/racing decisions are ~1e-4, so scoring candidates on a
    capped subsample changes the ranking far less than it changes the wall-clock. The chosen config
    is then refit on the FULL outer-train for every outer split — only the search sees fewer rows.

    The draw is:
      * **whole-group** when the bundle is grouped — a panel entity is never split between the
        search sample and the discarded remainder, matching the group-honest outer/inner splits;
      * **stratified** on the bundle's own stratify label (class for multiclass, claim-occurrence
        for frequency/pure-premium), so a capped sample keeps the positive-event share rather than
        drifting toward the zero mass;
      * **deterministic** in `(idx, max_rows, seed)` and returned in sorted row order.

    Returns `idx` unchanged when `max_rows` is falsy or already covers it, so the cap is a strict
    no-op below the threshold.
    """
    idx = np.asarray(idx)
    if not max_rows or idx.size <= int(max_rows):
        return idx
    target = int(max_rows)
    sub = bundle.subset(idx)
    strat = _stratify_label(sub)
    groups = sub.group_ids
    rng = np.random.default_rng([int(seed), 0xA3])

    if groups is None:
        units = np.arange(idx.size)[:, None]              # each row is its own unit
        unit_rows = np.ones(idx.size, dtype=np.int64)
        unit_label = strat if strat is not None else np.zeros(idx.size, dtype=np.int64)
        row_of_unit = None
    else:
        uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        unit_rows = counts.astype(np.int64)
        row_of_unit = inv
        if strat is None:
            unit_label = np.zeros(uniq.shape[0], dtype=np.int64)
        else:
            labels = np.asarray(strat).astype(np.int64)
            classes, lab_inv = np.unique(labels, return_inverse=True)
            tally = np.zeros((uniq.shape[0], classes.shape[0]), dtype=np.int64)
            np.add.at(tally, (inv, lab_inv), 1)           # a group's label = its most common one
            unit_label = classes[np.argmax(tally, axis=1)]
        units = np.arange(uniq.shape[0])[:, None]

    total = float(unit_rows.sum())
    keep_units: list[int] = []
    for lbl in np.unique(unit_label):
        members = np.flatnonzero(unit_label == lbl)
        stratum_rows = int(unit_rows[members].sum())
        want = max(1, int(round(target * stratum_rows / total)))
        order = members[rng.permutation(members.shape[0])]
        covered = 0
        for u in order:
            if covered >= want:
                break
            keep_units.append(int(u))
            covered += int(unit_rows[u])
    if not keep_units:                                     # degenerate: keep the first unit
        keep_units = [int(units[0, 0])]
    keep_units_arr = np.asarray(sorted(keep_units))
    rows = keep_units_arr if row_of_unit is None else np.flatnonzero(np.isin(row_of_unit, keep_units_arr))
    return idx[np.sort(rows)]


def _safe_strat(strat, min_per_class: int):
    """Use stratification only if every class has enough members; else fall back to unstratified."""
    if strat is None:
        return None
    _, counts = np.unique(strat, return_counts=True)
    return strat if (len(counts) >= 2 and counts.min() >= min_per_class) else None


def outer_splits(bundle: DatasetBundle, cfg: dict, seed: int):
    """Yield (train_idx, test_idx). Schedule depends on n_eff = number of groups."""
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedGroupKFold,
        StratifiedKFold,
    )

    n_eff = bundle.n_groups
    sched = cfg["small"] if n_eff < cfg.get("small_threshold", 2500) else cfg["large"]
    folds, repeats = int(sched["folds"]), int(sched["repeats"])
    strat = _safe_strat(_stratify_label(bundle), folds)
    groups = bundle.group_ids
    splits = []
    for r in range(repeats):
        rs = seed + 1000 * r
        if groups is not None and strat is not None:
            sp = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=rs)
            it = sp.split(bundle.X, strat, groups)
        elif groups is not None:
            # shuffle + the repeat's seed: without them GroupKFold is deterministic and every repeat
            # returns the same folds (fremotor_prem and euhealth_pp had 9 splits but 3 test sets).
            it = GroupKFold(n_splits=folds, shuffle=True, random_state=rs).split(bundle.X, groups=groups)
        elif strat is not None:
            it = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rs).split(bundle.X, strat)
        else:
            it = KFold(n_splits=folds, shuffle=True, random_state=rs).split(bundle.X)
        for tr, te in it:
            splits.append((np.asarray(tr), np.asarray(te)))
    return splits


def kfold(bundle: DatasetBundle, k: int, seed: int):
    """k (train_idx, val_idx) folds, stratify/group-aware — for out-of-fold HPO config selection.

    OOF scoring picks the tuned config; the deployed model is still a single refit (no bagging).
    """
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedGroupKFold,
        StratifiedKFold,
    )

    strat = _safe_strat(_stratify_label(bundle), k)
    groups = bundle.group_ids
    if groups is not None and strat is not None:
        it = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(bundle.X, strat, groups)
    elif groups is not None:
        it = GroupKFold(n_splits=k).split(bundle.X, groups=groups)
    elif strat is not None:
        it = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(bundle.X, strat)
    else:
        it = KFold(n_splits=k, shuffle=True, random_state=seed).split(bundle.X)
    return [(np.asarray(tr), np.asarray(va)) for tr, va in it]


def inner_holdout(bundle: DatasetBundle, frac: float = 0.125, seed: int = 0):
    """One grouped/stratified holdout of the outer-train pool for early stopping + HPO."""
    strat = _safe_strat(_stratify_label(bundle), 2)
    groups = bundle.group_ids
    idx = np.arange(len(bundle.X))
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit

        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        tr, val = next(gss.split(idx, groups=groups))
    else:
        from sklearn.model_selection import train_test_split

        tr, val = train_test_split(idx, test_size=frac, random_state=seed, stratify=strat)
    return np.asarray(tr), np.asarray(val)
