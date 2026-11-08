"""Declarative dataset manifest + generic loader.

A DatasetEntry describes a dataset purely as config (where to fetch it, which columns play
which role); build_bundle turns the raw DataFrame into a task-correct DatasetBundle. Adding a
dataset is therefore one manifest row, not a bespoke loader (the "automated" sourcing path).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.datasets.preprocessing import content_hash
from insurarena.datasets.sources import fetch
from insurarena.tasks import TASK_SPECS, TaskType

DATASET_CACHE_VERSION = "0.1"


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    task: TaskType
    source: str
    access_method: str             # openml_id | url_csv | rpackage_csv | casdatasets_rda
    locator: str
    target_col: str                # counts (freq) | total loss (pp) | severity (sev) | label (clf)
    # Full, post-task-shaping DatasetBundle dimensions. These drive benchmark eligibility only and
    # deliberately do not participate in the dataset-content cache key.
    full_n_rows: int | None = None
    full_n_predictors: int | None = None
    exposure_col: str | None = None
    claim_count_col: str | None = None   # sev weight / deriving binary
    group_col: str | None = None
    feature_cols: tuple[str, ...] | None = None   # None -> all minus role/drop cols
    drop_cols: tuple[str, ...] = ()
    cat_cols: tuple[str, ...] | None = None        # None -> auto (object/category dtype)
    # task shaping
    severity_is_average: bool = False    # sev: target_col already avg cost/claim
    binary_from: str | None = None       # clf: "count>0" to derive label from claim_count_col
    positive_label: str | None = None    # clf: value of target_col that means the positive event
    cap: dict[str, float] = field(default_factory=dict)   # column -> upper cap
    tweedie_power: float = 1.9
    license: str = ""
    redistributable: bool = False
    notes: str = ""


def _role_cols(e: DatasetEntry) -> set[str]:
    roles = {e.target_col, e.exposure_col, e.claim_count_col, e.group_col, *e.drop_cols}
    return {c for c in roles if c}


def _features(e: DatasetEntry, df: pd.DataFrame) -> list[str]:
    if e.feature_cols:
        return [c for c in e.feature_cols if c in df.columns]
    return [c for c in df.columns if c not in _role_cols(e)]


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _hygiene(X: pd.DataFrame) -> pd.DataFrame:
    """Data hygiene (TabArena §10): expand datetimes, flag-and-leave missing, drop constant/dup cols.

    This is the hygiene every board cell was scored on (the dataset bundles cached 2026-08-21).
    871dd0d7 briefly replaced it with an `__isna` flag on EVERY column and no constant/duplicate
    drop, but the dataset cache key does not cover this code, so the board never saw that version;
    only a run on an empty cache did (fremotor_prem 19 -> 36 columns, a different `source_hash`).
    Restored 2026-09-29 so a cold load reproduces the board's data.
    """
    X = X.copy()
    for c in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            dt = pd.to_datetime(X[c], errors="coerce")
            X[f"{c}__year"] = dt.dt.year.astype("float")
            X[f"{c}__month"] = dt.dt.month.astype("float")
            X[f"{c}__dow"] = dt.dt.dayofweek.astype("float")
            X = X.drop(columns=[c])
    for c in list(X.columns):                       # flag-and-leave missing indicators
        if X[c].isna().any():
            X[f"{c}__isna"] = X[c].isna().astype(float)
    nun = X.nunique(dropna=False)
    X = X.drop(columns=[c for c in X.columns if nun[c] <= 1])      # constant columns
    X = _drop_duplicate_columns(X)                                 # duplicate columns
    return X.reset_index(drop=True)


def _drop_duplicate_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate columns without materializing a full transposed copy.

    ``X.T.duplicated()`` is painfully expensive on tall datasets such as Porto Seguro. Hash each
    column, then only do exact equality checks within matching hash buckets.
    """
    if X.shape[1] <= 1:
        return X
    seen: dict[bytes, list[int]] = {}
    keep: list[int] = []
    for pos in range(X.shape[1]):
        s = X.iloc[:, pos]
        try:
            values = pd.util.hash_pandas_object(s, index=False).to_numpy(dtype=np.uint64, copy=False)
        except TypeError:
            values = pd.util.hash_pandas_object(s.astype(str), index=False).to_numpy(dtype=np.uint64, copy=False)
        digest = hashlib.blake2b(np.ascontiguousarray(values).view(np.uint8), digest_size=16).digest()
        bucket = seen.setdefault(digest, [])
        if any(s.equals(X.iloc[:, prev]) for prev in bucket):
            continue
        bucket.append(pos)
        keep.append(pos)
    return X.iloc[:, keep]


def _cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    root = cache_dir or os.environ.get("INSURARENA_DATASET_CACHE_DIR") or "~/.cache/insurarena"
    return Path(os.path.expanduser(str(root))) / "datasets"


def _entry_payload(entry: DatasetEntry) -> dict:
    payload = dataclasses.asdict(entry)
    payload["task"] = entry.task.value
    payload.pop("full_n_rows", None)
    payload.pop("full_n_predictors", None)
    return payload


def _bundle_cache_key(entry: DatasetEntry, subsample: int | None, seed: int) -> str:
    payload = {
        "version": DATASET_CACHE_VERSION,
        "entry": _entry_payload(entry),
        "subsample": subsample,
        "seed": seed,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _bundle_from_cache(path: Path, entry: DatasetEntry) -> DatasetBundle | None:
    meta_path = path / "meta.json"
    arrays_path = path / "arrays.npz"
    x_parquet = path / "X.parquet"
    x_pickle = path / "X.pkl"
    if not (meta_path.exists() and arrays_path.exists() and (x_parquet.exists() or x_pickle.exists())):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        X = pd.read_parquet(x_parquet) if x_parquet.exists() else pd.read_pickle(x_pickle)
        arrays = np.load(arrays_path, allow_pickle=False)
        exposure = arrays["exposure"]
        group_ids = arrays["group_ids"]
        task = TASK_SPECS[entry.task]
        if entry.task == TaskType.PP:
            task = dataclasses.replace(task, tweedie_power=entry.tweedie_power)
        return DatasetBundle(
            name=entry.name,
            X=X,
            y=arrays["y"],
            exposure=None if exposure.size == 0 else exposure,
            sample_weight=arrays["sample_weight"],
            task=task,
            group_ids=None if group_ids.size == 0 else group_ids,
            cat_features=list(meta.get("cat_features", [])),
            meta=dict(meta.get("bundle_meta", {})),
        )
    except Exception:
        return None


def _write_bundle_cache(path: Path, bundle: DatasetBundle) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        try:
            bundle.X.to_parquet(tmp / "X.parquet", index=False)
        except Exception:
            bundle.X.to_pickle(tmp / "X.pkl")
        group_ids = np.asarray(bundle.group_ids) if bundle.group_ids is not None else np.array([])
        if group_ids.dtype.kind == "O":
            group_ids = group_ids.astype(str)
        np.savez(
            tmp / "arrays.npz",
            y=np.asarray(bundle.y, dtype=float),
            exposure=np.asarray(bundle.exposure, dtype=float) if bundle.exposure is not None else np.array([]),
            sample_weight=np.asarray(bundle.sample_weight, dtype=float),
            group_ids=group_ids,
        )
        (tmp / "meta.json").write_text(json.dumps({
            "cat_features": list(bundle.cat_features),
            "bundle_meta": dict(bundle.meta),
        }, default=str))
        if path.exists():
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
            return
        os.replace(tmp, path)
    except Exception:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        raise


def build_bundle(entry: DatasetEntry, df: pd.DataFrame) -> DatasetBundle:
    df = df.copy()
    for col, upper in entry.cap.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(upper=upper)

    feats = _features(entry, df)
    cats = list(entry.cat_cols) if entry.cat_cols is not None else [
        c for c in feats  # any non-numeric, non-datetime column is categorical (catches object/category/string)
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    for c in cats:
        if c in df.columns:
            df[c] = df[c].astype(str)

    task = entry.task
    exposure = _num(df[entry.exposure_col]) if entry.exposure_col else None
    n_classes: int | None = None

    if task in (TaskType.FREQ, TaskType.PP):
        if exposure is None:
            exposure = np.ones(len(df))   # no exposure column -> full unit (one-year) exposure
        keep = exposure > 0
        y = _num(df[entry.target_col])               # counts (freq) or total loss (pp)
        weight = exposure
        df, y, exposure, weight = df[keep].reset_index(drop=True), y[keep], exposure[keep], weight[keep]
    elif task == TaskType.SEV:
        loss = _num(df[entry.target_col])
        cnt = _num(df[entry.claim_count_col]) if entry.claim_count_col else np.ones(len(df))
        keep = loss > 0
        sev = loss if entry.severity_is_average else np.divide(loss, np.where(cnt > 0, cnt, 1.0))
        df, y, weight, exposure = (
            df[keep].reset_index(drop=True), sev[keep],
            np.where(cnt[keep] > 0, cnt[keep], 1.0), None,
        )
    elif task == TaskType.CLF:
        if entry.binary_from == "count>0" and entry.claim_count_col:
            y = (_num(df[entry.claim_count_col]) > 0).astype(float)
        elif entry.positive_label is not None:
            y = (df[entry.target_col].astype(str) == str(entry.positive_label)).astype(float).to_numpy()
        else:
            y = _num(df[entry.target_col])
            uniq = set(np.unique(y[~np.isnan(y)]))
            if not uniq <= {0.0, 1.0}:
                y = (y > 0).astype(float)
        weight = exposure if exposure is not None else np.ones(len(df))
    elif task == TaskType.MULTICLF:
        codes, uniques = pd.factorize(df[entry.target_col], sort=True)   # dense codes 0..K-1; NaN -> -1
        keep = codes >= 0
        y = codes.astype(float)
        weight = exposure if exposure is not None else np.ones(len(df))
        df, y, weight = df[keep].reset_index(drop=True), y[keep], weight[keep]
        exposure = None
        n_classes = int(len(uniques))
    else:
        raise ValueError(f"unsupported task {task}")

    X = _hygiene(df[[c for c in feats if c in df.columns]].reset_index(drop=True))
    group_ids = df[entry.group_col].to_numpy() if entry.group_col and entry.group_col in df.columns else None
    spec = TASK_SPECS[task]
    if task == TaskType.PP:
        spec = dataclasses.replace(spec, tweedie_power=entry.tweedie_power)

    return DatasetBundle(
        name=entry.name, X=X, y=y, exposure=exposure, sample_weight=weight, task=spec,
        group_ids=group_ids, cat_features=[c for c in cats if c in X.columns],
        meta={
            "source": entry.source, "access": f"{entry.access_method}:{entry.locator}",
            "license": entry.license, "redistributable": entry.redistributable,
            "hash": content_hash(X.assign(_y=y)), "n_rows": int(len(X)),
            "stratify_on": (
                "class_label" if task == TaskType.MULTICLF
                else "claim_occurrence" if task in (TaskType.FREQ, TaskType.CLF) else None
            ),
            "n_classes": n_classes,
            "notes": entry.notes,
        },
    )


class GenericLoader:
    """Implements the DatasetLoader protocol for any DatasetEntry."""

    def __init__(
        self,
        entry: DatasetEntry,
        subsample: int | None = None,
        seed: int = 42,
        cache_dir: str | os.PathLike[str] | None = None,
        use_dataset_cache: bool | None = None,
    ):
        self.entry = entry
        self.name = entry.name
        self.subsample = subsample
        self.seed = seed
        self._hash: str | None = None
        self.cache_dir = cache_dir
        self.use_dataset_cache = (
            os.environ.get("INSURARENA_DATASET_CACHE", "1") != "0"
            if use_dataset_cache is None else bool(use_dataset_cache)
        )

    def load(self) -> DatasetBundle:
        cache_path: Path | None = None
        if self.use_dataset_cache:
            cache_path = _cache_root(self.cache_dir) / _bundle_cache_key(self.entry, self.subsample, self.seed)
            cached = _bundle_from_cache(cache_path, self.entry)
            if cached is not None:
                self._hash = cached.meta.get("hash")
                cached.meta["_dataset_cache"] = "hit"
                return cached
            if cache_path.exists():
                import shutil

                shutil.rmtree(cache_path, ignore_errors=True)

        df = fetch(self.entry.access_method, self.entry.locator)
        if self.subsample and len(df) > self.subsample:
            df = df.sample(n=self.subsample, random_state=self.seed).reset_index(drop=True)
        bundle = build_bundle(self.entry, df)
        self._hash = bundle.meta.get("hash")
        if cache_path is not None:
            bundle.meta["_dataset_cache"] = "miss"
            try:
                _write_bundle_cache(cache_path, bundle)
            except Exception:
                pass
        return bundle

    def source_hash(self) -> str:
        return self._hash or "unloaded"
