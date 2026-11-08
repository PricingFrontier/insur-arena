"""Dataset registry — resolves dataset names and benchmark eligibility.

Every row in ``MANIFEST`` remains loadable for historical reproducibility. The official benchmark
view applies minimum full-data dimensions. Synthetic loaders are kept as offline fixtures for tests
and smoke runs, but they are not benchmark datasets.
"""
from __future__ import annotations

from insurarena.datasets.base import DatasetLoader
from insurarena.datasets.loaders import (
    SyntheticClfLoader,
    SyntheticFreqLoader,
    SyntheticMulticlfLoader,
    SyntheticPPLoader,
    SyntheticSevLoader,
)
from insurarena.datasets.manifest import MANIFEST
from insurarena.datasets.spec import DatasetEntry, GenericLoader
from insurarena.tasks import TaskType

MIN_BENCHMARK_ROWS = 1_000
MIN_BENCHMARK_PREDICTORS = 4

# Bespoke loaders: offline synthetic data for TESTS + the `smoke` CI profile ONLY. These are
# deterministic fixtures, NOT benchmark datasets, and never appear on the leaderboard.
_BESPOKE = {
    "synthetic_freq": SyntheticFreqLoader,
    "synthetic_sev": SyntheticSevLoader,
    "synthetic_pp": SyntheticPPLoader,
    "synthetic_clf": SyntheticClfLoader,
    "synthetic_multiclf": SyntheticMulticlfLoader,
}

# Manifest-driven datasets (the automated sourcing path).
_MANIFEST: dict[str, DatasetEntry] = {e.name: e for e in MANIFEST}


def _entry_exclusion_reason(entry: DatasetEntry) -> str | None:
    if entry.full_n_rows is None or entry.full_n_predictors is None:
        return "full-data row and predictor counts are not recorded"
    failures = []
    if entry.full_n_rows < MIN_BENCHMARK_ROWS:
        failures.append(f"{entry.full_n_rows:,} rows")
    if entry.full_n_predictors < MIN_BENCHMARK_PREDICTORS:
        noun = "predictor" if entry.full_n_predictors == 1 else "predictors"
        failures.append(f"{entry.full_n_predictors} {noun}")
    if not failures:
        return None
    minimum = f"at least {MIN_BENCHMARK_ROWS:,} rows and {MIN_BENCHMARK_PREDICTORS} predictors"
    return f"full modeled data has {' and '.join(failures)}; benchmark requires {minimum}"


# The loader catalog includes every real dataset. The benchmark catalog is its eligible subset;
# task-specific lists are derived views, not separate benchmark groups.
DATASET_CATALOG: tuple[str, ...] = tuple(e.name for e in MANIFEST)
BENCHMARK_DATASET_CATALOG: tuple[str, ...] = tuple(
    e.name for e in MANIFEST if _entry_exclusion_reason(e) is None
)
DATASETS_BY_TASK: dict[TaskType, tuple[str, ...]] = {
    task: tuple(name for name in BENCHMARK_DATASET_CATALOG if _MANIFEST[name].task == task)
    for task in TaskType
}


def list_benchmark_datasets(task: TaskType | str | None = None) -> list[str]:
    """Return eligible real benchmark dataset names, optionally filtered by task."""
    if task is None:
        return list(BENCHMARK_DATASET_CATALOG)
    task_type = task if isinstance(task, TaskType) else TaskType(str(task))
    return list(DATASETS_BY_TASK[task_type])


def list_datasets(include_synthetic: bool = True) -> list[str]:
    names = set(DATASET_CATALOG)
    if include_synthetic:
        names |= set(_BESPOKE)
    return sorted(names)


def get_entry(name: str) -> DatasetEntry | None:
    return _MANIFEST.get(name)


def benchmark_exclusion_reason(name: str) -> str | None:
    """Explain why a real manifest dataset is excluded; non-manifest names return ``None``."""
    entry = get_entry(name)
    return None if entry is None else _entry_exclusion_reason(entry)


def is_benchmark_eligible(name: str) -> bool:
    """Return whether ``name`` is a real manifest dataset satisfying the size policy."""
    return name in _MANIFEST and benchmark_exclusion_reason(name) is None


def build_loader(name: str, **kwargs) -> DatasetLoader:
    if name in _BESPOKE:
        kwargs.pop("cache_dir", None)
        kwargs.pop("use_dataset_cache", None)
        return _BESPOKE[name](**kwargs)
    if name in _MANIFEST:
        return GenericLoader(_MANIFEST[name], **kwargs)
    raise KeyError(f"unknown dataset {name!r}; known: {list_datasets()}")
