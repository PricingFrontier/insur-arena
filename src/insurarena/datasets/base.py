"""The DatasetBundle contract every loader returns, plus the loader Protocol.

Invariant: ``X`` contains features ONLY — never exposure (it is scale, not a feature).
Loaders must enforce this; see ``DatasetBundle.validate``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from insurarena.tasks import TaskSpec


@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame                       # features ONLY
    y: np.ndarray                         # target per task contract (e.g. claim counts for freq)
    exposure: np.ndarray | None           # policy-years; None if not applicable
    sample_weight: np.ndarray             # exposure (freq/pp), ClaimNb (sev), or ones
    task: TaskSpec
    group_ids: np.ndarray | None = None   # policy IDs for group-aware splitting
    cat_features: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # source, license, hash, n_rows, notes_path, ...

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y, dtype=float)
        self.sample_weight = np.asarray(self.sample_weight, dtype=float)
        if self.exposure is not None:
            self.exposure = np.asarray(self.exposure, dtype=float)
        self.validate()

    # -- invariants -----------------------------------------------------------------------
    def validate(self) -> None:
        n = len(self.X)
        if not (len(self.y) == len(self.sample_weight) == n):
            raise ValueError("X, y and sample_weight length mismatch")
        if self.exposure is not None and len(self.exposure) != n:
            raise ValueError("exposure length mismatch")
        if self.group_ids is not None and len(self.group_ids) != n:
            raise ValueError("group_ids length mismatch")
        self._assert_no_exposure_leak()

    def _assert_no_exposure_leak(self) -> None:
        """Exposure (or log-exposure) must not appear as a feature (SPEC §12.1)."""
        if self.exposure is None or self.meta.get("_subset"):
            return
        banned = {"exposure", "expo", "exp_weights", "insured"}
        for col in self.X.columns:
            if str(col).strip().lower() in banned:
                raise ValueError(f"exposure-like column {col!r} present in X — exposure is scale, not a feature")
        log_exp = np.log(np.clip(self.exposure, 1e-12, None))
        for col in self.X.select_dtypes(include="number").columns:
            v = self.X[col].to_numpy(dtype=float)
            if np.nanstd(v) == 0:
                continue
            with np.errstate(invalid="ignore"):
                c = np.corrcoef(np.nan_to_num(v), log_exp)[0, 1]
            if np.isfinite(c) and abs(c) > 0.999:
                raise ValueError(f"feature {col!r} is ~collinear with log(exposure) (|corr|={c:.4f})")

    @property
    def n_groups(self) -> int:
        if self.group_ids is None:
            return len(self.X)
        return int(len(np.unique(self.group_ids)))

    def subset(self, idx) -> "DatasetBundle":
        """Cheap row subset for a CV split (skips the expensive leak re-check)."""
        idx = np.asarray(idx)
        return DatasetBundle(
            name=self.name,
            X=self.X.iloc[idx],
            y=self.y[idx],
            exposure=None if self.exposure is None else self.exposure[idx],
            sample_weight=self.sample_weight[idx],
            task=self.task,
            group_ids=None if self.group_ids is None else self.group_ids[idx],
            cat_features=list(self.cat_features),
            meta={**self.meta, "_subset": True},
        )


@runtime_checkable
class DatasetLoader(Protocol):
    name: str

    def load(self) -> DatasetBundle: ...

    def source_hash(self) -> str: ...
