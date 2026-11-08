"""Result + prediction storage. Parquet if pyarrow present, else CSV. No ensemble simulation."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from insurarena.results.schema import ResultRow


class ResultStore:
    def __init__(self, cache_dir: str = "~/.cache/insurarena", store_predictions: bool = True):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.pred_dir = self.cache_dir / "predictions"
        self.store_predictions = store_predictions
        self.rows: list[dict] = []           # flat result dicts (cached or fresh)
        self.trajectories: list[dict] = []   # tuning trajectories (best-so-far vs #configs)
        self.hpo_evaluations: list[dict] = []  # scored tuned candidate configs
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if store_predictions:
            self.pred_dir.mkdir(parents=True, exist_ok=True)

    def write(self, row: ResultRow) -> None:
        self.rows.append(row.to_dict())

    def add(self, row_dict: dict) -> None:
        self.rows.append(dict(row_dict))

    def write_trajectory(self, dataset: str, model: str, split_id: int, traj: list) -> None:
        # None-safe: traj may already carry None (NaN converted upstream / read back from cache).
        self.trajectories.append({"dataset": dataset, "model": model, "split_id": split_id,
                                  "trajectory": [None if (t is None or t != t) else float(t) for t in traj]})

    def write_hpo_evaluations(self, dataset: str, model: str, split_id: int, evaluations: list[dict]) -> None:
        for ev in evaluations:
            self.hpo_evaluations.append({
                "dataset": dataset,
                "model": model,
                "variant": "tuned",
                "split_id": int(split_id),
                **dict(ev),
            })

    def write_predictions(self, key: str, y_true, y_pred, exposure) -> None:
        if not self.store_predictions:
            return
        np.savez_compressed(
            self.pred_dir / f"{key}.npz",
            y_true=np.asarray(y_true, dtype=float),
            y_pred=np.asarray(y_pred, dtype=float),
            exposure=np.asarray(exposure, dtype=float) if exposure is not None else np.array([]),
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self) -> Path:
        df = self.to_frame()
        try:
            import pyarrow  # noqa: F401

            path = self.cache_dir / "results.parquet"
            df.to_parquet(path, index=False)
        except Exception:
            path = self.cache_dir / "results.csv"
            df.to_csv(path, index=False)
        if self.trajectories:
            import json

            (self.cache_dir / "trajectories.json").write_text(json.dumps(self.trajectories))
        if self.hpo_evaluations:
            import json

            def _default(v):
                if isinstance(v, np.generic):
                    return v.item()
                return str(v)

            (self.cache_dir / "hpo_evaluations.json").write_text(
                json.dumps(self.hpo_evaluations, default=_default)
            )
        return path
