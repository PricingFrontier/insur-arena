"""One result row per (dataset, task, model, variant, split)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResultRow:
    dataset: str
    task: str
    source_hash: str
    model: str
    variant: str            # default | tuned
    family: str             # neutral technique label (glm | gbm | ...)
    split_id: int
    n_train: int
    n_test: int
    primary_metric: str
    primary_value: float
    skill_vs_glm: float
    train_time_s: float
    seed: int
    inference_time_s: float = float("nan")   # predict cost (efficiency / Pareto view)
    failed: str = ""                          # "" if ok, else error type:message (A4)
    secondary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "secondary"}
        for name, val in self.secondary.items():
            d[f"sec_{name}"] = val
        return d
