"""Task taxonomy — each task is a contract (target, exposure role, distribution, metrics)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    FREQ = "freq"   # Poisson claim frequency (exposure offset/weight)
    SEV = "sev"     # Gamma severity (conditional on a claim)
    PP = "pp"       # Tweedie pure premium
    CLF = "clf"     # Bernoulli claim occurrence / default
    MULTICLF = "multiclf"  # multinomial: >2 classes (e.g. underwriting risk grade, coverage tier)


@dataclass(frozen=True)
class TaskSpec:
    """Immutable contract for a task on a given dataset."""

    task: TaskType
    distribution: str                 # "poisson" | "gamma" | "tweedie" | "bernoulli"
    link: str                         # "log" | "logit"
    primary_metric: str               # key into metrics.registry
    secondary_metrics: tuple[str, ...] = ()
    exposure_role: str = "weight"     # "weight" | "offset" | "none"
    tweedie_power: float | None = None  # fixed scoring power for pp (e.g. 1.9)


# --- Canonical task specs (MVP ships FREQ; others defined for Phase 2) -------------------

FREQ_SPEC = TaskSpec(
    task=TaskType.FREQ,
    distribution="poisson",
    link="log",
    primary_metric="poisson_deviance",
    secondary_metrics=("d2_poisson", "gini_exposure", "normalized_gini", "balance", "max_decile_balance_dev"),
    exposure_role="weight",
)

SEV_SPEC = TaskSpec(
    task=TaskType.SEV,
    distribution="gamma",
    link="log",
    primary_metric="gamma_deviance",
    secondary_metrics=("d2_gamma", "gini_exposure", "balance"),
    exposure_role="none",
)

PP_SPEC = TaskSpec(
    task=TaskType.PP,
    distribution="tweedie",
    link="log",
    primary_metric="tweedie_deviance",
    secondary_metrics=("d2_tweedie", "gini_exposure", "normalized_gini", "balance", "max_decile_balance_dev"),
    exposure_role="weight",
    tweedie_power=1.9,
)

CLF_SPEC = TaskSpec(
    task=TaskType.CLF,
    distribution="bernoulli",
    link="logit",
    primary_metric="log_loss",
    secondary_metrics=("normalized_gini", "roc_auc", "balance"),
    exposure_role="weight",
)

MULTICLF_SPEC = TaskSpec(
    task=TaskType.MULTICLF,
    distribution="multinomial",
    link="logit",
    primary_metric="multiclass_log_loss",   # multinomial cross-entropy, consumes an (n, K) prob matrix
    secondary_metrics=("multiclass_accuracy",),  # Gini/balance don't extend to K>2 -> omitted
    exposure_role="weight",
)

TASK_SPECS: dict[TaskType, TaskSpec] = {
    TaskType.FREQ: FREQ_SPEC,
    TaskType.SEV: SEV_SPEC,
    TaskType.PP: PP_SPEC,
    TaskType.CLF: CLF_SPEC,
    TaskType.MULTICLF: MULTICLF_SPEC,
}
