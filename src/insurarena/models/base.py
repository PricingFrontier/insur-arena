"""ModelAdapter contract. Each adapter = ONE fitted model, no benchmark-level ensembling."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.tasks import TaskType

MIN_FIT_EXPOSURE = 0.05


def filter_tiny_exposure_fit_rows(bundle: DatasetBundle) -> DatasetBundle:
    """Drop tiny exposure rows from GBM fit-time data, leaving benchmark scoring data untouched."""
    if bundle.task.task not in (TaskType.FREQ, TaskType.PP) or bundle.exposure is None:
        return bundle
    keep = np.isfinite(bundle.exposure) & (bundle.exposure >= MIN_FIT_EXPOSURE)
    if keep.all() or not keep.any():
        return bundle
    return bundle.subset(np.flatnonzero(keep))


@dataclass
class Explanation:
    """Optional model-inspection output (GLM coefficients, GBM gains). Diagnostic, not scored."""

    kind: str                                  # "coefficients" | "gain" | "none"
    global_importance: dict = field(default_factory=dict)
    artifact: object = None


class ModelAdapter(ABC):
    name: str = "abstract"
    family: str = "model"                      # neutral technique label, e.g. "glm", "gbm"
    adapter_version: str = "1"                 # bump to invalidate just this adapter's cached cells
    hpo_search_version: str = "1"              # bump for HPO search/evaluation changes only
    needs_es_probe: bool = True                # False for models that early-stop internally (skip the
                                               # inner-holdout probe in fit_one -> no wasted refit)
    supported_tasks: tuple[TaskType, ...] = ()

    def __init__(self, hp: dict | None = None):
        self.hp = dict(hp or {})
        self._fitted = False
        self.best_iteration_ = 0               # early-stopping iteration count (0 = N/A)

    # -- lifecycle ------------------------------------------------------------------------
    @abstractmethod
    def fit(self, train: DatasetBundle, val: DatasetBundle | None = None) -> "ModelAdapter": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, exposure: np.ndarray | None = None) -> np.ndarray:
        """Return predictions on the NATURAL scale (rate / cost / probability)."""

    def explain(self) -> Explanation:
        return Explanation(kind="none")

    # -- hyper-parameters -----------------------------------------------------------------
    @staticmethod
    def default_hp() -> dict:
        return {}

    @staticmethod
    def sample_hp(rng: np.random.Generator) -> dict:
        return {}

    @staticmethod
    def tuning_base_hp() -> dict | None:
        """Optional base config for greedy one-parameter-at-a-time tuning."""
        return None

    @staticmethod
    def coordinate_tuning_stages() -> list[dict]:
        """Optional ordered coordinate-tuning stages.

        Each stage is {"name": hp_name, "values": [...]}. The HPO loop evaluates one stage at a
        time, keeps the best value, then moves to the next stage.
        """
        return []

    @staticmethod
    def ship_gate_se_mult(inner_repeats: int, champion_hp: dict | None = None) -> float:
        """How many paired-fold SEs the champion must beat the default anchor by (hsv 19).

        Returning 1.0 — the default for every adapter — is the shared `_GATE_SE_MULT` rule that
        has always applied, so an adapter that does not override this keeps byte-identical
        selections and its cached tuned cells stay valid. Overriding it changes which config
        ships, so an adapter that does MUST bump its own `hpo_search_version`.

        `inner_repeats` is the RESOLVED repeat count for the cell (`hpo.resolve_inner_repeats`) —
        the arena's own measure of how far the tune split falls short of the 30k-scored-row
        target, i.e. a ready-made volatility flag that already keys the cell.

        `champion_hp` (hsv 20) is the hp dict of the candidate actually under test, so the bar can
        depend on WHAT is being shipped and not only on how noisy the cell is. It is None wherever
        the champion is not yet resolved; an adapter reading it must treat None as "no
        candidate-specific premium". Every adapter that ignores it — this base implementation
        included — stays byte-identical to hsv 19.
        """
        return 1.0

    @staticmethod
    def finale_rerace_spec() -> dict | None:
        """Opt in to the TOP-K FINALE RE-RACE (hpo_search_version 23, 2026-09-01).

        `None` — the default for every adapter — keeps the pre-hsv-23 protocol exactly: the scout's
        argmax is the only candidate re-scored at finale fidelity, and the cached tuned cells of
        every model that does not override this stay valid and byte-identical.

        Return `{"k": <max finale contenders>, "band": <admission width in paired-fold SEs>}` to
        turn it on. The scout ranks candidates at reduced fidelity, so where it cannot separate
        rank 1 from its runners-up, the runners-up are re-scored at finale fidelity too and the
        winner of THAT is the candidate the ship gate judges. See `hpo._RERACE_K` for the measured
        record behind k=4 / band=1.0 and for the pg16_payfreq counterexample that puts the
        max-of-N boundary at 4.

        Turning it on changes which config ships, so an adapter that does MUST bump its own
        `hpo_search_version` in the same change — the hook is explicit precisely so that the
        shared selection code cannot silently re-key a model whose version did not move.
        """
        return None

    @staticmethod
    def selection_hp() -> dict:
        """Optional selection-only hp overlay (roadmap B3, verified 2026-07-15).

        Merged into EVERY candidate's hp — the default anchor included, so the ship-gate
        stays a matched comparison — inside `hpo._oof_score` only: deploy fits and the
        selected/cached hp never carry these keys. Selection needs ranking, not deployment
        fidelity; adapters use this for cheaper selection-fit settings (e.g. a tighter
        early-stopping patience). Changing an adapter's selection_hp changes its inner
        scores -> bump `hpo_search_version`.
        """
        return {}
