"""Benchmark config (YAML) — profiles control runtime (SPEC §5.7)."""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml

_DEFAULT_SPLIT = {
    "small_threshold": 2500,
    "small": {"folds": 3, "repeats": 3},
    "large": {"folds": 3, "repeats": 1},
    "group_aware": True,
}


def _parse_inner_repeats(v) -> int | str:
    """"auto" (size-adaptive, see hpo.resolve_inner_repeats) or an explicit positive int."""
    if v is None:
        return "auto"
    if isinstance(v, str):
        if v.strip().lower() != "auto":
            raise ValueError(f"hpo.inner_repeats must be a positive int or 'auto', got {v!r}")
        return "auto"
    return max(1, int(v))


@dataclass
class Config:
    name: str = "insur-arena-v0.1"
    profile: str = "lite"
    datasets: list[str] = field(default_factory=lambda: ["freMTPL2freq"])
    models: list[dict] = field(default_factory=list)
    split: dict = field(default_factory=lambda: dict(_DEFAULT_SPLIT))
    n_configs: int = 32
    inner_folds: int = 4          # k-fold OOF for tuned config selection (TabArena uses 8)
    inner_repeats: int | str = "auto"  # repeat inner CV on small tune splits ("auto" = size-based)
    tune_split_id: int = 0        # legacy private helper only; campaign selects per outer split
    time_budget_s: float | None = None  # soft per tuned-once config-search budget; None = no cap
    # Tuning-speed levers (design/tuning-speed-roadmap.md in tri-boost, verified 2026-07-15):
    fold_workers: int = 1         # A1: threads per candidate eval for its inner folds (wall-clock
                                  # only, scores bit-identical). Total threads per search worker is
                                  # ~fold_workers x per-fit threads — size against max_workers.
    racing_se: float | None = None  # A2: fold-1 discard threshold in pooled paired-diff SDs
                                  # (2.0 replay-verified: 24-27% eval time saved, 0 selection
                                  # changes). None = off. Selection-affecting -> keyed in _hpo_proto.
    racing_min_rows: int | None = None  # A2 safety gate: suppress racing below this many dataset
                                  # rows. Fold-1 discard is unsound exactly where `inner_repeats:
                                  # auto` has already declared the tune split too noisy to score
                                  # once — see runner._racing_se_for for the 45k derivation.
                                  # None = no gate (racing wherever racing_se is set).
    selection_max_rows: int | None = None  # A3: cap the rows the SEARCH scores on (whole groups,
                                  # stratified, deterministic); outer-split deploy fits always
                                  # refit on the full outer-train. None = off (search on all rows).
                                  # Selection-affecting -> keyed in _proto AND _hpo_config_proto.
    subsample_rows: int | None = 100_000
    cache_dir: str = "~/.cache/insurarena"
    store_predictions: bool = True
    use_cache: bool = True        # reuse per-cell results across runs (only re-run changed cells)
    max_workers: int = 1          # >1 -> compute cells (model x variant x split) in parallel processes
    threads_per_worker: int = 0   # OMP threads inside each worker fit; 0 = auto (cpu // max_workers)
    evaluation_mode: str = "accuracy"  # "timing": serial fresh fits with warmed prediction repeats
    timing_repeats: int = 7
    timing_batch_size: int = 1024
    segment_columns: list[str] = field(default_factory=list)  # explicit prediction artifact allowlist
    include_ineligible_datasets: bool = False  # historical reproduction only

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        split = dict(_DEFAULT_SPLIT)
        split.update(raw.get("split", {}))
        storage = raw.get("storage", {})
        hpo = raw.get("hpo", {})
        return cls(
            name=raw.get("benchmark", "insur-arena-v0.1"),
            profile=raw.get("profile", "lite"),
            datasets=raw.get("datasets", ["freMTPL2freq"]),
            include_ineligible_datasets=bool(raw.get("include_ineligible_datasets", False)),
            models=raw.get("models", []),
            evaluation_mode=raw.get("evaluation_mode", "accuracy"),
            timing_repeats=int(raw.get("timing", {}).get("repeats", 7)),
            timing_batch_size=int(raw.get("timing", {}).get("batch_size", 1024)),
            segment_columns=list(storage.get("segment_columns", [])),
            split=split,
            n_configs=int(hpo.get("n_configs", raw.get("n_configs", 32))),
            inner_folds=int(hpo.get("inner_folds", 4)),
            inner_repeats=_parse_inner_repeats(hpo.get("inner_repeats", "auto")),
            tune_split_id=int(hpo.get("tune_split_id", 0)),
            time_budget_s=hpo.get("time_budget_s", None),
            fold_workers=int(hpo.get("fold_workers", 1)),
            racing_se=(None if hpo.get("racing_se") in (None, 0, False) else float(hpo["racing_se"])),
            racing_min_rows=(
                None
                if hpo.get("racing_min_rows") in (None, 0, False)
                else int(hpo["racing_min_rows"])
            ),
            selection_max_rows=(
                None
                if hpo.get("selection_max_rows") in (None, 0, False)
                else int(hpo["selection_max_rows"])
            ),
            subsample_rows=raw.get("subsample_rows", None),
            cache_dir=storage.get("cache_dir", "~/.cache/insurarena"),
            store_predictions=bool(storage.get("store_predictions", True)),
            use_cache=bool(storage.get("use_cache", True)),
            max_workers=int(storage.get("max_workers", raw.get("max_workers", 1))),
            threads_per_worker=int(storage.get("threads_per_worker", raw.get("threads_per_worker", 0))),
        )
