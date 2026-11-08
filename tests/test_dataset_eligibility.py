import pandas as pd
import pytest

import insurarena.eval.runner as runner
from insurarena.cli import _filter_ineligible_datasets
from insurarena.config import Config
from insurarena.datasets.manifest import MANIFEST
from insurarena.datasets.registry import (
    MIN_BENCHMARK_PREDICTORS,
    MIN_BENCHMARK_ROWS,
    benchmark_exclusion_reason,
    is_benchmark_eligible,
    list_benchmark_datasets,
    list_datasets,
)
from insurarena.datasets.spec import _entry_payload
from insurarena.eval.runner import _selected_dataset_names


EXCLUDED = {"mass_insurance", "danishuni", "norfire", "lossalae", "ohlsson_sev"}


def test_benchmark_catalog_applies_full_data_minimums():
    eligible = set(list_benchmark_datasets())
    all_real = set(list_datasets(include_synthetic=False))

    assert len(all_real) == 45
    assert len(eligible) == 40
    assert all_real - eligible == EXCLUDED
    assert all(entry.full_n_rows is not None for entry in MANIFEST)
    assert all(entry.full_n_predictors is not None for entry in MANIFEST)


def test_eligibility_boundaries_are_inclusive():
    assert MIN_BENCHMARK_ROWS == 1000
    assert MIN_BENCHMARK_PREDICTORS == 4
    assert is_benchmark_eligible("credit_g")  # exactly 1,000 rows
    assert is_benchmark_eligible("norauto")  # exactly 4 predictors
    assert not is_benchmark_eligible("mass_insurance")
    assert "64 rows" in benchmark_exclusion_reason("mass_insurance")
    assert "3 predictors" in benchmark_exclusion_reason("mass_insurance")


def test_ineligible_loaders_need_explicit_historical_opt_in():
    cfg = Config(datasets=["mass_insurance", "credit_g"])
    assert _selected_dataset_names(cfg, verbose=False) == ["credit_g"]

    cfg.include_ineligible_datasets = True
    assert _selected_dataset_names(cfg, verbose=False) == ["mass_insurance", "credit_g"]


@pytest.mark.parametrize("max_workers", [1, 2])
def test_runner_skips_ineligible_dataset_before_loading(tmp_path, monkeypatch, max_workers):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("ineligible dataset was loaded")

    monkeypatch.setattr(runner, "build_loader", unexpected_load)
    cfg = Config(
        datasets=["mass_insurance"],
        models=[],
        cache_dir=str(tmp_path),
        max_workers=max_workers,
    )

    assert runner.run_benchmark(cfg, verbose=False).empty


def test_historical_opt_in_loads_from_yaml(tmp_path):
    path = tmp_path / "historical.yaml"
    path.write_text("datasets: [mass_insurance]\ninclude_ineligible_datasets: true\n")

    cfg = Config.from_yaml(str(path))
    assert cfg.include_ineligible_datasets
    assert _selected_dataset_names(cfg, verbose=False) == ["mass_insurance"]


def test_cached_report_filter_defaults_to_eligible_datasets():
    frame = pd.DataFrame({"dataset": ["mass_insurance", "credit_g"], "primary_value": [1.0, 2.0]})
    filtered, excluded = _filter_ineligible_datasets(frame)

    assert filtered["dataset"].tolist() == ["credit_g"]
    assert excluded == ["mass_insurance"]


def test_eligibility_metadata_does_not_change_dataset_cache_identity():
    payload = _entry_payload(MANIFEST[0])
    assert "full_n_rows" not in payload
    assert "full_n_predictors" not in payload
