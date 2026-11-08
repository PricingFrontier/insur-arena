import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.models.base import MIN_FIT_EXPOSURE, filter_tiny_exposure_fit_rows
from insurarena.tasks import CLF_SPEC, FREQ_SPEC, PP_SPEC


def _bundle(task):
    exposure = np.array([0.0, MIN_FIT_EXPOSURE - 0.001, MIN_FIT_EXPOSURE, 0.2, np.nan])
    return DatasetBundle(
        name="fit_exposure_probe",
        X=pd.DataFrame({"x": [10, 20, 30, 40, 50]}),
        y=np.arange(5, dtype=float),
        exposure=exposure,
        sample_weight=exposure,
        task=task,
        meta={"_subset": True},
    )


def test_fit_exposure_filter_keeps_rate_task_rows_at_or_above_floor():
    bundle = _bundle(FREQ_SPEC)

    filtered = filter_tiny_exposure_fit_rows(bundle)

    assert filtered.X["x"].tolist() == [30, 40]
    assert filtered.y.tolist() == [2.0, 3.0]
    assert filtered.exposure.tolist() == [MIN_FIT_EXPOSURE, 0.2]
    assert filtered.sample_weight.tolist() == [MIN_FIT_EXPOSURE, 0.2]
    assert len(bundle.X) == 5


def test_fit_exposure_filter_applies_to_pure_premium():
    filtered = filter_tiny_exposure_fit_rows(_bundle(PP_SPEC))

    assert filtered.X["x"].tolist() == [30, 40]


def test_fit_exposure_filter_does_not_apply_to_classification():
    bundle = _bundle(CLF_SPEC)

    assert filter_tiny_exposure_fit_rows(bundle) is bundle
