"""Public contract and negative controls for ``calibration_report``."""

from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

from calibre import calibration_report


@pytest.fixture
def heldout_sample():
    """Independent held-out outcomes from calibrated probability forecasts."""
    rng = np.random.default_rng(83)
    predictions = rng.uniform(0.01, 0.99, 600)
    outcomes = rng.binomial(1, predictions).astype(float)
    return outcomes, predictions


def test_api_uses_descriptive_fields_and_keyword_only_options(heldout_sample):
    """The report should use the same names as its component functions."""
    signature = inspect.signature(calibration_report)
    assert list(signature.parameters) == [
        "y_true",
        "y_pred",
        "n_bins",
        "include_brier_interval",
        "interval_level",
        "interval_n_resamples",
        "random_state",
        "interval_method",
    ]
    for name in ("y_true", "y_pred"):
        assert (
            signature.parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
    for name in list(signature.parameters)[2:]:
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    report = calibration_report(*heldout_sample)
    assert [field.name for field in fields(report)] == [
        "n_observations",
        "base_rate",
        "mean_prediction",
        "mean_calibration_error",
        "brier_score",
        "miscalibration",
        "discrimination",
        "uncertainty",
        "smooth_calibration_error",
        "smooth_calibration_bandwidth",
        "debiased_calibration_error",
        "plugin_calibration_error",
        "sweep_calibration_error",
        "sweep_n_bins",
        "n_bins",
        "n_unique_predictions",
        "unique_prediction_ratio",
        "intervals",
    ]
    for old_name in (
        "bias",
        "brier",
        "mcb",
        "dsc",
        "unc",
        "smece",
        "smece_sigma",
        "debiased_ece",
        "plugin_ece",
        "sweep_ece",
        "sweep_bins",
        "n_distinct",
        "distinct_ratio",
    ):
        assert not hasattr(report, old_name)


@pytest.mark.parametrize("value", ["yes", 1, 0, None])
def test_interval_switch_must_be_boolean(heldout_sample, value):
    """Truthiness is not an input contract."""
    with pytest.raises(TypeError, match="include_brier_interval must be boolean"):
        calibration_report(*heldout_sample, include_brier_interval=value)


def test_report_intervals_are_deeply_immutable(heldout_sample):
    """A frozen report must not contain an editable confidence interval."""
    report = calibration_report(
        *heldout_sample,
        include_brier_interval=True,
        interval_n_resamples=30,
    )
    assert set(report.intervals) == {"brier_score"}
    with pytest.raises(TypeError):
        report.intervals["brier_score"]["lower"] = 999.0  # type: ignore[index]
    with pytest.raises(TypeError):
        report.intervals["new_metric"] = {}  # type: ignore[index]

    plain = report.to_dict()
    assert isinstance(plain["intervals"], dict)
    assert isinstance(plain["intervals"]["brier_score"], dict)


def test_reversed_forecast_exposes_the_sweep_assumption():
    """A one-bin sweep result must not masquerade as general calibration."""
    rng = np.random.default_rng(91)
    predictions = rng.uniform(0.01, 0.99, 2000)
    outcomes = rng.binomial(1, 1.0 - predictions).astype(float)

    report = calibration_report(outcomes, predictions)
    assert report.plugin_calibration_error > 0.5
    assert report.sweep_calibration_error < 0.02
    assert report.sweep_n_bins == 1
    text = str(report)
    assert "(1 bin; assumes a monotone calibration curve)" in text


def test_prediction_counts_are_described_as_granularity(heldout_sample):
    """Unique-value counts are not calibration or statistical resolution."""
    text = str(calibration_report(*heldout_sample))
    assert "prediction granularity" in text
    assert "distinct values" not in text
