"""Tests for ``bootstrap_ci`` and ``calibration_report``."""

from __future__ import annotations

import numpy as np
import pytest

from calibre import (
    IsotonicCalibrator,
    bootstrap_ci,
    calibration_report,
    cross_val_calibrate,
    score_decomposition,
)
from calibre.metrics import (
    brier_score,
    debiased_calibration_error,
    smooth_calibration_error,
)
from calibre.report import CalibrationReport


@pytest.fixture
def calibrated():
    """Predictions calibrated by construction.

    Returns
    -------
    tuple of ndarray
        Labels and predictions.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1500)
    return rng.binomial(1, p).astype(float), p


# --------------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------------- #


def test_interval_brackets_the_estimate_for_a_smooth_metric(calibrated):
    """For a mean-like functional the percentile interval contains the estimate."""
    y, p = calibrated
    result = bootstrap_ci(brier_score, y, p, n_resamples=300)
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_estimate_is_the_metric_on_the_observed_data(calibrated):
    """No resampling noise leaks into the point estimate."""
    y, p = calibrated
    result = bootstrap_ci(brier_score, y, p, n_resamples=50)
    assert result["estimate"] == pytest.approx(brier_score(y, p))


def test_interval_is_reproducible(calibrated):
    """Same seed, same interval."""
    y, p = calibrated
    a = bootstrap_ci(brier_score, y, p, n_resamples=100, random_state=7)
    b = bootstrap_ci(brier_score, y, p, n_resamples=100, random_state=7)
    assert a == b


def test_a_different_seed_moves_the_interval(calibrated):
    """The interval really is resampled, not computed analytically."""
    y, p = calibrated
    a = bootstrap_ci(brier_score, y, p, n_resamples=100, random_state=1)
    b = bootstrap_ci(brier_score, y, p, n_resamples=100, random_state=2)
    assert (a["lower"], a["upper"]) != (b["lower"], b["upper"])


def test_a_wider_level_gives_a_wider_interval(calibrated):
    """Nominal coverage has to do something."""
    y, p = calibrated
    narrow = bootstrap_ci(brier_score, y, p, level=0.5, n_resamples=400)
    wide = bootstrap_ci(brier_score, y, p, level=0.99, n_resamples=400)
    assert wide["upper"] - wide["lower"] > narrow["upper"] - narrow["lower"]


def test_more_data_gives_a_tighter_interval():
    """The interval must shrink as the sample grows."""
    rng = np.random.default_rng(0)
    widths = []
    for n in (500, 8000):
        p = rng.uniform(0, 1, n)
        y = rng.binomial(1, p).astype(float)
        result = bootstrap_ci(brier_score, y, p, n_resamples=300)
        widths.append(result["upper"] - result["lower"])
    assert widths[1] < widths[0]


def test_works_on_a_user_supplied_metric(calibrated):
    """Any callable of (y_true, y_pred) is accepted."""
    y, p = calibrated
    result = bootstrap_ci(
        lambda t, q: float(np.mean(q) - np.mean(t)), y, p, n_resamples=100
    )
    assert result["lower"] <= result["estimate"] <= result["upper"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"level": 0.0}, "level must be in"),
        ({"level": 1.0}, "level must be in"),
        ({"n_resamples": 1}, "n_resamples must be at least 2"),
    ],
)
def test_bad_arguments_are_rejected(calibrated, kwargs, match):
    """Nonsense arguments fail loudly."""
    y, p = calibrated
    with pytest.raises(ValueError, match=match):
        bootstrap_ci(brier_score, y, p, **kwargs)


# --------------------------------------------------------------------------- #
# calibration_report
# --------------------------------------------------------------------------- #


def test_report_fields_match_the_underlying_estimators(calibrated):
    """The report must not recompute anything differently."""
    y, p = calibrated
    report = calibration_report(y, p)
    decomposition = score_decomposition(y, p)

    assert report.n_observations == y.size
    assert report.brier_score == pytest.approx(brier_score(y, p))
    assert report.miscalibration == pytest.approx(decomposition["miscalibration"])
    assert report.discrimination == pytest.approx(decomposition["discrimination"])
    assert report.uncertainty == pytest.approx(decomposition["uncertainty"])
    assert report.smooth_calibration_error == pytest.approx(
        smooth_calibration_error(y, p)
    )
    assert report.debiased_calibration_error == pytest.approx(
        debiased_calibration_error(y, p, n_bins=15)
    )
    assert report.base_rate == pytest.approx(float(np.mean(y)))
    assert report.n_unique_predictions == int(np.unique(p).size)


def test_the_decomposition_identity_holds_in_the_report(calibrated):
    """``mean_score = MCB - DSC + UNC``, exactly."""
    y, p = calibrated
    report = calibration_report(y, p)
    assert (
        report.miscalibration - report.discrimination + report.uncertainty
    ) == pytest.approx(
        report.brier_score,
        abs=1e-12,
    )


def test_plugin_is_never_below_debiased(calibrated):
    """The correction only ever subtracts."""
    y, p = calibrated
    report = calibration_report(y, p)
    assert report.plugin_calibration_error >= report.debiased_calibration_error


def test_report_catches_an_overconfident_model(calibrated):
    """A distorted forecaster must score worse on every error estimator."""
    y, p = calibrated
    squashed = np.clip(2.0 * (p - 0.5) + 0.5, 0, 1)

    honest = calibration_report(y, p)
    bad = calibration_report(y, squashed)

    assert bad.miscalibration > honest.miscalibration
    assert bad.smooth_calibration_error > honest.smooth_calibration_error
    assert bad.debiased_calibration_error > honest.debiased_calibration_error
    assert bad.brier_score > honest.brier_score


def test_intervals_are_absent_by_default(calibrated):
    """Bootstrapping is opt-in; it costs n_resamples recomputations."""
    y, p = calibrated
    assert calibration_report(y, p).intervals == {}


def test_report_intervals_only_the_regular_proper_score(calibrated):
    """Ordinary row bootstrap is not offered for non-smooth calibration errors."""
    y, p = calibrated
    report = calibration_report(
        y,
        p,
        include_brier_interval=True,
        interval_n_resamples=100,
    )
    assert set(report.intervals) == {"brier_score"}


def test_the_brier_interval_brackets_its_estimate(calibrated):
    """A proper scoring rule is a plain mean, so the bootstrap behaves."""
    y, p = calibrated
    report = calibration_report(
        y,
        p,
        include_brier_interval=True,
        interval_n_resamples=200,
    )
    interval = report.intervals["brier_score"]
    assert interval["lower"] <= report.brier_score <= interval["upper"]


def test_the_printed_report_names_the_interval_scope(calibrated):
    """The text must not imply that every reported metric has an interval."""
    y, p = calibrated
    text = str(
        calibration_report(
            y,
            p,
            include_brier_interval=True,
            interval_n_resamples=50,
        )
    )
    assert "Brier only" in text
    assert "bca" in text


def test_report_prints_its_numbers(calibrated):
    """The text form is the point of the object; it must carry the values."""
    y, p = calibrated
    report = calibration_report(y, p)
    text = str(report)
    assert f"{report.brier_score:.4f}" in text
    assert f"{report.miscalibration:.4f}" in text
    assert f"{report.smooth_calibration_error:.4f}" in text
    assert "irreducible" in text
    assert repr(report) == text


def test_to_dict_round_trips(calibrated):
    """Every field survives conversion, for a DataFrame row or JSON."""
    y, p = calibrated
    report = calibration_report(y, p)
    as_dict = report.to_dict()
    assert as_dict["brier_score"] == report.brier_score
    assert set(as_dict) >= {
        "n_observations",
        "brier_score",
        "miscalibration",
        "discrimination",
        "uncertainty",
        "smooth_calibration_error",
    }


def test_report_is_immutable(calibrated):
    """A summary that can be edited after the fact is a liability."""
    y, p = calibrated
    with pytest.raises((AttributeError, TypeError)):
        calibration_report(y, p).brier_score = 0.0  # type: ignore[misc]


def test_in_sample_mcb_is_zero_but_out_of_fold_is_not():
    """The trap the report's docstring warns about, asserted.

    PAV is idempotent, so scoring an isotonic fit on its own training data gives
    MCB of exactly zero however badly the model generalizes.
    """
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 1500)
    labels = rng.binomial(1, scores).astype(float)

    in_sample = IsotonicCalibrator().fit(scores, labels).transform(scores)
    out_of_fold = cross_val_calibrate(IsotonicCalibrator(), scores, labels, cv=5)

    assert calibration_report(labels, in_sample).miscalibration == pytest.approx(
        0.0, abs=1e-12
    )
    assert calibration_report(labels, out_of_fold).miscalibration > 0.0


def test_bad_bin_count_is_rejected(calibrated):
    """Zero bins is not a binning."""
    y, p = calibrated
    with pytest.raises(ValueError, match="n_bins must be at least 1"):
        calibration_report(y, p, n_bins=0)


def test_report_is_a_dataclass_instance(calibrated):
    """Typed access, not a bare dict."""
    y, p = calibrated
    assert isinstance(calibration_report(y, p), CalibrationReport)
