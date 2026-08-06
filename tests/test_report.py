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


def test_the_naive_bootstrap_is_inconsistent_for_mcb(calibrated):
    """Document the trap that keeps ``MCB`` out of ``calibration_report``.

    Resampling rows with replacement leaves only about 63% of them distinct. PAV
    pools the duplicates and overfits, so the bootstrap distribution of ``MCB``
    sits well above the observed value -- the naive bootstrap is inconsistent for
    functionals of an isotonic fit.

    This test asserts the failure so that nobody "fixes" the report by adding MCB
    back without also fixing the resampling scheme.
    """
    y, p = calibrated
    observed = score_decomposition(p, y)["MCB"]
    result = bootstrap_ci(
        lambda t, q: score_decomposition(q, t)["MCB"], y, p, n_resamples=300
    )
    assert result["lower"] > observed, (
        "the naive bootstrap no longer inflates MCB; if the resampling scheme "
        "changed, revisit calibration_report's excluded targets"
    )


def test_the_same_bootstrap_is_unbiased_for_brier(calibrated):
    """The contrast that shows the inflation is the estimator, not the code."""
    y, p = calibrated
    rng = np.random.default_rng(1)
    n = y.size
    draws = np.array(
        [brier_score(y[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(300))]
    )
    assert draws.mean() == pytest.approx(brier_score(y, p), rel=0.02)


# --------------------------------------------------------------------------- #
# calibration_report
# --------------------------------------------------------------------------- #


def test_report_fields_match_the_underlying_estimators(calibrated):
    """The report must not recompute anything differently."""
    y, p = calibrated
    report = calibration_report(y, p)
    decomposition = score_decomposition(p, y)

    assert report.n == y.size
    assert report.brier == pytest.approx(brier_score(y, p))
    assert report.mcb == pytest.approx(decomposition["MCB"])
    assert report.dsc == pytest.approx(decomposition["DSC"])
    assert report.unc == pytest.approx(decomposition["UNC"])
    assert report.smece == pytest.approx(smooth_calibration_error(y, p))
    assert report.debiased_ece == pytest.approx(debiased_calibration_error(y, p, 15))
    assert report.base_rate == pytest.approx(float(np.mean(y)))
    assert report.n_distinct == int(np.unique(p).size)


def test_the_decomposition_identity_holds_in_the_report(calibrated):
    """``mean_score = MCB - DSC + UNC``, exactly."""
    y, p = calibrated
    report = calibration_report(y, p)
    assert report.mcb - report.dsc + report.unc == pytest.approx(
        report.brier, abs=1e-12
    )


def test_plugin_is_never_below_debiased(calibrated):
    """The correction only ever subtracts."""
    y, p = calibrated
    report = calibration_report(y, p)
    assert report.plugin_ece >= report.debiased_ece


def test_report_catches_an_overconfident_model(calibrated):
    """A distorted forecaster must score worse on every error estimator."""
    y, p = calibrated
    squashed = np.clip(2.0 * (p - 0.5) + 0.5, 0, 1)

    honest = calibration_report(y, p)
    bad = calibration_report(y, squashed)

    assert bad.mcb > honest.mcb
    assert bad.smece > honest.smece
    assert bad.debiased_ece > honest.debiased_ece
    assert bad.brier > honest.brier


def test_intervals_are_absent_by_default(calibrated):
    """Bootstrapping is opt-in; it costs n_resamples recomputations."""
    y, p = calibrated
    assert calibration_report(y, p).intervals == {}


def test_intervals_cover_the_reported_metrics(calibrated):
    """The decomposition is excluded; the rest get intervals.

    ``MCB`` and ``DSC`` are functionals of an isotonic fit, for which the naive
    bootstrap is inconsistent -- see ``tests/test_bootstrap_bias.py``.
    """
    y, p = calibrated
    report = calibration_report(y, p, ci=True, n_resamples=100)
    assert set(report.intervals) == {"brier", "smece", "debiased_ece"}
    assert "mcb" not in report.intervals
    assert "dsc" not in report.intervals


def test_the_brier_interval_brackets_its_estimate(calibrated):
    """A proper scoring rule is a plain mean, so the bootstrap behaves."""
    y, p = calibrated
    report = calibration_report(y, p, ci=True, n_resamples=200)
    interval = report.intervals["brier"]
    assert interval["lower"] <= report.brier <= interval["upper"]


@pytest.mark.parametrize("key", ["smece", "debiased_ece"])
def test_error_intervals_are_corrected_downward_on_calibrated_data(calibrated, key):
    """Bias correction pulls the interval *below* the point estimate.

    Surprising at first sight, and correct. A calibration error is a convex
    functional, so on well-calibrated data the plug-in estimate is biased upward;
    an interval for the true error therefore belongs below it. Before the default
    changed to ``"bc"`` the interval sat *above* the estimate, which is the
    failure this asserts against.

    The bound must still respect the range of the quantity: a calibration error
    cannot be negative.
    """
    y, p = calibrated
    report = calibration_report(y, p, ci=True, n_resamples=300)
    percentile = calibration_report(
        y, p, ci=True, n_resamples=300, ci_method="percentile"
    )
    assert report.intervals[key]["lower"] <= percentile.intervals[key]["lower"]
    assert report.intervals[key]["lower"] >= 0.0
    assert report.intervals[key]["bias"] > 0.0


def test_error_intervals_bracket_the_estimate_when_error_is_real(calibrated):
    """Away from zero the same intervals behave normally."""
    y, p = calibrated
    squashed = np.clip(2.0 * (p - 0.5) + 0.5, 0, 1)
    report = calibration_report(y, squashed, ci=True, n_resamples=200)
    for key in ("smece", "debiased_ece"):
        interval = report.intervals[key]
        estimate = getattr(report, key)
        assert interval["lower"] <= estimate <= interval["upper"], key


def test_the_printed_report_flags_the_bootstrap_caveat(calibrated):
    """A reader who never opens the docstring still has to see it."""
    y, p = calibrated
    text = str(calibration_report(y, p, ci=True, n_resamples=50))
    assert "convex functional" in text
    assert "well calibrated" in text
    assert "bc" in text


def test_report_prints_its_numbers(calibrated):
    """The text form is the point of the object; it must carry the values."""
    y, p = calibrated
    report = calibration_report(y, p)
    text = str(report)
    assert f"{report.brier:.4f}" in text
    assert f"{report.mcb:.4f}" in text
    assert f"{report.smece:.4f}" in text
    assert "irreducible" in text
    assert repr(report) == text


def test_to_dict_round_trips(calibrated):
    """Every field survives conversion, for a DataFrame row or JSON."""
    y, p = calibrated
    report = calibration_report(y, p)
    as_dict = report.to_dict()
    assert as_dict["brier"] == report.brier
    assert set(as_dict) >= {"n", "brier", "mcb", "dsc", "unc", "smece"}


def test_report_is_immutable(calibrated):
    """A summary that can be edited after the fact is a liability."""
    y, p = calibrated
    with pytest.raises((AttributeError, TypeError)):
        calibration_report(y, p).brier = 0.0  # type: ignore[misc]


def test_in_sample_mcb_is_zero_but_out_of_fold_is_not():
    """The trap the report's docstring warns about, asserted.

    PAV is idempotent, so scoring an isotonic fit on its own training data gives
    MCB of exactly zero however badly the model generalises.
    """
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 1500)
    labels = rng.binomial(1, scores).astype(float)

    in_sample = IsotonicCalibrator().fit(scores, labels).transform(scores)
    out_of_fold = cross_val_calibrate(IsotonicCalibrator(), scores, labels, cv=5)

    assert calibration_report(labels, in_sample).mcb == pytest.approx(0.0, abs=1e-12)
    assert calibration_report(labels, out_of_fold).mcb > 0.0


def test_bad_bin_count_is_rejected(calibrated):
    """Zero bins is not a binning."""
    y, p = calibrated
    with pytest.raises(ValueError, match="n_bins must be at least 1"):
        calibration_report(y, p, n_bins=0)


def test_report_is_a_dataclass_instance(calibrated):
    """Typed access, not a bare dict."""
    y, p = calibrated
    assert isinstance(calibration_report(y, p), CalibrationReport)
