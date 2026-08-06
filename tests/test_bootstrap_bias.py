"""Pin the mechanism behind the bootstrap bias in calibration-error intervals.

These tests assert *why* the bias exists, not merely that some number came out
where it did last time. The explanation in :func:`calibre.bootstrap_ci`'s
docstring is:

    The bootstrap resamples from the empirical measure, so ``E[F*] = F``. A
    linear functional therefore satisfies ``E[g(F*)] = g(F)`` exactly, while a
    convex one satisfies ``E[g(F*)] >= g(F)`` strictly, by Jensen. Calibration
    errors are convex; proper scoring rules are linear.

Each test below is a consequence of that claim, chosen so that it fails if the
claim stops holding. The controls matter more than the point estimates: if the
linear and convex cases ever behave the same, the explanation is wrong and the
docstring needs rewriting rather than the test needs relaxing.

The full investigation, with an order of magnitude more resamples, lives in
``experiments/bootstrap_bias/investigate.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibre import bootstrap_ci, score_decomposition
from calibre.metrics import (
    brier_score,
    debiased_calibration_error,
    plugin_calibration_error,
    smooth_calibration_error,
)

# Kept small so the suite stays fast. The effects here are large -- a factor of
# 1.4 -- so they survive modest resample counts comfortably.
N_RESAMPLES = 200


def calibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts that are calibrated by construction.

    Parameters
    ----------
    n
        Sample size.
    seed
        Seed.

    Returns
    -------
    tuple of ndarray
        ``(y_true, y_pred)``. The true calibration error is exactly zero.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    return rng.binomial(1, p).astype(float), p


def boot_mean(metric, y, p, seed: int = 1, n_resamples: int = N_RESAMPLES) -> float:
    """Mean of the metric over bootstrap resamples of the observations.

    Parameters
    ----------
    metric
        Callable of ``(y_true, y_pred)``.
    y
        Labels.
    p
        Predictions.
    seed
        Seed.
    n_resamples
        Resamples.

    Returns
    -------
    float
        The bootstrap mean.
    """
    rng = np.random.default_rng(seed)
    n = y.size
    return float(
        np.mean(
            [
                metric(y[idx], p[idx])
                for idx in (rng.integers(0, n, n) for _ in range(n_resamples))
            ]
        )
    )


# --------------------------------------------------------------------------- #
# The controls. These are the tests that carry the explanation.
# --------------------------------------------------------------------------- #


def test_a_linear_functional_is_unbiased_under_the_bootstrap():
    """Jensen holds with equality for a mean, so the Brier score must not shift.

    This is the control that proves the inflation seen elsewhere is a property of
    the estimator's shape rather than a bug in the resampling.
    """
    y, p = calibrated(1500, 0)
    observed = brier_score(y, p)
    assert boot_mean(brier_score, y, p) == pytest.approx(observed, rel=0.02)


def test_a_convex_functional_is_biased_upward_on_the_same_data():
    """The same data and the same resampling, differing only by an absolute value.

    ``mean(p) - mean(y)`` is linear in the empirical measure; wrapping it in an
    absolute value makes it convex and nothing else changes. If both came out the
    same, the convexity explanation would be wrong.
    """
    y, p = calibrated(1500, 0)

    def signed(t, q):
        return float(np.mean(q) - np.mean(t))

    def absolute(t, q):
        return float(abs(np.mean(q) - np.mean(t)))

    rng = np.random.default_rng(2)
    n = y.size
    index = [rng.integers(0, n, n) for _ in range(400)]
    signed_draws = np.array([signed(y[i], p[i]) for i in index])
    absolute_draws = np.array([absolute(y[i], p[i]) for i in index])

    # Standardised, because the signed statistic is ~0 and a ratio would be 0/0.
    z_signed = (signed_draws.mean() - signed(y, p)) / signed_draws.std(ddof=1)
    z_absolute = (absolute_draws.mean() - absolute(y, p)) / absolute_draws.std(ddof=1)

    assert abs(z_signed) < 0.15, "the linear functional should not shift"
    assert z_absolute > 0.2, "the convex functional must shift upward"
    assert z_absolute > abs(z_signed)


@pytest.mark.parametrize("n", [500, 4000])
def test_the_inflation_does_not_vanish_with_more_data(n):
    """It is structural, not a small-sample artefact.

    Both the observed value and the resampling noise scale as ``1/sqrt(n)``, so
    their ratio is roughly constant. A ratio that shrank with ``n`` would mean
    the effect was ordinary finite-sample noise and would not need documenting.
    """
    y, p = calibrated(n, 3)

    def metric(t, q):
        return plugin_calibration_error(t, q, 15, 2)

    assert boot_mean(metric, y, p) / metric(y, p) > 1.2


def test_the_inflation_is_worse_when_the_model_is_well_calibrated():
    """The gap is curvature at the kink, so it must fade as the error grows.

    This is the part of the explanation that matters practically: the interval is
    least trustworthy exactly when the answer "my model is calibrated" is the one
    being checked.
    """
    y, p = calibrated(4000, 4)
    distorted = np.clip(2.5 * (p - 0.5) + 0.5, 0.0, 1.0)

    def metric(t, q):
        return plugin_calibration_error(t, q, 15, 2)

    near_zero = boot_mean(metric, y, p) / metric(y, p)
    far_away = boot_mean(metric, y, distorted) / metric(y, distorted)

    assert near_zero > far_away + 0.15
    assert far_away < 1.15


def test_the_isotonic_decomposition_also_inflates():
    """``MCB`` is a supremum of linear functionals, hence convex -- and censored.

    It carries an extra problem on top: a resample keeps only ~63% of rows
    distinct and PAV overfits the duplicates. This is why ``calibration_report``
    does not offer an interval for it.
    """
    y, p = calibrated(1500, 5)

    def mcb(t, q):
        return float(score_decomposition(q, t)["MCB"])

    assert boot_mean(mcb, y, p) > mcb(y, p)


def test_calibration_report_offers_no_interval_for_the_decomposition():
    """The consequence of the test above, asserted at the API boundary."""
    from calibre import calibration_report

    y, p = calibrated(600, 6)
    report = calibration_report(y, p, ci=True, n_resamples=50)
    assert "mcb" not in report.intervals
    assert "dsc" not in report.intervals


# --------------------------------------------------------------------------- #
# The fix.
# --------------------------------------------------------------------------- #


def test_the_default_method_is_bias_corrected():
    """Percentile is the wrong default here; the default must not silently revert."""
    y, p = calibrated(800, 7)
    assert bootstrap_ci(brier_score, y, p, n_resamples=50)["method"] == "bc"


@pytest.mark.parametrize("method", ["percentile", "basic", "bc", "bca"])
def test_every_method_returns_an_ordered_interval(method):
    """Whatever the method, the bounds must not cross.

    ``bca`` is run on a deliberately tiny sample: its jackknife costs one metric
    evaluation per observation.
    """
    y, p = calibrated(60 if method == "bca" else 800, 8)
    result = bootstrap_ci(brier_score, y, p, n_resamples=50, method=method)
    assert result["lower"] <= result["upper"]
    assert result["method"] == method


def test_bias_correction_moves_the_interval_down():
    """The correction has to act in the direction the inflation goes."""
    y, p = calibrated(1500, 9)

    def metric(t, q):
        return plugin_calibration_error(t, q, 15, 2)

    percentile = bootstrap_ci(metric, y, p, n_resamples=300, method="percentile")
    corrected = bootstrap_ci(metric, y, p, n_resamples=300, method="bc")
    assert corrected["lower"] <= percentile["lower"]


@pytest.mark.parametrize("method", ["bc", "bca"])
def test_bias_corrected_bounds_stay_in_range(method):
    """A calibration error cannot be negative, and the interval must respect that.

    This is why ``bc`` is the default rather than ``basic``: both correct the
    bias, but the reverse-percentile interval routinely reports a negative lower
    bound for a non-negative quantity.
    """
    y, p = calibrated(60 if method == "bca" else 1200, 10)
    result = bootstrap_ci(smooth_calibration_error, y, p, n_resamples=60, method=method)
    assert result["lower"] >= 0.0


def test_the_basic_interval_is_the_one_that_goes_negative():
    """Stated as a test so the reason for the default choice is not folklore."""
    y, p = calibrated(1500, 11)

    def metric(t, q):
        return debiased_calibration_error(t, q, 15)

    result = bootstrap_ci(metric, y, p, n_resamples=300, method="basic")
    assert result["lower"] < 0.0


def test_the_reported_bias_is_the_measured_shift():
    """``bias`` must be the bootstrap mean minus the estimate, not a guess."""
    y, p = calibrated(1200, 12)

    def metric(t, q):
        return plugin_calibration_error(t, q, 15, 2)

    result = bootstrap_ci(metric, y, p, n_resamples=200, method="bc")
    expected = boot_mean(metric, y, p, seed=0, n_resamples=200) - result["estimate"]
    assert result["bias"] == pytest.approx(expected, rel=0.15)
    assert result["bias"] > 0.0


def test_a_degenerate_interval_is_flagged_rather_than_hidden():
    """A censored statistic on its boundary can collapse the interval to a point.

    That is a legitimate reading -- "no detectable miscalibration at this
    resolution" -- but a zero-width interval must never pass for a measurement
    without saying so.
    """
    y, p = calibrated(1500, 13)
    result = bootstrap_ci(brier_score, y, p, n_resamples=100)
    assert result["degenerate"] is False
    assert result["upper"] > result["lower"]


def test_tied_draws_do_not_collapse_the_bias_correction():
    """The tie correction, which is what keeps the default usable.

    Without counting draws equal to the estimate as half, a floored estimator
    sitting at zero has no draw strictly below it, the bias correction pins to
    its clamp, and the interval collapses to ``[0, 0]``.
    """
    # Seed chosen so the estimator actually floors; the assertion below guards
    # that it still does, rather than letting the test pass vacuously.
    y, p = calibrated(2000, 0)

    def metric(t, q):
        return debiased_calibration_error(t, q, 15)

    result = bootstrap_ci(metric, y, p, n_resamples=400, method="bc")
    assert result["estimate"] == pytest.approx(0.0, abs=1e-12), (
        "fixture no longer exercises the floored case"
    )
    assert result["upper"] > 0.0, "the interval collapsed; tie correction lost"


def test_unknown_method_is_rejected():
    """A typo must not silently fall back to a different interval."""
    y, p = calibrated(200, 15)
    with pytest.raises(ValueError, match="method must be one of"):
        bootstrap_ci(brier_score, y, p, n_resamples=20, method="bootstrap")
