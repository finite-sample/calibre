"""
Comprehensive tests for metrics module.
"""

import numpy as np
import pytest

import calibre.metrics
from calibre.metrics import (
    binned_calibration_error,
    brier_score,
    calibration_curve,
    correlation_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    mean_calibration_error,
    unique_value_counts,
)


@pytest.mark.parametrize(
    "metric",
    [
        binned_calibration_error,
        expected_calibration_error,
        maximum_calibration_error,
        calibration_curve,
    ],
)
def test_binned_metrics_reject_zero_bins(metric):
    """Zero bins cannot be reported as zero calibration error."""
    with pytest.raises(ValueError, match="n_bins must be at least 1"):
        metric(np.array([0.0, 1.0]), np.array([0.2, 0.8]), n_bins=0)


class TestMeanCalibrationError:
    """Test mean_calibration_error function."""

    def test_perfect_calibration(self):
        """Test with perfectly calibrated predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        error = mean_calibration_error(y_true, y_pred)
        assert error == 0.0

    def test_known_values(self):
        """The metric is the bias: |mean(prediction) - base rate|.

        This test previously asserted ``mean(|y_pred - y_true|)``, i.e. mean
        absolute error, and so certified a formula that is not a calibration
        error at all -- see test_perfectly_calibrated_but_unsharp_scores_zero.
        """
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
        error = mean_calibration_error(y_true, y_pred)
        expected = abs(np.mean(y_pred) - np.mean(y_true))  # |0.54 - 0.6|
        assert np.isclose(error, expected)
        assert np.isclose(error, 0.06)

    def test_perfectly_calibrated_but_unsharp_scores_zero(self):
        """A calibration error must be zero for a calibrated predictor.

        Predicting the base rate for everyone is perfectly calibrated in the
        large, however uninformative. The old mean-absolute-error formula scored
        this 0.48, and scored a confident-but-wrong predictor better.
        """
        y_true = np.array([0, 0, 1, 1, 1])
        base_rate = float(np.mean(y_true))
        constant = np.full_like(y_true, base_rate, dtype=float)
        assert mean_calibration_error(y_true, constant) == pytest.approx(0.0)

    def test_penalises_systematic_bias(self):
        """Shifting every prediction up must raise the error by that shift."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.full(4, 0.5)
        assert mean_calibration_error(y_true, y_pred) == pytest.approx(0.0)
        assert mean_calibration_error(y_true, y_pred + 0.2) == pytest.approx(0.2)
        assert mean_calibration_error(y_true, y_pred - 0.3) == pytest.approx(0.3)

    def test_input_validation(self):
        """Test input validation."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.1, 0.9])  # Different length

        with pytest.raises(ValueError, match="should have the same shape"):
            mean_calibration_error(y_true, y_pred)

    def test_edge_cases(self):
        """Test edge cases."""
        # Single point: |0.8 - 1| = 0.2
        error = mean_calibration_error([1], [0.8])
        assert error == pytest.approx(0.2)

        # All zeros
        error = mean_calibration_error([0, 0, 0], [0, 0, 0])
        assert error == 0.0


class TestBinnedCalibrationError:
    """Test binned_calibration_error function."""

    def test_uniform_strategy(self):
        """Test uniform binning strategy."""
        y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4])

        error = binned_calibration_error(y_true, y_pred, n_bins=4, strategy="uniform")
        assert isinstance(error, float)
        assert error >= 0

    def test_quantile_strategy(self):
        """Test quantile binning strategy."""
        y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4])

        error = binned_calibration_error(y_true, y_pred, n_bins=4, strategy="quantile")
        assert isinstance(error, float)
        assert error >= 0

    def test_return_details(self):
        """Test returning detailed bin information."""
        y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4])

        result = binned_calibration_error(y_true, y_pred, n_bins=4, return_details=True)
        assert isinstance(result, dict)
        assert "bce" in result
        assert "bin_counts" in result
        assert "bin_centers" in result

    def test_invalid_strategy(self):
        """Test invalid strategy parameter."""
        y_true = np.array([0, 1])
        y_pred = np.array([0.1, 0.9])

        with pytest.raises(ValueError, match="Unknown binning strategy"):
            binned_calibration_error(y_true, y_pred, strategy="invalid")


class TestExpectedCalibrationError:
    """Test expected_calibration_error function."""

    def test_basic_functionality(self):
        """Test basic ECE calculation."""
        y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4])

        ece = expected_calibration_error(y_true, y_pred, n_bins=4)
        assert isinstance(ece, float)
        assert ece >= 0

    def test_perfect_calibration_ece(self):
        """Test ECE with perfect calibration."""
        n = 1000
        np.random.seed(42)
        y_pred = np.random.uniform(0, 1, n)
        y_true = np.random.binomial(1, y_pred, n)

        ece = expected_calibration_error(y_true, y_pred, n_bins=10)
        # Should be low for perfect calibration (allowing some variance)
        assert ece < 0.1


class TestMaximumCalibrationError:
    """Test maximum_calibration_error function."""

    def test_basic_functionality(self):
        """Test basic MCE calculation."""
        y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4])

        mce = maximum_calibration_error(y_true, y_pred, n_bins=4)
        assert isinstance(mce, float)
        assert mce >= 0
        assert mce <= 1

    def test_mce_greater_than_ece(self):
        """Test that MCE >= ECE."""
        y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4])

        ece = expected_calibration_error(y_true, y_pred, n_bins=4)
        mce = maximum_calibration_error(y_true, y_pred, n_bins=4)
        assert mce >= ece


class TestBrierScore:
    """Test brier_score function."""

    def test_perfect_predictions(self):
        """Test Brier score with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])

        bs = brier_score(y_true, y_pred)
        assert bs == 0.0

    def test_worst_predictions(self):
        """Test Brier score with worst possible predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1])

        bs = brier_score(y_true, y_pred)
        assert bs == 1.0

    def test_known_values(self):
        """Test with known values."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0.8, 0.2, 0.9])

        bs = brier_score(y_true, y_pred)
        expected = np.mean([(0.8 - 1) ** 2, (0.2 - 0) ** 2, (0.9 - 1) ** 2])
        assert np.isclose(bs, expected)


class TestCorrelationMetrics:
    """Test correlation_metrics function."""

    def test_basic_correlation(self):
        """Test basic correlation calculation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.7, 0.8, 0.9])

        metrics = correlation_metrics(y_true, y_pred)
        assert isinstance(metrics, dict)
        assert "spearman_corr_to_y_true" in metrics
        assert -1 <= metrics["spearman_corr_to_y_true"] <= 1

    def test_with_original_predictions(self):
        """Test with original predictions provided."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
        y_orig = np.array([0.2, 0.3, 0.6, 0.7, 0.8])

        metrics = correlation_metrics(y_true, y_pred, y_orig=y_orig)
        assert "spearman_corr_to_y_orig" in metrics
        assert "spearman_corr_to_y_orig" in metrics

    def test_perfect_correlation(self):
        """Test perfect correlation case."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])

        metrics = correlation_metrics(y_true, y_pred)
        assert np.isclose(metrics["spearman_corr_to_y_true"], 1.0)


class TestUniqueValueCounts:
    """Test unique_value_counts function."""

    def test_basic_counting(self):
        """Test basic unique value counting."""
        y_pred = np.array([0.1, 0.2, 0.1, 0.3, 0.2])

        counts = unique_value_counts(y_pred)
        assert isinstance(counts, dict)
        assert "n_unique_y_pred" in counts
        assert counts["n_unique_y_pred"] == 3

    def test_with_original(self):
        """Test counting with original predictions."""
        y_pred = np.array([0.1, 0.2, 0.1, 0.3])
        y_orig = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        counts = unique_value_counts(y_pred, y_orig)
        assert "n_unique_y_orig" in counts
        assert "unique_value_ratio" in counts
        assert counts["n_unique_y_orig"] == 5

    def test_precision_rounding(self):
        """Test precision rounding."""
        y_pred = np.array([0.123456789, 0.123456780])

        counts_low_precision = unique_value_counts(y_pred, precision=6)
        counts_high_precision = unique_value_counts(y_pred, precision=9)

        assert counts_low_precision["n_unique_y_pred"] == 1
        assert counts_high_precision["n_unique_y_pred"] == 2


class TestCalibrationCurve:
    """Test calibration_curve function."""

    def test_uniform_strategy(self):
        """Test calibration curve with uniform strategy."""
        y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4])

        fraction_pos, mean_pred, _counts = calibration_curve(
            y_true, y_pred, n_bins=4, strategy="uniform"
        )

        assert len(fraction_pos) == len(mean_pred)
        assert all(0 <= frac <= 1 for frac in fraction_pos)
        assert all(0 <= pred <= 1 for pred in mean_pred)

    def test_quantile_strategy(self):
        """Test calibration curve with quantile strategy."""
        y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4])

        fraction_pos, mean_pred, _counts = calibration_curve(
            y_true, y_pred, n_bins=4, strategy="quantile"
        )

        assert len(fraction_pos) == len(mean_pred)
        assert all(0 <= frac <= 1 for frac in fraction_pos)

    def test_perfect_calibration_curve(self):
        """Test calibration curve with perfect calibration."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])

        fraction_pos, mean_pred, _counts = calibration_curve(y_true, y_pred, n_bins=2)
        # Should be close to perfect diagonal
        np.testing.assert_array_almost_equal(fraction_pos, mean_pred, decimal=1)


class TestEdgeCases:
    """Test edge cases across all metrics functions."""

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        # Most functions should handle empty arrays gracefully or raise errors
        with pytest.raises((ValueError, IndexError)):
            mean_calibration_error(y_true, y_pred)

    def test_single_value(self):
        """Test behavior with single values."""
        y_true = np.array([1])
        y_pred = np.array([0.8])

        # Should work for most functions
        error = mean_calibration_error(y_true, y_pred)
        assert error == pytest.approx(0.2)

        bs = brier_score(y_true, y_pred)
        assert bs == pytest.approx(0.04)

    def test_constant_predictions(self):
        """Test behavior with constant predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        # Constant 0.5 against a base rate of 0.4 is a bias of 0.1.
        error = mean_calibration_error(y_true, y_pred)
        assert error == pytest.approx(0.1)

        counts = unique_value_counts(y_pred)
        assert counts["n_unique_y_pred"] == 1

    def test_nan_handling(self):
        """Test behavior with NaN values."""
        y_true = np.array([0, 1, np.nan])
        y_pred = np.array([0.1, 0.9, 0.5])

        # Should either handle NaN gracefully or raise appropriate error
        with pytest.raises((ValueError, TypeError)):
            mean_calibration_error(y_true, y_pred)


# --------------------------------------------------------------------------- #
# Debiased and monotonic-sweep calibration error
# --------------------------------------------------------------------------- #


def _calibrated_sample(seed: int, n: int):
    """Generate predictions that are calibrated by construction.

    Parameters
    ----------
    seed
        Random seed.
    n
        Number of observations.

    Returns
    -------
    tuple of ndarray
        Predictions and outcomes.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, n)
    return p, rng.binomial(1, p).astype(float)


def test_equal_mass_bins_never_split_a_tie_group():
    """Identical predictions must share a bin.

    Splitting a tie group compares a bin's mean prediction against a mean label
    drawn from an arbitrary subset of observations carrying that same
    prediction, which measures the sort order rather than calibration. Clipping
    makes this the common case, not an exotic one.
    """
    from calibre.metrics import _equal_mass_bins

    rng = np.random.default_rng(7)
    p = rng.uniform(0.0, 1.0, 500)
    clipped = np.clip(1.6 * (p - 0.5) + 0.5, 0.0, 1.0)
    assert (clipped == 0.0).sum() > 50, "fixture should have heavy ties"

    bin_id, _ = _equal_mass_bins(clipped, 15)
    for value in np.unique(clipped):
        assert len(np.unique(bin_id[clipped == value])) == 1


def test_equal_mass_bins_are_balanced_without_ties():
    """With distinct values the bins really are equal mass."""
    from calibre.metrics import _equal_mass_bins

    rng = np.random.default_rng(1)
    p = rng.uniform(0.0, 1.0, 1000)
    bin_id, n_used = _equal_mass_bins(p, 10)
    counts = np.bincount(bin_id, minlength=n_used)
    assert n_used == 10
    assert counts.min() == counts.max() == 100


def test_debiasing_removes_error_that_is_not_there():
    """On calibrated data the plugin estimator reports error; debiasing does not.

    The plugin bias grows with the bin count, so this is checked at 15 bins
    where it is large enough to be unmistakable.
    """
    p, y = _calibrated_sample(0, 4000)
    plugin = expected_calibration_error(y, p, n_bins=15)
    debiased = calibre.metrics.debiased_calibration_error(y, p, n_bins=15)
    assert plugin > 0.01
    assert debiased < plugin


def test_debiased_error_detects_real_miscalibration():
    """Debiasing must not flatten genuine miscalibration to zero."""
    p, y = _calibrated_sample(1, 4000)
    squashed = 0.4 * (p - 0.5) + 0.5
    assert calibre.metrics.debiased_calibration_error(y, squashed, n_bins=15) > 0.1


def test_debiased_error_is_never_negative():
    """The correction can overshoot; the reported error is floored at zero."""
    for seed in range(10):
        p, y = _calibrated_sample(100 + seed, 200)
        assert calibre.metrics.debiased_calibration_error(y, p, n_bins=15) >= 0.0


def test_debiased_error_rejects_mismatched_lengths():
    """Length mismatch is an error, not a broadcast."""
    with pytest.raises(ValueError, match="same length"):
        calibre.metrics.debiased_calibration_error(
            np.array([0.0, 1.0]), np.array([0.5])
        )


def test_sweep_stops_before_the_bins_read_noise():
    """ECE_sweep picks a bin count and reports a small error when calibrated."""
    p, y = _calibrated_sample(2, 4000)
    assert calibre.metrics.sweep_calibration_error(y, p) < 0.05


def test_sweep_detects_real_miscalibration():
    """A distorted forecast must score much worse than an honest one."""
    p, y = _calibrated_sample(3, 4000)
    squashed = 0.4 * (p - 0.5) + 0.5
    honest = calibre.metrics.sweep_calibration_error(y, p)
    distorted = calibre.metrics.sweep_calibration_error(y, squashed)
    assert distorted > honest * 3


def test_sweep_is_order_invariant():
    """Shuffling rows must not change the answer."""
    p, y = _calibrated_sample(4, 1000)
    rng = np.random.default_rng(0)
    perm = rng.permutation(p.size)
    assert calibre.metrics.sweep_calibration_error(y, p) == pytest.approx(
        calibre.metrics.sweep_calibration_error(y[perm], p[perm])
    )


def test_sweep_handles_degenerate_input():
    """One observation supports no binning at all."""
    assert (
        calibre.metrics.sweep_calibration_error(np.array([1.0]), np.array([0.5])) == 0.0
    )


def test_sweep_rejects_a_norm_below_one():
    """p < 1 is not a norm."""
    p, y = _calibrated_sample(5, 100)
    with pytest.raises(ValueError, match="p must be at least 1"):
        calibre.metrics.sweep_calibration_error(y, p, p=0)


def test_all_names_are_defined():
    """Every name promised by ``__all__`` must exist."""
    missing = [n for n in calibre.metrics.__all__ if not hasattr(calibre.metrics, n)]
    assert missing == []


def test_all_covers_every_public_function():
    """No public metric may be missing from ``__all__``.

    ``__all__`` used to be declared partway down the module, above
    ``debiased_calibration_error`` and ``sweep_calibration_error``, so those two
    were silently absent from ``from calibre.metrics import *``. This test fails
    if a public metric is ever added without being exported.
    """
    import inspect

    public = {
        name
        for name, obj in inspect.getmembers(calibre.metrics, inspect.isfunction)
        if not name.startswith("_") and inspect.getmodule(obj) is calibre.metrics
    }
    assert public - set(calibre.metrics.__all__) == set()
