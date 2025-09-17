"""
Comprehensive tests for metrics module.
"""

import numpy as np
import pytest

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


@pytest.fixture
def perfect_calibration_data():
    """Data where predictions exactly match true probabilities."""
    np.random.seed(42)
    n = 100
    y_pred = np.random.uniform(0, 1, n)
    y_true = np.random.binomial(1, y_pred, n)
    return y_true, y_pred


@pytest.fixture
def poorly_calibrated_data():
    """Data with systematic calibration bias."""
    np.random.seed(42)
    n = 100
    true_probs = np.random.uniform(0, 1, n)
    # Add systematic bias: overconfident predictions
    y_pred = np.clip(true_probs + 0.3 * (true_probs - 0.5), 0, 1)
    y_true = np.random.binomial(1, true_probs, n)
    return y_true, y_pred


class TestMeanCalibrationError:
    """Test mean_calibration_error function."""

    def test_perfect_calibration(self):
        """Test with perfectly calibrated predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        error = mean_calibration_error(y_true, y_pred)
        assert error == 0.0

    def test_known_values(self):
        """Test with known values."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
        error = mean_calibration_error(y_true, y_pred)
        expected = np.mean([0.2, 0.3, 0.2, 0.4, 0.4])
        assert np.isclose(error, expected)

    def test_input_validation(self):
        """Test input validation."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.1, 0.9])  # Different length

        with pytest.raises(ValueError, match="should have the same shape"):
            mean_calibration_error(y_true, y_pred)

    def test_edge_cases(self):
        """Test edge cases."""
        # Single point
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
        assert "spearman_corr_orig_to_calib" in metrics
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

        fraction_pos, mean_pred, counts = calibration_curve(
            y_true, y_pred, n_bins=4, strategy="uniform"
        )

        assert len(fraction_pos) == len(mean_pred)
        assert all(0 <= frac <= 1 for frac in fraction_pos)
        assert all(0 <= pred <= 1 for pred in mean_pred)

    def test_quantile_strategy(self):
        """Test calibration curve with quantile strategy."""
        y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4])

        fraction_pos, mean_pred, counts = calibration_curve(
            y_true, y_pred, n_bins=4, strategy="quantile"
        )

        assert len(fraction_pos) == len(mean_pred)
        assert all(0 <= frac <= 1 for frac in fraction_pos)

    def test_perfect_calibration_curve(self):
        """Test calibration curve with perfect calibration."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])

        fraction_pos, mean_pred, counts = calibration_curve(y_true, y_pred, n_bins=2)
        # Should be close to perfect diagonal
        np.testing.assert_array_almost_equal(fraction_pos, mean_pred, decimal=1)


class TestEdgeCases:
    """Test edge cases across all metrics functions."""

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        # Most functions should handle empty arrays gracefully or raise appropriate errors
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

        # Should handle constant predictions
        error = mean_calibration_error(y_true, y_pred)
        assert error == 0.5

        counts = unique_value_counts(y_pred)
        assert counts["n_unique_y_pred"] == 1

    def test_nan_handling(self):
        """Test behavior with NaN values."""
        y_true = np.array([0, 1, np.nan])
        y_pred = np.array([0.1, 0.9, 0.5])

        # Should either handle NaN gracefully or raise appropriate error
        with pytest.raises((ValueError, TypeError)):
            mean_calibration_error(y_true, y_pred)
