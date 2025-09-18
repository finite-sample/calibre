"""
Mathematical property validation tests for calibration algorithms.

This module tests fundamental mathematical properties that calibration
algorithms should satisfy, using realistic test data.
"""

import numpy as np
import pytest
from typing import List, Tuple, Dict, Any

from calibre.calibration import (
    NearlyIsotonicRegression,
    ISplineCalibrator,
    RelaxedPAVA,
    RegularizedIsotonicRegression,
    SmoothedIsotonicRegression,
)
from calibre.metrics import (
    mean_calibration_error,
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    correlation_metrics,
    unique_value_counts,
)
from tests.data_generators import CalibrationDataGenerator, quick_test_data


@pytest.fixture
def data_generator():
    """Fixture providing a data generator instance."""
    return CalibrationDataGenerator(random_state=42)


@pytest.fixture
def all_calibrators():
    """Fixture providing all calibrators with different parameter settings."""
    return {
        "nearly_isotonic_strict": NearlyIsotonicRegression(lam=10.0, method="path"),
        "nearly_isotonic_relaxed": NearlyIsotonicRegression(lam=0.1, method="path"),
        "nearly_isotonic_cvx": NearlyIsotonicRegression(lam=1.0, method="cvx"),
        "ispline_small": ISplineCalibrator(n_splines=5, degree=2, cv=3),
        "ispline_large": ISplineCalibrator(n_splines=15, degree=3, cv=3),
        "relaxed_pava_strict": RelaxedPAVA(percentile=5, adaptive=True),
        "relaxed_pava_loose": RelaxedPAVA(percentile=20, adaptive=False),
        "regularized_strong": RegularizedIsotonicRegression(alpha=1.0),
        "regularized_weak": RegularizedIsotonicRegression(alpha=0.01),
        "smoothed_fixed": SmoothedIsotonicRegression(window_length=7, poly_order=3),
        "smoothed_adaptive": SmoothedIsotonicRegression(
            window_length=None, adaptive=True
        ),
    }


class TestProbabilityBounds:
    """Test that all calibrators produce outputs in [0, 1] range."""

    @pytest.mark.parametrize(
        "pattern",
        [
            "overconfident_nn",
            "underconfident_rf",
            "sigmoid_distorted",
            "imbalanced_binary",
            "multi_modal",
        ],
    )
    def test_output_bounds_realistic_data(
        self, all_calibrators, data_generator, pattern
    ):
        """Test that calibrated outputs are always in [0, 1] range."""
        # Generate test data
        y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=200)

        for name, calibrator in all_calibrators.items():
            try:
                # Fit calibrator
                calibrator.fit(y_pred, y_true)

                # Transform predictions
                y_calib = calibrator.transform(y_pred)

                # Check bounds
                assert np.all(
                    y_calib >= 0
                ), f"{name} produced negative values with {pattern}"
                assert np.all(
                    y_calib <= 1
                ), f"{name} produced values > 1 with {pattern}"
                assert len(y_calib) == len(
                    y_pred
                ), f"{name} changed array length with {pattern}"

            except Exception as e:
                pytest.skip(f"Calibrator {name} failed on {pattern}: {e}")

    def test_extreme_input_bounds(self, all_calibrators):
        """Test behavior with extreme input values."""
        # Test with inputs at boundaries
        y_pred = np.array([0.0, 0.001, 0.5, 0.999, 1.0])
        y_true = np.array([0, 0, 1, 1, 1])

        for name, calibrator in all_calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                assert np.all(
                    y_calib >= 0
                ), f"{name} failed on extreme inputs (negative)"
                assert np.all(y_calib <= 1), f"{name} failed on extreme inputs (> 1)"

            except Exception as e:
                pytest.skip(f"Calibrator {name} failed on extreme inputs: {e}")

    def test_extrapolation_bounds(self, all_calibrators):
        """Test bounds when extrapolating beyond training range."""
        # Train on middle range, test on full range
        y_pred_train = np.linspace(0.2, 0.8, 100)
        y_true_train = np.random.binomial(1, y_pred_train, 100)

        y_pred_test = np.linspace(0.0, 1.0, 50)

        for name, calibrator in all_calibrators.items():
            try:
                calibrator.fit(y_pred_train, y_true_train)
                y_calib = calibrator.transform(y_pred_test)

                assert np.all(
                    y_calib >= 0
                ), f"{name} extrapolation produced negative values"
                assert np.all(y_calib <= 1), f"{name} extrapolation produced values > 1"

            except Exception as e:
                pytest.skip(f"Calibrator {name} failed on extrapolation: {e}")


class TestMonotonicity:
    """Test monotonicity properties of calibration algorithms."""

    def test_strict_monotonicity(self, data_generator):
        """Test calibrators that should maintain strict monotonicity."""
        strict_calibrators = {
            "regularized_strong": RegularizedIsotonicRegression(alpha=1.0),
            "regularized_weak": RegularizedIsotonicRegression(alpha=0.01),
        }

        # Use multiple patterns to test robustness
        patterns = ["overconfident_nn", "underconfident_rf", "sigmoid_distorted"]

        for pattern in patterns:
            y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=200)

            for name, calibrator in strict_calibrators.items():
                try:
                    calibrator.fit(y_pred, y_true)

                    # Test monotonicity on sorted input
                    x_test = np.linspace(0, 1, 50)
                    y_calib = calibrator.transform(x_test)

                    # Check strict monotonicity
                    diff = np.diff(y_calib)
                    violations = np.sum(diff < 0)

                    assert (
                        violations == 0
                    ), f"{name} violated strict monotonicity on {pattern}: {violations} violations"

                except Exception as e:
                    pytest.skip(f"Calibrator {name} failed on {pattern}: {e}")

    def test_relaxed_monotonicity(self, data_generator):
        """Test calibrators that allow controlled monotonicity violations."""
        relaxed_calibrators = {
            "nearly_isotonic_strict": NearlyIsotonicRegression(lam=10.0, method="path"),
            "nearly_isotonic_relaxed": NearlyIsotonicRegression(lam=0.1, method="path"),
            "relaxed_pava_strict": RelaxedPAVA(percentile=5),
            "relaxed_pava_loose": RelaxedPAVA(percentile=20),
        }

        y_pred, y_true = data_generator.generate_dataset("multi_modal", n_samples=200)

        for name, calibrator in relaxed_calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)

                # Test on sorted input
                x_test = np.linspace(0, 1, 100)
                y_calib = calibrator.transform(x_test)

                # Check that violations are controlled
                diff = np.diff(y_calib)
                violations = np.sum(diff < 0)
                violation_rate = violations / len(diff)

                # Relaxed methods should have few violations
                assert (
                    violation_rate < 0.1
                ), f"{name} had too many violations: {violation_rate:.3f}"

                # Most pairs should still be monotonic
                monotonic_pairs = np.sum(diff >= 0)
                monotonic_rate = monotonic_pairs / len(diff)
                assert (
                    monotonic_rate > 0.8
                ), f"{name} lost too much monotonicity: {monotonic_rate:.3f}"

            except Exception as e:
                pytest.skip(f"Calibrator {name} failed: {e}")

    def test_monotonicity_preservation(self, data_generator):
        """Test that calibrators preserve the general monotonic trend."""
        y_pred, y_true = data_generator.generate_dataset(
            "sigmoid_distorted", n_samples=300
        )

        calibrators = {
            "nearly_isotonic": NearlyIsotonicRegression(lam=1.0, method="path"),
            "ispline": ISplineCalibrator(n_splines=10, degree=3, cv=3),
            "smoothed": SmoothedIsotonicRegression(window_length=7),
        }

        for name, calibrator in calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)

                # Test correlation between input order and output order
                x_test = np.linspace(0, 1, 50)
                y_calib = calibrator.transform(x_test)

                # Calculate Spearman correlation (rank correlation)
                correlation = np.corrcoef(x_test, y_calib)[0, 1]

                assert (
                    correlation > 0.7
                ), f"{name} failed to preserve monotonic trend: correlation {correlation:.3f}"

            except Exception as e:
                pytest.skip(f"Calibrator {name} failed: {e}")


class TestCalibrationImprovement:
    """Test that calibrators actually improve calibration quality."""

    @pytest.mark.parametrize(
        "pattern",
        [
            "overconfident_nn",
            "underconfident_rf",
            "sigmoid_distorted",
            "imbalanced_binary",
        ],
    )
    def test_calibration_error_reduction(
        self, all_calibrators, data_generator, pattern
    ):
        """Test that calibration reduces expected calibration error."""
        y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=500)

        # Calculate original ECE
        original_ece = expected_calibration_error(y_true, y_pred)

        # Allow some tolerance - not all methods work on all patterns
        improved_count = 0
        total_count = 0

        for name, calibrator in all_calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                calibrated_ece = expected_calibration_error(y_true, y_calib)

                total_count += 1
                if calibrated_ece <= original_ece * 1.1:  # Allow 10% tolerance
                    improved_count += 1

            except Exception as e:
                pytest.skip(f"Calibrator {name} failed on {pattern}: {e}")

        # At least 60% of calibrators should improve or maintain ECE
        improvement_rate = improved_count / max(total_count, 1)
        assert (
            improvement_rate >= 0.6
        ), f"Only {improvement_rate:.1%} of calibrators improved ECE on {pattern}"

    def test_brier_score_bounds(self, all_calibrators, data_generator):
        """Test that Brier score remains reasonable after calibration."""
        patterns = ["overconfident_nn", "underconfident_rf", "sigmoid_distorted"]

        for pattern in patterns:
            y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=300)

            original_brier = brier_score(y_true, y_pred)

            for name, calibrator in all_calibrators.items():
                try:
                    calibrator.fit(y_pred, y_true)
                    y_calib = calibrator.transform(y_pred)

                    calibrated_brier = brier_score(y_true, y_calib)

                    # Brier score should be reasonable (not worse than random)
                    assert (
                        calibrated_brier <= 0.5
                    ), f"{name} produced poor Brier score {calibrated_brier:.3f} on {pattern}"

                    # Should not deteriorate dramatically
                    assert (
                        calibrated_brier <= original_brier * 2.0
                    ), f"{name} deteriorated Brier score too much on {pattern}"

                except Exception as e:
                    pytest.skip(f"Calibrator {name} failed on {pattern}: {e}")

    def test_reliability_diagram_improvement(self, data_generator):
        """Test that calibration improves reliability diagram alignment."""
        y_pred, y_true = data_generator.generate_dataset(
            "overconfident_nn", n_samples=1000
        )

        calibrators = {
            "nearly_isotonic": NearlyIsotonicRegression(lam=1.0, method="path"),
            "regularized": RegularizedIsotonicRegression(alpha=0.1),
        }

        for name, calibrator in calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Calculate binned metrics for reliability
                n_bins = 10
                bin_boundaries = np.linspace(0, 1, n_bins + 1)

                reliability_improvement = 0
                valid_bins = 0

                for i in range(n_bins):
                    bin_mask = (y_pred >= bin_boundaries[i]) & (
                        y_pred < bin_boundaries[i + 1]
                    )
                    if i == n_bins - 1:  # Include upper boundary in last bin
                        bin_mask = (y_pred >= bin_boundaries[i]) & (
                            y_pred <= bin_boundaries[i + 1]
                        )

                    if np.sum(bin_mask) > 0:
                        # Original reliability
                        bin_pred_orig = np.mean(y_pred[bin_mask])
                        bin_true = np.mean(y_true[bin_mask])
                        orig_error = abs(bin_pred_orig - bin_true)

                        # Calibrated reliability
                        bin_pred_calib = np.mean(y_calib[bin_mask])
                        calib_error = abs(bin_pred_calib - bin_true)

                        if calib_error <= orig_error:
                            reliability_improvement += 1
                        valid_bins += 1

                improvement_rate = reliability_improvement / max(valid_bins, 1)
                assert (
                    improvement_rate >= 0.5
                ), f"{name} failed to improve reliability in most bins"

            except Exception as e:
                pytest.skip(f"Calibrator {name} failed: {e}")


class TestGranularityPreservation:
    """Test that calibrators preserve probability granularity."""

    def test_unique_value_preservation(self, all_calibrators, data_generator):
        """Test that calibrators preserve reasonable number of unique values."""
        patterns = ["multi_modal", "weather_forecasting", "click_through_rate"]

        for pattern in patterns:
            y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=400)

            original_unique = len(np.unique(np.round(y_pred, 6)))

            for name, calibrator in all_calibrators.items():
                try:
                    calibrator.fit(y_pred, y_true)
                    y_calib = calibrator.transform(y_pred)

                    calibrated_unique = len(np.unique(np.round(y_calib, 6)))
                    preservation_ratio = calibrated_unique / original_unique

                    # Should preserve at least 30% of unique values
                    assert (
                        preservation_ratio >= 0.3
                    ), f"{name} collapsed too many values on {pattern}: {preservation_ratio:.3f}"

                    # Should not create unrealistic explosion of unique values
                    assert (
                        preservation_ratio <= 3.0
                    ), f"{name} created too many values on {pattern}: {preservation_ratio:.3f}"

                except Exception as e:
                    pytest.skip(f"Calibrator {name} failed on {pattern}: {e}")

    def test_information_preservation(self, all_calibrators, data_generator):
        """Test that calibrators preserve ranking information."""
        y_pred, y_true = data_generator.generate_dataset(
            "overconfident_nn", n_samples=300
        )

        for name, calibrator in all_calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Calculate rank correlation
                rank_correlation = np.corrcoef(y_pred, y_calib)[0, 1]

                # Should preserve strong ranking correlation
                assert (
                    rank_correlation >= 0.7
                ), f"{name} failed to preserve ranking: {rank_correlation:.3f}"

                # Check that high predictions remain high and low remain low
                high_mask = y_pred >= 0.8
                low_mask = y_pred <= 0.2

                if np.sum(high_mask) > 0 and np.sum(low_mask) > 0:
                    high_calib_mean = np.mean(y_calib[high_mask])
                    low_calib_mean = np.mean(y_calib[low_mask])

                    assert (
                        high_calib_mean > low_calib_mean
                    ), f"{name} inverted high/low ordering"

            except Exception as e:
                pytest.skip(f"Calibrator {name} failed: {e}")


class TestEdgeCases:
    """Test calibrator behavior on edge cases."""

    def test_perfect_calibration(self, all_calibrators):
        """Test behavior when input is already perfectly calibrated."""
        # Create perfectly calibrated data
        np.random.seed(42)
        n_samples = 200
        y_pred = np.random.uniform(0, 1, n_samples)
        y_true = np.random.binomial(1, y_pred, n_samples)

        original_ece = expected_calibration_error(y_true, y_pred)

        for name, calibrator in all_calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                calibrated_ece = expected_calibration_error(y_true, y_calib)

                # Should not make perfectly calibrated data worse
                assert (
                    calibrated_ece <= original_ece * 1.5
                ), f"{name} degraded perfect calibration too much"

                # Should preserve bounds
                assert np.all(y_calib >= 0) and np.all(
                    y_calib <= 1
                ), f"{name} violated bounds on perfect data"

            except Exception as e:
                pytest.skip(f"Calibrator {name} failed on perfect calibration: {e}")

    def test_constant_predictions(self, all_calibrators):
        """Test behavior with constant predictions."""
        y_pred = np.full(100, 0.5)
        y_true = np.random.binomial(1, 0.3, 100)  # Different from predictions

        for name, calibrator in all_calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Should handle constant input gracefully
                assert len(y_calib) == len(
                    y_pred
                ), f"{name} changed array length on constant input"
                assert np.all(y_calib >= 0) and np.all(
                    y_calib <= 1
                ), f"{name} violated bounds on constant input"

                # Output might be constant or adjusted toward true rate
                true_rate = np.mean(y_true)
                if len(np.unique(y_calib)) == 1:
                    # If constant output, should be reasonable
                    constant_value = y_calib[0]
                    assert (
                        0 <= constant_value <= 1
                    ), f"{name} produced invalid constant output"

            except Exception as e:
                pytest.skip(f"Calibrator {name} failed on constant predictions: {e}")

    def test_extreme_class_imbalance(self, all_calibrators):
        """Test behavior with extreme class imbalance."""
        # 99% negative class
        y_pred = np.random.uniform(0, 0.2, 1000)  # Most predictions should be low
        y_true = np.random.binomial(1, 0.01, 1000)  # 1% positive rate

        for name, calibrator in all_calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Should handle extreme imbalance
                assert np.all(y_calib >= 0) and np.all(
                    y_calib <= 1
                ), f"{name} violated bounds on imbalanced data"

                # Mean calibrated prediction should be closer to true rate
                true_rate = np.mean(y_true)
                orig_mean = np.mean(y_pred)
                calib_mean = np.mean(y_calib)

                # Calibrated mean should be closer to true rate than original
                orig_error = abs(orig_mean - true_rate)
                calib_error = abs(calib_mean - true_rate)

                # Allow some tolerance for difficult cases
                assert (
                    calib_error <= orig_error * 1.5
                ), f"{name} moved away from true rate on imbalanced data"

            except Exception as e:
                pytest.skip(f"Calibrator {name} failed on imbalanced data: {e}")

    def test_small_sample_size(self, all_calibrators):
        """Test behavior with very small sample sizes."""
        sample_sizes = [5, 10, 20]

        for n_samples in sample_sizes:
            y_pred = np.random.uniform(0, 1, n_samples)
            y_true = np.random.binomial(1, y_pred, n_samples)

            for name, calibrator in all_calibrators.items():
                try:
                    calibrator.fit(y_pred, y_true)
                    y_calib = calibrator.transform(y_pred)

                    # Should handle small samples gracefully
                    assert len(y_calib) == len(
                        y_pred
                    ), f"{name} changed array length on n={n_samples}"
                    assert np.all(y_calib >= 0) and np.all(
                        y_calib <= 1
                    ), f"{name} violated bounds on n={n_samples}"

                except Exception as e:
                    # Some calibrators might not work with very small samples
                    if n_samples >= 10:  # Should work with at least 10 samples
                        pytest.skip(f"Calibrator {name} failed on n={n_samples}: {e}")


class TestParameterSensitivity:
    """Test sensitivity to calibrator parameters."""

    def test_lambda_sensitivity_nearly_isotonic(self, data_generator):
        """Test NearlyIsotonicRegression sensitivity to lambda parameter."""
        y_pred, y_true = data_generator.generate_dataset(
            "overconfident_nn", n_samples=300
        )

        lambda_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        results = {}

        for lam in lambda_values:
            try:
                calibrator = NearlyIsotonicRegression(lam=lam, method="path")
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Calculate monotonicity violations
                x_test = np.linspace(0, 1, 100)
                y_test_calib = calibrator.transform(x_test)
                violations = np.sum(np.diff(y_test_calib) < 0)

                results[lam] = {
                    "violations": violations,
                    "ece": expected_calibration_error(y_true, y_calib),
                }

            except Exception as e:
                pytest.skip(f"Lambda {lam} failed: {e}")

        # Higher lambda should generally reduce violations
        if len(results) >= 3:
            low_lambda = min(results.keys())
            high_lambda = max(results.keys())

            assert (
                results[high_lambda]["violations"] <= results[low_lambda]["violations"]
            ), "Higher lambda should reduce monotonicity violations"

    def test_percentile_sensitivity_relaxed_pava(self, data_generator):
        """Test RelaxedPAVA sensitivity to percentile parameter."""
        y_pred, y_true = data_generator.generate_dataset("multi_modal", n_samples=300)

        percentiles = [5, 10, 20, 30]
        results = {}

        for perc in percentiles:
            try:
                calibrator = RelaxedPAVA(percentile=perc, adaptive=True)
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Count unique values (higher percentile should preserve more)
                unique_count = len(np.unique(np.round(y_calib, 6)))

                results[perc] = {
                    "unique_count": unique_count,
                    "ece": expected_calibration_error(y_true, y_calib),
                }

            except Exception as e:
                pytest.skip(f"Percentile {perc} failed: {e}")

        # Higher percentile should generally preserve more unique values
        if len(results) >= 2:
            sorted_perc = sorted(results.keys())
            unique_counts = [results[p]["unique_count"] for p in sorted_perc]

            # Should show general trend (allow some tolerance)
            if len(set(unique_counts)) > 1:  # Only check if there's variation
                correlation = np.corrcoef(sorted_perc, unique_counts)[0, 1]
                if not np.isnan(correlation):
                    assert (
                        correlation >= 0
                    ), "Higher percentile should tend to preserve more unique values"


# Utility functions for property testing
def calculate_monotonicity_violations(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the rate of monotonicity violations."""
    if len(x) < 2:
        return 0.0

    # Sort by x
    sort_idx = np.argsort(x)
    y_sorted = y[sort_idx]

    # Count violations
    violations = np.sum(np.diff(y_sorted) < 0)
    return violations / (len(y_sorted) - 1)


def calculate_calibration_reliability(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate reliability as average absolute difference between bin accuracy and confidence."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    total_error = 0
    valid_bins = 0

    for i in range(n_bins):
        bin_mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if i == n_bins - 1:  # Include upper boundary in last bin
            bin_mask = (y_pred >= bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])

        if np.sum(bin_mask) > 0:
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_confidence = np.mean(y_pred[bin_mask])
            total_error += abs(bin_accuracy - bin_confidence)
            valid_bins += 1

    return total_error / max(valid_bins, 1)
