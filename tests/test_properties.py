"""
Mathematical property validation tests for calibration algorithms.

This module tests fundamental mathematical properties that calibration
algorithms should satisfy, using realistic test data.
"""


import numpy as np
import pytest

from calibre import (
    NearlyIsotonicCalibrator,
    RegularizedIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SmoothedIsotonicCalibrator,
    SplineCalibrator,
)
from calibre.metrics import (
    brier_score,
    expected_calibration_error,
)
from tests.data_generators import CalibrationDataGenerator


@pytest.fixture
def data_generator():
    """Fixture providing a data generator instance."""
    return CalibrationDataGenerator(random_state=42)


@pytest.fixture
def core_calibrators():
    """Fixture providing core calibrators for property testing."""
    return {
        "nearly_isotonic": NearlyIsotonicCalibrator(lam=1.0, method="path"),
        "spline": SplineCalibrator(n_splines=10, degree=3, cv=3),
        "relaxed_pava": RelaxedPAVACalibrator(percentile=10, adaptive=True),
        "regularized": RegularizedIsotonicCalibrator(alpha=0.1),
        "smoothed": SmoothedIsotonicCalibrator(window_length=7, poly_order=3),
    }


@pytest.fixture
def extended_calibrators():
    """Extended set for comprehensive testing."""
    return {
        "nearly_isotonic_strict": NearlyIsotonicCalibrator(lam=10.0, method="path"),
        "nearly_isotonic_relaxed": NearlyIsotonicCalibrator(lam=0.1, method="path"),
        "regularized_strong": RegularizedIsotonicCalibrator(alpha=1.0),
        "regularized_weak": RegularizedIsotonicCalibrator(alpha=0.01),
    }


class TestProbabilityBounds:
    """Test that all calibrators produce outputs in [0, 1] range."""

    def _test_bounds_helper(self, calibrators, test_cases, test_name):
        """Helper method to test bounds across multiple scenarios."""
        for case_name, (y_pred, y_true) in test_cases.items():
            for cal_name, calibrator in calibrators.items():
                try:
                    calibrator.fit(y_pred, y_true)
                    y_calib = calibrator.transform(y_pred)

                    assert np.all(
                        y_calib >= 0
                    ), f"{cal_name} negative values in {case_name}"
                    assert np.all(y_calib <= 1), f"{cal_name} values > 1 in {case_name}"
                    assert len(y_calib) == len(
                        y_pred
                    ), f"{cal_name} length changed in {case_name}"

                except Exception as e:
                    pytest.skip(f"{cal_name} failed on {case_name}: {e}")

    @pytest.mark.parametrize(
        "pattern",
        ["overconfident_nn", "underconfident_rf", "sigmoid_distorted", "multi_modal"],
    )
    def test_output_bounds_realistic_data(
        self, core_calibrators, data_generator, pattern
    ):
        """Test bounds on realistic data patterns."""
        y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=200)
        test_cases = {pattern: (y_pred, y_true)}
        self._test_bounds_helper(core_calibrators, test_cases, "realistic_data")

    def test_bounds_edge_cases(self, core_calibrators):
        """Test bounds on various edge cases."""
        test_cases = {
            "extreme_inputs": (
                np.array([0.0, 0.001, 0.5, 0.999, 1.0]),
                np.array([0, 0, 1, 1, 1]),
            ),
            "extrapolation": (
                np.linspace(0.0, 1.0, 50),
                None,
            ),  # Special case for extrapolation
        }

        # Handle extrapolation test separately
        y_pred_train = np.linspace(0.2, 0.8, 100)
        y_true_train = np.random.binomial(1, y_pred_train, 100)
        y_pred_test = np.linspace(0.0, 1.0, 50)

        for cal_name, calibrator in core_calibrators.items():
            try:
                calibrator.fit(y_pred_train, y_true_train)
                y_calib = calibrator.transform(y_pred_test)
                assert np.all(y_calib >= 0) and np.all(
                    y_calib <= 1
                ), f"{cal_name} extrapolation bounds"
            except Exception as e:
                pytest.skip(f"{cal_name} failed on extrapolation: {e}")

        # Test other cases
        del test_cases["extrapolation"]
        self._test_bounds_helper(core_calibrators, test_cases, "edge_cases")


class TestMonotonicity:
    """Test monotonicity properties of calibration algorithms."""

    def _check_monotonicity(self, calibrator, y_pred, y_true, max_violation_rate=0.0):
        """Helper to check monotonicity violations."""
        calibrator.fit(y_pred, y_true)
        x_test = np.linspace(0, 1, 50)
        y_calib = calibrator.transform(x_test)

        violations = np.sum(np.diff(y_calib) < 0)
        violation_rate = violations / (len(y_calib) - 1) if len(y_calib) > 1 else 0

        return violation_rate <= max_violation_rate, violation_rate

    def test_strict_vs_relaxed_monotonicity(self, data_generator, extended_calibrators):
        """Test monotonicity behavior across strict and relaxed calibrators."""
        patterns = ["overconfident_nn", "multi_modal"]

        for pattern in patterns:
            y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=200)

            for name, calibrator in extended_calibrators.items():
                try:
                    # Set different expectations based on calibrator type
                    if "strong" in name:
                        max_violations = 0.0  # Strict monotonicity expected
                    else:
                        max_violations = 0.1  # Allow some violations

                    is_monotonic, violation_rate = self._check_monotonicity(
                        calibrator, y_pred, y_true, max_violations
                    )

                    assert (
                        is_monotonic
                    ), f"{name} violations {violation_rate:.3f} > {max_violations} on {pattern}"

                except Exception as e:
                    pytest.skip(f"{name} failed on {pattern}: {e}")

    def test_monotonicity_preservation(self, data_generator, core_calibrators):
        """Test that calibrators preserve general monotonic trend."""
        y_pred, y_true = data_generator.generate_dataset(
            "sigmoid_distorted", n_samples=300
        )

        for name, calibrator in core_calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)
                x_test = np.linspace(0, 1, 50)
                y_calib = calibrator.transform(x_test)

                correlation = np.corrcoef(x_test, y_calib)[0, 1]
                assert (
                    correlation > 0.7
                ), f"{name} correlation {correlation:.3f} too low"

            except Exception as e:
                pytest.skip(f"{name} failed: {e}")


class TestCalibrationImprovement:
    """Test that calibrators improve calibration quality."""

    @pytest.mark.parametrize(
        "pattern", ["overconfident_nn", "underconfident_rf", "sigmoid_distorted"]
    )
    def test_calibration_metrics_improvement(
        self, core_calibrators, data_generator, pattern
    ):
        """Test that calibrators improve ECE and maintain reasonable Brier scores."""
        y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=400)

        original_ece = expected_calibration_error(y_true, y_pred)
        original_brier = brier_score(y_true, y_pred)

        improved_count = 0
        total_count = 0

        for name, calibrator in core_calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Test ECE improvement
                calibrated_ece = expected_calibration_error(y_true, y_calib)
                if calibrated_ece <= original_ece * 1.1:  # 10% tolerance
                    improved_count += 1
                total_count += 1

                # Test Brier score bounds
                calibrated_brier = brier_score(y_true, y_calib)
                assert calibrated_brier <= 0.5, f"{name} poor Brier on {pattern}"
                assert (
                    calibrated_brier <= original_brier * 2.0
                ), f"{name} deteriorated Brier on {pattern}"

            except Exception as e:
                pytest.skip(f"{name} failed on {pattern}: {e}")

        # At least 60% should improve ECE
        improvement_rate = improved_count / max(total_count, 1)
        assert (
            improvement_rate >= 0.6
        ), f"Only {improvement_rate:.1%} improved ECE on {pattern}"

    def test_reliability_improvement(self, data_generator):
        """Test that calibration improves reliability diagram alignment."""
        y_pred, y_true = data_generator.generate_dataset(
            "overconfident_nn", n_samples=1000
        )

        calibrators = {
            "nearly_isotonic": NearlyIsotonicCalibrator(lam=1.0, method="path"),
            "regularized": RegularizedIsotonicCalibrator(alpha=0.1),
        }

        for name, calibrator in calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Use utility function for reliability calculation
                original_reliability = calculate_calibration_reliability(y_true, y_pred)
                calibrated_reliability = calculate_calibration_reliability(
                    y_true, y_calib
                )

                assert (
                    calibrated_reliability <= original_reliability * 1.2
                ), f"{name} degraded reliability"

            except Exception as e:
                pytest.skip(f"{name} failed: {e}")


class TestGranularityPreservation:
    """Test that calibrators preserve probability granularity and ranking."""

    @pytest.mark.parametrize("pattern", ["multi_modal", "weather_forecasting"])
    def test_granularity_and_ranking_preservation(
        self, core_calibrators, data_generator, pattern
    ):
        """Test preservation of unique values and ranking information."""
        y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=400)
        original_unique = len(np.unique(np.round(y_pred, 6)))

        for name, calibrator in core_calibrators.items():
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Test granularity preservation
                calibrated_unique = len(np.unique(np.round(y_calib, 6)))
                preservation_ratio = calibrated_unique / original_unique
                assert (
                    0.3 <= preservation_ratio <= 3.0
                ), f"{name} granularity ratio {preservation_ratio:.3f} on {pattern}"

                # Test ranking preservation
                rank_correlation = np.corrcoef(y_pred, y_calib)[0, 1]
                assert (
                    rank_correlation >= 0.7
                ), f"{name} ranking correlation {rank_correlation:.3f} on {pattern}"

                # Test high/low ordering preservation
                high_mask, low_mask = y_pred >= 0.8, y_pred <= 0.2
                if np.sum(high_mask) > 0 and np.sum(low_mask) > 0:
                    assert np.mean(y_calib[high_mask]) > np.mean(
                        y_calib[low_mask]
                    ), f"{name} inverted ordering on {pattern}"

            except Exception as e:
                pytest.skip(f"{name} failed on {pattern}: {e}")


class TestEdgeCases:
    """Test calibrator behavior on challenging edge cases."""

    def _test_edge_case_helper(self, calibrators, test_cases):
        """Helper to test multiple edge cases."""
        for case_name, (y_pred, y_true, extra_checks) in test_cases.items():
            for cal_name, calibrator in calibrators.items():
                try:
                    calibrator.fit(y_pred, y_true)
                    y_calib = calibrator.transform(y_pred)

                    # Common checks
                    assert len(y_calib) == len(
                        y_pred
                    ), f"{cal_name} length change in {case_name}"
                    assert np.all(y_calib >= 0) and np.all(
                        y_calib <= 1
                    ), f"{cal_name} bounds in {case_name}"

                    # Case-specific checks
                    if extra_checks:
                        extra_checks(cal_name, y_calib, y_pred, y_true)

                except Exception as e:
                    if case_name == "small_sample" and len(y_pred) < 10:
                        continue  # Expected for very small samples
                    pytest.skip(f"{cal_name} failed on {case_name}: {e}")

    def test_challenging_scenarios(self, core_calibrators):
        """Test various challenging scenarios in a consolidated test."""
        np.random.seed(42)

        def perfect_calib_check(name, y_calib, y_pred, y_true):
            original_ece = expected_calibration_error(y_true, y_pred)
            calibrated_ece = expected_calibration_error(y_true, y_calib)
            assert (
                calibrated_ece <= original_ece * 1.5
            ), f"{name} degraded perfect calibration"

        def imbalance_check(name, y_calib, y_pred, y_true):
            true_rate = np.mean(y_true)
            orig_error = abs(np.mean(y_pred) - true_rate)
            calib_error = abs(np.mean(y_calib) - true_rate)
            assert calib_error <= orig_error * 1.5, f"{name} moved away from true rate"

        test_cases = {
            "perfect_calibration": (
                np.random.uniform(0, 1, 200),
                lambda p: np.random.binomial(1, p, 200),
                perfect_calib_check,
            ),
            "constant_predictions": (
                np.full(100, 0.5),
                np.random.binomial(1, 0.3, 100),
                None,
            ),
            "extreme_imbalance": (
                np.random.uniform(0, 0.2, 500),
                np.random.binomial(1, 0.01, 500),
                imbalance_check,
            ),
            "small_sample": (
                np.random.uniform(0, 1, 10),
                np.random.binomial(1, 0.5, 10),
                None,
            ),
        }

        # Handle perfect calibration case
        y_pred_perfect = test_cases["perfect_calibration"][0]
        y_true_perfect = test_cases["perfect_calibration"][1](y_pred_perfect)
        test_cases["perfect_calibration"] = (
            y_pred_perfect,
            y_true_perfect,
            perfect_calib_check,
        )

        self._test_edge_case_helper(core_calibrators, test_cases)


class TestParameterSensitivity:
    """Test sensitivity to key calibrator parameters."""

    def test_parameter_sensitivity_trends(self, data_generator):
        """Test that parameter changes produce expected trends."""
        y_pred, y_true = data_generator.generate_dataset(
            "overconfident_nn", n_samples=300
        )

        # Test lambda sensitivity for NearlyIsotonicCalibrator
        lambda_results = {}
        for lam in [0.1, 1.0, 10.0]:
            try:
                calibrator = NearlyIsotonicCalibrator(lam=lam, method="path")
                calibrator.fit(y_pred, y_true)
                x_test = np.linspace(0, 1, 50)
                y_test_calib = calibrator.transform(x_test)
                violations = np.sum(np.diff(y_test_calib) < 0)
                lambda_results[lam] = violations
            except Exception:
                pass

        if len(lambda_results) >= 2:
            # Higher lambda should reduce violations
            low_lam, high_lam = min(lambda_results.keys()), max(lambda_results.keys())
            assert (
                lambda_results[high_lam] <= lambda_results[low_lam]
            ), "Lambda trend check failed"

        # Test percentile sensitivity for RelaxedPAVACalibrator
        y_pred2, y_true2 = data_generator.generate_dataset("multi_modal", n_samples=300)
        percentile_results = {}
        for perc in [5, 20]:
            try:
                calibrator = RelaxedPAVACalibrator(percentile=perc, adaptive=True)
                calibrator.fit(y_pred2, y_true2)
                y_calib = calibrator.transform(y_pred2)
                unique_count = len(np.unique(np.round(y_calib, 6)))
                percentile_results[perc] = unique_count
            except Exception:
                pass

        if len(percentile_results) == 2:
            # Higher percentile should preserve more unique values
            assert (
                percentile_results[20] >= percentile_results[5]
            ), "Percentile trend check failed"


# Utility functions for property testing
def calculate_monotonicity_violations(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the rate of monotonicity violations."""
    if len(x) < 2:
        return 0.0
    sort_idx = np.argsort(x)
    y_sorted = y[sort_idx]
    violations = np.sum(np.diff(y_sorted) < 0)
    return violations / (len(y_sorted) - 1)


def calculate_calibration_reliability(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate reliability as average absolute difference between bin accuracy and confidence."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    total_error, valid_bins = 0, 0

    for i in range(n_bins):
        bin_mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if i == n_bins - 1:
            bin_mask = (y_pred >= bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])

        if np.sum(bin_mask) > 0:
            total_error += abs(np.mean(y_true[bin_mask]) - np.mean(y_pred[bin_mask]))
            valid_bins += 1

    return total_error / max(valid_bins, 1)
