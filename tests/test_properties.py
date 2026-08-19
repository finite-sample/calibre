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
        "spline": SplineCalibrator(n_knots=10, degree=3, cv=3),
        "relaxed_pava": RelaxedPAVACalibrator(epsilon=0.02),
        "regularized": RegularizedIsotonicCalibrator(alpha=0.1),
        "smoothed": SmoothedIsotonicCalibrator(window_length=7, poly_order=3),
    }


class TestProbabilityBounds:
    """Test that all calibrators produce outputs in [0, 1] range."""

    def _test_bounds_helper(self, calibrators, test_cases):
        """Helper method to test bounds across multiple scenarios.

        No try/except: every calibrator here clips to [0, 1], so a raise or an
        out-of-range value is the finding, not a reason to skip.
        """
        for case_name, (y_pred, y_true) in test_cases.items():
            for cal_name, calibrator in calibrators.items():
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                assert np.all(y_calib >= 0), (
                    f"{cal_name} negative values in {case_name}"
                )
                assert np.all(y_calib <= 1), f"{cal_name} values > 1 in {case_name}"
                assert len(y_calib) == len(y_pred), (
                    f"{cal_name} length changed in {case_name}"
                )

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
        self._test_bounds_helper(core_calibrators, test_cases)

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
            calibrator.fit(y_pred_train, y_true_train)
            y_calib = calibrator.transform(y_pred_test)
            assert np.all(y_calib >= 0), f"{cal_name} extrapolated below 0"
            assert np.all(y_calib <= 1), f"{cal_name} extrapolated above 1"

        # Test other cases
        del test_cases["extrapolation"]
        self._test_bounds_helper(core_calibrators, test_cases)


class TestMonotonicity:
    """Test monotonicity properties of calibration algorithms."""

    def _check_monotonicity(
        self, calibrator, y_pred, y_true, max_violation_rate=0.0, x_test=None
    ):
        """Helper to check monotonicity violations.

        The -1e-10 floor is representation noise, not a tolerance on the
        property: a monotone fit routinely produces one adjacent difference of
        around -5e-17, which is a single unit in the last place, not a decrease.
        """
        calibrator.fit(y_pred, y_true)
        if x_test is None:
            x_test = np.linspace(0, 1, 50)
        y_calib = calibrator.transform(x_test)

        violations = np.sum(np.diff(y_calib) < -1e-10)
        violation_rate = violations / (len(y_calib) - 1) if len(y_calib) > 1 else 0

        return violation_rate <= max_violation_rate, violation_rate

    @pytest.mark.parametrize("pattern", ["overconfident_nn", "multi_modal"])
    @pytest.mark.parametrize("alpha", [0.01, 1.0])
    def test_regularized_is_monotone_at_every_alpha(
        self, data_generator, pattern, alpha
    ):
        """RegularizedIsotonicCalibrator is monotone by construction.

        The penalty controls curvature, not the order constraint, so no value of
        alpha may produce a violation. Exactly zero, not a tolerance.
        """
        y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=200)
        _, violation_rate = self._check_monotonicity(
            RegularizedIsotonicCalibrator(alpha=alpha), y_pred, y_true
        )
        assert violation_rate == 0.0, (
            f"alpha={alpha} produced violations at rate {violation_rate:.3f} "
            f"on {pattern}"
        )

    @pytest.mark.parametrize("pattern", ["overconfident_nn", "multi_modal"])
    def test_nearly_isotonic_violations_shrink_with_lambda(
        self, data_generator, pattern
    ):
        """NearlyIsotonicCalibrator trades monotonicity for fit, by design.

        Violations are the point of the estimator, so a fixed tolerance would be
        asserting the wrong thing -- an earlier version of this test demanded a
        rate below 0.1 and swallowed the resulting failure with pytest.skip. The
        property that actually holds is that raising lambda cannot increase the
        violation rate.
        """
        y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=200)

        # Measured on the fitted curve at the training scores, not on a coarse
        # uniform grid: sampling a near-step function at 50 evenly spaced points
        # counts grid artifacts rather than the estimator's own violations.
        x_test = np.sort(y_pred)
        rates = [
            self._check_monotonicity(
                NearlyIsotonicCalibrator(lam=lam, method="path"),
                y_pred,
                y_true,
                x_test=x_test,
            )[1]
            for lam in (0.1, 1.0, 10.0)
        ]

        assert rates[0] >= rates[1] >= rates[2], (
            f"violation rate must be non-increasing in lambda, got {rates} on {pattern}"
        )

    def test_monotonicity_preservation(self, data_generator, core_calibrators):
        """Test that calibrators preserve general monotonic trend."""
        y_pred, y_true = data_generator.generate_dataset(
            "sigmoid_distorted", n_samples=300
        )

        for name, calibrator in core_calibrators.items():
            calibrator.fit(y_pred, y_true)
            x_test = np.linspace(0, 1, 50)
            y_calib = calibrator.transform(x_test)

            correlation = np.corrcoef(x_test, y_calib)[0, 1]
            assert correlation > 0.7, f"{name} correlation {correlation:.3f} too low"


class TestCalibrationImprovement:
    """Test that calibrators improve calibration quality."""

    @pytest.mark.parametrize(
        "pattern", ["overconfident_nn", "underconfident_rf", "sigmoid_distorted"]
    )
    def test_calibration_does_not_deteriorate_scores(
        self, core_calibrators, data_generator, pattern
    ):
        """A single draw can only support a non-deterioration bound.

        This test used to claim it verified improvement, on one dataset of 400
        observations, with `calibrated_brier <= original_brier * 2.0` and an ECE
        comparison that counted a 10% worsening as an improvement. A calibrator
        that doubled the Brier score passed.

        Tightening those thresholds does not fix it. `NearlyIsotonicCalibrator`
        at `lam=1.0` genuinely reports a worse ECE than the uncalibrated scores on
        this particular draw (0.110 against 0.071) while improving in 40 of 40
        independent draws at a larger sample size -- the statistic is simply too
        noisy at n=400, fitted and scored on the same data, to carry the claim.

        So the improvement claim lives in ``tests/test_monte_carlo.py``, paired
        across replications and with a Monte Carlo standard error on it. What
        remains here is the cheap guard a single draw *can* support: calibration
        must not make the proper score materially worse.
        """
        y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=400)
        original_brier = brier_score(y_true, y_pred)

        for name, calibrator in core_calibrators.items():
            calibrator.fit(y_pred, y_true)
            y_calib = calibrator.transform(y_pred)
            calibrated_brier = brier_score(y_true, y_calib)
            assert calibrated_brier <= original_brier + 0.02, (
                f"{name} deteriorated Brier on {pattern}: "
                f"{calibrated_brier:.4f} against {original_brier:.4f} uncalibrated"
            )

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
            calibrator.fit(y_pred, y_true)
            y_calib = calibrator.transform(y_pred)

            # Use utility function for reliability calculation
            original_reliability = calculate_calibration_reliability(y_true, y_pred)
            calibrated_reliability = calculate_calibration_reliability(y_true, y_calib)

            assert calibrated_reliability <= original_reliability * 1.2, (
                f"{name} degraded reliability"
            )


class TestGranularityPreservation:
    """Test that calibrators preserve probability granularity and ranking."""

    @pytest.mark.parametrize("pattern", ["multi_modal", "weather_forecasting"])
    def test_granularity_and_ranking_preservation(
        self, core_calibrators, data_generator, pattern
    ):
        """Test preservation of unique values and ranking information."""
        y_pred, y_true = data_generator.generate_dataset(pattern, n_samples=400)
        original_unique = len(np.unique(np.round(y_pred, 6)))

        # Which calibrators are held to the granularity floor, and why the other
        # two are not. An earlier version applied the floor to everything and hid
        # the failures behind pytest.skip -- which also meant the loop aborted at
        # the first calibrator, so the ones after it were never checked at all.
        #
        #   nearly_isotonic: lambda is a knob on how hard to pool. At lam=1.0 it
        #     pools these datasets to three levels, by design.
        #   smoothed: smooths the isotonic staircase and then repairs the result
        #     with a running maximum, which re-flattens wherever the filter dipped.
        #     Measured ratios are 0.13 (multi_modal) and 0.16 (weather_forecasting),
        #     and the same numbers come out of the released 0.7.0 wheel -- this is
        #     the estimator's long-standing behaviour, not a regression.
        #     CenteredIsotonicCalibrator is the granularity-preserving answer.
        granularity_preserving = {"spline", "relaxed_pava", "regularized"}

        for name, calibrator in core_calibrators.items():
            calibrator.fit(y_pred, y_true)
            y_calib = calibrator.transform(y_pred)

            if name in granularity_preserving:
                calibrated_unique = len(np.unique(np.round(y_calib, 6)))
                preservation_ratio = calibrated_unique / original_unique
                assert 0.3 <= preservation_ratio <= 3.0, (
                    f"{name} granularity ratio {preservation_ratio:.3f} on {pattern}"
                )

                # A correlation floor only means something once granularity is
                # preserved. At lam=1.0 nearly_isotonic pools these datasets to
                # three levels, so its correlation with a continuous score is
                # near 0.5 by construction -- for Spearman as much as Pearson.
                rank_correlation = np.corrcoef(y_pred, y_calib)[0, 1]
                assert rank_correlation >= 0.7, (
                    f"{name} ranking correlation {rank_correlation:.3f} on {pattern}"
                )

            # Ordering must hold for every calibrator, pooling or not.
            high_mask, low_mask = y_pred >= 0.8, y_pred <= 0.2
            if np.sum(high_mask) > 0 and np.sum(low_mask) > 0:
                assert np.mean(y_calib[high_mask]) > np.mean(y_calib[low_mask]), (
                    f"{name} inverted ordering on {pattern}"
                )


class TestEdgeCases:
    """Test calibrator behavior on challenging edge cases."""

    def _test_edge_case_helper(self, calibrators, test_cases):
        """Helper to test multiple edge cases."""
        for case_name, (y_pred, y_true, extra_checks) in test_cases.items():
            for cal_name, calibrator in calibrators.items():
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Common checks
                assert len(y_calib) == len(y_pred), (
                    f"{cal_name} length change in {case_name}"
                )
                assert np.all(y_calib >= 0), f"{cal_name} below 0 in {case_name}"
                assert np.all(y_calib <= 1), f"{cal_name} above 1 in {case_name}"

                # Case-specific checks
                if extra_checks:
                    extra_checks(cal_name, y_calib, y_pred, y_true)

    def test_challenging_scenarios(self, core_calibrators):
        """Test various challenging scenarios in a consolidated test."""
        np.random.seed(42)

        def perfect_calib_check(name, y_calib, y_pred, y_true):
            original_ece = expected_calibration_error(y_true, y_pred)
            calibrated_ece = expected_calibration_error(y_true, y_calib)
            assert calibrated_ece <= original_ece * 1.5, (
                f"{name} degraded perfect calibration"
            )

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
            except Exception as exc:
                pytest.fail(f"lam={lam} raised {type(exc).__name__}: {exc}")

        assert len(lambda_results) >= 2, "the sweep must produce comparable fits"
        if len(lambda_results) >= 2:
            # Higher lambda should reduce violations
            low_lam, high_lam = min(lambda_results.keys()), max(lambda_results.keys())
            assert lambda_results[high_lam] <= lambda_results[low_lam], (
                "Lambda trend check failed"
            )

        # Epsilon sensitivity for RelaxedPAVACalibrator. A larger tolerance means
        # less pooling, hence at least as many distinct calibrated values. No
        # try/except here: if the fit raises, that is the finding.
        y_pred2, y_true2 = data_generator.generate_dataset("multi_modal", n_samples=300)
        epsilon_results = {}
        for eps in [0.0, 0.05]:
            calibrator = RelaxedPAVACalibrator(epsilon=eps)
            calibrator.fit(y_pred2, y_true2)
            y_calib = calibrator.transform(y_pred2)
            epsilon_results[eps] = len(np.unique(np.round(y_calib, 6)))

        assert epsilon_results[0.05] >= epsilon_results[0.0], (
            f"a larger epsilon pooled more, not less: {epsilon_results}"
        )


# Utility functions for property testing
def calculate_calibration_reliability(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate reliability as average absolute difference between bin
    accuracy and confidence."""
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
