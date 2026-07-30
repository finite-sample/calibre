"""
Unit tests for individual calibrator classes.

This module provides focused unit tests for each calibrator implementation,
testing basic functionality, parameter validation, and core behavior.
"""

import numpy as np
import pytest

from calibre import (
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    RegularizedIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SmoothedIsotonicCalibrator,
    SplineCalibrator,
)


@pytest.fixture
def calibration_data():
    """Generate synthetic test data for calibration with realistic bias."""
    np.random.seed(42)
    n = 100
    # Sorted x values as the underlying true signal
    x = np.sort(np.random.uniform(0, 1, n))
    # True underlying probabilities (monotonic)
    y_true = x.copy()
    # Introduce non-linear bias: quadratic term that increases for higher x
    bias = 0.5 * x**2
    # Add Gaussian noise (small relative to the bias)
    noise = np.random.normal(0, 0.05, size=n)
    # Observed predictions are biased: true + bias + noise
    y_observed = y_true + bias + noise
    return x, y_observed, y_true


@pytest.fixture
def binary_data():
    """Generate binary classification data for testing."""
    np.random.seed(42)
    n = 100
    x = np.random.uniform(0, 1, n)
    y = (x + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    return x, y


class TestIsotonicCalibrator:
    """Test IsotonicCalibrator functionality."""

    def test_basic_fitting(self, binary_data):
        """Test basic fit and transform operations."""
        x, y = binary_data
        cal = IsotonicCalibrator()
        cal.fit(x, y)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        assert np.all((y_calib >= 0) & (y_calib <= 1))

    def test_with_diagnostics(self, binary_data):
        """Test calibrator with diagnostics enabled."""
        x, y = binary_data
        cal = IsotonicCalibrator(enable_diagnostics=True)
        cal.fit(x, y)

        assert cal.has_diagnostics()
        diagnostics = cal.get_diagnostics()
        assert isinstance(diagnostics, dict)
        assert "n_plateaus" in diagnostics

    def test_parameter_bounds(self, binary_data):
        """Test with y_min and y_max parameters."""
        x, y = binary_data
        cal = IsotonicCalibrator(y_min=0.1, y_max=0.9)
        cal.fit(x, y)
        y_calib = cal.transform(x)

        assert np.all(y_calib >= 0.1)
        assert np.all(y_calib <= 0.9)


class TestNearlyIsotonicCalibrator:
    """Test NearlyIsotonicCalibrator functionality."""

    def test_cvx_method(self, calibration_data):
        """Test NearlyIsotonicCalibrator with CVX method."""
        x, y_observed, y_true = calibration_data
        cal = NearlyIsotonicCalibrator(lam=10.0, method="cvx")
        cal.fit(x, y_observed)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        corr = np.corrcoef(y_true, y_calib)[0, 1]
        assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"

    def test_path_method(self, calibration_data):
        """Test NearlyIsotonicCalibrator with path method."""
        x, y_observed, y_true = calibration_data
        cal = NearlyIsotonicCalibrator(lam=0.1, method="path")
        cal.fit(x, y_observed)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        corr = np.corrcoef(y_true, y_calib)[0, 1]
        assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"

    def test_invalid_method(self, calibration_data):
        """An unknown solver name must be rejected by fit, not deferred.

        Validating in ``fit`` rather than in ``transform`` means a bad
        configuration surfaces where the user can act on it, instead of after a
        model has apparently trained successfully.
        """
        x, y_observed, _ = calibration_data
        cal = NearlyIsotonicCalibrator(lam=1.0, method="invalid")

        with pytest.raises(ValueError, match="method must be"):
            cal.fit(x, y_observed)


class TestSplineCalibrator:
    """Test SplineCalibrator functionality."""

    def test_basic_functionality(self, calibration_data):
        """Test SplineCalibrator basic operations."""
        x, y_observed, y_true = calibration_data
        # The `calibration_data` fixture's targets run outside [0, 1] (up to
        # ~1.52), so they are not probabilities and the Bernoulli likelihood
        # does not apply -- fit on the identity scale instead.
        cal = SplineCalibrator(n_knots=10, degree=3, cv=5, link="identity")
        cal.fit(x, y_observed)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        assert np.all((y_calib >= 0) & (y_calib <= 1))
        corr = np.corrcoef(y_true, y_calib)[0, 1]
        assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"

    def test_parameter_variations(self, calibration_data):
        """Test different parameter combinations."""
        x, y_observed, y_true = calibration_data

        # Fitted on the identity scale: see test_basic_functionality.
        configs = [
            {"n_knots": 5, "degree": 2},
            {"n_knots": 15, "degree": 3},
            {"n_knots": 8, "degree": 1},
        ]

        for config in configs:
            cal = SplineCalibrator(**config, link="identity")
            cal.fit(x, y_observed)
            y_calib = cal.transform(x)
            assert len(y_calib) == len(x)
            assert np.all((y_calib >= 0) & (y_calib <= 1))


class TestRelaxedPAVACalibrator:
    """Test RelaxedPAVACalibrator functionality."""

    def test_basic_functionality(self, calibration_data):
        """Test RelaxedPAVACalibrator basic operations."""
        x, y_observed, y_true = calibration_data
        cal = RelaxedPAVACalibrator(epsilon=0.02)
        cal.fit(x, y_observed)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        assert np.all((y_calib >= 0) & (y_calib <= 1))
        corr = np.corrcoef(y_true, y_calib)[0, 1]
        assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"

    def test_epsilon_relaxes_monotonicity_monotonically(self, calibration_data):
        """A larger epsilon must permit at least as much total decrease.

        `epsilon` bounds the decrease allowed between adjacent unique scores, so
        the total decrease in the fit is non-decreasing in epsilon, and epsilon=0
        must reproduce plain isotonic regression exactly.
        """
        x, y_observed, _ = calibration_data
        grid = np.unique(x)

        totals = []
        for epsilon in [0.0, 0.01, 0.05, 0.2]:
            cal = RelaxedPAVACalibrator(epsilon=epsilon, clip_output=False)
            fitted = cal.fit(x, y_observed).transform(grid)
            assert len(cal.transform(x)) == len(x)
            totals.append(float(np.sum(np.maximum(0.0, -np.diff(fitted)))))

        assert totals[0] == 0.0, "epsilon=0 must be exactly monotone"
        for lo, hi in zip(totals[:-1], totals[1:], strict=True):
            assert hi >= lo - 1e-12, f"total decrease fell as epsilon rose: {totals}"

    def test_epsilon_zero_equals_isotonic(self, calibration_data):
        """epsilon=0 is standard isotonic regression, so it must match sklearn."""
        from sklearn.isotonic import IsotonicRegression

        x, y_observed, _ = calibration_data
        grid = np.unique(x)

        # clip_output=False so the comparison is against the same estimator:
        # sklearn does not clip, and this fixture's targets dip below 0.
        got = (
            RelaxedPAVACalibrator(epsilon=0.0, clip_output=False)
            .fit(x, y_observed)
            .transform(grid)
        )
        expected = (
            IsotonicRegression(out_of_bounds="clip").fit(x, y_observed).transform(grid)
        )
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)

    def test_min_slope_removes_plateaus(self, calibration_data):
        """A positive min_slope must leave no plateau on the fitted grid."""
        x, y_observed, _ = calibration_data
        grid = np.unique(x)

        plain = RelaxedPAVACalibrator().fit(x, y_observed).transform(grid)
        sloped = (
            RelaxedPAVACalibrator(min_slope=1e-4, clip_output=False)
            .fit(x, y_observed)
            .transform(grid)
        )

        assert np.any(np.diff(plain) == 0), "fixture should produce plateaus"
        assert np.all(np.diff(sloped) > 0), "min_slope must eliminate plateaus"


class TestRegularizedIsotonicCalibrator:
    """Test RegularizedIsotonicCalibrator functionality."""

    def test_basic_functionality(self, calibration_data):
        """Test RegularizedIsotonicCalibrator basic operations."""
        x, y_observed, y_true = calibration_data
        # The `calibration_data` fixture's targets run outside [0, 1] (up to
        # ~1.52), so they are not probabilities and the Bernoulli likelihood
        # does not apply -- fit on the identity scale instead.
        cal = RegularizedIsotonicCalibrator(alpha=0.1, link="identity")
        cal.fit(x, y_observed)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        assert np.all((y_calib >= 0) & (y_calib <= 1))
        corr = np.corrcoef(y_true, y_calib)[0, 1]
        assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"

    def test_regularization_strength(self, calibration_data):
        """Test different regularization strengths."""
        x, y_observed, y_true = calibration_data

        for alpha in [0.01, 0.1, 1.0]:
            cal = RegularizedIsotonicCalibrator(alpha=alpha, link="identity")
            cal.fit(x, y_observed)
            y_calib = cal.transform(x)
            assert len(y_calib) == len(x)
            assert np.all((y_calib >= 0) & (y_calib <= 1))


class TestSmoothedIsotonicCalibrator:
    """Test SmoothedIsotonicCalibrator functionality."""

    def test_basic_functionality(self, calibration_data):
        """Test SmoothedIsotonicCalibrator basic operations."""
        x, y_observed, y_true = calibration_data
        cal = SmoothedIsotonicCalibrator(
            window_length=7, poly_order=3, interp_method="linear"
        )
        cal.fit(x, y_observed)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        assert np.all((y_calib >= 0) & (y_calib <= 1))
        corr = np.corrcoef(y_true, y_calib)[0, 1]
        assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"

    def test_smoothing_parameters(self, calibration_data):
        """Test different smoothing configurations."""
        x, y_observed, y_true = calibration_data

        configs = [
            {"window_length": 5, "poly_order": 2},
            {"window_length": 9, "poly_order": 3},
            {"window_length": 7, "poly_order": 1},
        ]

        for config in configs:
            cal = SmoothedIsotonicCalibrator(**config)
            cal.fit(x, y_observed)
            y_calib = cal.transform(x)
            assert len(y_calib) == len(x)
            assert np.all((y_calib >= 0) & (y_calib <= 1))


class TestCalibratorErrorHandling:
    """Test error handling across all calibrators."""

    def test_mismatched_array_lengths(self):
        """Test error handling for mismatched array lengths."""
        np.array([1, 2, 3, 4, 5])
        y_good = np.array([1, 2, 3, 4, 5])
        x_bad = np.array([1, 2, 3])  # mismatched length

        calibrators = [
            NearlyIsotonicCalibrator(lam=1.0, method="cvx"),
            SplineCalibrator(n_knots=5),
            RelaxedPAVACalibrator(epsilon=0.02),
            RegularizedIsotonicCalibrator(alpha=0.1),
            SmoothedIsotonicCalibrator(window_length=5),
        ]

        for cal in calibrators:
            with pytest.raises(ValueError):
                cal.fit(x_bad, y_good)

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        x_empty = np.array([])
        y_empty = np.array([])

        calibrators = [
            IsotonicCalibrator(),
            NearlyIsotonicCalibrator(lam=1.0, method="cvx"),
            SplineCalibrator(n_knots=5),
        ]

        for cal in calibrators:
            with pytest.raises(ValueError):
                cal.fit(x_empty, y_empty)

    def test_transform_before_fit(self):
        """Test that transform raises error when called before fit."""
        x = np.array([0.1, 0.2, 0.3])

        calibrators = [
            IsotonicCalibrator(),
            NearlyIsotonicCalibrator(),
            RegularizedIsotonicCalibrator(),
        ]

        for cal in calibrators:
            with pytest.raises((ValueError, AttributeError)):
                cal.transform(x)


class TestCalibratorCommonInterface:
    """Test that all calibrators follow the common interface."""

    @pytest.fixture
    def all_calibrators(self):
        """Fixture providing all calibrator instances."""
        return [
            IsotonicCalibrator(),
            NearlyIsotonicCalibrator(lam=1.0, method="path"),
            SplineCalibrator(n_knots=8, cv=3),
            RelaxedPAVACalibrator(epsilon=0.02),
            RegularizedIsotonicCalibrator(alpha=0.1),
            SmoothedIsotonicCalibrator(window_length=7),
        ]

    def test_fit_returns_self(self, all_calibrators, binary_data):
        """Test that fit() returns self for method chaining."""
        x, y = binary_data

        for cal in all_calibrators:
            result = cal.fit(x, y)
            assert result is cal

    def test_transform_output_shape(self, all_calibrators, binary_data):
        """Test that transform() returns correct output shape."""
        x, y = binary_data

        for cal in all_calibrators:
            cal.fit(x, y)
            y_calib = cal.transform(x)
            assert len(y_calib) == len(x)
            assert isinstance(y_calib, np.ndarray)

    def test_fit_transform_equivalence(self, all_calibrators, binary_data):
        """Test that fit_transform gives same result as fit + transform."""
        x, y = binary_data

        for cal in all_calibrators:
            # Test fit_transform
            cal1 = cal.__class__(**cal.get_params())
            y_calib_1 = cal1.fit_transform(x, y)

            # Test fit + transform
            cal2 = cal.__class__(**cal.get_params())
            cal2.fit(x, y)
            y_calib_2 = cal2.transform(x)

            np.testing.assert_array_almost_equal(y_calib_1, y_calib_2)

    def test_diagnostics_toggle(self, all_calibrators, binary_data):
        """Test that diagnostics can be enabled/disabled."""
        x, y = binary_data

        for cal in all_calibrators:
            # Get base parameters
            params = cal.get_params()

            # Test with diagnostics disabled
            params_no_diag = params.copy()
            params_no_diag.pop("enable_diagnostics", None)  # Remove to avoid conflict
            params_no_diag["enable_diagnostics"] = False
            cal_no_diag = cal.__class__(**params_no_diag)
            cal_no_diag.fit(x, y)
            assert not cal_no_diag.has_diagnostics()

            # Test with diagnostics enabled
            params_with_diag = params.copy()
            params_with_diag.pop("enable_diagnostics", None)  # Remove to avoid conflict
            params_with_diag["enable_diagnostics"] = True
            cal_with_diag = cal.__class__(**params_with_diag)
            cal_with_diag.fit(x, y)
            assert cal_with_diag.has_diagnostics()
            assert isinstance(cal_with_diag.get_diagnostics(), dict)
