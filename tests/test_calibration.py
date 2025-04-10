"""
Basic tests for the calibre package calibrators and metrics.

To run these tests, install pytest and run:
    pytest -xvs tests/
"""

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

# Import calibrator classes and metrics from the package.
# Adjust these imports if your package structure differs.
from calibre.calibration import (
    NearlyIsotonicRegression,
    ISplineCalibrator,
    RelaxedPAVA,
    RegularizedIsotonicRegression,
    SmoothedIsotonicRegression
)
from calibre.metrics import (
    mean_calibration_error,
    binned_calibration_error,
    correlation_metrics,
    unique_value_counts
)


@pytest.fixture
def test_data():
    """Generate synthetic test data for calibration tests with increasing bias."""
    np.random.seed(42)
    n = 100
    # Sorted x values as the underlying true signal.
    x = np.sort(np.random.uniform(0, 1, n))
    # The true underlying probabilities (monotonic).
    y_true = x.copy()
    # Introduce a non-linear bias: a quadratic term that increases for higher x.
    bias = 0.5 * x**2  
    # Add some Gaussian noise (small relative to the bias).
    noise = np.random.normal(0, 0.05, size=n)
    # Observed predictions are biased: true + bias + noise.
    y_observed = y_true + bias + noise
    # Optionally, you might want to clip or sort the observations to ensure a mostly monotonic trend.
    # Here we intentionally do not re-sort y_observed so the calibrator has to correct for bias plus small non-monotonicity.
    return x, y_observed, y_true

def test_nearly_isotonic_regression_cvx(test_data):
    x, y_observed, y_true = test_data
    from calibre.calibration import NearlyIsotonicRegression
    cal = NearlyIsotonicRegression(lam=10.0, method='cvx')
    cal.fit(x, y_observed)
    y_calib = cal.transform(x)
    assert len(y_calib) == len(x)
    # Compare calibrated output with y_true (which in this case is simply x)
    corr = np.corrcoef(y_true, y_calib)[0, 1]
    assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"


def test_nearly_isotonic_regression_path(test_data):
    """Test NearlyIsotonicRegression using the path algorithm."""
    x, y, y_true = test_data
    cal = NearlyIsotonicRegression(lam=0.1, method='path')
    cal.fit(x, y)
    y_calib = cal.transform(x)
    assert len(y_calib) == len(x)
    corr = np.corrcoef(y_true, y_calib)[0, 1]
    assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"


def test_ispline_calibrator(test_data):
    """Test the ISplineCalibrator."""
    x, y, y_true = test_data
    cal = ISplineCalibrator(n_splines=10, degree=3, cv=5)
    cal.fit(x, y)
    y_calib = cal.transform(x)
    assert len(y_calib) == len(x)
    corr = np.corrcoef(y_true, y_calib)[0, 1]
    assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"


def test_relaxed_pava(test_data):
    """Test the RelaxedPAVA calibrator."""
    x, y, y_true = test_data
    cal = RelaxedPAVA(percentile=10, adaptive=True)
    cal.fit(x, y)
    y_calib = cal.transform(x)
    assert len(y_calib) == len(x)
    corr = np.corrcoef(y_true, y_calib)[0, 1]
    assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"


def test_regularized_isotonic_regression(test_data):
    """Test RegularizedIsotonicRegression."""
    x, y, y_true = test_data
    cal = RegularizedIsotonicRegression(alpha=0.1)
    cal.fit(x, y)
    y_calib = cal.transform(x)
    assert len(y_calib) == len(x)
    corr = np.corrcoef(y_true, y_calib)[0, 1]
    assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"


def test_smoothed_isotonic_regression(test_data):
    """Test SmoothedIsotonicRegression."""
    x, y, y_true = test_data
    cal = SmoothedIsotonicRegression(window_length=7, poly_order=3, interp_method='linear')
    cal.fit(x, y)
    y_calib = cal.transform(x)
    assert len(y_calib) == len(x)
    corr = np.corrcoef(y_true, y_calib)[0, 1]
    assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"


def test_metrics_functions(test_data):
    """Test calibration metrics functions."""
    x, y, y_true = test_data
    error = mean_calibration_error(y_true, y)
    assert isinstance(error, float)
    assert error >= 0

    perfect_error = mean_calibration_error(y_true, y_true)
    assert perfect_error == 0

    error_bin = binned_calibration_error(y_true, y, n_bins=10)
    assert isinstance(error_bin, float)
    assert error_bin >= 0

    metrics = correlation_metrics(y_true, y, x=x, y_orig=y)
    # Check the basic keys exist and values are in expected range.
    assert 'spearman_corr_to_y_true' in metrics
    assert -1 <= metrics['spearman_corr_to_y_true'] <= 1

    counts = unique_value_counts(y, y_orig=y)
    assert 'n_unique_y_pred' in counts
    assert 'n_unique_y_orig' in counts
    assert counts['n_unique_y_pred'] <= counts['n_unique_y_orig']


def test_error_handling():
    """Test error handling for invalid input lengths."""
    x_good = np.array([1, 2, 3, 4, 5])
    y_good = np.array([1, 2, 3, 4, 5])
    x_bad = np.array([1, 2, 3])  # mismatched length
    
    with pytest.raises(ValueError):
        NearlyIsotonicRegression(lam=1.0, method='cvx').fit(x_bad, y_good)


if __name__ == "__main__":
    pytest.main()
