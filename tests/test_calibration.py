"""
Basic tests for the calibre package.

To run these tests, install pytest and run:
    pytest -xvs tests/
"""
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

# Import functions to test
from calibre import (
    nearly_isotonic_opt,
    nearly_isotonic_path,
    ispline_calib,
    relax_pava,
    mean_calibration_error,
    binned_calibration_error,
    correlation_metrics,
    unique_value_counts
)

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic test data
@pytest.fixture
def test_data():
    """Generate synthetic test data for calibration tests."""
    n = 100
    x = np.sort(np.random.uniform(0, 1, n))
    y_true = np.sin(2 * np.pi * x)
    y = y_true + np.random.normal(0, 0.1, size=n)
    return x, y, y_true

# Test nearly_isotonic_opt
def test_nearly_isotonic_opt(test_data):
    """Test nearly_isotonic_opt function."""
    x, y, y_true = test_data
    
    # Test with different lambda values
    y_calib_strict = nearly_isotonic_opt(x, y, lam=10.0)
    y_calib_relaxed = nearly_isotonic_opt(x, y, lam=0.1)
    
    # Verify output shape
    assert len(y_calib_strict) == len(y)
    assert len(y_calib_relaxed) == len(y)
    
    # Verify strict calibration has fewer unique values than relaxed
    assert len(np.unique(y_calib_strict)) <= len(np.unique(y_calib_relaxed))
    
    # Verify correlation with true values is reasonable
    assert np.corrcoef(y_true, y_calib_strict)[0, 1] > 0.5
    assert np.corrcoef(y_true, y_calib_relaxed)[0, 1] > 0.5

# Test nearly_isotonic_path
def test_nearly_isotonic_path(test_data):
    """Test nearly_isotonic_path function."""
    x, y, y_true = test_data
    
    # Test with different lambda values
    y_calib_strict = nearly_isotonic_path(x, y, lam=10.0)
    y_calib_relaxed = nearly_isotonic_path(x, y, lam=0.1)
    
    # Verify output shape
    assert len(y_calib_strict) == len(y)
    assert len(y_calib_relaxed) == len(y)
    
    # Verify strict calibration has fewer unique values than relaxed
    assert len(np.unique(y_calib_strict)) <= len(np.unique(y_calib_relaxed))
    
    # Verify correlation with true values is reasonable
    assert np.corrcoef(y_true, y_calib_strict)[0, 1] > 0.5
    assert np.corrcoef(y_true, y_calib_relaxed)[0, 1] > 0.5

# Test ispline_calib
def test_ispline_calib(test_data):
    """Test ispline_calib function."""
    x, y, y_true = test_data
    
    # Test with default parameters
    y_calib = ispline_calib(x, y)
    
    # Verify output shape
    assert len(y_calib) == len(y)
    
    # Verify correlation with true values is reasonable
    assert np.corrcoef(y_true, y_calib)[0, 1] > 0.5
    
    # Test with different parameters
    y_calib_more_splines = ispline_calib(x, y, n_splines=15)
    assert len(y_calib_more_splines) == len(y)

# Test relax_pava
def test_relax_pava(test_data):
    """Test relax_pava function."""
    x, y, y_true = test_data
    
    # Test with different percentile values
    y_calib_strict = relax_pava(x, y, percentile=5)
    y_calib_relaxed = relax_pava(x, y, percentile=20)
    
    # Verify output shape
    assert len(y_calib_strict) == len(y)
    assert len(y_calib_relaxed) == len(y)
    
    # Compare with standard isotonic regression
    ir = IsotonicRegression()
    y_iso = ir.fit_transform(x, y)
    
    # Relaxed PAVA should have more unique values than standard isotonic
    assert len(np.unique(y_calib_relaxed)) >= len(np.unique(y_iso))
    
    # Verify correlation with true values is reasonable
    assert np.corrcoef(y_true, y_calib_strict)[0, 1] > 0.5
    assert np.corrcoef(y_true, y_calib_relaxed)[0, 1] > 0.5

# Test mean_calibration_error
def test_mean_calibration_error(test_data):
    """Test mean_calibration_error function."""
    x, y, y_true = test_data
    
    # Calculate error
    error = mean_calibration_error(y_true, y)
    
    # Error should be a non-negative float
    assert isinstance(error, float)
    assert error >= 0
    
    # Perfect predictions should have zero error
    perfect_error = mean_calibration_error(y_true, y_true)
    assert perfect_error == 0
    
    # Constant predictions should have higher error
    const_y = np.ones_like(y_true) * np.mean(y_true)
    const_error = mean_calibration_error(y_true, const_y)
    assert const_error > error

# Test binned_calibration_error
def test_binned_calibration_error(test_data):
    """Test binned_calibration_error function."""
    x, y, y_true = test_data
    
    # Calculate error with different numbers of bins
    error_5bins = binned_calibration_error(y_true, y, n_bins=5)
    error_10bins = binned_calibration_error(y_true, y, n_bins=10)
    
    # Error should be a non-negative float
    assert isinstance(error_5bins, float)
    assert isinstance(error_10bins, float)
    assert error_5bins >= 0
    assert error_10bins >= 0
    
    # Perfect predictions should have zero error
    perfect_error = binned_calibration_error(y_true, y_true)
    assert perfect_error == 0
    
    # Using x for binning should work too
    error_with_x = binned_calibration_error(y_true, y, x=x)
    assert isinstance(error_with_x, float)
    assert error_with_x >= 0

# Test correlation_metrics
def test_correlation_metrics(test_data):
    """Test correlation_metrics function."""
    x, y, y_true = test_data
    
    # Calculate metrics with all parameters
    metrics = correlation_metrics(y_true, y, x=x, y_orig=y)
    
    # Should return a dictionary with expected keys
    assert isinstance(metrics, dict)
    assert 'spearman_corr_to_y_true' in metrics
    assert 'spearman_corr_to_y_orig' in metrics
    assert 'spearman_corr_to_x' in metrics
    
    # Values should be between -1 and 1
    assert -1 <= metrics['spearman_corr_to_y_true'] <= 1
    assert -1 <= metrics['spearman_corr_to_y_orig'] <= 1
    assert -1 <= metrics['spearman_corr_to_x'] <= 1
    
    # Correlation with self should be 1 (using approximate comparison)
    self_metrics = correlation_metrics(y_true, y_true)
    assert np.isclose(self_metrics['spearman_corr_to_y_true'], 1.0)

# Test unique_value_counts
def test_unique_value_counts(test_data):
    """Test unique_value_counts function."""
    x, y, y_true = test_data
    
    # Apply standard isotonic regression
    ir = IsotonicRegression()
    y_iso = ir.fit_transform(x, y)
    
    # Calculate counts
    counts = unique_value_counts(y_iso, y_orig=y)
    
    # Should return a dictionary with expected keys
    assert isinstance(counts, dict)
    assert 'n_unique_y_pred' in counts
    assert 'n_unique_y_orig' in counts
    assert 'unique_value_ratio' in counts
    
    # Values should be positive
    assert counts['n_unique_y_pred'] > 0
    assert counts['n_unique_y_orig'] > 0
    assert counts['unique_value_ratio'] > 0
    
    # Isotonic regression typically reduces the number of unique values
    assert counts['n_unique_y_pred'] <= counts['n_unique_y_orig']
    
    # Test with different precision
    counts_lower_precision = unique_value_counts(y_iso, y_orig=y, precision=2)
    counts_higher_precision = unique_value_counts(y_iso, y_orig=y, precision=10)
    
    # Higher precision should find more unique values
    assert counts_lower_precision['n_unique_y_pred'] <= counts_higher_precision['n_unique_y_pred']

# Test error handling
def test_error_handling():
    """Test error handling in various functions."""
    # Create invalid inputs
    x_good = np.array([1, 2, 3, 4, 5])
    y_good = np.array([1, 2, 3, 4, 5])
    x_bad = np.array([1, 2, 3])  # Wrong length
    
    # Functions should raise ValueError for mismatched lengths
    with pytest.raises(ValueError):
        nearly_isotonic_opt(x_bad, y_good)
    
    with pytest.raises(ValueError):
        nearly_isotonic_path(x_bad, y_good)
    
    with pytest.raises(ValueError):
        ispline_calib(x_bad, y_good)
    
    with pytest.raises(ValueError):
        relax_pava(x_bad, y_good)
    
    # Metrics should also check input validity
    with pytest.raises(ValueError):
        mean_calibration_error(x_bad, y_good)
    
    with pytest.raises(ValueError):
        binned_calibration_error(x_bad, y_good)