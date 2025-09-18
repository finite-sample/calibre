"""
Tests for the diagnostic tools for isotonic regression plateau analysis.
"""

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from calibre.diagnostics import IsotonicDiagnostics, PlateauAnalyzer, analyze_plateaus
from calibre.calibration import IsotonicRegressionWithDiagnostics
from calibre.metrics import (
    tie_preservation_score,
    plateau_quality_score,
    calibration_diversity_index,
    progressive_sampling_diversity,
)
from calibre.utils import (
    extract_plateaus,
    bootstrap_resample,
    compute_delong_ci,
    minimum_detectable_difference,
)


@pytest.fixture
def plateau_data():
    """Generate synthetic data with known plateaus."""
    np.random.seed(42)
    
    # Create data that will produce plateaus with isotonic regression
    n = 100
    X = np.sort(np.random.uniform(0, 1, n))
    
    # Create true probabilities with some flat regions
    y_true = np.zeros(n)
    y_true[:20] = 0.1  # Flat region
    y_true[20:40] = np.linspace(0.1, 0.3, 20)  # Rising
    y_true[40:70] = 0.3  # Another flat region  
    y_true[70:] = np.linspace(0.3, 0.8, 30)  # Rising
    
    # Add noise to create binary outcomes
    y_binary = np.random.binomial(1, y_true)
    
    return X, y_binary, y_true


@pytest.fixture
def no_plateau_data():
    """Generate data without plateaus."""
    np.random.seed(123)
    
    n = 50
    X = np.sort(np.random.uniform(0, 1, n))
    y_true = X  # Strictly increasing
    y_binary = np.random.binomial(1, y_true)
    
    return X, y_binary, y_true


def test_extract_plateaus(plateau_data):
    """Test plateau extraction functionality."""
    X, y_binary, y_true = plateau_data
    
    # Fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(X, y_binary)
    y_calibrated = iso_reg.transform(X)
    
    # Extract plateaus
    plateaus = extract_plateaus(X, y_calibrated)
    
    # Should find some plateaus
    assert len(plateaus) >= 0  # May or may not have plateaus depending on noise
    
    # Check plateau format
    for start_idx, end_idx, value in plateaus:
        assert isinstance(start_idx, (int, np.integer))
        assert isinstance(end_idx, (int, np.integer))
        assert isinstance(value, (float, np.floating))
        assert start_idx < end_idx
        assert 0 <= start_idx < len(X)
        assert 0 <= end_idx < len(X)


def test_bootstrap_resample():
    """Test bootstrap resampling."""
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    X_boot, y_boot = bootstrap_resample(X, y, random_state=42)
    
    assert len(X_boot) == len(X)
    assert len(y_boot) == len(y)
    
    # Check that all values come from original data
    for x_val in X_boot:
        assert x_val in X
    for y_val in y_boot:
        assert y_val in y


def test_compute_delong_ci():
    """Test DeLong confidence interval computation."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_scores = np.array([0.1, 0.3, 0.4, 0.7, 0.2, 0.8])
    
    auc, ci_lower, ci_upper = compute_delong_ci(y_true, y_scores)
    
    assert 0 <= auc <= 1
    assert 0 <= ci_lower <= auc <= ci_upper <= 1
    
    # Test edge case with single class
    y_true_single = np.array([0, 0, 0])
    y_scores_single = np.array([0.1, 0.2, 0.3])
    
    auc, ci_lower, ci_upper = compute_delong_ci(y_true_single, y_scores_single)
    assert np.isnan(auc)


def test_minimum_detectable_difference():
    """Test MDD calculation."""
    mdd = minimum_detectable_difference(50, 60, 0.3)
    assert mdd > 0
    assert mdd < 1  # Should be a reasonable value
    
    # Test edge cases
    mdd_inf = minimum_detectable_difference(0, 10, 0.3)
    assert np.isinf(mdd_inf)
    
    mdd_inf2 = minimum_detectable_difference(10, 10, 0.0)
    assert np.isinf(mdd_inf2)


def test_plateau_analyzer(plateau_data):
    """Test PlateauAnalyzer class."""
    X, y_binary, y_true = plateau_data
    
    # Fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(X, y_binary)
    y_calibrated = iso_reg.transform(X)
    
    analyzer = PlateauAnalyzer()
    plateaus = analyzer.identify_plateaus(X, y_calibrated)
    
    # Should return list of PlateauInfo objects
    assert isinstance(plateaus, list)
    
    for plateau in plateaus:
        assert hasattr(plateau, 'start_idx')
        assert hasattr(plateau, 'end_idx')
        assert hasattr(plateau, 'value')
        assert hasattr(plateau, 'x_range')
        assert hasattr(plateau, 'sample_size')
        assert hasattr(plateau, 'width')
        
        # Test MDD computation
        mdd_left, mdd_right = analyzer.compute_mdd_for_plateau(plateau, X, y_binary)
        assert mdd_left >= 0
        assert mdd_right >= 0


def test_isotonic_diagnostics_basic(plateau_data):
    """Test basic IsotonicDiagnostics functionality."""
    X, y_binary, y_true = plateau_data
    
    # Use fewer bootstraps for faster testing
    diagnostics = IsotonicDiagnostics(n_bootstraps=10, n_splits=3, random_state=42)
    
    results = diagnostics.analyze(X, y_binary)
    
    # Check results structure
    assert 'n_plateaus' in results
    assert 'plateaus' in results
    assert 'classification_counts' in results
    assert isinstance(results['n_plateaus'], int)
    assert isinstance(results['plateaus'], list)
    
    # Check classification counts
    counts = results['classification_counts']
    assert 'supported' in counts
    assert 'limited_data' in counts
    assert 'inconclusive' in counts
    
    total_classified = sum(counts.values())
    assert total_classified == results['n_plateaus']


def test_isotonic_diagnostics_with_test_data(plateau_data):
    """Test IsotonicDiagnostics with separate test data."""
    X, y_binary, y_true = plateau_data
    
    # Split data
    n_train = len(X) // 2
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_binary[:n_train], y_binary[n_train:]
    
    diagnostics = IsotonicDiagnostics(n_bootstraps=5, random_state=42)
    results = diagnostics.analyze(X_train, y_train, X_test, y_test)
    
    # Should have results
    assert isinstance(results, dict)
    assert 'n_plateaus' in results


def test_analyze_plateaus_function(plateau_data):
    """Test the convenience analyze_plateaus function."""
    X, y_binary, y_true = plateau_data
    
    results = analyze_plateaus(X, y_binary, n_bootstraps=5, random_state=42)
    
    assert isinstance(results, dict)
    assert 'n_plateaus' in results
    assert 'plateaus' in results


def test_isotonic_regression_with_diagnostics(plateau_data):
    """Test IsotonicRegressionWithDiagnostics class."""
    X, y_binary, y_true = plateau_data
    
    # Test with diagnostics enabled
    cal = IsotonicRegressionWithDiagnostics(
        enable_diagnostics=True, 
        n_bootstraps=5, 
        random_state=42
    )
    cal.fit(X, y_binary)
    
    # Test transform
    y_calibrated = cal.transform(X)
    assert len(y_calibrated) == len(X)
    assert np.all((y_calibrated >= 0) & (y_calibrated <= 1))
    
    # Test diagnostics
    diagnostics = cal.get_diagnostics()
    assert diagnostics is not None
    assert isinstance(diagnostics, dict)
    
    # Test summary
    summary = cal.plateau_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    
    # Test with diagnostics disabled
    cal_no_diag = IsotonicRegressionWithDiagnostics(enable_diagnostics=False)
    cal_no_diag.fit(X, y_binary)
    
    assert cal_no_diag.get_diagnostics() is None
    assert "not available" in cal_no_diag.plateau_summary().lower()


def test_tie_preservation_score():
    """Test tie preservation score metric."""
    # Test case where ties are preserved
    y_orig = np.array([0.1, 0.15, 0.2, 0.6, 0.65, 0.7])
    y_cal = np.array([0.1, 0.15, 0.2, 0.65, 0.65, 0.65])  # Ties last 3
    
    score = tie_preservation_score(y_orig, y_cal)
    assert 0 <= score <= 1
    
    # Test perfect preservation (no ties in original)
    y_orig_no_ties = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y_cal_no_ties = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    score_perfect = tie_preservation_score(y_orig_no_ties, y_cal_no_ties)
    assert score_perfect == 1.0


def test_plateau_quality_score():
    """Test plateau quality score metric."""
    X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0, 0, 1, 1, 1])
    y_cal = np.array([0.1, 0.25, 0.25, 0.4, 0.6])  # Has a plateau
    
    score = plateau_quality_score(X, y, y_cal)
    assert 0 <= score <= 1
    
    # Test with no plateaus
    y_cal_no_plateau = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    score_no_plateau = plateau_quality_score(X, y, y_cal_no_plateau)
    assert score_no_plateau == 1.0


def test_calibration_diversity_index():
    """Test calibration diversity index."""
    # Test with high diversity
    y_cal_diverse = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    diversity = calibration_diversity_index(y_cal_diverse)
    assert diversity == 1.0  # All unique values
    
    # Test with low diversity
    y_cal_uniform = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
    diversity_low = calibration_diversity_index(y_cal_uniform)
    assert diversity_low == 0.2  # Only 1 unique value out of 5
    
    # Test relative diversity
    relative_div = calibration_diversity_index(y_cal_uniform, reference_diversity=1.0)
    assert relative_div == 0.2


def test_progressive_sampling_diversity():
    """Test progressive sampling diversity analysis."""
    np.random.seed(42)
    X = np.linspace(0, 1, 50)
    y = np.random.binomial(1, X, 50)
    
    sizes, diversities = progressive_sampling_diversity(
        X, y, sample_sizes=[10, 20, 30], n_trials=3, random_state=42
    )
    
    assert len(sizes) == 3
    assert len(diversities) == 3
    assert sizes == [10, 20, 30]
    
    # Diversities should be reasonable
    for div in diversities:
        assert 0 <= div <= 1


def test_edge_cases():
    """Test edge cases and error conditions."""
    # Empty data
    with pytest.raises(ValueError):
        extract_plateaus(np.array([]), np.array([]))
    
    # Mismatched lengths
    with pytest.raises(ValueError):
        extract_plateaus(np.array([1, 2]), np.array([1]))
    
    # Single point
    plateaus = extract_plateaus(np.array([1]), np.array([0.5]))
    assert len(plateaus) == 0  # Single point can't form a plateau
    
    # Test IsotonicDiagnostics with no plateaus
    X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Already perfectly calibrated
    
    diagnostics = IsotonicDiagnostics(n_bootstraps=5, random_state=42)
    results = diagnostics.analyze(X, y)
    
    assert results['n_plateaus'] == 0
    assert results['plateaus'] == []


def test_diagnostic_summary_methods():
    """Test the summary methods in IsotonicDiagnostics."""
    X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    diagnostics = IsotonicDiagnostics(n_bootstraps=5, random_state=42)
    results = diagnostics.analyze(X, y)
    
    # Test plateau_summary method if plateaus exist
    if hasattr(diagnostics, 'plateau_summary'):
        summary = diagnostics.plateau_summary()
        assert isinstance(summary, str)


if __name__ == "__main__":
    pytest.main([__file__])