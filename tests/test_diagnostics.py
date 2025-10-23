"""
Tests for the diagnostic tools for calibration plateau analysis.
"""

import numpy as np
import pytest

from calibre import IsotonicCalibrator
from calibre.diagnostics import detect_plateaus, run_plateau_diagnostics


@pytest.fixture
def plateau_data():
    """Generate synthetic data with known plateaus."""
    np.random.seed(42)

    # Create data that will produce plateaus with isotonic regression
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    # Add some noise to create realistic calibration data
    noise = np.random.normal(0, 0.05, len(x))
    y_observed = y_true + noise

    # Create binary labels for testing
    y_binary = (y_observed > 0.5).astype(int)

    return x, y_binary, y_true


def test_built_in_diagnostics_basic(plateau_data):
    """Test built-in diagnostics functionality."""
    X, y_binary, y_true = plateau_data

    # Test with diagnostics enabled
    cal = IsotonicCalibrator(enable_diagnostics=True)
    cal.fit(X, y_binary)

    # Test transform
    y_calibrated = cal.transform(X)
    assert len(y_calibrated) == len(X)
    assert np.all((y_calibrated >= 0) & (y_calibrated <= 1))

    # Test diagnostics
    assert cal.has_diagnostics()
    diagnostics = cal.get_diagnostics()
    assert diagnostics is not None
    assert isinstance(diagnostics, dict)
    assert "n_plateaus" in diagnostics
    assert "plateaus" in diagnostics
    assert "warnings" in diagnostics

    # Test summary
    summary = cal.diagnostic_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_built_in_diagnostics_disabled(plateau_data):
    """Test calibrator without diagnostics."""
    X, y_binary, y_true = plateau_data

    # Test with diagnostics disabled (default)
    cal = IsotonicCalibrator(enable_diagnostics=False)
    cal.fit(X, y_binary)

    # No diagnostics should be available
    assert not cal.has_diagnostics()
    assert cal.get_diagnostics() is None
    summary = cal.diagnostic_summary()
    assert "not available" in summary.lower()


def test_standalone_run_plateau_diagnostics(plateau_data):
    """Test standalone run_plateau_diagnostics function."""
    X, y_binary, y_true = plateau_data

    # Fit a calibrator first
    cal = IsotonicCalibrator()
    cal.fit(X, y_binary)
    y_calibrated = cal.transform(X)

    # Run standalone diagnostics
    diagnostics = run_plateau_diagnostics(X, y_binary, y_calibrated)

    assert isinstance(diagnostics, dict)
    assert "n_plateaus" in diagnostics
    assert "plateaus" in diagnostics
    assert "warnings" in diagnostics

    # Check structure
    assert isinstance(diagnostics["warnings"], list)


def test_detect_plateaus():
    """Test plateau detection function."""
    y_calibrated = np.array([0.2, 0.2, 0.5, 0.5, 0.5])  # Two plateaus

    plateaus = detect_plateaus(y_calibrated)

    assert isinstance(plateaus, list)
    assert len(plateaus) >= 1  # Should detect at least one plateau

    for plateau in plateaus:
        assert len(plateau) == 3  # (start_idx, end_idx, value)
        start_idx, end_idx, value = plateau
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        assert isinstance(value, (int, float))
        assert end_idx >= start_idx
        assert (end_idx - start_idx + 1) >= 2  # Minimum width for a plateau


def test_diagnostic_workflow_comparison():
    """Test that built-in and standalone diagnostics give similar results."""
    X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y = np.array([0, 0, 1, 1, 1])

    # Built-in approach
    cal_builtin = IsotonicCalibrator(enable_diagnostics=True)
    cal_builtin.fit(X, y)
    builtin_diagnostics = cal_builtin.get_diagnostics()

    # Standalone approach
    cal_standalone = IsotonicCalibrator()
    cal_standalone.fit(X, y)
    y_calibrated = cal_standalone.transform(X)
    standalone_diagnostics = run_plateau_diagnostics(X, y, y_calibrated)

    # Should have same structure
    assert builtin_diagnostics.keys() == standalone_diagnostics.keys()
    assert builtin_diagnostics["n_plateaus"] == standalone_diagnostics["n_plateaus"]


def test_edge_cases():
    """Test edge cases for diagnostics."""
    # Empty arrays
    X_empty = np.array([])
    y_empty = np.array([])
    y_cal_empty = np.array([])

    # Should handle gracefully
    try:
        diagnostics = run_plateau_diagnostics(X_empty, y_empty, y_cal_empty)
        # If it doesn't raise an error, check it returns reasonable results
        assert isinstance(diagnostics, dict)
    except (ValueError, IndexError):
        # It's okay if it raises an error for empty arrays
        pass

    # Single value
    X_single = np.array([0.5])
    y_single = np.array([1])
    y_cal_single = np.array([0.5])

    plateaus = detect_plateaus(y_cal_single, min_width=2)
    assert isinstance(plateaus, list)
    assert (
        len(plateaus) == 0
    )  # No plateaus possible with single point using min_width=2


def test_calibrator_with_different_parameters():
    """Test diagnostics work with different calibrator parameters."""
    X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])

    # Test with different calibrator configurations
    for y_min, y_max in [(None, None), (0.0, 1.0), (0.1, 0.9)]:
        cal = IsotonicCalibrator(
            y_min=y_min,
            y_max=y_max,
            enable_diagnostics=True,
        )
        cal.fit(X, y)

        assert cal.has_diagnostics()
        diagnostics = cal.get_diagnostics()
        assert isinstance(diagnostics, dict)
        assert "n_plateaus" in diagnostics
