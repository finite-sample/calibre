"""
Calibre: Model Probability Calibration Library

This library provides various methods for calibrating probability predictions
from machine learning models to improve their reliability.
"""

from __future__ import annotations

# Get version from pyproject.toml - single source of truth
import importlib.metadata

# Import modules (users can do: from calibre import metrics)
from . import metrics

# Import base classes
from .base import BaseCalibrator, MonotonicMixin

# Import all calibrators (including cvxpy-dependent ones)
from .calibrators import (
    CDIIsotonicCalibrator,
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    RegularizedIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SmoothedIsotonicCalibrator,
    SplineCalibrator,
)

# Import diagnostic functions
from .diagnostics import detect_plateaus, run_plateau_diagnostics

# Import all metrics functions directly for convenient access
from .metrics import (
    binned_calibration_error,
    brier_score,
    calibration_curve,
    calibration_diversity_index,
    correlation_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    mean_calibration_error,
    plateau_quality_score,
    progressive_sampling_diversity,
    tie_preservation_score,
    unique_value_counts,
)

__version__ = importlib.metadata.version("calibre")

__all__ = [
    # Base classes
    "BaseCalibrator",
    "MonotonicMixin",
    # Calibrators
    "CDIIsotonicCalibrator",
    "IsotonicCalibrator",
    "NearlyIsotonicCalibrator",
    "SplineCalibrator",
    "RelaxedPAVACalibrator",
    "RegularizedIsotonicCalibrator",
    "SmoothedIsotonicCalibrator",
    # Diagnostic functions
    "run_plateau_diagnostics",
    "detect_plateaus",
    # Metrics functions
    "binned_calibration_error",
    "brier_score",
    "calibration_curve",
    "calibration_diversity_index",
    "correlation_metrics",
    "expected_calibration_error",
    "maximum_calibration_error",
    "mean_calibration_error",
    "plateau_quality_score",
    "progressive_sampling_diversity",
    "tie_preservation_score",
    "unique_value_counts",
    # Modules
    "metrics",
]
