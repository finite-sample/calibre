"""
Calibre: Model Probability Calibration Library.

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
    CenteredIsotonicCalibrator,
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    RegularizedIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SmoothedIsotonicCalibrator,
    SplineCalibrator,
)

# Import diagnostic functions
from .diagnostics import detect_plateaus, run_plateau_diagnostics

# Import the CORP evaluation stack
from .evaluation import (
    confidence_bands,
    consistency_bands,
    corp_reliability,
    score_decomposition,
)

# Import all metrics functions directly for convenient access
from .metrics import (
    binned_calibration_error,
    brier_score,
    calibration_curve,
    calibration_diversity_index,
    correlation_metrics,
    debiased_calibration_error,
    expected_calibration_error,
    maximum_calibration_error,
    mean_calibration_error,
    plateau_quality_score,
    progressive_sampling_diversity,
    sweep_calibration_error,
    tie_preservation_score,
    unique_value_counts,
)

# Import the shared cross-validation machinery
from .selection import cross_val_calibrate, make_folds, select_by_cv

__version__ = importlib.metadata.version("calibre")

__all__ = [
    # Base classes
    "BaseCalibrator",
    # Calibrators
    "CDIIsotonicCalibrator",
    "CenteredIsotonicCalibrator",
    "IsotonicCalibrator",
    "MonotonicMixin",
    "NearlyIsotonicCalibrator",
    "RegularizedIsotonicCalibrator",
    "RelaxedPAVACalibrator",
    "SmoothedIsotonicCalibrator",
    "SplineCalibrator",
    # Metrics functions
    "binned_calibration_error",
    "brier_score",
    "calibration_curve",
    "calibration_diversity_index",
    # CORP evaluation
    "confidence_bands",
    "consistency_bands",
    "corp_reliability",
    "correlation_metrics",
    # Cross-validation
    "cross_val_calibrate",
    "debiased_calibration_error",
    "detect_plateaus",
    "expected_calibration_error",
    "make_folds",
    "maximum_calibration_error",
    "mean_calibration_error",
    # Modules
    "metrics",
    "plateau_quality_score",
    "progressive_sampling_diversity",
    # Diagnostic functions
    "run_plateau_diagnostics",
    "score_decomposition",
    "select_by_cv",
    "sweep_calibration_error",
    "tie_preservation_score",
    "unique_value_counts",
]
