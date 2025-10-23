"""
Calibre: Model Probability Calibration Library

This library provides various methods for calibrating probability predictions
from machine learning models to improve their reliability.
"""

# Import modules (users can do: from calibre import metrics)
from . import metrics

# Import base classes
from .base import BaseCalibrator, MonotonicMixin

# Import all calibrators (including cvxpy-dependent ones)
from .calibrators import (
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    RegularizedIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SmoothedIsotonicCalibrator,
    SplineCalibrator,
)

# Import diagnostic functions
from .diagnostics import detect_plateaus, run_plateau_diagnostics

# Get version from pyproject.toml - single source of truth
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("calibre")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development/editable installs
    __version__ = "0.4.1-dev"

__all__ = [
    # Base classes
    "BaseCalibrator",
    "MonotonicMixin",
    # Calibrators
    "IsotonicCalibrator",
    "NearlyIsotonicCalibrator",
    "SplineCalibrator",
    "RelaxedPAVACalibrator",
    "RegularizedIsotonicCalibrator",
    "SmoothedIsotonicCalibrator",
    # Diagnostic functions
    "run_plateau_diagnostics",
    "detect_plateaus",
    # Modules
    "metrics",
]
