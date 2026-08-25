"""Calibre: Model Probability Calibration Library.

This library provides various methods for calibrating probability predictions
from machine learning models to improve their reliability.
"""

from __future__ import annotations

# Get version from pyproject.toml - single source of truth
import importlib
import importlib.metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Resolved lazily at runtime by __getattr__ below, so that matplotlib stays
    # an optional dependency. Imported here only so type checkers can see it.
    from . import plots

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
    bootstrap_ci,
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
    correlation_metrics,
    debiased_calibration_error,
    expected_calibration_error,
    maximum_calibration_error,
    mean_calibration_error,
    plugin_calibration_error,
    smooth_calibration_error,
    sweep_calibration_error,
    tie_preservation_score,
    unique_value_counts,
)

# Import the multiclass evaluation surface
from .multiclass import (
    TemperatureScaler,
    classwise_decomposition,
    classwise_ece,
    classwise_reliability,
    miscalibration_profile,
    top_label_ece,
)

# Import the one-call summary
from .report import CalibrationReport, calibration_report

# Import the shared cross-validation machinery
from .selection import cross_val_calibrate, make_folds, select_by_cv

__version__ = importlib.metadata.version("calibre")

__all__ = [
    # Base classes
    "BaseCalibrator",
    # Calibrators
    "CDIIsotonicCalibrator",
    "CalibrationReport",
    "CenteredIsotonicCalibrator",
    "IsotonicCalibrator",
    "MonotonicMixin",
    "NearlyIsotonicCalibrator",
    "RegularizedIsotonicCalibrator",
    "RelaxedPAVACalibrator",
    "SmoothedIsotonicCalibrator",
    "SplineCalibrator",
    "TemperatureScaler",
    # Metrics functions
    "binned_calibration_error",
    "bootstrap_ci",
    "brier_score",
    "calibration_curve",
    "calibration_report",
    "classwise_decomposition",
    "classwise_ece",
    "classwise_reliability",
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
    "miscalibration_profile",
    # Plotting (optional: needs `pip install 'calibre[plots]'`)
    "plots",
    "plugin_calibration_error",
    # Diagnostic functions
    "run_plateau_diagnostics",
    "score_decomposition",
    "select_by_cv",
    "smooth_calibration_error",
    "sweep_calibration_error",
    "tie_preservation_score",
    "top_label_ece",
    "unique_value_counts",
]


def __getattr__(name: str) -> object:
    """Resolve :mod:`calibre.plots` on first access.

    :pep:`562` module-level lookup, so that ``calibre.plots`` works after a plain
    ``import calibre`` without the plotting subpackage -- and therefore
    matplotlib -- being imported when nobody asks for it. matplotlib is an
    optional dependency and must stay one.

    Args:
        name: Attribute being looked up.

    Returns:
        object: The requested attribute.

    Raises:
        AttributeError: If ``name`` is not a lazily-exposed attribute.
    """
    if name == "plots":
        return importlib.import_module(f"{__name__}.plots")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the module's attributes, including the lazy ones.

    Returns:
        list of str: Sorted attribute names.
    """
    return sorted(set(__all__) | set(globals()))
