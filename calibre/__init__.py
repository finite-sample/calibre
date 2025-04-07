"""
Calibre: Advanced probability calibration methods for machine learning
"""
from .calibration import (
    nearly_isotonic_opt,
    nearly_isotonic_path,
    ispline_calib,
    relax_pava,
    regularized_isotonic,
    locally_smoothed_isotonic,
    adaptive_smoothed_isotonic
)
from .metrics import (
    mean_calibration_error,
    binned_calibration_error,
    correlation_metrics,
    unique_value_counts
)

__all__ = [
    'nearly_isotonic_opt',
    'nearly_isotonic_path',
    'ispline_calib',
    'relax_pava',
    'regularized_isotonic',
    'locally_smoothed_isotonic',
    'adaptive_smoothed_isotonic',
    'mean_calibration_error',
    'binned_calibration_error',
    'correlation_metrics',
    'unique_value_counts'
]

__version__ = '0.1.2'