"""
Calibre: Advanced probability calibration methods for machine learning
"""
from .calibration import (
    BaseCalibrator,
    NearlyIsotonicRegression,
    ISplineCalibrator,
    RelaxedPAVA,
    RegularizedIsotonicRegression,
    SmoothedIsotonicRegression,
    check_arrays,
    sort_by_x
)
from .metrics import (
    mean_calibration_error,
    binned_calibration_error,
    correlation_metrics,
    unique_value_counts
)

__all__ = [
    # Calibrators
    'BaseCalibrator',
    'NearlyIsotonicRegression',
    'ISplineCalibrator',
    'RelaxedPAVA',
    'RegularizedIsotonicRegression',
    'SmoothedIsotonicRegression',
    
    # Utility functions
    'check_arrays',
    'sort_by_x',
    
    # Metrics
    'mean_calibration_error',
    'binned_calibration_error',
    'correlation_metrics',
    'unique_value_counts'
]

__version__ = '0.2.0' 
