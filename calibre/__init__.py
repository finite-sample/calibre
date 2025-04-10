"""
Calibre: Advanced probability calibration methods for machine learning
"""
from .calibration import (
    BaseCalibrator,
    NearlyIsotonicRegression,
    ISplineCalibrator,
    RelaxedPAVA,
    RegularizedIsotonicRegression,
    SmoothedIsotonicRegression
)
from .metrics import (
    mean_calibration_error,
    binned_calibration_error,
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    calibration_curve,
    correlation_metrics,
    unique_value_counts
)
from .utils import (
    check_arrays,
    sort_by_x,
    create_bins,
    bin_data
)

__all__ = [
    # Calibrators
    'BaseCalibrator',
    'NearlyIsotonicRegression',
    'ISplineCalibrator',
    'RelaxedPAVA',
    'RegularizedIsotonicRegression',
    'SmoothedIsotonicRegression',
    
    # Metrics
    'mean_calibration_error',
    'binned_calibration_error',
    'expected_calibration_error',
    'maximum_calibration_error', 
    'brier_score',
    'calibration_curve',
    'correlation_metrics',
    'unique_value_counts',
    
    # Utility functions
    'check_arrays',
    'sort_by_x',
    'create_bins',
    'bin_data'
]

__version__ = '0.2.0'
