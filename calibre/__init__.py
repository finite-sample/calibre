"""
Calibre: Advanced probability calibration methods for machine learning
"""

from .calibration import (
    BaseCalibrator,
    IsotonicRegressionWithDiagnostics,
    ISplineCalibrator,
    NearlyIsotonicRegression,
    RegularizedIsotonicRegression,
    RelaxedPAVA,
    SmoothedIsotonicRegression,
)

# Import diagnostic functionality
from .diagnostics import IsotonicDiagnostics, PlateauAnalyzer, analyze_plateaus
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
from .utils import (
    bin_data,
    bootstrap_resample,
    check_arrays,
    compute_delong_ci,
    create_bins,
    extract_plateaus,
    minimum_detectable_difference,
    sort_by_x,
)

__all__ = [
    # Calibrators
    "BaseCalibrator",
    "IsotonicRegressionWithDiagnostics",
    "NearlyIsotonicRegression",
    "ISplineCalibrator",
    "RelaxedPAVA",
    "RegularizedIsotonicRegression",
    "SmoothedIsotonicRegression",
    # Diagnostics
    "IsotonicDiagnostics",
    "PlateauAnalyzer",
    "analyze_plateaus",
    # Metrics
    "mean_calibration_error",
    "binned_calibration_error",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "calibration_curve",
    "correlation_metrics",
    "unique_value_counts",
    "tie_preservation_score",
    "plateau_quality_score",
    "calibration_diversity_index",
    "progressive_sampling_diversity",
    # Utility functions
    "check_arrays",
    "sort_by_x",
    "create_bins",
    "bin_data",
    "extract_plateaus",
    "bootstrap_resample",
    "compute_delong_ci",
    "minimum_detectable_difference",
]

__version__ = "0.4.0"
