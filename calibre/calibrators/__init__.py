"""
Calibrators package - collection of calibration algorithms.

This package provides various calibration methods for improving probability
predictions from machine learning models. All calibrators follow the sklearn
transformer interface with fit/transform methods.

Available Calibrators
--------------------
IsotonicCalibrator
    Isotonic regression calibration
NearlyIsotonicCalibrator
    Nearly-isotonic regression with soft monotonicity
SplineCalibrator
    I-Spline calibration with cross-validation
RelaxedPAVACalibrator
    Relaxed Pool Adjacent Violators Algorithm
RegularizedIsotonicCalibrator
    Isotonic regression with L2 regularization
SmoothedIsotonicCalibrator
    Isotonic regression with Savitzky-Golay smoothing

Examples
--------
>>> from calibre import IsotonicCalibrator
>>> import numpy as np
>>>
>>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
>>> y = np.array([0, 0, 1, 1, 1])
>>>
>>> cal = IsotonicCalibrator()
>>> cal.fit(X, y)
>>> X_calibrated = cal.transform(X)
"""

from __future__ import annotations

# Import all calibrators
from .cdi_iso import CDIIsotonicCalibrator
from .isotonic import IsotonicCalibrator
from .nearly_isotonic import NearlyIsotonicCalibrator
from .regularized import RegularizedIsotonicCalibrator
from .relaxed_pava import RelaxedPAVACalibrator
from .smoothed import SmoothedIsotonicCalibrator
from .spline import SplineCalibrator

# Define public API
__all__ = [
    "CDIIsotonicCalibrator",
    "IsotonicCalibrator",
    "NearlyIsotonicCalibrator",
    "SplineCalibrator",
    "RelaxedPAVACalibrator",
    "RegularizedIsotonicCalibrator",
    "SmoothedIsotonicCalibrator",
]
