"""Calibrators package - collection of calibration algorithms.

This package provides various calibration methods for improving probability
predictions from machine learning models. All calibrators follow the sklearn
transformer interface with fit/transform methods.

Available Calibrators
--------------------
IsotonicCalibrator
    Isotonic regression calibration
CenteredIsotonicCalibrator
    Centered isotonic regression with interpolation between pooled blocks
NearlyIsotonicCalibrator
    Nearly-isotonic regression with soft monotonicity
SplineCalibrator
    I-Spline calibration with cross-validation
RelaxedPAVACalibrator
    Relaxed Pool Adjacent Violators Algorithm

Examples:
--------
>>> from calibre import IsotonicCalibrator
>>> import numpy as np
>>>
>>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
>>> y = np.array([0, 0, 1, 1, 1])
>>>
>>> cal = IsotonicCalibrator()
>>> _ = cal.fit(X, y)
>>> X_calibrated = cal.transform(X)
"""

from __future__ import annotations

# Import all calibrators
from .cdi_iso import CDIIsotonicCalibrator
from .centered_isotonic import CenteredIsotonicCalibrator
from .isotonic import IsotonicCalibrator
from .nearly_isotonic import NearlyIsotonicCalibrator
from .relaxed_pava import RelaxedPAVACalibrator
from .spline import SplineCalibrator

# Define public API
__all__ = [
    "CDIIsotonicCalibrator",
    "CenteredIsotonicCalibrator",
    "IsotonicCalibrator",
    "NearlyIsotonicCalibrator",
    "RelaxedPAVACalibrator",
    "SplineCalibrator",
]
