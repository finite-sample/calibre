"""
Isotonic regression calibrator with optional plateau diagnostics.

This module provides isotonic regression calibration, which is a non-parametric
method that fits a monotonically increasing function to data. It can optionally
perform sophisticated plateau analysis to distinguish between genuine flat
regions and artifacts of limited data.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..base import BaseCalibrator
from ..utils import check_arrays

logger = logging.getLogger(__name__)


class IsotonicCalibrator(BaseCalibrator):
    """
    Isotonic regression calibrator.

    This calibrator wraps sklearn's IsotonicRegression for probability calibration.

    Parameters
    ----------
    y_min : float, default=None
        Lower bound for the calibrated values.
    y_max : float, default=None
        Upper bound for the calibrated values.
    increasing : bool, default=True
        Whether the calibration function should be increasing.
    out_of_bounds : {'nan', 'clip', 'raise'}, default='clip'
        How to handle out-of-bounds values in transform.
    enable_diagnostics : bool, default=False
        Whether to enable plateau diagnostics analysis.

    Attributes
    ----------
    isotonic_ : IsotonicRegression
        The fitted isotonic regression model.
    diagnostics_ : dict or None
        Diagnostic results (only available if enable_diagnostics=True).

    Examples
    --------
    >>> import numpy as np
    >>> from calibre import IsotonicCalibrator
    >>>
    >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> y = np.array([0, 0, 1, 1, 1])
    >>>
    >>> # Basic usage
    >>> cal = IsotonicCalibrator()
    >>> cal.fit(X, y)
    >>> X_calibrated = cal.transform(X)
    >>>
    >>> # With diagnostics
    >>> cal = IsotonicCalibrator(enable_diagnostics=True)
    >>> cal.fit(X, y)
    >>> if cal.has_diagnostics():
    ...     print(cal.diagnostic_summary())

    Notes
    -----
    Isotonic regression finds the best monotonic fit to the data, which is
    particularly useful for calibration because well-calibrated predictions
    should maintain the rank order of predictions while improving probability
    estimates.

    See Also
    --------
    NearlyIsotonicCalibrator : Relaxed monotonicity constraint
    SmoothedIsotonicCalibrator : Isotonic with smoothing
    """

    def __init__(
        self,
        y_min: float | None = None,
        y_max: float | None = None,
        increasing: bool = True,
        out_of_bounds: str = "clip",
        enable_diagnostics: bool = False,
    ):
        # Call base class __init__ for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

        self.isotonic_: IsotonicRegression | None = None

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Implement the isotonic regression fitting logic.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The training input samples (predicted probabilities).
        y : array-like of shape (n_samples,)
            The target values (true labels).

        Notes
        -----
        This method implements the actual fitting logic. Data storage,
        diagnostics, and return value are handled by the base class fit() method.
        """
        X, y = check_arrays(X, y)

        # Fit standard isotonic regression
        self.isotonic_ = IsotonicRegression(
            y_min=self.y_min,
            y_max=self.y_max,
            increasing=self.increasing,
            out_of_bounds=self.out_of_bounds,
        )
        self.isotonic_.fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration to new data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The values to be calibrated.

        Returns
        -------
        X_calibrated : array-like of shape (n_samples,)
            Calibrated values.

        Raises
        ------
        ValueError
            If called before fit().
        """
        if self.isotonic_ is None:
            raise ValueError("Model must be fitted before transform")

        X = np.asarray(X).ravel()
        return self.isotonic_.transform(X)
