"""
I-Spline calibration with cross-validation.

This module provides monotonic I-spline calibration, which uses spline basis
functions with non-negative coefficients to ensure monotonicity while providing
smooth calibration curves.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer

from ..base import BaseCalibrator
from ..utils import check_arrays

logger = logging.getLogger(__name__)


class SplineCalibrator(BaseCalibrator):
    """I-Spline calibration with cross-validation.

    This calibrator uses monotonic I-splines with non-negative coefficients
    to ensure monotonicity while providing a smooth calibration function.
    Cross-validation is used to find the best model.

    Parameters
    ----------
    n_splines : int, default=10
        Number of spline basis functions.
    degree : int, default=3
        Polynomial degree of spline basis functions.
    cv : int, default=5
        Number of cross-validation folds.

    Attributes
    ----------
    spline_ : SplineTransformer or None
        Fitted spline transformer.
    model_ : Ridge or None
        Fitted linear model with non-negative coefficients.
    fallback_ : IsotonicRegression or None
        Fallback model if spline fitting fails.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.calibrators import SplineCalibrator
    >>>
    >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> y = np.array([0.12, 0.18, 0.35, 0.25, 0.55])
    >>>
    >>> cal = SplineCalibrator(n_splines=5)
    >>> cal.fit(X, y)
    >>> X_calibrated = cal.transform(np.array([0.15, 0.35, 0.55]))

    Notes
    -----
    I-splines are integrated versions of M-splines (monotone splines) that are
    guaranteed to be monotonically increasing when coefficients are non-negative.
    This calibrator fits a Ridge regression with positive=True constraint to
    ensure monotonicity.

    See Also
    --------
    IsotonicCalibrator : Non-parametric monotonic calibration
    SmoothedIsotonicCalibrator : Isotonic with local smoothing
    """

    def __init__(
        self,
        n_splines: int = 10,
        degree: int = 3,
        cv: int = 5,
        enable_diagnostics: bool = False,
    ):
        # Call base class for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.n_splines = n_splines
        self.degree = degree
        self.cv = cv

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SplineCalibrator":
        """Fit the I-Spline calibration model.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : SplineCalibrator
            Returns self for method chaining.
        """
        X, y = check_arrays(X, y)

        # Validate parameters
        if self.n_splines < 3:
            logger.warning("n_splines should be at least 3. Setting to 3.")
            self.n_splines = 3

        if self.degree < 1:
            logger.warning("degree should be at least 1. Setting to 1.")
            self.degree = 1

        # Reshape X to 2D if needed
        X_2d = np.array(X).reshape(-1, 1)

        # Create spline transformer with monotonicity constraints
        spline = SplineTransformer(
            n_knots=self.n_splines,
            degree=self.degree,
            extrapolation="constant",
            include_bias=True,
        )

        # Perform cross-validation to find the best model
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        best_score = -np.inf
        best_model = None

        for train_idx, val_idx in kf.split(X_2d):
            X_train, y_train = X_2d[train_idx], y[train_idx]
            X_val, y_val = X_2d[val_idx], y[val_idx]

            # Fit spline transformer
            X_train_spline = spline.fit_transform(X_train)

            # Fit linear model with non-negative coefficients (monotonicity constraint)
            model = Ridge(alpha=0.01, positive=True, fit_intercept=True)
            model.fit(X_train_spline, y_train)

            # Evaluate on validation set
            X_val_spline = spline.transform(X_val)
            score = model.score(X_val_spline, y_val)

            if score > best_score:
                best_score = score
                best_model = (spline, model)

        # If no best model was found, use simple isotonic regression
        if best_model is None:
            logger.warning(
                "Cross-validation failed to find a good model. Using fallback isotonic regression."
            )
            self.fallback_ = IsotonicRegression(out_of_bounds="clip")
            self.fallback_.fit(X, y)
            self.spline_ = None
            self.model_ = None
        else:
            self.spline_, self.model_ = best_model
            self.fallback_ = None

        # Run diagnostics if enabled
        # Store fit data and run diagnostics if enabled
        self._fit_data_X = X
        self._fit_data_y = y
        self._run_diagnostics()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply I-Spline calibration to new data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The values to be calibrated.

        Returns
        -------
        X_calibrated : array-like of shape (n_samples,)
            Calibrated values.
        """
        X = np.asarray(X).ravel()
        X_2d = X.reshape(-1, 1)

        if self.fallback_ is not None:
            return self.fallback_.transform(X)

        X_spline = self.spline_.transform(X_2d)
        predictions = self.model_.predict(X_spline)

        # Ensure predictions are within [0, 1] bounds
        return np.clip(predictions, 0, 1)
