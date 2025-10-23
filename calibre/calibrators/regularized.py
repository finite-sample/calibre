"""
Regularized isotonic regression with L2 regularization.

This module provides isotonic regression with L2 regularization to prevent
overfitting and produce smoother calibration curves.
"""

import logging
from typing import Optional

import cvxpy as cp
import numpy as np
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

from ..base import BaseCalibrator
from ..utils import check_arrays, sort_by_x

logger = logging.getLogger(__name__)


class RegularizedIsotonicCalibrator(BaseCalibrator):
    """Regularized isotonic regression with L2 regularization.

    This calibrator adds L2 regularization to standard isotonic regression to
    prevent overfitting and produce smoother calibration curves.

    Parameters
    ----------
    alpha : float, default=0.1
        Regularization strength. Higher values result in smoother curves.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples,)
        The training input samples.
    y_ : ndarray of shape (n_samples,)
        The target values.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.calibrators import RegularizedIsotonicCalibrator
    >>>
    >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> y = np.array([0.12, 0.18, 0.35, 0.25, 0.55])
    >>>
    >>> cal = RegularizedIsotonicCalibrator(alpha=0.2)
    >>> cal.fit(X, y)
    >>> X_calibrated = cal.transform(np.array([0.15, 0.35, 0.55]))

    See Also
    --------
    IsotonicCalibrator : No regularization
    NearlyIsotonicCalibrator : Penalizes monotonicity violations
    """

    def __init__(
        self,
        alpha: float = 0.1,
        enable_diagnostics: bool = False,
    ):
        # Call base class for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularizedIsotonicCalibrator":
        """Fit the regularized isotonic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : RegularizedIsotonicCalibrator
            Returns self for method chaining.
        """
        X, y = check_arrays(X, y)

        # Validate alpha parameter
        if self.alpha < 0:
            logger.warning(
                f"alpha should be non-negative. Got {self.alpha}. Setting to 0."
            )
            self.alpha = 0

        self.X_ = X
        self.y_ = y

        # Run diagnostics if enabled
        # Store fit data and run diagnostics if enabled
        self._fit_data_X = X
        self._fit_data_y = y
        self._run_diagnostics()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply regularized isotonic calibration to new data.

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

        # Calculate calibration function
        order, X_sorted, y_sorted = sort_by_x(self.X_, self.y_)

        # Define variables
        beta = cp.Variable(len(y_sorted))

        # Monotonicity constraints: each value should be greater than or equal to the previous
        constraints = [beta[:-1] <= beta[1:]]

        # Objective: minimize squared error + alpha * L2 regularization
        obj = cp.Minimize(
            cp.sum_squares(beta - y_sorted) + self.alpha * cp.sum_squares(beta)
        )

        # Create and solve the problem
        prob = cp.Problem(obj, constraints)

        try:
            # Solve the problem
            prob.solve(solver=cp.OSQP, polishing=True)

            # Check if solution is found and is optimal
            if prob.status in ["optimal", "optimal_inaccurate"]:
                # Create interpolation function
                cal_func = interp1d(
                    X_sorted,
                    beta.value,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(beta.value[0], beta.value[-1]),
                )

                # Apply interpolation to get values at X points
                return np.clip(cal_func(X), 0, 1)

        except Exception as e:
            logger.warning(f"Regularized isotonic optimization failed: {e}")

        # Fallback to standard isotonic regression
        logger.warning("Falling back to standard isotonic regression")
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(self.X_, self.y_)
        return ir.transform(X)
