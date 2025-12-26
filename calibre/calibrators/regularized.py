"""
Regularized isotonic regression with L2 regularization.

This module provides isotonic regression with L2 regularization to prevent
overfitting and produce smoother calibration curves.
"""

from __future__ import annotations

import logging

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

    The optimization problem solved is:

    .. math::
        \\min_{\\beta} \\sum_{i=1}^{n} (y_i - \\beta_i)^2 + \\alpha \\sum_{i=1}^{n} \\beta_i^2

    subject to :math:`\\beta_i \\leq \\beta_{i+1}` for all :math:`i`.

    Parameters
    ----------
    alpha
        Regularization strength. Higher values result in smoother curves.
    enable_diagnostics
        Whether to enable plateau diagnostics analysis.


    Examples
    --------
    >>> import numpy as np
    >>> from calibre import RegularizedIsotonicCalibrator
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

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Implement the regularized isotonic regression fitting logic.

        Parameters
        ----------
        X
            The training input samples.
        y
            The target values.

        Notes
        -----
        This method implements the actual fitting logic. Data storage,
        diagnostics, and return value are handled by the base class fit() method.
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

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply regularized isotonic calibration to new data.

        Parameters
        ----------
        X
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
            if prob.status in ["optimal", "optimal_inaccurate"] and beta.value is not None:
                # Create interpolation function
                cal_func = interp1d(
                    X_sorted,
                    beta.value,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(beta.value[0], beta.value[-1]),
                )

                # Apply interpolation to get values at X points
                return np.asarray(np.clip(cal_func(X), 0, 1))

        except Exception as e:
            logger.warning(f"Regularized isotonic optimization failed: {e}")

        # Fallback to standard isotonic regression
        logger.warning("Falling back to standard isotonic regression")
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(self.X_, self.y_)
        return np.asarray(ir.transform(X))
