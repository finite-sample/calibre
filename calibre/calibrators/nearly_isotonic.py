"""
Nearly-isotonic regression for flexible monotonic calibration.

This module provides nearly-isotonic regression, which relaxes the strict
monotonicity constraint by penalizing rather than prohibiting violations.
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


class NearlyIsotonicCalibrator(BaseCalibrator):
    """Nearly-isotonic regression for flexible monotonic calibration.

    This calibrator implements nearly-isotonic regression, which relaxes the
    strict monotonicity constraint of standard isotonic regression by penalizing
    rather than prohibiting violations. This allows for a more flexible fit
    while still maintaining a generally monotonic trend.

    Parameters
    ----------
    lam : float, default=1.0
        Regularization parameter controlling the strength of monotonicity constraint.
        Higher values enforce stricter monotonicity.
    method : {'cvx', 'path'}, default='cvx'
        Method to use for solving the optimization problem:
        - 'cvx': Uses convex optimization with CVXPY
        - 'path': Uses a path algorithm similar to the original nearly-isotonic paper

    Attributes
    ----------
    X_ : ndarray of shape (n_samples,)
        The training input samples.
    y_ : ndarray of shape (n_samples,)
        The target values.

    Notes
    -----
    Nearly-isotonic regression solves the following optimization problem:

        minimize sum((y_i - beta_i)^2) + lambda * sum(max(0, beta_i - beta_{i+1}))

    This formulation penalizes violations of monotonicity proportionally to their
    magnitude, allowing small violations when they significantly improve the fit.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.calibrators import NearlyIsotonicCalibrator
    >>>
    >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> y = np.array([0.12, 0.18, 0.35, 0.25, 0.55])
    >>>
    >>> cal = NearlyIsotonicCalibrator(lam=0.5)
    >>> cal.fit(X, y)
    >>> X_calibrated = cal.transform(np.array([0.15, 0.35, 0.55]))

    See Also
    --------
    IsotonicCalibrator : Strict monotonicity constraint
    RegularizedIsotonicCalibrator : L2 regularization with strict monotonicity
    """

    def __init__(
        self,
        lam: float = 1.0,
        method: str = "cvx",
        enable_diagnostics: bool = False,
    ):
        # Call base class for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.lam = lam
        self.method = method

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearlyIsotonicCalibrator":
        """Fit the nearly-isotonic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : NearlyIsotonicCalibrator
            Returns self for method chaining.
        """
        X, y = check_arrays(X, y)
        self.X_ = X
        self.y_ = y

        # Store fit data and run diagnostics if enabled
        self._fit_data_X = X
        self._fit_data_y = y
        self._run_diagnostics()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply nearly-isotonic calibration to new data.

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
            If method is not 'cvx' or 'path'.
        """
        X = np.asarray(X).ravel()

        if self.method == "cvx":
            return self._transform_cvx(X)
        elif self.method == "path":
            return self._transform_path(X)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'cvx' or 'path'.")

    def _transform_cvx(self, X: np.ndarray) -> np.ndarray:
        """Implement nearly-isotonic regression using convex optimization."""
        order, X_sorted, y_sorted = sort_by_x(self.X_, self.y_)

        # Define variables
        beta = cp.Variable(len(y_sorted))

        # Penalty for non-monotonicity: sum of positive parts of decreases
        monotonicity_penalty = cp.sum(cp.maximum(0, beta[:-1] - beta[1:]))

        # Objective: minimize squared error + lambda * monotonicity penalty
        obj = cp.Minimize(
            cp.sum_squares(beta - y_sorted) + self.lam * monotonicity_penalty
        )

        # Create and solve the problem
        prob = cp.Problem(obj)

        try:
            prob.solve(solver=cp.OSQP, polishing=True)

            # Check if solution is found and is optimal
            if prob.status in ["optimal", "optimal_inaccurate"]:
                # Create interpolation function based on sorted values
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
            logger.warning(f"Optimization failed: {e}")

        # Fallback to standard isotonic regression if optimization fails
        logger.warning("Falling back to standard isotonic regression")
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(self.X_, self.y_)
        return ir.transform(X)

    def _transform_path(self, X: np.ndarray) -> np.ndarray:
        """Implement nearly-isotonic regression using a path algorithm."""
        order, X_sorted, y_sorted = sort_by_x(self.X_, self.y_)
        n = len(y_sorted)

        # Initialize solution with original values
        beta = y_sorted.copy()

        # Initialize groups and number of groups
        groups = [[i] for i in range(n)]

        # Initialize current lambda
        lambda_curr = 0

        while True:
            # Compute collision times
            collisions = []

            for i in range(len(groups) - 1):
                g1 = groups[i]
                g2 = groups[i + 1]

                # Calculate average values for each group
                avg1 = np.mean([beta[j] for j in g1])
                avg2 = np.mean([beta[j] for j in g2])

                # Check if collision will occur (if first group has higher value)
                if avg1 > avg2:
                    # Calculate collision time
                    t = avg1 - avg2
                    collisions.append((i, t))
                else:
                    # No collision will occur
                    collisions.append((i, np.inf))

            # Check termination condition
            if all(t[1] > self.lam - lambda_curr for t in collisions):
                break

            # Find minimum collision time
            valid_times = [(i, t) for i, t in collisions if t < np.inf]
            if not valid_times:
                break

            idx, t_min = min(valid_times, key=lambda x: x[1])

            # Compute new lambda value (critical point)
            lambda_star = lambda_curr + t_min

            # Check if we've exceeded lambda or reached max iterations
            if lambda_star > self.lam or len(groups) <= 1:
                break

            # Update current lambda
            lambda_curr = lambda_star

            # Merge groups
            new_group = groups[idx] + groups[idx + 1]
            avg = np.mean([beta[j] for j in new_group])
            for j in new_group:
                beta[j] = avg

            groups = groups[:idx] + [new_group] + groups[idx + 2 :]

        # Create interpolation function based on sorted values
        cal_func = interp1d(
            X_sorted,
            beta,
            kind="linear",
            bounds_error=False,
            fill_value=(beta[0], beta[-1]),
        )

        # Apply interpolation to get values at X points
        return np.clip(cal_func(X), 0, 1)
