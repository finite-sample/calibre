"""
Nearly-isotonic regression for flexible monotonic calibration.

This module provides nearly-isotonic regression, which relaxes the strict
monotonicity constraint by penalizing rather than prohibiting violations.
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


class NearlyIsotonicCalibrator(BaseCalibrator):
    """Nearly-isotonic regression for flexible monotonic calibration.

    This calibrator implements nearly-isotonic regression, which relaxes the
    strict monotonicity constraint of standard isotonic regression by penalizing
    rather than prohibiting violations. This allows for a more flexible fit
    while still maintaining a generally monotonic trend.

    Parameters
    ----------
    lam
        Regularization parameter controlling the strength of monotonicity constraint.
        Higher values enforce stricter monotonicity.
    method
        Method to use for solving the optimization problem:
        - 'cvx': Uses convex optimization with CVXPY
        - 'path': Uses a path algorithm similar to the original nearly-isotonic paper
    enable_diagnostics
        Whether to enable plateau diagnostics analysis.


    Notes
    -----
    Nearly-isotonic regression solves the following optimization problem:

    .. math::
        \\min_{\\beta} \\sum_{i=1}^{n} (y_i - \\beta_i)^2 + \\lambda \\sum_{i=1}^{n-1} \\max(0, \\beta_i - \\beta_{i+1})

    where :math:`\\beta` is the calibrated output, :math:`y` are the true labels,
    and :math:`\\lambda > 0` controls the strength of the monotonicity penalty.

    This formulation penalizes violations of monotonicity proportionally to their
    magnitude, allowing small violations when they significantly improve the fit.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre import NearlyIsotonicCalibrator
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

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Implement the nearly-isotonic regression fitting logic.

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
        self.X_ = X
        self.y_ = y

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply nearly-isotonic calibration to new data.

        Parameters
        ----------
        X
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
        """Implement nearly-isotonic regression using convex optimization.

        This method solves the convex optimization problem:
        minimize ||β - y||² + λ * Σ max(0, β[i] - β[i+1])

        Parameters
        ----------
        X
            Input values to calibrate.

        Returns
        -------
        ndarray of shape (n_samples,)
            Calibrated values obtained by linear interpolation of the
            optimal solution on the training grid.
        """
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
            if prob.status in ["optimal", "optimal_inaccurate"] and beta.value is not None:
                # Create interpolation function based on sorted values
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
            logger.warning(f"Optimization failed: {e}")

        # Fallback to standard isotonic regression if optimization fails
        logger.warning("Falling back to standard isotonic regression")
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(self.X_, self.y_)
        return np.asarray(ir.transform(X))

    def _transform_path(self, X: np.ndarray) -> np.ndarray:
        """Implement nearly-isotonic regression using a path algorithm.

        This method implements the path algorithm from the original
        nearly-isotonic regression paper, which iteratively merges
        groups of points that violate monotonicity until the penalty
        budget λ is exhausted.

        Parameters
        ----------
        X
            Input values to calibrate.

        Returns
        -------
        ndarray of shape (n_samples,)
            Calibrated values obtained by linear interpolation after
            applying the path algorithm.
        """
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
        return np.asarray(np.clip(cal_func(X), 0, 1))
