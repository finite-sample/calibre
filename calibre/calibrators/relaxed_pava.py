"""
Relaxed Pool Adjacent Violators Algorithm (PAVA) for calibration.

This module provides a relaxed version of PAVA that allows small monotonicity
violations, creating smoother calibration curves while maintaining general
monotonic trends.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d

from ..base import BaseCalibrator
from ..utils import check_arrays

logger = logging.getLogger(__name__)


class RelaxedPAVACalibrator(BaseCalibrator):
    """Relaxed Pool Adjacent Violators Algorithm (PAVA) for calibration.

    This calibrator implements a relaxed version of PAVA that allows small
    monotonicity violations up to a threshold determined by the percentile
    of differences between adjacent sorted points.

    Parameters
    ----------
    percentile
        Percentile of absolute differences to use as threshold.
        Lower values enforce stricter monotonicity.
    adaptive
        Whether to use the adaptive implementation (recommended) or the
        block-merging implementation.
    enable_diagnostics
        Whether to enable plateau diagnostics analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre import RelaxedPAVACalibrator
    >>>
    >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> y = np.array([0.12, 0.18, 0.35, 0.25, 0.55])
    >>>
    >>> cal = RelaxedPAVACalibrator(percentile=20)
    >>> cal.fit(X, y)
    >>> X_calibrated = cal.transform(np.array([0.15, 0.35, 0.55]))

    See Also
    --------
    IsotonicCalibrator : Strict monotonicity constraint
    NearlyIsotonicCalibrator : Penalized monotonicity violations
    """

    def __init__(
        self,
        percentile: float = 10,
        adaptive: bool = True,
        enable_diagnostics: bool = False,
    ):
        # Call base class for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.percentile = percentile
        self.adaptive = adaptive

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Implement the relaxed PAVA fitting logic.

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

        # Validate percentile parameter
        if not 0 <= self.percentile <= 100:
            logger.warning(
                f"percentile should be between 0 and 100. Got {self.percentile}. Clipping to range."
            )
            self.percentile = np.clip(self.percentile, 0, 100)

        self.X_ = X
        self.y_ = y

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply relaxed PAVA calibration to new data.

        Parameters
        ----------
        X
            The values to be calibrated.

        Returns
        -------
        Calibrated values.
        """
        X = np.asarray(X).ravel()

        # Apply relaxed PAVA to get calibrated values for training data
        if self.adaptive:
            y_calibrated = self._relaxed_pava_adaptive()
        else:
            y_calibrated = self._relaxed_pava_block()

        # Create interpolation function
        cal_func = interp1d(
            self.X_,
            y_calibrated,
            kind="linear",
            bounds_error=False,
            fill_value=(np.min(y_calibrated), np.max(y_calibrated)),
        )

        # Apply interpolation to get values at X points
        return np.asarray(np.clip(cal_func(X), 0, 1))

    def _relaxed_pava_adaptive(self) -> np.ndarray:
        """Implement relaxed PAVA with adaptive threshold."""
        X, y = self.X_, self.y_

        # Sort by X values
        sort_idx = np.argsort(X)
        y_sorted = y[sort_idx]

        # Calculate absolute differences between adjacent points
        diffs = np.abs(np.diff(y_sorted))

        # Handle edge cases
        if len(diffs) == 0:
            return y.copy()

        # Handle case where all differences are zero
        if np.all(diffs == 0):
            return y.copy()

        # Find relaxation threshold based on percentile of differences
        relaxation = np.percentile(diffs, self.percentile)

        n = len(y_sorted)
        y_smoothed = y_sorted.copy()

        # Iteratively pool adjacent violators that exceed the relaxation threshold
        max_iterations = min(n, 100)  # Prevent infinite loops
        for _iteration in range(max_iterations):
            changed = False
            for i in range(n - 1):
                # Check if monotonicity is violated by more than the threshold
                if y_smoothed[i] > y_smoothed[i + 1] + relaxation:
                    # Average adjacent violators
                    avg = (y_smoothed[i] + y_smoothed[i + 1]) / 2
                    y_smoothed[i] = avg
                    y_smoothed[i + 1] = avg
                    changed = True

            # If no changes in this iteration, we've converged
            if not changed:
                break

        # Restore original order
        y_result = np.empty_like(y)
        y_result[sort_idx] = y_smoothed

        return np.clip(y_result, 0, 1)

    def _relaxed_pava_block(self) -> np.ndarray:
        """Implement relaxed PAVA with block merging approach."""
        X, y = self.X_, self.y_

        # Sort by X values
        sort_idx = np.argsort(X)
        y_sorted = y[sort_idx]
        n = len(y_sorted)

        # Calculate threshold based on the percentile of sorted differences
        diffs = np.abs(np.diff(y_sorted))
        if len(diffs) > 0:
            epsilon = np.percentile(diffs, self.percentile)
        else:
            epsilon = 0.0

        # Apply modified PAVA with epsilon threshold
        y_fit = y_sorted.copy()

        # Use a more efficient approach with block tracking via indices
        block_starts = np.arange(n)
        block_ends = np.arange(n) + 1
        block_values = y_sorted.copy()

        changed = True
        max_iterations = min(n, 50)  # Prevent excessive iterations
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            i = 0
            while i < len(block_starts) - 1:
                if block_values[i] > block_values[i + 1] + epsilon:
                    # Merge blocks i and i+1
                    start = block_starts[i]
                    end = block_ends[i + 1]
                    merged_avg = np.mean(y_sorted[start:end])

                    # Update arrays
                    block_starts = np.concatenate(
                        [block_starts[:i], [start], block_starts[i + 2 :]]
                    )
                    block_ends = np.concatenate(
                        [block_ends[:i], [end], block_ends[i + 2 :]]
                    )
                    block_values = np.concatenate(
                        [block_values[:i], [merged_avg], block_values[i + 2 :]]
                    )

                    # Update y_fit for all merged indices
                    for j in range(start, end):
                        y_fit[j] = merged_avg

                    changed = True
                    # Don't increment i, check this position again
                else:
                    i += 1

        # Restore original order
        y_result = np.empty_like(y_fit)
        y_result[sort_idx] = y_fit

        return np.clip(y_result, 0, 1)
