"""
Locally smoothed isotonic regression.

This module provides isotonic regression with Savitzky-Golay smoothing to
reduce jaggedness while preserving monotonicity.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression

from ..base import (
    DEFAULT_POLY_ORDER,
    MIN_VARIANCE_THRESHOLD,
    BaseCalibrator,
    MonotonicMixin,
)
from ..utils import check_arrays, sort_by_x

logger = logging.getLogger(__name__)


class SmoothedIsotonicCalibrator(BaseCalibrator, MonotonicMixin):
    """Locally smoothed isotonic regression.

    This calibrator applies standard isotonic regression and then smooths
    the result using a Savitzky-Golay filter, which preserves the monotonicity
    properties while reducing jaggedness.

    Parameters
    ----------
    window_length
        Window length for Savitzky-Golay filter. Should be odd.
        If None, window_length is set to max(5, len(X)//10)
    poly_order
        Polynomial order for the Savitzky-Golay filter.
        Must be less than window_length.
    interp_method
        Interpolation method to use ('linear', 'cubic', etc.)
    adaptive
        Whether to use adaptive window sizes based on local density.
    min_window
        Minimum window length when using adaptive=True.
    max_window
        Maximum window length when using adaptive=True.
        If None, max_window is set to len(X)//5.
    enable_diagnostics
        Whether to enable plateau diagnostics analysis.


    Examples
    --------
    >>> import numpy as np
    >>> from calibre import SmoothedIsotonicCalibrator
    >>>
    >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> y = np.array([0.12, 0.18, 0.35, 0.25, 0.55])
    >>>
    >>> cal = SmoothedIsotonicCalibrator(window_length=7)
    >>> cal.fit(X, y)
    >>> X_calibrated = cal.transform(X)

    See Also
    --------
    IsotonicCalibrator : Isotonic regression without smoothing
    RegularizedIsotonicCalibrator : L2 regularization instead of smoothing
    """

    def __init__(
        self,
        window_length: int | None = None,
        poly_order: int = DEFAULT_POLY_ORDER,
        interp_method: str = "linear",
        adaptive: bool = False,
        min_window: int = 5,
        max_window: int | None = None,
        enable_diagnostics: bool = False,
    ):
        # Call base class for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.window_length = window_length
        self.poly_order = poly_order
        self.interp_method = interp_method
        self.adaptive = adaptive
        self.min_window = min_window
        self.max_window = max_window

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Implement the smoothed isotonic regression fitting logic.

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

        if self.poly_order < 1:
            logger.warning(
                f"poly_order should be at least 1. Got {self.poly_order}. Setting to 1."
            )
            self.poly_order = 1

        if self.min_window < 3:
            logger.warning(
                f"min_window should be at least 3. Got {self.min_window}. Setting to 3."
            )
            self.min_window = 3

        self.X_ = X
        self.y_ = y

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply smoothed isotonic calibration to new data.

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
        if self.adaptive:
            y_smoothed = self._transform_adaptive()
        else:
            y_smoothed = self._transform_fixed()

        cal_func = interp1d(
            self.X_,
            y_smoothed,
            kind=self.interp_method,
            bounds_error=False,
            fill_value=(np.min(y_smoothed), np.max(y_smoothed)),
        )

        return np.asarray(np.clip(cal_func(X), 0, 1))

    def _transform_fixed(self) -> np.ndarray:
        """Implement smoothed isotonic regression with fixed window size."""
        order, X_sorted, y_sorted = sort_by_x(self.X_, self.y_)
        ir = IsotonicRegression(out_of_bounds="clip")
        y_iso = ir.fit_transform(X_sorted, y_sorted)

        n = len(X_sorted)
        window_length = (
            self.window_length if self.window_length is not None else max(5, n // 10)
        )
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, n - (n % 2 == 0))
        poly_order = min(self.poly_order, window_length - 1)

        if n >= window_length:
            try:
                y_smoothed = savgol_filter(y_iso, window_length, poly_order)
                # Check for low variance in the smoothed output
                if np.var(y_smoothed) < MIN_VARIANCE_THRESHOLD:
                    logger.warning(
                        "Smoothed output has low variance; falling back to isotonic regression result."
                    )
                    y_smoothed = y_iso
                else:
                    # Enforce monotonicity post-smoothing
                    y_smoothed = self.enforce_monotonicity(y_smoothed)
            except Exception as e:
                logger.warning(f"Savitzky-Golay smoothing failed: {e}")
                y_smoothed = y_iso
        else:
            logger.info(
                f"Not enough points for smoothing (need {window_length}, have {n}). Using isotonic regression without smoothing."
            )
            y_smoothed = y_iso

        y_result = np.empty_like(y_smoothed)
        y_result[order] = y_smoothed
        return np.asarray(np.clip(y_result, 0, 1))

    def _transform_adaptive(self) -> np.ndarray:
        """Implement smoothed isotonic regression with adaptive window size."""
        order, X_sorted, y_sorted = sort_by_x(self.X_, self.y_)
        ir = IsotonicRegression(out_of_bounds="clip")
        y_iso = ir.fit_transform(X_sorted, y_sorted)

        n = len(X_sorted)
        max_window = (
            self.max_window
            if self.max_window is not None
            else max(self.min_window, n // 5)
        )
        if max_window % 2 == 0:
            max_window += 1

        y_smoothed = np.array(y_iso)
        if n <= 1:
            y_result = np.empty_like(y_smoothed)
            y_result[order] = y_smoothed
            return y_result

        x_range = X_sorted[-1] - X_sorted[0]
        if x_range <= 0:
            return np.asarray(np.clip(y_iso, 0, 1))
        x_norm = (X_sorted - X_sorted[0]) / x_range

        for i in range(n):
            distances = np.abs(x_norm[i] - x_norm)
            window_size = self._find_optimal_window_size(
                distances, self.min_window, max_window, n
            )
            if window_size >= 5:
                y_smoothed[i] = self._apply_local_smoothing(
                    i, window_size, X_sorted, y_iso, n
                )

        # Enforce monotonicity
        y_smoothed = self.enforce_monotonicity(y_smoothed)

        y_result = np.empty_like(y_smoothed)
        y_result[order] = y_smoothed
        return np.asarray(np.clip(y_result, 0, 1))

    def _find_optimal_window_size(
        self, distances: np.ndarray, min_window: int, max_window: int, n: int
    ) -> int:
        window_size = min_window
        for w in range(min_window, max_window + 2, 2):
            width = w / n
            count = np.sum(distances <= width)
            if count >= w:
                window_size = w
            else:
                break
        return window_size

    def _apply_local_smoothing(
        self, i: int, window_size: int, X_sorted: np.ndarray, y_iso: np.ndarray, n: int
    ) -> float:
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        if end_idx - start_idx < 5:
            return float(y_iso[i])

        x_local = X_sorted[start_idx:end_idx]
        y_local = y_iso[start_idx:end_idx]
        window_len = len(x_local)
        if window_len % 2 == 0:
            window_len -= 1
        if window_len < 5:
            return float(y_iso[i])

        poly_ord = min(self.poly_order, window_len - 1)
        try:
            y_local_smooth = savgol_filter(y_local, window_len, poly_ord)
            local_idx = i - start_idx
            if 0 <= local_idx < len(y_local_smooth):
                return float(y_local_smooth[local_idx])
        except Exception as e:
            logger.debug(f"Local smoothing failed for point {i}: {e}")
        return float(y_iso[i])
