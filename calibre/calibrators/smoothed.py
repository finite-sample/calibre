"""Locally smoothed isotonic regression.

Isotonic regression's fitted curve is a staircase. This calibrator runs a
Savitzky-Golay filter over that staircase to soften the steps, then restores
monotonicity with a running maximum.

Savitzky-Golay is a local polynomial fit and does **not** preserve monotonicity
on its own -- a rising staircase can be smoothed into something with a local dip.
The running maximum afterwards is what makes the result monotone, not the filter.

This estimator is retained for continuity. :class:`CenteredIsotonicCalibrator`
addresses the same jaggedness with a construction that is monotone by design
rather than by repair.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import savgol_filter

from .._core import PiecewiseLinear, aggregate_ties, weighted_pava
from ..base import (
    DEFAULT_POLY_ORDER,
    MIN_VARIANCE_THRESHOLD,
    BaseCalibrator,
    MonotonicMixin,
)
from ..utils import check_arrays

logger = logging.getLogger(__name__)

__all__ = ["SmoothedIsotonicCalibrator"]


class SmoothedIsotonicCalibrator(BaseCalibrator, MonotonicMixin):
    """Isotonic regression with Savitzky-Golay smoothing.

    Fits weighted PAVA on the pooled unique scores, smooths the fitted values,
    restores monotonicity with a running maximum, and interpolates linearly
    between the resulting knots.

    Parameters
    ----------
    window_length
        Window length for the Savitzky-Golay filter, in **distinct** scores.
        Forced odd and capped at the number of distinct scores. If None, uses
        ``max(5, n_distinct // 10)``.
    poly_order
        Polynomial order for the filter. Values below 1 are raised to 1.
    adaptive
        Size the window per point from local density instead of using one
        fixed window.
    min_window
        Minimum window length when ``adaptive=True``. Values below 3 are
        raised to 3.
    max_window
        Maximum window length when ``adaptive=True``. If None, uses
        ``n_distinct // 5``.
    enable_diagnostics
        Run plateau diagnostics after fitting.

    Attributes
    ----------
    calibration_curve_ : PiecewiseLinear
        The fitted calibration map, on the distinct training scores.
    poly_order_ : int
        ``poly_order`` after validation.
    min_window_ : int
        ``min_window`` after validation.
    n_features_in_ : int
        Always 1. Present for scikit-learn compatibility.

    Notes
    -----
    Window lengths count *distinct* scores, not observations. Tied scores are
    pooled before smoothing, because a filter applied across repeated abscissae
    smooths over points that carry no separate information, and an interpolant
    cannot be built on repeated abscissae at all.

    This estimator does not preserve granularity well. Restoring monotonicity
    with a running maximum re-flattens the curve wherever the filter introduced
    a dip, so plateaus come back: on the package's test datasets it retains
    roughly 13-16% of the distinct input values, against 100% for
    :class:`CenteredIsotonicCalibrator`. If granularity is why you are here,
    use that instead.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre import SmoothedIsotonicCalibrator
    >>>
    >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> y = np.array([0.12, 0.18, 0.35, 0.25, 0.55])
    >>>
    >>> cal = SmoothedIsotonicCalibrator(window_length=7)
    >>> _ = cal.fit(X, y)
    >>> p = cal.transform(np.array([0.15, 0.45]))
    >>> bool(p[0] <= p[1])
    True

    See Also
    --------
    IsotonicCalibrator : Isotonic regression without smoothing.
    CenteredIsotonicCalibrator : Smooth by construction rather than by repair.
    """

    def __init__(
        self,
        window_length: int | None = None,
        poly_order: int = DEFAULT_POLY_ORDER,
        adaptive: bool = False,
        min_window: int = 5,
        max_window: int | None = None,
        enable_diagnostics: bool = False,
    ):
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.window_length = window_length
        self.poly_order = poly_order
        self.adaptive = adaptive
        self.min_window = min_window
        self.max_window = max_window

    def _fit_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit the smoothed isotonic calibration map.

        Parameters
        ----------
        X
            Uncalibrated scores.
        y
            Targets: binary labels, or probabilities in ``[0, 1]``.
        sample_weight
            Not supported by this calibrator.

        Notes
        -----
        All of the work happens here so that ``transform`` is a pure lookup.
        Validated parameters are written to trailing-underscore attributes; the
        constructor arguments are left untouched so that ``get_params`` round
        trips and ``sklearn.base.clone`` reproduces the estimator.
        """
        self._reject_sample_weight(sample_weight)
        X, y = check_arrays(X, y)

        self.poly_order_ = self.poly_order
        if self.poly_order_ < 1:
            logger.warning(
                f"poly_order should be at least 1. Got {self.poly_order}. Using 1."
            )
            self.poly_order_ = 1

        self.min_window_ = self.min_window
        if self.min_window_ < 3:
            logger.warning(
                f"min_window should be at least 3. Got {self.min_window}. Using 3."
            )
            self.min_window_ = 3

        # Pool tied scores first: the estimator is defined on an ordered
        # sequence, and an interpolant cannot be built on repeated abscissae.
        x_unique, y_mean, weight = aggregate_ties(X, y)
        y_iso = weighted_pava(y_mean, weight)

        if self.adaptive:
            y_smoothed = self._smooth_adaptive(x_unique, y_iso)
        else:
            y_smoothed = self._smooth_fixed(y_iso)

        y_smoothed = np.clip(self.enforce_monotonicity(y_smoothed), 0.0, 1.0)

        self.calibration_curve_ = PiecewiseLinear(x_unique, y_smoothed)
        self.n_features_in_ = 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map scores through the fitted calibration curve.

        Parameters
        ----------
        X
            Scores to calibrate.

        Returns
        -------
        ndarray of shape (n_samples,)
            Calibrated probabilities.

        Raises
        ------
        AttributeError
            If called before :meth:`fit`.
        """
        if not hasattr(self, "calibration_curve_"):
            raise AttributeError(
                f"{type(self).__name__} is not fitted yet. Call fit() first."
            )
        return self.calibration_curve_(np.asarray(X, dtype=float).ravel())

    def _smooth_fixed(self, y_iso: np.ndarray) -> np.ndarray:
        """Smooth the isotonic fit with one window length.

        Parameters
        ----------
        y_iso
            Isotonic fit on the distinct scores.

        Returns
        -------
        ndarray
            Smoothed values, not yet made monotone.
        """
        m = y_iso.size
        window_length = (
            self.window_length if self.window_length is not None else max(5, m // 10)
        )
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, m - (m % 2 == 0))
        poly_order = min(self.poly_order_, window_length - 1)

        if m < window_length or window_length < 3:
            logger.info(
                f"Not enough distinct scores for smoothing (need {window_length}, "
                f"have {m}). Using the isotonic fit."
            )
            return y_iso

        try:
            y_smoothed = np.asarray(
                savgol_filter(y_iso, window_length, poly_order), dtype=float
            )
        except Exception as e:
            logger.warning(f"Savitzky-Golay smoothing failed: {e}")
            return y_iso

        # A near-constant smoothed curve means the filter has flattened the fit
        # rather than softened it; the isotonic values are the better answer.
        if np.var(y_smoothed) < MIN_VARIANCE_THRESHOLD:
            logger.warning(
                "Smoothed output has low variance; falling back to the isotonic fit."
            )
            return y_iso
        return y_smoothed

    def _smooth_adaptive(self, x: np.ndarray, y_iso: np.ndarray) -> np.ndarray:
        """Smooth the isotonic fit with a per-point window from local density.

        Parameters
        ----------
        x
            Distinct scores, strictly increasing.
        y_iso
            Isotonic fit on those scores.

        Returns
        -------
        ndarray
            Smoothed values, not yet made monotone.
        """
        m = y_iso.size
        if m <= 1:
            return y_iso

        x_range = x[-1] - x[0]
        if x_range <= 0:
            return y_iso

        max_window = (
            self.max_window
            if self.max_window is not None
            else max(self.min_window_, m // 5)
        )
        if max_window % 2 == 0:
            max_window += 1

        x_norm = (x - x[0]) / x_range
        y_smoothed = np.array(y_iso, dtype=float)
        for i in range(m):
            distances = np.abs(x_norm[i] - x_norm)
            window_size = self._find_optimal_window_size(
                distances, self.min_window_, max_window, m
            )
            if window_size >= 5:
                y_smoothed[i] = self._apply_local_smoothing(i, window_size, y_iso, m)
        return y_smoothed

    def _find_optimal_window_size(
        self, distances: np.ndarray, min_window: int, max_window: int, m: int
    ) -> int:
        """Return the largest window whose span holds at least that many points.

        Parameters
        ----------
        distances
            Normalised distances from the point of interest to every score.
        min_window
            Smallest window to consider.
        max_window
            Largest window to consider.
        m
            Number of distinct scores.

        Returns
        -------
        int
            Chosen window length.
        """
        window_size = min_window
        for w in range(min_window, max_window + 2, 2):
            if np.sum(distances <= w / m) >= w:
                window_size = w
            else:
                break
        return window_size

    def _apply_local_smoothing(
        self, i: int, window_size: int, y_iso: np.ndarray, m: int
    ) -> float:
        """Smooth one point against its local window.

        Parameters
        ----------
        i
            Index of the point to smooth.
        window_size
            Window length to centre on ``i``.
        y_iso
            Isotonic fit on the distinct scores.
        m
            Number of distinct scores.

        Returns
        -------
        float
            Smoothed value, or the isotonic value if the window is too small.
        """
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(m, i + half_window + 1)

        window_len = end_idx - start_idx
        if window_len % 2 == 0:
            window_len -= 1
        if window_len < 5:
            return float(y_iso[i])

        poly_ord = min(self.poly_order_, window_len - 1)
        try:
            y_local_smooth = np.asarray(
                savgol_filter(y_iso[start_idx:end_idx], window_len, poly_ord),
                dtype=float,
            )
        except Exception as e:
            logger.debug(f"Local smoothing failed for point {i}: {e}")
            return float(y_iso[i])

        local_idx = i - start_idx
        if 0 <= local_idx < y_local_smooth.size:
            return float(y_local_smooth[local_idx])
        return float(y_iso[i])
