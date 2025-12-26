# calibre/calibrators/cdi_iso.py
# Copyright (c) ...
# Licensed under the ... license.

"""
CDI-ISO: Cost- and Data-Informed Isotonic Calibration

This calibrator solves:
    min_z  sum_i w_i (y_i - z_i)^2
    s.t.   z_{i+1} - z_i >= L_i   for i = 1..m-1

on the unique sorted training scores (optionally aggregated),
where the local lower bounds are L_i = phi_i - epsilon_i.

- phi_i  (>=0): economically weighted minimum slope near operating thresholds,
               gated by statistical evidence of a positive adjacent-block difference.
- epsilon_i (>=0): variance-aware relaxation away from thresholds.

The constrained problem reduces to a single weighted isotonic regression
via a cumulative shift (shift-to-PAVA). Prediction is stepwise-constant,
as in standard isotonic calibration.

References (for context; not imported):
- Weighted PAVA under order constraints (Barlow et al. 1972; Robertson et al. 1988)
- Decision-curve analysis and threshold odds (Vickers & Elkin, 2006)
- Two-proportion normal approximation for difference SE (textbook)

Usage
-----
cal = CDIIsotonicCalibrator(
    thresholds=[0.2, 0.5], threshold_weights=[0.4, 0.6], bandwidth=0.05,
    alpha=0.05, gamma=0.15, window=25
)
cal.fit(scores_train, y_train)
p_test = cal.transform(scores_test)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

try:
    # Optional: for sklearn-style get_params/set_params compatibility
    from sklearn.base import BaseEstimator, TransformerMixin

    _SK_AVAILABLE = True
except Exception:
    BaseEstimator = object  # type: ignore
    TransformerMixin = object  # type: ignore
    _SK_AVAILABLE = False


# ----------------------------- utilities ---------------------------------- #


def _triangular_kernel(dist: np.ndarray, h: float) -> np.ndarray:
    """Triangular kernel K(d; h) = max(0, 1 - d/h) for d >= 0."""
    if h <= 0:
        # Degenerates to a point mass: weight only exact matches at zero distance
        return np.asarray((dist == 0).astype(float))
    w = 1.0 - dist / float(h)
    w[w < 0.0] = 0.0
    return w


def _inv_std_normal_cdf(p: float) -> float:
    """
    Inverse CDF of the standard normal, high-accuracy rational approximation.
    Source: Peter J. Acklam's algorithm (public domain).
    """
    if not (0.0 < p < 1.0):
        if p <= 0.0:
            return -np.inf
        if p >= 1.0:
            return np.inf

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    # Define break-points
    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return float((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        ))
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        return float(
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    else:
        q = np.sqrt(-2 * np.log(1 - p))
        return float(-(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))


def _z_value(alpha: float) -> float:
    """Two-sided z such that P(|Z| <= z) = 1 - alpha."""
    alpha = float(alpha)
    alpha = min(max(alpha, 1e-9), 0.999999999)  # guard
    return _inv_std_normal_cdf(1 - alpha / 2.0)


def _weighted_pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted PAVA for nondecreasing fit on a total order.
    Returns fitted values of length len(y).
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    assert y.ndim == 1 and w.ndim == 1 and y.size == w.size and y.size > 0

    means: list[float] = []
    weights: list[float] = []
    counts: list[int] = []

    for i in range(y.size):
        means.append(y[i])
        weights.append(w[i])
        counts.append(1)

        # Merge while the monotonicity constraint is violated
        while len(means) >= 2 and means[-2] > means[-1]:
            m2, w2, c2 = means.pop(), weights.pop(), counts.pop()
            m1, w1, c1 = means.pop(), weights.pop(), counts.pop()
            m = (w1 * m1 + w2 * m2) / (w1 + w2)
            w_new = w1 + w2
            c_new = c1 + c2
            means.append(m)
            weights.append(w_new)
            counts.append(c_new)

    # Expand block means to per-index fitted values
    fitted = np.repeat(np.asarray(means), np.asarray(counts))
    return fitted


# ----------------------------- calibrator ---------------------------------- #


@dataclass
class CDIIsotonicCalibrator(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """
    Cost- and Data-Informed Isotonic calibrator (CDI-ISO).

    Parameters
    ----------
    thresholds
        Operating thresholds in [0,1] that matter economically. If None,
        uniform attention across the score range is assumed.
    threshold_weights
        Nonnegative weights matching thresholds. If None, equal weights.
    bandwidth
        Half-width h of the triangular kernel around each threshold (in score units,
        after optional min-max normalization). Defaults to 0.05.
    alpha
        Significance level for the two-proportion normal approximation used to
        gate minimum-slope enforcement (default 0.05 -> z≈1.96).
    gamma
        Global multiplier in [0,1] for the minimum-slope budget phi_i (default 0.15).
    window
        Number of adjacent unique-score points used on each side to form the
        left/right evidence blocks (default 25). Automatically clipped at edges.
    normalize_scores
        If True (default), min-max normalize training scores to [0,1] for the
        economics kernel; the same affine scaling is applied at transform time.
    clip_output
        If True (default), clip calibrated outputs to [0,1].

    Notes
    -----
    - Builds local bounds L_i = phi_i - epsilon_i on sorted unique training scores.
    - Solves a single weighted PAVA on shifted labels (O(n)) and shifts back.
    - Predictions are stepwise-constant in the training score order.
    """

    thresholds: Iterable[float] | None = None
    threshold_weights: Iterable[float] | None = None
    bandwidth: float = 0.05
    alpha: float = 0.05
    gamma: float = 0.15
    window: int = 25
    normalize_scores: bool = True
    clip_output: bool = True

    # Fitted attributes
    _fitted: bool = False
    _s_min: float = 0.0
    _s_max: float = 1.0
    _x_unique: np.ndarray | None = None  # unique sorted scores (train scale)
    _x_unique_scaled: np.ndarray | None = None  # scaled to [0,1] if normalize_scores
    _z_fit: np.ndarray | None = None  # calibrated values per unique score
    _L: np.ndarray | None = None  # local bounds per adjacency
    _R: np.ndarray | None = None  # cumulative shift
    _w_block: np.ndarray | None = None  # weights per unique score (counts)

    # ----------------------------- core API -------------------------------- #

    def fit(
        self,
        scores: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> CDIIsotonicCalibrator:
        """
        Fit CDI-ISO on (scores, y).

        Parameters
        ----------
        scores
            Raw model scores; will be sorted internally. If normalize_scores=True,
            an affine min-max transform to [0,1] is learned and applied in transform.
        y
            Binary labels {0,1}.
        sample_weight
            Nonnegative per-sample weights.

        Returns
        -------
        Returns self for method chaining.

        Raises
        ------
        ValueError
            If scores and y have different lengths, y contains invalid values,
            or sample_weight has invalid values.
        """
        s = np.asarray(scores, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if s.shape[0] != y.shape[0]:
            raise ValueError("scores and y must have the same length")
        if np.any((y < 0) | (y > 1)):
            raise ValueError(
                "y must be in {0,1} (or in [0,1] for probabilistic labels)"
            )

        if sample_weight is None:
            w = np.ones_like(y, dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            if w.shape[0] != y.shape[0]:
                raise ValueError("sample_weight must match length of y")
            if np.any(w < 0):
                raise ValueError("sample_weight must be nonnegative")

        # Sort by scores
        order = np.argsort(s, kind="mergesort")
        s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]

        # Aggregate duplicates to unique-score blocks for stability/efficiency
        uniq_vals, idx_first, counts = np.unique(
            s_sorted, return_index=True, return_counts=True
        )

        sum_w = np.add.reduceat(w_sorted, idx_first)
        sum_yw = np.add.reduceat(y_sorted * w_sorted, idx_first)
        y_bar = np.divide(sum_yw, sum_w, out=np.zeros_like(sum_yw), where=sum_w > 0.0)

        x_unique = uniq_vals.astype(float)  # shape (m,)
        w_block = sum_w.astype(float)  # shape (m,)
        m = x_unique.size
        if m < 2:
            # Degenerate: constant mapping
            self._store_fit(
                x_unique,
                x_unique,
                np.full(m, y_bar[0]),
                np.zeros(0),
                np.zeros(m),
                w_block,
            )
            return self

        # Score scaling for economics kernel
        s_min, s_max = float(x_unique.min()), float(x_unique.max())
        self._s_min, self._s_max = s_min, s_max
        if self.normalize_scores and s_max > s_min:
            x_scaled = (x_unique - s_min) / (s_max - s_min)
        else:
            x_scaled = (
                x_unique - s_min
            )  # could be [0, range], but distances are consistent
        self._x_unique = x_unique
        self._x_unique_scaled = x_scaled

        # Build local bounds L_i = phi_i - epsilon_i on adjacencies
        L = self._compute_local_bounds(
            x_scaled=x_scaled,
            y_bar=y_bar,
            w_block=w_block,
        )  # shape (m-1,)

        # Shift-to-PAVA reduction: cumulative shift R_i, shifted labels y' = y_bar - R
        R = np.zeros(m, dtype=float)
        if L.size > 0:
            R[1:] = np.cumsum(L)
        y_shifted = y_bar - R

        # Solve weighted isotonic on shifted labels
        u_fit = _weighted_pava(y_shifted, w_block)
        z_fit = u_fit + R  # undo the shift

        if self.clip_output:
            z_fit = np.clip(z_fit, 0.0, 1.0)

        self._store_fit(x_unique, x_scaled, z_fit, L, R, w_block)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Map new scores to calibrated probabilities (stepwise-constant).

        Parameters
        ----------
        scores
            Input scores to calibrate.

        Returns
        -------
        Calibrated probabilities in [0,1] (if clip_output=True).

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        x_unique = self._x_unique  # train scale
        z_fit = self._z_fit
        assert x_unique is not None and z_fit is not None

        s = np.asarray(scores, dtype=float).reshape(-1)
        # Apply the same affine scaling for threshold distances if needed (not required for prediction)
        # For prediction, we step on the ORIGINAL train-scale breakpoints.
        # Stepwise rule: right-closed intervals (like sklearn IsotonicRegression)
        idx = np.searchsorted(x_unique, s, side="right") - 1
        idx[idx < 0] = 0
        idx[idx >= x_unique.size] = x_unique.size - 1
        p = z_fit[idx]
        if self.clip_output:
            p = np.clip(p, 0.0, 1.0)
        return p

    # --------------------------- diagnostics -------------------------------- #

    def adjacency_bounds_(self) -> np.ndarray | None:
        """Return the learned local bounds L_i per adjacency (shape: m-1) or None if not fitted."""
        return None if not self._fitted else self._L

    def cumulative_shift_(self) -> np.ndarray | None:
        """Return the cumulative shift R_i (shape: m) or None if not fitted."""
        return None if not self._fitted else self._R

    def breakpoints_(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (unique_scores, calibrated_values) on the training grid."""
        if not self._fitted or self._x_unique is None or self._z_fit is None:
            return None
        return (self._x_unique, self._z_fit)

    # --------------------------- internals ---------------------------------- #

    def _store_fit(
        self,
        x_unique: np.ndarray,
        x_scaled: np.ndarray,
        z_fit: np.ndarray,
        L: np.ndarray,
        R: np.ndarray,
        w_block: np.ndarray,
    ) -> None:
        self._x_unique = x_unique
        self._x_unique_scaled = x_scaled
        self._z_fit = z_fit
        self._L = L
        self._R = R
        self._w_block = w_block
        self._fitted = True

    def _compute_local_bounds(
        self,
        x_scaled: np.ndarray,  # shape (m,)
        y_bar: np.ndarray,  # shape (m,)
        w_block: np.ndarray,  # shape (m,)
    ) -> np.ndarray:
        """
        Build L_i = phi_i - epsilon_i for i=0..m-2 using:
          - economics kernel weight w_econ_i in [0,1],
          - two-proportion SE across adjacent aggregated blocks in a sliding band,
          - gamma and alpha hyperparameters.

        Parameters
        ----------
        x_scaled
            Scaled unique scores, shape (m,).
        y_bar
            Average target values per unique score, shape (m,).
        w_block
            Weights per unique score (counts), shape (m,).

        Returns
        -------
        Local bounds array of shape (m-1,).
        """
        m = x_scaled.size
        z = _z_value(self.alpha)

        # Economics weights per adjacency: center at midpoints between x_i and x_{i+1}
        mids = 0.5 * (x_scaled[:-1] + x_scaled[1:])  # shape (m-1,)
        w_econ = self._economics_weight(mids)  # in [0,1], shape (m-1,)

        # Evidence on adjacent-block differences within a sliding window
        win = int(max(1, self.window))
        L_list = []

        # Precompute cumulative sums for fast band aggregation
        c_w = np.cumsum(w_block)  # weights
        c_yw = np.cumsum(y_bar * w_block)  # weighted positives

        def band_sums(lo: int, hi: int) -> tuple[float, float]:
            """Inclusive band [lo, hi], return (sum_w, sum_pos_weighted)."""
            if lo > hi:
                return 0.0, 0.0
            sw = c_w[hi] - (c_w[lo - 1] if lo > 0 else 0.0)
            sy = c_yw[hi] - (c_yw[lo - 1] if lo > 0 else 0.0)
            return sw, sy

        for i in range(m - 1):
            # Left band: indices [i-win+1, i]
            l_lo = max(0, i - win + 1)
            l_hi = i
            # Right band: indices [i+1, i+win]
            r_lo = i + 1
            r_hi = min(m - 1, i + win)

            n_l, y_l = band_sums(l_lo, l_hi)
            n_r, y_r = band_sums(r_lo, r_hi)

            # Guard small bands
            if n_l <= 0 or n_r <= 0:
                L_list.append(0.0)
                continue

            p_l = y_l / n_l
            p_r = y_r / n_r
            p_pool = (y_l + y_r) / (n_l + n_r)

            # Two-proportion pooled SE
            se = np.sqrt(max(p_pool * (1 - p_pool), 0.0) * (1.0 / n_l + 1.0 / n_r))
            # Evidence-gated minimum slope (lower confidence bound, truncated at 0)
            delta_lcb = max(0.0, (p_r - p_l) - z * se)

            phi = self.gamma * w_econ[i] * delta_lcb
            eps = (1.0 - w_econ[i]) * z * se

            L_list.append(phi - eps)

        return np.asarray(L_list, dtype=float)

    def _economics_weight(self, mids: np.ndarray) -> np.ndarray:
        """
        Compute w_econ(mid) in [0,1] from thresholds and a triangular kernel of half-width h.

        If thresholds is None, return ones (uniform attention). Otherwise, for thresholds T_j
        with weights a_j, set
            w(mid) = sum_j a_j * K(|mid - T_j|; h) / sum_j a_j
        and normalize to have max 1 across mids.
        """
        if self.thresholds is None:
            return np.ones_like(mids, dtype=float)

        T = np.asarray(list(self.thresholds), dtype=float).reshape(-1)
        if T.size == 0:
            return np.ones_like(mids, dtype=float)

        if self.threshold_weights is None:
            A = np.ones_like(T, dtype=float)
        else:
            A = np.asarray(list(self.threshold_weights), dtype=float).reshape(-1)
            if A.size != T.size:
                raise ValueError("threshold_weights must match thresholds length")
            if np.any(A < 0):
                raise ValueError("threshold_weights must be nonnegative")
            if A.sum() == 0:
                A = np.ones_like(T, dtype=float)

        # Broadcast kernel contributions and average by total weight
        # mids: (k,), T: (J,) -> |mids[:,None] - T[None,:]| -> (k,J)
        d = np.abs(mids[:, None] - T[None, :])
        K = _triangular_kernel(d, self.bandwidth)  # (k,J)
        weighted = K * A[None, :]
        w = weighted.sum(axis=1) / (A.sum() + 1e-12)  # in [0,1] but may not reach 1.0

        # Normalize to [0,1] by max; if max is 0 (all outside bandwidth), return zeros.
        maxw = float(np.max(w)) if w.size > 0 else 0.0
        if maxw > 0:
            w = w / maxw
        return np.asarray(w.astype(float))
