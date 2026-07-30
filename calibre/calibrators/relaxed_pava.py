"""Epsilon-monotone isotonic regression via a cumulative-shift reduction.

Isotonic regression forces every adjacent increment to be non-negative, which is
exactly what produces its plateaus. Relaxing that to a lower *bound* on each
increment turns one signed parameter into a family of estimators, all solvable by
a single weighted PAVA call.
"""

from __future__ import annotations

import logging

import numpy as np

from .._core import PiecewiseLinear, aggregate_ties, shift_to_pava
from ..base import BaseCalibrator
from ..utils import check_arrays

logger = logging.getLogger(__name__)

__all__ = ["RelaxedPAVACalibrator"]


class RelaxedPAVACalibrator(BaseCalibrator):
    r"""Isotonic regression with a lower bound on each adjacent increment.

    Solves

    .. math::
        \min_{z} \sum_i w_i (y_i - z_i)^2
        \quad\text{s.t.}\quad z_{i+1} - z_i \ge L_i

    in O(n) via the cumulative-shift reduction (see
    :func:`calibre._core.shift_to_pava`): substituting
    :math:`u_i = z_i - \sum_{j<i} L_j` turns the constraint into
    :math:`u_{i+1} \ge u_i`, so one weighted PAVA on the shifted targets solves
    it exactly.

    One signed bound spans three estimators:

    ==================  =====================================================
    ``epsilon = 0``     standard isotonic regression
    ``epsilon > 0``     epsilon-monotone: decreases up to ``epsilon`` allowed
    ``min_slope > 0``   strictly increasing, so no plateau can form at all
    ==================  =====================================================

    Parameters
    ----------
    epsilon
        Largest decrease permitted between adjacent unique scores, in the units
        of ``y``. So ``epsilon=0.02`` means "tolerate a drop of up to 2
        percentage points".
    min_slope
        Minimum required increase between adjacent unique scores. Mutually
        exclusive with a non-zero ``epsilon``; this is the direction that
        eliminates plateaus.
    clip_output
        Clip calibrated values into ``[0, 1]``.
    enable_diagnostics
        Whether to enable plateau diagnostics analysis.

    Attributes
    ----------
    calibration_curve_ : PiecewiseLinear
        The fitted calibration map.
    n_features_in_ : int
        Always 1. Present for scikit-learn compatibility.

    Notes
    -----
    ``epsilon`` is an absolute tolerance on the target scale, deliberately. An
    earlier version of this class derived its threshold as a percentile of
    ``|diff(y)|`` over the score-sorted targets, which cannot work for this
    package's primary use case: with binary labels those differences are all 0 or
    1, so any percentile collapses to either 0 -- the relaxation never binds and
    the estimator is silently just PAVA -- or 1, where it never constrains
    anything. There is no intermediate setting to choose.

    Relaxing monotonicity is not free: a decrease in the calibration map reverses
    the ranking of every score pair it spans, which costs discrimination. To
    preserve granularity, ``min_slope`` is usually the better direction, since it
    removes plateaus while keeping the map strictly increasing.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre import RelaxedPAVACalibrator
    >>>
    >>> x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> y = np.array([0, 0, 1, 0, 1])
    >>>
    >>> RelaxedPAVACalibrator(epsilon=0.0).fit_transform(x, y)
    array([0. , 0. , 0.5, 0.5, 1. ])

    A minimum slope leaves no plateau anywhere:

    >>> fitted = RelaxedPAVACalibrator(min_slope=0.05).fit_transform(x, y)
    >>> bool(np.all(np.diff(fitted) > 0))
    True

    The bound itself is exact only without clipping. Clipping into ``[0, 1]``
    can shorten the increments that straddle a boundary, so the guarantee
    degrades from ">= min_slope" to "> 0" there:

    >>> exact = RelaxedPAVACalibrator(
    ...     min_slope=0.05, clip_output=False
    ... ).fit_transform(x, y)
    >>> bool(np.all(np.diff(exact) >= 0.05 - 1e-12))
    True
    >>> float(exact.min())                      # below 0, hence the clipping
    -0.025

    See Also
    --------
    IsotonicCalibrator : The ``epsilon = 0`` special case.
    CenteredIsotonicCalibrator : Removes plateaus without relaxing monotonicity.
    NearlyIsotonicCalibrator : Penalises violations instead of bounding them.
    """

    def __init__(
        self,
        epsilon: float = 0.0,
        min_slope: float = 0.0,
        clip_output: bool = True,
        enable_diagnostics: bool = False,
    ):
        # Call base class for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.epsilon = epsilon
        self.min_slope = min_slope
        self.clip_output = clip_output

    def _fit_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Solve the epsilon-monotone problem once and store the fitted curve.

        Parameters
        ----------
        X
            Uncalibrated scores.
        y
            Targets.
        sample_weight
            Non-negative per-observation weights.

        Raises
        ------
        ValueError
            If ``epsilon`` or ``min_slope`` is negative, or both are non-zero.
        """
        X, y = check_arrays(X, y)

        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if self.min_slope < 0:
            raise ValueError(f"min_slope must be non-negative, got {self.min_slope}")
        if self.epsilon > 0 and self.min_slope > 0:
            raise ValueError(
                "epsilon and min_slope pull in opposite directions; set at most "
                f"one (got epsilon={self.epsilon}, min_slope={self.min_slope})"
            )

        # Pool tied scores. Beyond removing the interpolation hazard, this is what
        # makes the bound mean "per distinct score" rather than "per observation".
        x_unique, y_mean, weight = aggregate_ties(X, y, sample_weight)

        # Lower bound on each increment: negative permits decreases, positive
        # forces strict growth.
        bound = self.min_slope - self.epsilon
        fitted = shift_to_pava(y_mean, weight, L=bound)

        if self.clip_output:
            # Clipping can flatten an enforced minimum slope at the boundaries,
            # but returning probabilities outside [0, 1] is worse for a calibrator.
            fitted = np.clip(fitted, 0.0, 1.0)

        self.calibration_curve_ = PiecewiseLinear(x_unique, fitted)
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
            Calibrated values.

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
