"""Isotonic regression with a lower bound on adjacent fitted increments."""

from __future__ import annotations

import logging

import numpy as np

from .._core import PiecewiseLinear, aggregate_ties, shift_to_pava
from ..base import BaseCalibrator
from ..utils import check_array_1d, check_arrays, check_fitted

logger = logging.getLogger(__name__)

__all__ = ["RelaxedPAVACalibrator"]


class RelaxedPAVACalibrator(BaseCalibrator):
    r"""Isotonic regression with a bound on each adjacent fitted increment.

    On the sorted unique input scores, this estimator solves

    .. math::
        \min_{z} \sum_i w_i (y_i - z_i)^2
        \quad\text{s.t.}\quad z_{i+1} - z_i \ge L,

    where ``min_increment`` is :math:`L`. Substituting
    :math:`u_i = z_i - iL` reduces the problem to weighted isotonic regression,
    so a single PAVA solve gives the exact constrained least-squares fit.

    The sign of the one constraint parameter determines the behavior:

    ========================  ================================================
    ``min_increment == 0``    ordinary isotonic regression
    ``min_increment < 0``     bounded decreases are permitted
    ``min_increment > 0``     adjacent fitted values must increase
    ========================  ================================================

    Args:
        min_increment: Lower bound on the fitted-value change between adjacent
            unique scores. For example, ``-0.02`` permits a decrease of at most
            two percentage points, while ``0.02`` requires an increase of at
            least two percentage points. There is no automatic default because
            the appropriate bound depends on the application and score grid.
        clip_output: Clip calibrated values into ``[0, 1]``. Clipping preserves
            non-positive increment bounds but can flatten a positive bound at
            the output boundaries.
        enable_diagnostics: Whether to enable plateau diagnostics analysis.

    Attributes:
        min_increment_: Bound selected during :meth:`fit`.
        calibration_curve_: Fitted piecewise-linear calibration map.
        n_features_in_: Always 1. Present for scikit-learn compatibility.

    Notes:
        The bound is measured per adjacent *unique score*, not per unit of input
        score. It therefore depends on the score grid. Set it explicitly only
        when that interpretation is appropriate.

        Negative bounds relax monotonicity and can reverse the ranking of scores.
        Positive bounds preserve ranking but may push an unclipped fit outside the
        probability range. Select the bound using held-out data and a proper score.

        The cumulative-shift construction is the total-order special case of the
        margin-separated isotonic projection described by Gunasekar, Koyejo, and
        Ghosh (2016). That result supports the optimization method, not a claim that
        any particular bound improves probability calibration.

    Examples:
        >>> import numpy as np
        >>> from calibre import RelaxedPAVACalibrator
        >>> x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> y = np.array([0, 0, 1, 0, 1])
        >>> RelaxedPAVACalibrator(min_increment=0.0).fit_transform(x, y)
        array([0. , 0. , 0.5, 0.5, 1. ])

        A negative bound permits small fitted decreases:

        >>> relaxed = RelaxedPAVACalibrator(
        ...     min_increment=-0.05, clip_output=False
        ... ).fit_transform(x, y)
        >>> bool(np.all(np.diff(relaxed) >= -0.05 - 1e-12))
        True

        A positive bound separates adjacent fitted values before clipping:

        >>> strict = RelaxedPAVACalibrator(
        ...     min_increment=0.05, clip_output=False
        ... ).fit_transform(x, y)
        >>> bool(np.all(np.diff(strict) >= 0.05 - 1e-12))
        True

    See Also:
        IsotonicCalibrator : The ``min_increment=0`` special case.
        CenteredIsotonicCalibrator : Monotone interpolation between pooled blocks.
        NearlyIsotonicCalibrator : Penalises rather than bounds violations.

    References:
        Gunasekar, S., Koyejo, O., & Ghosh, J. (2016). Preference completion from
        partial rankings. *Advances in Neural Information Processing Systems 29*.
    """

    def __init__(
        self,
        min_increment: float,
        *,
        clip_output: bool = True,
        enable_diagnostics: bool = False,
    ):
        super().__init__(enable_diagnostics=enable_diagnostics)
        self.min_increment = min_increment
        self.clip_output = clip_output

    def _fit_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit the bounded-increment least-squares projection."""
        X, y = check_arrays(X, y)
        try:
            self.min_increment_ = float(self.min_increment)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"min_increment must be finite, got {self.min_increment!r}"
            ) from exc
        if not np.isfinite(self.min_increment_):
            raise ValueError(
                f"min_increment must be finite, got {self.min_increment!r}"
            )

        x_unique, y_mean, weight = aggregate_ties(X, y, sample_weight)
        fitted = shift_to_pava(y_mean, weight, L=self.min_increment_)

        if self.clip_output:
            fitted = np.clip(fitted, 0.0, 1.0)

        self.calibration_curve_ = PiecewiseLinear(x_unique, fitted)
        self.n_features_in_ = 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map scores through the fitted calibration curve.

        Args:
            X: Scores to calibrate.

        Returns:
            ndarray of shape (n_samples,): Calibrated values.
        """
        check_fitted(self, ["calibration_curve_"])
        return self.calibration_curve_(check_array_1d(X))
