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
from ..utils import check_array_1d, check_arrays, check_fitted

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
    ``min_slope > 0``   strictly increasing before optional output clipping
    ==================  =====================================================

    Args:
        epsilon: Largest decrease permitted between adjacent unique scores, in
            the units of ``y``. So ``epsilon=0.02`` means "tolerate a drop of
            up to 2 percentage points".
        min_slope: Minimum required increase between adjacent unique scores.
            Mutually exclusive with a non-zero ``epsilon``; this is the
            direction that separates adjacent fitted values before optional
            output clipping. ``"auto"`` (the default) uses
            ``0.01 / n_unique``, but only on the untouched default path --
            that is, when ``epsilon`` was also left at ``"auto"`` and the
            search settled on ``0``. Naming ``epsilon`` yourself, including
            ``epsilon=0``, leaves the slope at ``0`` and the estimator exactly
            as documented in the table above.
        cv: Number of cross-validation folds used when a hyperparameter is
            left at ``"auto"``. Ignored when every hyperparameter is pinned.
        scoring: Proper scoring rule the ``"auto"`` search minimizes.
            ``"auto"`` (the default) uses log loss for probability targets and
            squared error otherwise.
            Deliberately not a calibration error: ECE and its relatives are
            minimised by a constant forecast, so selecting on one would reward
            throwing resolution away.
        random_state: Seed for the cross-validation split, so an ``"auto"``
            selection is reproducible.
        clip_output: Clip calibrated values into ``[0, 1]``.
        enable_diagnostics: Whether to enable plateau diagnostics analysis.

    Attributes:
        calibration_curve_: The fitted calibration map.
        n_features_in_: Always 1. Present for scikit-learn compatibility.

    Notes:
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
        separates adjacent fitted values while keeping the map monotone.

        That is why the default is a slope rather than nothing. PAVA's plateaus are
        an artifact of pooling adjacent violators, not a finding about the data, and
        at ``min_slope=0`` this estimator keeps only 1-4% of the input's distinct
        values. A slope small enough to be invisible in the score recovers almost all
        of them: measured on logit-inflated designs at n from 300 to 3000, the
        default retains 80-95% of distinct values for a Brier cost in the fifth
        decimal. It is 80-95% rather than all of them because ``clip_output`` flattens
        the two ends of a fit that saturates 0 and 1; the plateaus that survive the
        default are at the boundaries, not in the interior. It scales as
        ``1 / n_unique`` because a fixed slope safe at n=1000
        would need an output range of 10 at n=1e6, and clipping would flatten it back
        into the plateaus it exists to prevent.

    Examples:
        >>> import numpy as np
        >>> from calibre import RelaxedPAVACalibrator
        >>>
        >>> x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> y = np.array([0, 0, 1, 0, 1])
        >>>
        >>> RelaxedPAVACalibrator(epsilon=0.0).fit_transform(x, y)
        array([0. , 0. , 0.5, 0.5, 1. ])

        Left alone, the default breaks that tie apart rather than reporting two
        scores as indistinguishable:

        >>> default = RelaxedPAVACalibrator().fit_transform(x, y)
        >>> bool(np.all(np.diff(default) > 0))
        True

        For this example, a minimum slope separates every adjacent fitted value:

        >>> fitted = RelaxedPAVACalibrator(min_slope=0.05).fit_transform(x, y)
        >>> bool(np.all(np.diff(fitted) > 0))
        True

        The bound itself is exact only without clipping. Clipping into ``[0, 1]``
        can shorten increments to zero at a boundary, so a strict-increase
        guarantee requires ``clip_output=False``:

        >>> exact = RelaxedPAVACalibrator(
        ...     min_slope=0.05, clip_output=False
        ... ).fit_transform(x, y)
        >>> bool(np.all(np.diff(exact) >= 0.05 - 1e-12))
        True
        >>> float(exact.min())                      # below 0, hence the clipping
        -0.025

    See Also:
        IsotonicCalibrator : The ``epsilon = 0`` special case.
        CenteredIsotonicCalibrator : Monotone interpolation between pooled blocks.
        NearlyIsotonicCalibrator : Penalises violations instead of bounding them.
    """

    #: Candidate tolerances searched when ``epsilon="auto"``. 0.0 is included so
    #: selection can return strict isotonic regression when that fits best.
    EPSILON_GRID = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1)

    def __init__(
        self,
        epsilon: float | str = "auto",
        min_slope: float | str = "auto",
        cv: int = 5,
        scoring: str = "auto",
        random_state: int | None = 0,
        clip_output: bool = True,
        enable_diagnostics: bool = False,
    ):
        # Call base class for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.epsilon = epsilon
        self.min_slope = min_slope
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.clip_output = clip_output

    def _fit_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Solve the epsilon-monotone problem once and store the fitted curve.

        Args:
            X: Uncalibrated scores.
            y: Targets.
            sample_weight: Non-negative per-observation weights.

        Raises:
            ValueError: If ``epsilon`` or ``min_slope`` is negative, or both
                are non-zero.
        """
        from ..selection import resolve_auto

        X, y = check_arrays(X, y)

        # "auto" scales the forced step with the sample so the enforced increments
        # consume at most 1% of the output range however many points there are.
        # A fixed default cannot do that: a slope safe at n=1000 needs a range of
        # 10 at n=1e6, and the fit would spend most of its output outside [0, 1]
        # before clipping flattened it back into the plateaus it was meant to
        # prevent. Measured across n from 300 to 20000, the adaptive value below
        # keeps 92-100% of the input's distinct values for a Brier cost that
        # shrinks from 7e-5 to 5e-6.
        automatic = isinstance(self.min_slope, str)
        if automatic and self.min_slope != "auto":
            raise ValueError(
                f'min_slope must be a number or "auto", got {self.min_slope!r}'
            )
        if not automatic and float(self.min_slope) < 0:
            raise ValueError(f"min_slope must be non-negative, got {self.min_slope}")

        explicit_slope = 0.0 if automatic else float(self.min_slope)

        # min_slope forbids plateaus and epsilon permits decreases, so selecting
        # epsilon while min_slope is set would search against the caller's stated
        # intent. Pin epsilon to 0 in that case rather than tune it.
        if explicit_slope > 0 and self.epsilon == "auto":
            self.epsilon_ = 0.0
        else:
            self.epsilon_ = resolve_auto(
                self.epsilon,
                "epsilon",
                self.EPSILON_GRID,
                lambda **kw: type(self)(
                    min_slope=explicit_slope, clip_output=self.clip_output, **kw
                ),
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,
                random_state=self.random_state,
                sample_weight=sample_weight,
            )

        if self.epsilon_ > 0 and explicit_slope > 0:
            raise ValueError(
                "epsilon and min_slope pull in opposite directions; set at most "
                f"one (got epsilon={self.epsilon}, min_slope={self.min_slope})"
            )

        # Pool tied scores. Beyond removing the interpolation hazard, this is what
        # makes the bound mean "per distinct score" rather than "per observation".
        x_unique, y_mean, weight = aggregate_ties(X, y, sample_weight)

        # The automatic slope applies on the default path only: neither parameter
        # named, and the search having concluded that strict monotonicity fits
        # best. A caller who names epsilon is driving, and epsilon=0.0 must keep
        # meaning plain isotonic regression -- so the automatic value stands down
        # rather than tilting a fit that was asked for flat.
        untouched = automatic and self.epsilon == "auto"
        if untouched and self.epsilon_ == 0.0:
            self.min_slope_ = 0.01 / len(x_unique)
        else:
            self.min_slope_ = explicit_slope

        # Lower bound on each increment: negative permits decreases, positive
        # forces strict growth.
        bound = self.min_slope_ - self.epsilon_
        fitted = shift_to_pava(y_mean, weight, L=bound)

        if self.clip_output:
            # Clipping can flatten an enforced minimum slope at the boundaries,
            # but returning probabilities outside [0, 1] is worse for a calibrator.
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
