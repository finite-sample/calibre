"""Cost- and data-informed isotonic calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from .._core import StepFunction, aggregate_ties, shift_to_pava
from ..base import BaseCalibrator
from ..utils import check_array_1d, check_arrays, check_fitted

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["CDIIsotonicCalibrator"]


def _effective_sample_size_by_score(
    scores: np.ndarray,
    sample_weight: np.ndarray | None,
    unique_scores: np.ndarray,
) -> np.ndarray:
    """Return Kish effective sample sizes for positive-mass score groups."""
    weight = (
        np.ones(scores.size, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    positive = weight > 0.0
    group = np.searchsorted(unique_scores, scores[positive])
    sum_weight = np.bincount(
        group, weights=weight[positive], minlength=unique_scores.size
    )
    sum_squared_weight = np.bincount(
        group, weights=weight[positive] ** 2, minlength=unique_scores.size
    )
    return np.asarray(sum_weight**2 / sum_squared_weight, dtype=float)


class CDIIsotonicCalibrator(BaseCalibrator):
    r"""Cost- and data-informed isotonic calibration (CDI-ISO).

    On the sorted unique score groups, this estimator solves

    .. math::
        \min_z \sum_i w_i(y_i-z_i)^2
        \quad\text{s.t.}\quad z_{i+1}-z_i \ge L_i,

    where :math:`L_i=\phi_i-\epsilon_i`. Near user-supplied operating
    thresholds, :math:`\phi_i` can require a positive increment when the
    adjacent empirical rates differ by more than their pooled normal-approximation
    uncertainty. Away from those thresholds, :math:`\epsilon_i` permits a bounded
    decrease. A cumulative shift reduces the problem exactly to weighted PAVA.

    This is the estimator defined in Sood (2025). It is research-grade: there is
    no independent reference implementation or evidence for universal defaults.
    Choose ``bandwidth``, ``alpha``, and ``gamma`` using validation data that was
    not used to fit the final calibration map, and compare proper scores as well
    as decision utility at the intended threshold.

    Args:
        thresholds: Non-empty operating thresholds in ``[0, 1]``. Thresholds
            and input scores use the same probability-score scale.
        threshold_weights: Finite non-negative relative importance weights, one
            per threshold. At least one must be positive. Equal weights are used
            when omitted.
        bandwidth: Positive half-width of the triangular threshold kernel, in
            probability-score units.
        alpha: Two-sided significance level in ``(0, 1)`` for the pooled normal
            approximation used in the adjacent-rate lower confidence bound.
        gamma: Minimum-increment multiplier in ``[0, 1]``.
        clip_output: Clip fitted values into ``[0, 1]``. Clipping preserves
            ordering but can flatten a positive local increment at a boundary.
        enable_diagnostics: Whether to enable plateau diagnostics.

    Attributes:
        adjacency_bounds_: Learned lower bound for each adjacent fitted
            increment.
        block_rate_: Weighted event rate at each unique training score.
        block_weight_: Objective weight at each unique training score.
        calibration_curve_: Fitted right-continuous step function.
        cumulative_shift_: Cumulative shift used by the PAVA reduction.
        economics_weight_: Threshold-kernel weight for each adjacency.
        effective_sample_size_: Kish effective sample size used in each block's
            uncertainty calculation. This makes the inference invariant to a
            common rescaling of ``sample_weight``.
        thresholds_: Validated operating thresholds.
        threshold_weights_: Validated threshold weights normalized to sum to one.

    Notes:
        CDI-ISO estimates adjacent event rates at repeated score values. With
        continuous scores, most effective block sizes are one and the normal
        approximation is weak; discretize scores using a prespecified scheme or
        use another calibrator rather than interpreting those bounds as strong
        evidence.

        The returned map is the right-continuous step function specified in the
        paper. It is not scikit-learn's piecewise-linear isotonic interpolant.

    Examples:
        >>> import numpy as np
        >>> from calibre import CDIIsotonicCalibrator
        >>> scores = np.repeat([0.2, 0.5, 0.8], 20)
        >>> outcomes = np.r_[np.zeros(16), np.ones(4),
        ...                  np.zeros(10), np.ones(10),
        ...                  np.zeros(4), np.ones(16)]
        >>> calibrator = CDIIsotonicCalibrator(thresholds=[0.5])
        >>> calibrated = calibrator.fit_transform(scores, outcomes)
        >>> bool(np.all((calibrated >= 0.0) & (calibrated <= 1.0)))
        True

    References:
        Sood, G. (2025). *Calibration Where It Counts: Cost- and Data-Informed
        Isotonic Regression*. https://gsood.com/research/papers/calibre.pdf

        Kish, L. (1965). *Survey Sampling*. John Wiley & Sons.

        Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: A novel
        method for evaluating prediction models. *Medical Decision Making*,
        26(6), 565--574.
    """

    def __init__(
        self,
        thresholds: Sequence[float],
        *,
        threshold_weights: Sequence[float] | None = None,
        bandwidth: float = 0.05,
        alpha: float = 0.05,
        gamma: float = 0.15,
        clip_output: bool = True,
        enable_diagnostics: bool = False,
    ) -> None:
        super().__init__(enable_diagnostics=enable_diagnostics)
        self.thresholds = thresholds
        self.threshold_weights = threshold_weights
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.gamma = gamma
        self.clip_output = clip_output

    def _fit_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit the paper's adjacent-block constrained projection."""
        X, y = check_arrays(X, y)
        positive_mass = (
            np.ones(X.size, dtype=bool)
            if sample_weight is None
            else sample_weight > 0.0
        )
        if np.any((X[positive_mass] < 0.0) | (X[positive_mass] > 1.0)):
            raise ValueError("X must contain probability scores in [0, 1]")
        if np.any((y[positive_mass] != 0.0) & (y[positive_mass] != 1.0)):
            raise ValueError("y must contain binary outcomes in {0, 1}")

        self._validate_hyperparameters()
        x_unique, block_rate, block_weight = aggregate_ties(X, y, sample_weight)
        effective_size = _effective_sample_size_by_score(X, sample_weight, x_unique)
        bounds, economics_weight = self._local_bounds(
            x_unique, block_rate, effective_size
        )
        fitted = shift_to_pava(block_rate, block_weight, L=bounds)
        if self.clip_output:
            fitted = np.clip(fitted, 0.0, 1.0)

        cumulative_shift = np.zeros(x_unique.size, dtype=float)
        if bounds.size:
            np.cumsum(bounds, out=cumulative_shift[1:])

        self.adjacency_bounds_ = bounds
        self.block_rate_ = block_rate
        self.block_weight_ = block_weight
        self.calibration_curve_ = StepFunction(x_unique, fitted)
        self.cumulative_shift_ = cumulative_shift
        self.economics_weight_ = economics_weight
        self.effective_sample_size_ = effective_size
        self.n_features_in_ = 1

    def _validate_hyperparameters(self) -> None:
        """Validate and store the estimator's statistical design parameters."""
        try:
            thresholds = np.asarray(self.thresholds, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "thresholds must be a one-dimensional numeric sequence"
            ) from exc
        if thresholds.ndim != 1 or thresholds.size == 0:
            raise ValueError("thresholds must be a non-empty one-dimensional sequence")
        if not np.all(np.isfinite(thresholds)) or np.any(
            (thresholds < 0.0) | (thresholds > 1.0)
        ):
            raise ValueError("thresholds must contain finite values in [0, 1]")

        if self.threshold_weights is None:
            threshold_weights = np.ones(thresholds.size, dtype=float)
        else:
            try:
                threshold_weights = np.asarray(self.threshold_weights, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "threshold_weights must be a one-dimensional numeric sequence"
                ) from exc
            if (
                threshold_weights.ndim != 1
                or threshold_weights.shape != thresholds.shape
            ):
                raise ValueError("threshold_weights must match thresholds in shape")
            if not np.all(np.isfinite(threshold_weights)) or np.any(
                threshold_weights < 0.0
            ):
                raise ValueError(
                    "threshold_weights must contain finite non-negative values"
                )
            if not np.any(threshold_weights > 0.0):
                raise ValueError(
                    "threshold_weights must contain at least one positive value"
                )

        self.bandwidth_ = self._finite_scalar("bandwidth", self.bandwidth)
        if self.bandwidth_ <= 0.0:
            raise ValueError("bandwidth must be greater than zero")
        self.alpha_ = self._finite_scalar("alpha", self.alpha)
        if not 0.0 < self.alpha_ < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        self.gamma_ = self._finite_scalar("gamma", self.gamma)
        if not 0.0 <= self.gamma_ <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if not isinstance(self.clip_output, (bool, np.bool_)):
            raise ValueError("clip_output must be a boolean")

        self.thresholds_ = thresholds.copy()
        self.threshold_weights_ = threshold_weights / np.sum(threshold_weights)

    @staticmethod
    def _finite_scalar(name: str, value: float) -> float:
        """Coerce one numeric hyperparameter and reject non-finite values."""
        try:
            validated = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not np.isfinite(validated):
            raise ValueError(f"{name} must be a finite number")
        return validated

    def _local_bounds(
        self,
        scores: np.ndarray,
        rates: np.ndarray,
        effective_size: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the paper's adjacent-block bounds and economics weights."""
        if scores.size < 2:
            empty = np.empty(0, dtype=float)
            return empty, empty.copy()

        midpoint = 0.5 * (scores[:-1] + scores[1:])
        distance = np.abs(midpoint[:, None] - self.thresholds_[None, :])
        kernel = np.maximum(0.0, 1.0 - distance / self.bandwidth_)
        economics_weight = kernel @ self.threshold_weights_

        left_size = effective_size[:-1]
        right_size = effective_size[1:]
        pooled_rate = (left_size * rates[:-1] + right_size * rates[1:]) / (
            left_size + right_size
        )
        standard_error = np.sqrt(
            pooled_rate * (1.0 - pooled_rate) * (1.0 / left_size + 1.0 / right_size)
        )
        z_value = float(norm.ppf(1.0 - self.alpha_ / 2.0))
        lower_difference = rates[1:] - rates[:-1] - z_value * standard_error
        minimum_increment = (
            self.gamma_ * economics_weight * np.maximum(lower_difference, 0.0)
        )
        relaxation = (1.0 - economics_weight) * z_value * standard_error
        return (
            np.asarray(minimum_increment - relaxation, dtype=float),
            np.asarray(economics_weight, dtype=float),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map probability scores through the fitted CDI-ISO step function.

        Args:
            X: Probability scores in ``[0, 1]``.

        Returns:
            ndarray of shape (n_samples,): Calibrated values.

        Raises:
            ValueError: If a score is outside ``[0, 1]``.
        """
        check_fitted(self, ["calibration_curve_"])
        X = check_array_1d(X)
        if np.any((X < 0.0) | (X > 1.0)):
            raise ValueError("X must contain probability scores in [0, 1]")
        return self.calibration_curve_(X)
