"""A compact summary of held-out calibration diagnostics and proper scores.

Everything here is assembled from the estimators in :mod:`calibre.metrics` and
:mod:`calibre.evaluation`; nothing new is computed. The report does not perform a
hypothesis test or turn several diagnostics into a calibration verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

from .evaluation import bootstrap_ci, score_decomposition
from .metrics import (
    brier_score,
    debiased_calibration_error,
    mean_calibration_error,
    plugin_calibration_error,
    smooth_calibration_error,
    sweep_calibration_error,
)
from .utils import check_arrays

__all__ = ["CalibrationReport", "calibration_report"]

_DEFAULT_BINS = 15


@dataclass(frozen=True)
class CalibrationReport:
    """Held-out diagnostics for one set of probability forecasts.

    Attributes:
        n_observations: Number of observations.
        base_rate: Observed event frequency.
        mean_prediction: Mean forecast probability.
        mean_calibration_error: Absolute gap between the mean forecast and base rate.
        brier_score: Mean Brier score.
        miscalibration: CORP miscalibration component (MCB).
        discrimination: CORP discrimination component (DSC).
        uncertainty: CORP uncertainty component (UNC).
        smooth_calibration_error: Smooth calibration error (smECE).
        smooth_calibration_bandwidth: Bandwidth selected for smECE.
        debiased_calibration_error: Bias-corrected binned error at ``n_bins``.
        plugin_calibration_error: Uncorrected binned error on the same bins.
        sweep_calibration_error: Monotonic sweep calibration error. Its
            interpretation assumes a non-decreasing population calibration curve.
        sweep_n_bins: Bin count selected by the monotonic sweep.
        n_bins: Bin count used for the debiased and plugin errors.
        n_unique_predictions: Number of distinct forecast probabilities.
        unique_prediction_ratio: ``n_unique_predictions / n``. This describes
            prediction granularity, not calibration or statistical resolution.
        intervals: Read-only bootstrap intervals, empty unless
            ``include_brier_interval=True`` was passed. Only ``brier_score`` is
            intervalled.
    """

    n_observations: int
    base_rate: float
    mean_prediction: float
    mean_calibration_error: float
    brier_score: float
    miscalibration: float
    discrimination: float
    uncertainty: float
    smooth_calibration_error: float
    smooth_calibration_bandwidth: float
    debiased_calibration_error: float
    plugin_calibration_error: float
    sweep_calibration_error: float
    sweep_n_bins: int
    n_bins: int
    n_unique_predictions: int
    unique_prediction_ratio: float
    intervals: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the interval mapping and each nested interval result."""
        frozen = {
            name: MappingProxyType(dict(interval))
            for name, interval in self.intervals.items()
        }
        object.__setattr__(self, "intervals", MappingProxyType(frozen))

    def _interval_text(self, key: str) -> str:
        """Format the interval for ``key``, or an empty string.

        Args:
            key: Metric name.

        Returns:
            str: ``"  [lo, hi]"`` or ``""``.
        """
        if key not in self.intervals:
            return ""
        interval = self.intervals[key]
        return f"  [{interval['lower']:.4f}, {interval['upper']:.4f}]"

    def __str__(self) -> str:
        """Return the report as an aligned block of text."""
        sweep_bin_label = "bin" if self.sweep_n_bins == 1 else "bins"
        lines = [
            "CalibrationReport  "
            f"n={self.n_observations:,}  base rate {self.base_rate:.4f}",
            "",
            f"  Brier            {self.brier_score:.4f}"
            f"{self._interval_text('brier_score')}",
            f"    = MCB          {self.miscalibration:.4f}"
            "   (recalibration recovers this)",
            f"    - DSC          {self.discrimination:.4f}   (earned by the forecasts)",
            f"    + UNC          {self.uncertainty:.4f}   (irreducible)",
            "",
            f"  mean cal. error  {self.mean_calibration_error:.4f}   "
            f"(mean forecast {self.mean_prediction:.4f})",
            f"  smECE            {self.smooth_calibration_error:.4f}"
            f"   (bandwidth {self.smooth_calibration_bandwidth:.4f}, chosen)",
            f"  debiased ECE     {self.debiased_calibration_error:.4f}   "
            f"({self.n_bins} bins)",
            f"  plugin ECE       {self.plugin_calibration_error:.4f}   "
            f"({self.n_bins} bins, uncorrected)",
            f"  sweep ECE        {self.sweep_calibration_error:.4f}   "
            f"({self.sweep_n_bins} {sweep_bin_label}; "
            "assumes a monotone calibration curve)",
            "",
            "  prediction granularity  "
            f"{self.n_unique_predictions:,} of {self.n_observations:,} "
            f"values unique ({self.unique_prediction_ratio:.1%})",
        ]
        if self.intervals:
            lines += [
                "",
                f"  Intervals: {self.intervals['brier_score']['level']:.0%} "
                f"bootstrap, method '{self.intervals['brier_score']['method']}', "
                "Brier only.",
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return the same text as :meth:`__str__`, so it reads well in a REPL."""
        return self.__str__()

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a plain dictionary.

        Returns:
            dict: Every field, suitable for a DataFrame row or JSON.
        """
        result = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "intervals"
        }
        result["intervals"] = {
            name: dict(interval) for name, interval in self.intervals.items()
        }
        return result


def calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = _DEFAULT_BINS,
    include_brier_interval: bool = False,
    interval_level: float = 0.95,
    interval_n_resamples: int = 1000,
    random_state: int | None = 0,
    interval_method: str = "bca",
) -> CalibrationReport:
    """Summarize the calibration of one set of probabilities.

    Gathers the CORP decomposition, four calibration-error estimators, and
    prediction granularity. It does not perform a hypothesis test or issue a
    calibrated/not-calibrated verdict.

    Args:
        y_true: Ground truth values (0 or 1).
        y_pred: Predicted probabilities.
        n_bins: Bin count for the two fixed-bin estimators. The sweep chooses
            its own and smECE needs none.
        include_brier_interval: Whether to bootstrap an interval for the mean
            Brier score. Calibration-error and CORP-component intervals are
            omitted because the ordinary row bootstrap is not generally valid
            for those non-smooth estimators.
        interval_level: Nominal coverage for the Brier interval.
        interval_n_resamples: Number of bootstrap resamples.
        random_state: Bootstrap random seed.
        interval_method: Method passed to :func:`~calibre.bootstrap_ci`.
            Defaults to ``"bca"``.

    Returns:
        CalibrationReport: The summary. Print it, or read fields off it.

    Raises:
        ValueError: If the evaluation data or ``n_bins`` are invalid.
        TypeError: If ``include_brier_interval`` is not boolean.

    Warnings:
        Run this on independent, **held-out** predictions. On the data a calibrator
        was fitted to, any isotonic-family method reports ``MCB`` of exactly zero by
        construction -- the calibrator and this diagnostic are the same PAV
        projection, and PAV is idempotent -- no matter how badly the model
        generalizes. Use :func:`~calibre.cross_val_calibrate` for out-of-fold
        probabilities.

        Sweep calibration error assumes a non-decreasing population calibration
        curve. It can be near zero for strongly nonmonotone miscalibration; compare
        it with the other diagnostics rather than treating it as a general verdict.

    Examples:
        >>> import numpy as np
        >>> from calibre import calibration_report
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 2000)
        >>> y = rng.binomial(1, p).astype(float)
        >>> report = calibration_report(y, p)
        >>> report.n_observations
        2000

        These are calibrated by construction, so miscalibration is small next to the
        discrimination the forecasts earn:

        >>> bool(report.miscalibration < 0.1 * report.discrimination)
        True

        And the uncorrected estimator reports more error than the corrected one:

        >>> bool(
        ...     report.plugin_calibration_error
        ...     >= report.debiased_calibration_error
        ... )
        True
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ):
        raise ValueError("n_bins must be a positive integer")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if not isinstance(include_brier_interval, (bool, np.bool_)):
        raise TypeError("include_brier_interval must be boolean")
    y_true, y_pred = check_arrays(y_true, y_pred)

    decomposition = score_decomposition(y_true, y_pred)
    smece, bandwidth = smooth_calibration_error(y_true, y_pred, return_bandwidth=True)
    sweep, sweep_bins = sweep_calibration_error(y_true, y_pred, return_n_bins=True)
    n_unique_predictions = int(np.unique(y_pred).size)
    n_observations = int(y_true.size)

    intervals: dict[str, dict[str, Any]] = {}
    if include_brier_interval:
        intervals["brier_score"] = bootstrap_ci(
            brier_score,
            y_true,
            y_pred,
            level=interval_level,
            n_resamples=interval_n_resamples,
            random_state=random_state,
            method=interval_method,
        )

    return CalibrationReport(
        n_observations=n_observations,
        base_rate=float(np.mean(y_true)),
        mean_prediction=float(np.mean(y_pred)),
        mean_calibration_error=mean_calibration_error(y_true, y_pred),
        brier_score=brier_score(y_true, y_pred),
        miscalibration=float(decomposition["miscalibration"]),
        discrimination=float(decomposition["discrimination"]),
        uncertainty=float(decomposition["uncertainty"]),
        smooth_calibration_error=float(smece),
        smooth_calibration_bandwidth=float(bandwidth),
        debiased_calibration_error=debiased_calibration_error(
            y_true, y_pred, n_bins=n_bins
        ),
        plugin_calibration_error=plugin_calibration_error(
            y_true, y_pred, n_bins=n_bins, norm=2
        ),
        sweep_calibration_error=float(sweep),
        sweep_n_bins=int(sweep_bins),
        n_bins=int(n_bins),
        n_unique_predictions=n_unique_predictions,
        unique_prediction_ratio=(
            float(n_unique_predictions / n_observations) if n_observations else 0.0
        ),
        intervals=intervals,
    )
