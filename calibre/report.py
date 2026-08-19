"""One call that says whether a model is calibrated, and how much it cost.

Everything here is assembled from the estimators in :mod:`calibre.metrics` and
:mod:`calibre.evaluation`; nothing new is computed. It exists because the answer
to "is my model calibrated?" needs more than one number, and gathering them by
hand invites the two mistakes this package keeps warning about: quoting a binned
ECE without its bin count, and scoring a calibrator on the data it was fit to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    """Everything worth knowing about one set of probabilities.

    Attributes:
        n: Number of observations.
        base_rate: Observed event frequency.
        mean_prediction: Mean forecast. Compare with ``base_rate``: the gap is ``bias``.
        bias: Calibration in the large, ``|mean_prediction - base_rate|``.
        brier: Brier score. The proper scoring rule to optimise.
        mcb: Miscalibration, from the CORP decomposition. What recalibration recovers.
        dsc: Discrimination. What the forecasts buy over predicting the base rate.
        unc: Uncertainty. The difficulty of the problem; no forecaster changes it.
        smece: Smooth calibration error, with no bin count and no bandwidth to choose.
        smece_sigma: The bandwidth smECE selected.
        debiased_ece: Bias-corrected binned error at ``n_bins``.
        plugin_ece: Uncorrected binned error at ``n_bins``, on the same bins.
            The gap between this and ``debiased_ece`` is the bias you would
            have reported.
        sweep_ece: Binned error at the bin count the monotone sweep selected.
        sweep_bins: That bin count.
        n_bins: The bin count used for ``debiased_ece`` and ``plugin_ece``.
        n_distinct: Distinct forecast values. Isotonic regression collapses
            this; the point of most of this package is not to.
        distinct_ratio: ``n_distinct / n``.
        intervals: Bootstrap confidence intervals, empty unless ``ci=True``
            was passed. Each holds ``lower``, ``upper``, ``bias`` and
            ``degenerate``.
    """

    n: int
    base_rate: float
    mean_prediction: float
    bias: float
    brier: float
    mcb: float
    dsc: float
    unc: float
    smece: float
    smece_sigma: float
    debiased_ece: float
    plugin_ece: float
    sweep_ece: float
    sweep_bins: int
    n_bins: int
    n_distinct: int
    distinct_ratio: float
    intervals: dict[str, dict[str, float]] = field(default_factory=dict)

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
        lines = [
            f"CalibrationReport  n={self.n:,}  base rate {self.base_rate:.4f}",
            "",
            f"  Brier            {self.brier:.4f}{self._interval_text('brier')}",
            f"    = MCB          {self.mcb:.4f}{self._interval_text('mcb')}"
            "   (recalibration recovers this)",
            f"    - DSC          {self.dsc:.4f}   (earned by the forecasts)",
            f"    + UNC          {self.unc:.4f}   (irreducible)",
            "",
            f"  bias             {self.bias:.4f}   "
            f"(mean forecast {self.mean_prediction:.4f})",
            f"  smECE            {self.smece:.4f}{self._interval_text('smece')}"
            f"   (bandwidth {self.smece_sigma:.4f}, chosen)",
            f"  debiased ECE     {self.debiased_ece:.4f}"
            f"{self._interval_text('debiased_ece')}   ({self.n_bins} bins)",
            f"  plugin ECE       {self.plugin_ece:.4f}   "
            f"({self.n_bins} bins, uncorrected)",
            f"  sweep ECE        {self.sweep_ece:.4f}   "
            f"({self.sweep_bins} bins, chosen)",
            "",
            f"  distinct values  {self.n_distinct:,} of {self.n:,} "
            f"({self.distinct_ratio:.1%})",
        ]
        if self.intervals:
            lines += [
                "",
                f"  Intervals: {self.intervals['brier']['level']:.0%} bootstrap, "
                f"method '{self.intervals['brier']['method']}'. A calibration",
                "  error is a convex functional, so resampling inflates it -- worst "
                "when the model",
                "  is well calibrated. See calibre.bootstrap_ci for why, and for "
                "what that costs.",
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
        from dataclasses import asdict

        return asdict(self)


def calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = _DEFAULT_BINS,
    ci: bool = False,
    level: float = 0.95,
    n_resamples: int = 1000,
    random_state: int | None = 0,
    ci_method: str = "bc",
) -> CalibrationReport:
    """Summarise the calibration of one set of probabilities.

    Gathers the CORP decomposition, three calibration-error estimators that
    disagree in instructive ways, and the resolution the forecasts retain.

    Args:
        y_true: Ground truth values (0 or 1).
        y_pred: Predicted probabilities.
        n_bins: Bin count for the two fixed-bin estimators. The sweep chooses
            its own and smECE needs none.
        ci: Whether to bootstrap confidence intervals for ``brier``, ``smece``
            and ``debiased_ece``. Off by default because it costs
            ``n_resamples`` recomputations of each. ``MCB`` and ``DSC`` are
            excluded on purpose: the naive bootstrap is inconsistent for
            functionals of an isotonic fit, and would report an interval that
            can sit above the estimate. Use
            :func:`~calibre.consistency_bands` or
            :func:`~calibre.confidence_bands` for those.
        level: Nominal coverage for those intervals.
        n_resamples: Bootstrap resamples.
        random_state: Seed.
        ci_method: Interval method, passed to :func:`~calibre.bootstrap_ci`.
            Defaults to ``"bc"``, which is bias-corrected; the plain
            percentile interval under-covers badly here, for reasons that
            function documents.

    Returns:
        CalibrationReport: The summary. Print it, or read fields off it.

    Raises:
        ValueError: If the arrays disagree in length or ``n_bins`` is below 1.

    Warnings:
        Run this on **held-out** predictions. On the data a calibrator was fitted to,
        any isotonic-family method reports ``MCB`` of exactly zero by construction --
        the calibrator and this diagnostic are the same PAV projection, and PAV is
        idempotent -- no matter how badly the model generalises. Use
        :func:`~calibre.cross_val_calibrate` for out-of-fold probabilities.

    Examples:
        >>> import numpy as np
        >>> from calibre import calibration_report
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 2000)
        >>> y = rng.binomial(1, p).astype(float)
        >>> report = calibration_report(y, p)
        >>> report.n
        2000

        These are calibrated by construction, so miscalibration is small next to the
        discrimination the forecasts earn:

        >>> bool(report.mcb < 0.1 * report.dsc)
        True

        And the uncorrected estimator reports more error than the corrected one:

        >>> bool(report.plugin_ece >= report.debiased_ece)
        True
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    y_true, y_pred = check_arrays(y_true, y_pred)

    decomposition = score_decomposition(y_pred, y_true)
    smece, sigma = smooth_calibration_error(y_true, y_pred, return_sigma=True)
    sweep, sweep_bins = sweep_calibration_error(y_true, y_pred, return_n_bins=True)
    n_distinct = int(np.unique(y_pred).size)
    n = int(y_true.size)

    intervals: dict[str, dict[str, float]] = {}
    if ci:
        # MCB and DSC are absent. They are functionals of an isotonic fit, for
        # which the naive n-out-of-n bootstrap is inconsistent: a resample keeps
        # only ~63% of rows distinct and PAV overfits the duplicates, so the
        # inflation tracks effective sample size rather than sampling error. In
        # practice that produced an interval both degenerate and sitting *above*
        # its own estimate, which is worse than no interval. consistency_bands
        # and confidence_bands resample outcomes instead and are correct there.
        targets = {
            "brier": brier_score,
            "smece": lambda t, p: smooth_calibration_error(t, p),
            "debiased_ece": lambda t, p: debiased_calibration_error(t, p, n_bins),
        }
        for key, metric in targets.items():
            intervals[key] = bootstrap_ci(
                metric,
                y_true,
                y_pred,
                level=level,
                n_resamples=n_resamples,
                random_state=random_state,
                method=ci_method,
            )

    return CalibrationReport(
        n=n,
        base_rate=float(np.mean(y_true)),
        mean_prediction=float(np.mean(y_pred)),
        bias=mean_calibration_error(y_true, y_pred),
        brier=brier_score(y_true, y_pred),
        mcb=float(decomposition["MCB"]),
        dsc=float(decomposition["DSC"]),
        unc=float(decomposition["UNC"]),
        smece=float(smece),
        smece_sigma=float(sigma),
        debiased_ece=debiased_calibration_error(y_true, y_pred, n_bins),
        plugin_ece=plugin_calibration_error(y_true, y_pred, n_bins, 2),
        sweep_ece=float(sweep),
        sweep_bins=int(sweep_bins),
        n_bins=int(n_bins),
        n_distinct=n_distinct,
        distinct_ratio=float(n_distinct / n) if n else 0.0,
        intervals=intervals,
    )
