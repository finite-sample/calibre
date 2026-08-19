"""Plots that make the binned-ECE bias visible."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..metrics import (
    debiased_calibration_error,
    plugin_calibration_error,
    sweep_calibration_error,
)
from ._deps import require_matplotlib
from ._style import SEMANTIC, finalize, get_axes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from matplotlib.axes import Axes

__all__ = ["plot_ece_bin_sensitivity"]

_ESTIMATORS = ("plugin", "debiased", "sweep")


def plot_ece_bin_sensitivity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ax: Axes | None = None,
    n_bins: Sequence[int] | None = None,
    norm: int = 2,
    estimators: Sequence[str] = _ESTIMATORS,
    reference: float | None = None,
    log_x: bool = False,
) -> Axes:
    r"""Plot calibration error against the number of bins.

    Binned calibration error is biased upward, because part of every bin's gap is
    sampling noise in the label mean rather than miscalibration. The bias grows
    with the bin count -- precisely when a finer picture of the curve is wanted.
    Plotting the estimators against the bin count shows this directly: the plugin
    curve climbs, the debiased curve does not, and any single ECE number quoted
    without its bin count is a point on a rising line.

    All three series are computed at the same norm and on the same equal-mass,
    tie-safe bins, so the only thing separating them is the bias correction.
    Reaching for :func:`~calibre.expected_calibration_error` instead would mix
    :math:`\ell_1` with :math:`\ell_2` and uniform-width bins with equal-mass
    ones, and the resulting picture would show three different quantities
    disagreeing rather than one estimator being biased.

    This is one of only two plots in :mod:`calibre.plots` that computes anything,
    because sweeping the computation *is* the plot.

    Args:
        y_true: Ground truth values (0 or 1).
        y_pred: Predicted probabilities.
        ax: Axes to draw on. A new figure is created when omitted.
        n_bins: Bin counts to evaluate. Defaults to ``range(2, 51)``.
        norm: The :math:`\ell_p` norm shared by every series.
        estimators: Which of ``"plugin"``, ``"debiased"`` and ``"sweep"`` to
            draw. The sweep chooses its own bin count, so it appears as a
            horizontal line annotated with the count it settled on.
        reference: A known true calibration error, drawn as a horizontal rule.
            On data that is calibrated by construction this is 0, and
            everything above it is bias.
        log_x: Whether to put the bin count on a log scale.

    Returns:
        Axes: The axes drawn on.

    Raises:
        ValueError: If ``estimators`` names something unknown, ``n_bins`` is
            empty or holds a value below 1, or the arrays disagree in length.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> from calibre.plots import plot_ece_bin_sensitivity
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 2000)
        >>> y = rng.binomial(1, p).astype(float)
        >>> ax = plot_ece_bin_sensitivity(y, p, n_bins=range(2, 21), reference=0.0)
        >>> ax.get_xlabel()
        'number of bins'
    """
    unknown = [e for e in estimators if e not in _ESTIMATORS]
    if unknown:
        raise ValueError(f"unknown estimators {unknown}; expected {list(_ESTIMATORS)}")

    counts = list(range(2, 51) if n_bins is None else n_bins)
    if not counts:
        raise ValueError("n_bins is empty; nothing to evaluate")
    if min(counts) < 1:
        raise ValueError(f"n_bins must all be at least 1, got {min(counts)}")

    require_matplotlib()
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    axes = get_axes(ax, figsize=(6.5, 4.5))

    if reference is not None:
        axes.axhline(
            reference,
            color=SEMANTIC["reference"],
            linestyle="--",
            linewidth=1.0,
            zorder=1,
            label=f"true error ({reference:g})",
        )

    if "plugin" in estimators:
        values = [plugin_calibration_error(y_true, y_pred, b, norm) for b in counts]
        axes.plot(
            counts,
            values,
            color=SEMANTIC["mcb"],
            linewidth=1.8,
            zorder=3,
            label="plugin (uncorrected)",
        )

    if "debiased" in estimators:
        values = [debiased_calibration_error(y_true, y_pred, b) for b in counts]
        axes.plot(
            counts,
            values,
            color=SEMANTIC["dsc"],
            linewidth=1.8,
            zorder=3,
            label="debiased",
        )

    if "sweep" in estimators:
        error, chosen = sweep_calibration_error(
            y_true, y_pred, norm, return_n_bins=True
        )
        axes.axhline(
            error,
            color=SEMANTIC["calibre"],
            linestyle=":",
            linewidth=1.4,
            zorder=2,
            label=f"sweep (chose {chosen} bins)",
        )
        if min(counts) <= chosen <= max(counts):
            axes.plot(
                [chosen],
                [error],
                marker="o",
                markersize=5,
                color=SEMANTIC["calibre"],
                zorder=4,
                label="_calibre:sweep_point",
            )

    if log_x:
        axes.set_xscale("log")
    axes.set_ylim(bottom=0.0)
    finalize(
        axes,
        xlabel="number of bins",
        ylabel=rf"$\ell_{norm}$ calibration error",
    )
    return axes
