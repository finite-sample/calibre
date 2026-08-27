"""Plots of what calibration cost you in resolution.

Isotonic regression is a step function, so it maps many distinct scores onto one
value. The usual reliability diagram cannot show this -- a step function and a
strictly increasing curve can sit on top of each other and score identically --
which is why the loss goes unnoticed. These two plots make it visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._deps import require_matplotlib
from ._style import SEMANTIC, finalize, get_axes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes

__all__ = ["plot_resolution_frontier", "plot_resolution_loss"]


def _distinct_positions(values: np.ndarray, precision: int) -> np.ndarray:
    """Return the index of each position where the output value changes.

    Args:
        values: Calibrated outputs, in ascending order of the input score.
        precision: Decimal places at which two outputs count as equal. Matches
            :func:`~calibre.unique_value_counts`.

    Returns:
        ndarray: Indices at which a new distinct output value begins, including 0.
    """
    rounded = np.round(values, precision)
    if rounded.size == 0:
        return np.empty(0, dtype=int)
    changed = np.flatnonzero(np.diff(rounded) != 0.0) + 1
    return np.concatenate([[0], changed])


def plot_resolution_loss(
    calibrated_predictions: Mapping[str, np.ndarray],
    *,
    input_scores: np.ndarray | None = None,
    ax: Axes | None = None,
    annotate_counts: bool = True,
    precision: int = 6,
    sort: bool = True,
) -> Axes:
    """Draw one "collapse barcode" per method.

    Each method gets a horizontal strip. Inside it, one thin vertical tick marks
    every place along the input range where the calibrated output *changes* --
    so the number of ticks is exactly the number of distinct output values, and
    their spacing is where the resolution went.

    Isotonic regression's strip is sparse enough to count by eye; a
    resolution-preserving calibrator's is solid ink. That contrast is the claim
    this package is built on, drawn rather than asserted, and it needs no legend.

    Args:
        calibrated_predictions: Mapping from method name to calibrated predictions.
            Every array must be the same length, being the same observations
            calibrated different ways.
        input_scores: Scores the outputs came from, used for the horizontal
            axis. When omitted, rank position is used instead.
        ax: Axes to draw on. A new figure is created when omitted.
        annotate_counts: Whether to print the distinct-value count at the
            right of each strip.
        precision: Decimal places at which two outputs count as equal.
        sort: Whether to order strips by distinct count, most granular at the top.

    Returns:
        Axes: The axes drawn on.

    Raises:
        ValueError: If ``calibrated_predictions`` is empty, the arrays disagree in
            length, or ``input_scores`` does not match them.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> from calibre import CenteredIsotonicCalibrator, IsotonicCalibrator
        >>> from calibre.plots import plot_resolution_loss
        >>> rng = np.random.default_rng(0)
        >>> scores = rng.uniform(0, 1, 800)
        >>> labels = rng.binomial(1, scores).astype(float)
        >>> ax = plot_resolution_loss({
        ...     "isotonic": IsotonicCalibrator().fit(scores, labels).transform(scores),
        ...     "centered": (
        ...         CenteredIsotonicCalibrator().fit(scores, labels).transform(scores)
        ...     ),
        ... }, input_scores=scores)
        >>> ax.get_xlabel()
        'input score'
    """
    require_matplotlib()
    if not calibrated_predictions:
        raise ValueError("calibrated_predictions is empty; nothing to plot")

    arrays = {
        name: np.asarray(values, dtype=float).ravel()
        for name, values in calibrated_predictions.items()
    }
    lengths = {v.size for v in arrays.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"every output array must have the same length, got {sorted(lengths)}"
        )
    n = lengths.pop()

    if input_scores is None:
        axis_values = np.arange(n, dtype=float)
        xlabel = "rank position"
    else:
        axis_values = np.asarray(input_scores, dtype=float).ravel()
        if axis_values.size != n:
            raise ValueError(
                f"input_scores has {axis_values.size} entries but the calibrated "
                f"predictions have {n}"
            )
        xlabel = "input score"

    order = np.argsort(axis_values, kind="mergesort")
    axis_sorted = axis_values[order]

    counts = {
        name: int(np.unique(np.round(v, precision)).size) for name, v in arrays.items()
    }
    names = list(arrays)
    if sort:
        names.sort(key=lambda name: counts[name], reverse=True)

    axes = get_axes(ax, figsize=(8.0, 0.75 * len(names) + 1.4))

    # One ink color for every row, deliberately. The message of this plot is
    # carried by ink density, and density is read as lightness: a black strip
    # looks denser than a sky-blue one holding exactly the same number of ticks.
    # Colouring the rows by method would make two calibrators with identical
    # granularity look different. Rows are identified by their axis label.
    ink = SEMANTIC["score"]

    for row, name in enumerate(names):
        position = len(names) - 1 - row
        ticks = _distinct_positions(arrays[name][order], precision)
        axes.vlines(
            axis_sorted[ticks],
            position - 0.35,
            position + 0.35,
            color=ink,
            linewidth=0.4,
            # Semi-transparent so the strip reads as a density ramp rather than
            # clipping to solid black. At full opacity a row holding 800 ticks
            # and one holding 4000 are both saturated black and the comparison
            # the plot exists to make disappears; at this alpha they are clearly
            # different greys.
            alpha=0.5,
            label=f"_calibre:ticks:{name}",
        )
        if annotate_counts:
            axes.annotate(
                f"{counts[name]:,} distinct ({counts[name] / n:.0%})",
                xy=(1.005, position),
                xycoords=("axes fraction", "data"),
                va="center",
                fontsize="small",
            )

    axes.set_yticks(list(range(len(names))))
    axes.set_yticklabels(names[::-1])
    axes.set_ylim(-0.6, len(names) - 0.4)
    # One observation, or a single distinct score, gives a degenerate range;
    # matplotlib warns and expands it. Pad it ourselves so the plot stays quiet
    # on exactly the tied data this package cares most about.
    low, high = float(axis_sorted[0]), float(axis_sorted[-1])
    if low == high:
        low, high = low - 0.5, high + 0.5
    axes.set_xlim(low, high)
    finalize(axes, xlabel=xlabel, legend=False, grid=False)
    return axes


def plot_resolution_frontier(
    results: Mapping[str, tuple[int, float]],
    *,
    ax: Axes | None = None,
    error_bars: Mapping[str, tuple[float, float]] | None = None,
    score_label: str = "held-out Brier score",
    highlight: Sequence[str] = (),
) -> Axes:
    """Plot held-out score against the number of distinct values retained.

    The barcode invites the objection that the extra values might be noise. This
    answers it: methods that keep far more distinct values sit at the *same*
    height, meaning they cost nothing in score. Down is better, right is better,
    and the interesting finding is usually that the frontier is flat.

    Args:
        results: Mapping from method name to ``(n_distinct, score)``.
        ax: Axes to draw on. A new figure is created when omitted.
        error_bars: Optional mapping from method name to ``(low, high)``
            absolute score bounds, for instance a bootstrap interval.
        score_label: Label for the y-axis.
        highlight: Names to draw in the accent color.

    Returns:
        Axes: The axes drawn on.

    Raises:
        ValueError: If ``results`` is empty or a distinct count is not positive.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from calibre.plots import plot_resolution_frontier
        >>> ax = plot_resolution_frontier({
        ...     "isotonic": (56, 0.1515),
        ...     "centered": (1874, 0.1511),
        ... }, highlight=["centered"])
        >>> ax.get_xscale()
        'log'
    """
    require_matplotlib()
    if not results:
        raise ValueError("results is empty; nothing to plot")

    bad = {n: c for n, (c, _) in results.items() if c <= 0}
    if bad:
        raise ValueError(f"distinct counts must be positive, got {bad}")

    axes = get_axes(ax, figsize=(6.5, 4.5))
    highlighted = set(highlight)

    # Methods that keep similar resolution land on top of each other -- which is
    # the point of the plot, and also what makes their labels collide. Two
    # simpler schemes were tried and both failed on real benchmark output:
    # alternating the offset breaks down past two coincident points, and
    # laddering within position-clusters cannot see *other* clusters, so two
    # nearby stacks interleave. What follows instead packs the labels exactly, in
    # data coordinates, which cannot overlap by construction.
    ordered = sorted(results.items(), key=lambda item: (item[1][1], item[1][0]))

    spread_y = max(np.ptp([s for _, s in results.values()]), 1e-12)

    # Labels point at the middle of the plot rather than always trailing right,
    # so the right-hand cluster -- where the resolution-preserving methods sit,
    # and the ones a reader most wants to identify -- does not run off the edge.
    # Each side is packed separately: labels on opposite sides are far enough
    # apart horizontally that they cannot collide.
    midpoint_x = float(np.mean(np.log10([c for c, _ in results.values()])))
    minimum_gap = 0.055 * spread_y

    sides = {
        leftward: [
            (name, count, score)
            for name, (count, score) in ordered
            if bool(np.log10(count) > midpoint_x) is leftward
        ]
        for leftward in (False, True)
    }

    # Each side's labels share one x, placed just inside its innermost marker so
    # the whole column sits in the empty middle of the plot. Nudging each label
    # individually off its own marker is not enough: within a dense cluster a
    # label then runs across its *neighbours'* markers instead. Multiplicative
    # because the axis is logarithmic, so it is a constant pixel gap at any x.
    nudge = 1.11
    columns = {
        leftward: (
            min(c for _, c, _ in side) / nudge
            if leftward
            else max(c for _, c, _ in side) * nudge
        )
        for leftward, side in sides.items()
        if side
    }
    # Two clusters close together in x would put the columns on the wrong sides
    # of each other. Fall back to a per-marker nudge, which is worse but never
    # inverted.
    if len(columns) == 2 and columns[False] >= columns[True]:
        columns = {}

    placements: dict[str, tuple[float, float, str]] = {}
    for leftward, side in sides.items():
        if not side:
            continue
        # Pack upward from the lowest label, then shift the whole stack back down
        # by half of whatever it grew, so a crowded side stays centered on its
        # points instead of drifting off the top of the axes.
        packed: list[float] = []
        for _, _, score in side:
            floor = packed[-1] + minimum_gap if packed else -np.inf
            packed.append(max(score, floor))
        drift = (packed[-1] - side[-1][2]) / 2.0
        for (name, count, _), label_y in zip(side, packed, strict=True):
            fallback = count / nudge if leftward else count * nudge
            placements[name] = (
                columns.get(leftward, fallback),
                label_y - drift,
                "right" if leftward else "left",
            )

    for name, (n_distinct, score) in ordered:
        color = SEMANTIC["highlight"] if name in highlighted else SEMANTIC["calibre"]
        if error_bars is not None and name in error_bars:
            low, high = error_bars[name]
            axes.plot(
                [n_distinct, n_distinct],
                [low, high],
                color=color,
                linewidth=1.0,
                zorder=2,
                label="_calibre:errorbar",
            )
        axes.scatter(
            [n_distinct],
            [score],
            s=60,
            color=color,
            zorder=3,
            label="_calibre:frontier",
        )
        label_x, label_y, alignment = placements[name]
        # A hairline back to the marker, drawn only when the label had to move
        # far enough that which point it belongs to stops being obvious.
        displaced = abs(label_y - score) > 0.4 * minimum_gap
        axes.annotate(
            name,
            xy=(n_distinct, score),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize="small",
            ha=alignment,
            va="center",
            zorder=4,
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.5,
                "color": "0.6",
                "shrinkB": 6.0,
            }
            if displaced
            else None,
        )

    # Room on both sides now that labels point inward from either edge.
    axes.margins(x=0.14, y=0.10)

    axes.set_xscale("log")
    finalize(
        axes,
        xlabel="distinct calibrated values retained",
        ylabel=score_label,
        legend=False,
    )
    return axes
