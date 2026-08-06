"""Plots of what calibration cost you in resolution.

Isotonic regression is a step function, so it maps many distinct scores onto one
value. The usual reliability diagram cannot show this -- a step function and a
strictly increasing curve can sit on top of each other and score identically --
which is why the loss goes unnoticed. These two plots make it visible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ._deps import require_matplotlib
from ._style import SEMANTIC, finalize, get_axes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = ["plot_resolution_frontier", "plot_resolution_loss"]


def _distinct_positions(values: np.ndarray, precision: int) -> np.ndarray:
    """Return the index of each position where the output value changes.

    Parameters
    ----------
    values
        Calibrated outputs, in ascending order of the input score.
    precision
        Decimal places at which two outputs count as equal. Matches
        :func:`~calibre.unique_value_counts`.

    Returns
    -------
    ndarray
        Indices at which a new distinct output value begins, including 0.
    """
    rounded = np.round(values, precision)
    if rounded.size == 0:
        return np.empty(0, dtype=int)
    changed = np.flatnonzero(np.diff(rounded) != 0.0) + 1
    return np.concatenate([[0], changed])


def plot_resolution_loss(
    outputs: Mapping[str, np.ndarray],
    x: np.ndarray | None = None,
    *,
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

    Parameters
    ----------
    outputs
        Mapping from method name to that method's calibrated outputs. Every
        array must be the same length, being the same observations calibrated
        different ways.
    x
        The input scores the outputs came from, used for the horizontal axis.
        When omitted, rank position is used instead.
    ax
        Axes to draw on. A new figure is created when omitted.
    annotate_counts
        Whether to print the distinct-value count at the right of each strip.
    precision
        Decimal places at which two outputs count as equal.
    sort
        Whether to order strips by distinct count, most granular at the top.

    Returns
    -------
    Axes
        The axes drawn on.

    Raises
    ------
    ValueError
        If ``outputs`` is empty, the arrays disagree in length, or ``x`` does not
        match them.

    Examples
    --------
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
    ... }, scores)
    >>> ax.get_xlabel()
    'input score'
    """
    require_matplotlib()
    if not outputs:
        raise ValueError("outputs is empty; nothing to plot")

    arrays = {name: np.asarray(v, dtype=float).ravel() for name, v in outputs.items()}
    lengths = {v.size for v in arrays.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"every output array must have the same length, got {sorted(lengths)}"
        )
    n = lengths.pop()

    if x is None:
        axis_values = np.arange(n, dtype=float)
        xlabel = "rank position"
    else:
        axis_values = np.asarray(x, dtype=float).ravel()
        if axis_values.size != n:
            raise ValueError(
                f"x has {axis_values.size} entries but the outputs have {n}"
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

    # One ink colour for every row, deliberately. The message of this plot is
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
    errorbars: Mapping[str, tuple[float, float]] | None = None,
    score_label: str = "held-out Brier score",
    highlight: Sequence[str] = (),
) -> Axes:
    """Plot held-out score against the number of distinct values retained.

    The barcode invites the objection that the extra values might be noise. This
    answers it: methods that keep far more distinct values sit at the *same*
    height, meaning they cost nothing in score. Down is better, right is better,
    and the interesting finding is usually that the frontier is flat.

    Parameters
    ----------
    results
        Mapping from method name to ``(n_distinct, score)``.
    ax
        Axes to draw on. A new figure is created when omitted.
    errorbars
        Optional mapping from method name to ``(low, high)`` absolute score
        bounds, for instance a bootstrap interval.
    score_label
        Label for the y-axis.
    highlight
        Names to draw in the accent colour.

    Returns
    -------
    Axes
        The axes drawn on.

    Raises
    ------
    ValueError
        If ``results`` is empty or a distinct count is not positive.

    Examples
    --------
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
    # the point of the plot, and also what makes their labels collide. Alternating
    # the offset is not enough once several methods share a spot, which is the
    # normal case: a benchmark comparing ten calibrators typically has two tight
    # clusters. So points are grouped by position and each cluster's labels are
    # stacked in a ladder, which needs no layout solver and cannot overlap.
    ordered = sorted(results.items(), key=lambda item: (item[1][1], item[1][0]))

    spread_x = max(np.ptp(np.log10([c for c, _ in results.values()])), 1e-9)
    spread_y = max(np.ptp([s for _, s in results.values()]), 1e-12)

    offsets: list[tuple[float, float]] = []
    clusters: list[tuple[float, float, int]] = []
    for n_distinct, score in (value for _, value in ordered):
        position = (np.log10(n_distinct), score)
        for index, (cx, cy, count) in enumerate(clusters):
            near = (
                abs(position[0] - cx) < 0.06 * spread_x
                and abs(position[1] - cy) < 0.06 * spread_y
            )
            if near:
                offsets.append((8.0, 4.0 - 11.0 * count))
                clusters[index] = (cx, cy, count + 1)
                break
        else:
            offsets.append((8.0, 4.0))
            clusters.append((position[0], position[1], 1))

    for (name, (n_distinct, score)), offset in zip(ordered, offsets, strict=True):
        color = SEMANTIC["highlight"] if name in highlighted else SEMANTIC["calibre"]
        if errorbars is not None and name in errorbars:
            low, high = errorbars[name]
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
        axes.annotate(
            name,
            xy=(n_distinct, score),
            xytext=offset,
            textcoords="offset points",
            fontsize="small",
        )

    # Room on the right for labels that trail off the last point.
    axes.margins(x=0.12)

    axes.set_xscale("log")
    finalize(
        axes,
        xlabel="distinct calibrated values retained",
        ylabel=score_label,
        legend=False,
    )
    return axes
