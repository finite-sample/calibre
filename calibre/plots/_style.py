"""Shared styling for calibre's plots.

Colours are fixed here rather than left to matplotlib's cycle so that a given
quantity keeps the same colour across every figure in the documentation: ``MCB``
is the same red whether it appears in a decomposition panel, a benchmark scatter
or a notebook.

Nothing in this module mutates global matplotlib state. The one exception is
:func:`style_context`, which is a context manager and therefore restores whatever
it changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._deps import require_matplotlib

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator
    from contextlib import AbstractContextManager

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.container import Container

# Okabe-Ito, the standard colourblind-safe qualitative palette. matplotlib's
# default `tab10` is not colourblind-safe: its red and green are indistinguishable
# under deuteranopia, and a calibration plot that compares methods by colour has
# to survive that. Yellow is last because it is illegible on a white background.
PALETTE: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
)

# Roles, so that the same quantity is the same colour in every figure.
SEMANTIC: dict[str, str] = {
    # The line of perfect calibration, and other things that are not data.
    "reference": "#666666",
    # Uncertainty bands, drawn behind everything.
    "band": "#9ecae1",
    # Observation density (histogram bars, rug ticks).
    "density": "#BBBBBB",
    # The score decomposition. MCB is what you can fix, so it is the warm one.
    "mcb": "#D55E00",
    "dsc": "#0072B2",
    "unc": "#999999",
    "score": "#000000",
    # Method families in comparisons.
    "isotonic": "#E69F00",
    "calibre": "#0072B2",
    "uncalibrated": "#666666",
    # Whatever the figure is drawing attention to.
    "highlight": "#D55E00",
}


def color_cycle(n: int) -> list[str]:
    """Return ``n`` distinguishable colours, cycling if more are asked for.

    Args:
        n: How many colours are needed.

    Returns:
        list of str: Hex colour strings.

    Raises:
        ValueError: If ``n`` is negative.

    Examples:
        >>> color_cycle(3)
        ['#000000', '#E69F00', '#56B4E9']

        Asking for more than the palette holds wraps around rather than failing:

        >>> len(color_cycle(12))
        12
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def get_axes(
    ax: Axes | None,
    *,
    figsize: tuple[float, float] = (5.0, 5.0),
) -> Axes:
    """Return the axes to draw on, creating a figure only if needed.

    This is the only place in :mod:`calibre.plots` that touches ``pyplot``, and
    it does so only when ``ax`` is None. ``pyplot`` is required rather than a
    bare :class:`~matplotlib.figure.Figure` because a figure created outside
    ``pyplot`` never renders in a Jupyter inline backend.

    Args:
        ax: Existing axes, or None to create a new figure.
        figsize: Size of the new figure, in inches. Ignored when ``ax`` is given.

    Returns:
        Axes: ``ax`` itself when it was supplied, otherwise freshly created axes.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> ax = get_axes(None)
        >>> get_axes(ax) is ax
        True
    """
    if ax is not None:
        return ax
    _, plt = require_matplotlib()
    _, new_ax = plt.subplots(figsize=figsize)
    return new_ax


def probability_axes(ax: Axes, *, square: bool = True) -> Axes:
    """Set both axes to the unit interval, optionally with an equal aspect.

    A reliability diagram is only readable against the diagonal if one unit on
    the x-axis is one unit on the y-axis, so the aspect is locked by default.

    Args:
        ax: Axes to configure.
        square: Whether to force an equal aspect ratio.

    Returns:
        Axes: The same axes, configured.
    """
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if square:
        ax.set_aspect("equal", adjustable="box")
    return ax


def add_diagonal(ax: Axes) -> Axes:
    """Draw the line of perfect calibration.

    Args:
        ax: Axes to draw on.

    Returns:
        Axes: The same axes.
    """
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.0,
        color=SEMANTIC["reference"],
        zorder=1,
        label="_calibre:diagonal",
    )
    return ax


def finalize(
    ax: Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend: bool = True,
    grid: bool = True,
) -> Axes:
    """Apply labels, grid and legend without touching global state.

    The legend is added only when at least one artist carries a public label.
    matplotlib hides labels beginning with an underscore, which is how calibre's
    internal ``_calibre:`` artists stay out of the legend while remaining
    findable by tests.

    Args:
        ax: Axes to finish.
        xlabel: Label for the x-axis, if any.
        ylabel: Label for the y-axis, if any.
        title: Title, if any.
        legend: Whether to add a legend when there is something to put in it.
        grid: Whether to draw a light y-axis grid.

    Returns:
        Axes: The same axes.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if grid:
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    if legend:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize="small")
    return ax


def style_context(**overrides: Any) -> AbstractContextManager[None]:
    """Return a context manager applying publication-friendly rcParams.

    Provided because some users do want global settings; making it a context
    manager means the settings are restored on exit, so calibre never leaves a
    session's ``rcParams`` altered.

    Args:
        **overrides: Additional rcParams, overriding the defaults below.

    Returns:
        contextlib.AbstractContextManager: A context manager, as returned by :func:`matplotlib.rc_context`.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> before = matplotlib.rcParams["savefig.dpi"]
        >>> with style_context():
        ...     pass
        >>> matplotlib.rcParams["savefig.dpi"] == before
        True
    """
    mpl, _ = require_matplotlib()
    params: dict[str, Any] = {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Vector output should stay vector: no rasterized layers in a PDF.
        "path.simplify": False,
    }
    params.update(overrides)
    return mpl.rc_context(params)


def artists(ax: Axes) -> Iterator[Artist | Container]:
    """Iterate over every artist calibre may have labelled on ``ax``.

    Used by the test suite to locate a drawn element by its label rather than by
    its index, since indices shift whenever drawing order changes.

    Args:
        ax: Axes to scan.

    Yields:
        Artist | Container: Lines, collections, patches and containers in
            drawing order. Containers are not Artists, hence the union.
    """
    yield from ax.lines
    yield from ax.collections
    yield from ax.patches
    yield from ax.containers
