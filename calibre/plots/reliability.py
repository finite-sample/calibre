"""CORP reliability diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ._deps import require_matplotlib
from ._style import SEMANTIC, add_diagonal, finalize, get_axes, probability_axes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

    from calibre.evaluation import ReliabilityDiagram

__all__ = ["plot_reliability_diagram"]

_DENSITY_CHOICES = ("hist", "rug", "none")
_STYLE_CHOICES = ("line", "step")


def _draw_bands(
    ax: Axes,
    bands: Mapping[str, np.ndarray] | Sequence[Mapping[str, np.ndarray]],
) -> None:
    """Fill one or more uncertainty bands behind the estimate.

    Args:
        ax: Axes to draw on.
        bands: A mapping with ``x``, ``lower`` and ``upper``, or a sequence of
            them. Later bands are drawn with the same colour at lower opacity,
            so nested levels read as nested.

    Raises:
        ValueError: If a band is missing a key or its arrays disagree in length.
    """
    band_list: list[Mapping[str, np.ndarray]]
    band_list = [bands] if isinstance(bands, Mapping) else list(bands)

    for depth, band in enumerate(band_list):
        missing = {"x", "lower", "upper"} - set(band)
        if missing:
            raise ValueError(
                f"band is missing {sorted(missing)}; expected the mapping returned "
                "by calibre.consistency_bands or calibre.confidence_bands"
            )
        grid = np.asarray(band["x"], dtype=float)
        low = np.asarray(band["lower"], dtype=float)
        high = np.asarray(band["upper"], dtype=float)
        if not (grid.shape == low.shape == high.shape):
            raise ValueError(
                f"band arrays must have equal length, got x={grid.shape}, "
                f"lower={low.shape}, upper={high.shape}"
            )
        ax.fill_between(
            grid,
            low,
            high,
            color=SEMANTIC["band"],
            alpha=0.55 / (depth + 1),
            linewidth=0.0,
            zorder=0,
            label="_calibre:band",
        )


def _draw_density(
    ax: Axes,
    diagram: ReliabilityDiagram,
    density: str,
    density_bins: int,
) -> None:
    """Show where the observations actually are.

    A reliability diagram invites the question "is that excursion near 0.9 built
    on twelve points or twelve hundred?", and the curve alone cannot answer it.
    ``diagram.weight`` is the exact mass carried by each distinct forecast value,
    so this is the density of the estimator's own support rather than of an
    arbitrary re-binning.

    Args:
        ax: Axes to draw on.
        diagram: Fitted diagram, supplying ``x`` and ``weight``.
        density: ``"hist"`` for a marginal histogram in a panel below the
            axes, ``"rug"`` for ticks inside the axes, ``"none"`` to draw
            nothing.
        density_bins: Number of bins for ``"hist"``.
    """
    if density == "none":
        return

    weight = np.asarray(diagram.weight, dtype=float)
    values = np.asarray(diagram.x, dtype=float)

    if density == "rug":
        # Inside the main axes, so the caller's grid geometry is untouched.
        ax.vlines(
            values,
            0.0,
            0.03,
            color=SEMANTIC["density"],
            linewidth=0.6,
            zorder=2,
            label="_calibre:density",
        )
        return

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    panel = make_axes_locatable(ax).append_axes(
        "bottom", size="18%", pad=0.08, sharex=ax
    )
    panel.hist(
        values,
        bins=density_bins,
        range=(0.0, 1.0),
        weights=weight,
        color=SEMANTIC["density"],
        label="_calibre:density",
    )
    panel.set_yticks([])
    panel.set_xlim(0.0, 1.0)
    for side in ("top", "right", "left"):
        panel.spines[side].set_visible(False)
    panel.set_xlabel(ax.get_xlabel())
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)


def plot_reliability_diagram(
    diagram: ReliabilityDiagram,
    *,
    ax: Axes | None = None,
    bands: Mapping[str, np.ndarray] | Sequence[Mapping[str, np.ndarray]] | None = None,
    density: str = "hist",
    density_bins: int = 30,
    style: str = "line",
    diagonal: bool = True,
    color: str | None = None,
    label: str | None = None,
) -> Axes:
    """Draw a CORP reliability diagram.

    The curve is the PAV-recalibrated conditional event probability at each
    distinct forecast value. Where it sits above the diagonal the forecasts were
    too low; below, too high. Unlike a binned reliability diagram there is no bin
    count to choose, so the picture cannot be changed by choosing a different one.

    Args:
        diagram: A fitted diagram from :func:`~calibre.corp_reliability`.
        ax: Axes to draw on. A new figure is created when omitted.
        bands: Uncertainty bands from :func:`~calibre.consistency_bands` or
            :func:`~calibre.confidence_bands`, or a sequence of them to nest
            several levels. Bands are never computed here: they cost a
            thousand PAV refits, which must not happen as a side effect of
            drawing.
        density: How to show where the observations are: ``"hist"`` (a
            marginal histogram below the axes), ``"rug"`` (ticks inside the
            axes) or ``"none"``.
        density_bins: Number of bins when ``density="hist"``.
        style: ``"line"`` to interpolate between the estimated points,
            matching :meth:`~calibre.evaluation.ReliabilityDiagram.as_function`,
            or ``"step"`` to show the PAV blocks as the step function they
            are.
        diagonal: Whether to draw the line of perfect calibration.
        color: Colour of the curve. Defaults to calibre's semantic blue.
        label: Legend label. Omit for no legend entry.

    Returns:
        Axes: The axes drawn on -- the very object passed as ``ax``, when one was.

    Raises:
        ValueError: If ``density`` or ``style`` is not one of the documented
            choices, or a band mapping is malformed.

    Notes:
        ``density="hist"`` appends a panel below ``ax``, which shrinks ``ax`` itself.
        Inside a ``subplot_mosaic`` or similar fixed grid, prefer ``"rug"`` or
        ``"none"`` so the layout is left alone.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> from calibre import corp_reliability
        >>> from calibre.plots.reliability import plot_reliability_diagram
        >>> rng = np.random.default_rng(0)
        >>> x = rng.uniform(0, 1, 500)
        >>> y = rng.binomial(1, x).astype(float)
        >>> ax = plot_reliability_diagram(corp_reliability(x, y), density="none")
        >>> ax.get_ylabel()
        'observed event frequency'
    """
    if density not in _DENSITY_CHOICES:
        raise ValueError(f"density must be one of {_DENSITY_CHOICES}, got {density!r}")
    if style not in _STYLE_CHOICES:
        raise ValueError(f"style must be one of {_STYLE_CHOICES}, got {style!r}")

    require_matplotlib()
    # Bound to a separate name so the type narrows: reassigning `ax` would leave
    # `None` in its declared union for every use below.
    axes = get_axes(ax)
    probability_axes(axes)

    if diagonal:
        add_diagonal(axes)
    if bands is not None:
        _draw_bands(axes, bands)

    x = np.asarray(diagram.x, dtype=float)
    cep = np.asarray(diagram.cep, dtype=float)
    curve_color = SEMANTIC["calibre"] if color is None else color

    draw: Any = axes.step if style == "step" else axes.plot
    kwargs: dict[str, Any] = {
        "color": curve_color,
        "linewidth": 1.8,
        "zorder": 3,
        "label": label if label is not None else "_calibre:cep",
    }
    if style == "step":
        kwargs["where"] = "post"
    draw(x, cep, **kwargs)

    finalize(
        axes,
        xlabel="forecast probability",
        ylabel="observed event frequency",
        grid=False,
    )
    _draw_density(axes, diagram, density, density_bins)
    return axes
