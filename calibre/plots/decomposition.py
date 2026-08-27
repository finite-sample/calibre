"""Plots of the CORP score decomposition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._deps import require_matplotlib
from ._style import SEMANTIC, finalize, get_axes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_mcb_dsc_plane", "plot_score_decomposition"]

_COMPONENTS = (
    "mean_score",
    "miscalibration",
    "discrimination",
    "uncertainty",
)


def _as_mapping_of_decompositions(
    decompositions: Mapping[str, object],
) -> dict[str, Mapping[str, float]]:
    """Accept either one decomposition or a named collection of them.

    Args:
        decompositions: A single :func:`~calibre.score_decomposition` result,
            or a mapping from forecaster name to such a result.

    Returns:
        dict: Always a name-to-decomposition mapping.

    Raises:
        ValueError: If a decomposition is missing a component.
    """
    # Unambiguous: a decomposition always has "miscalibration", and a named
    # mapping never does, because the keys are forecaster names.
    single = "miscalibration" in decompositions
    named: dict[str, Mapping[str, float]] = (
        {"": decompositions} if single else dict(decompositions)  # type: ignore[dict-item, arg-type]
    )

    for name, decomposition in named.items():
        missing = [k for k in _COMPONENTS if k not in decomposition]
        if missing:
            label = name or "the decomposition"
            raise ValueError(
                f"{label} is missing {missing}; expected the mapping returned by "
                "calibre.score_decomposition"
            )
    return named


def plot_score_decomposition(
    decompositions: Mapping[str, object],
    *,
    axes: Sequence[Axes] | None = None,
    score_label: str = "Brier score",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw the ``MCB``/``DSC``/``UNC`` split as three comparable panels.

    ``mean_score = UNC + MCB - DSC``. The three terms answer three different
    questions, and the plot gives each its own panel, sharing the forecaster
    axis:

    - **MCB**, miscalibration: what recalibration would recover. Less is better.
    - **DSC**, discrimination: what the forecasts buy over always predicting the
      base rate. More is better.
    - the achieved score itself.

    ``UNC`` is reported in the title rather than drawn. It depends on the
    outcomes, not on the forecaster, so every panel would show the same number.

    Notes:
        Separate panels rather than one stacked bar, because the terms differ in
        magnitude by one to two orders: for any competent model ``DSC`` is around
        0.10 while ``MCB`` is around 0.001. On a shared linear axis the bar for the
        quantity this decomposition exists to expose is thinner than its own outline,
        and a stacked rendering hides it entirely behind the discrimination bar. Each
        panel therefore carries its own scale, and every panel starts at zero so bar
        lengths remain honest.

    Args:
        decompositions: One :func:`~calibre.score_decomposition` result, or a
            mapping from forecaster name to result to draw one row each.
        axes: Three existing axes to draw into. A new figure is created when
            omitted.
        score_label: Name of the score, used for the third panel's label.
        figsize: Size of the new figure. Scales with the number of forecasters
            by default.

    Returns:
        Figure: The figure holding the three panels.

    Raises:
        ValueError: If a decomposition is missing a component, or ``axes`` is
            not length 3.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> from calibre import score_decomposition
        >>> from calibre.plots import plot_score_decomposition
        >>> rng = np.random.default_rng(0)
        >>> x = rng.uniform(0, 1, 500)
        >>> y = rng.binomial(1, x).astype(float)
        >>> fig = plot_score_decomposition(score_decomposition(y, x))
        >>> len(fig.axes)
        3
    """
    _, plt = require_matplotlib()
    named = _as_mapping_of_decompositions(decompositions)

    names = list(named)
    positions = np.arange(len(names))[::-1]
    uncertainties = {round(float(d["uncertainty"]), 12) for d in named.values()}

    panels: Sequence[Axes]
    if axes is None:
        size = figsize or (9.0, 0.55 * len(names) + 2.0)
        figure, grid = plt.subplots(1, 3, figsize=size, sharey=True)
        panels = list(grid)
    else:
        panels = list(axes)
        if len(panels) != 3:
            raise ValueError(f"axes must hold exactly 3 axes, got {len(panels)}")
        root = panels[0].get_figure(root=True)
        if root is None:
            raise ValueError("the supplied axes are not attached to a figure")
        figure = root

    specs = (
        (
            "miscalibration",
            SEMANTIC["mcb"],
            "MCB -- recalibration recovers this",
            "mcb",
        ),
        (
            "discrimination",
            SEMANTIC["dsc"],
            "DSC -- earned by the forecasts",
            "dsc",
        ),
        ("mean_score", SEMANTIC["score"], score_label, "mean_score"),
    )

    for panel, (key, color, title, artist_name) in zip(panels, specs, strict=True):
        values = [float(named[name][key]) for name in names]
        panel.barh(
            positions,
            values,
            height=0.6,
            color=color,
            zorder=2,
            label=f"_calibre:{artist_name}",
        )
        for position, value in zip(positions, values, strict=True):
            panel.annotate(
                f"{value:.4f}",
                xy=(value, position),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize="x-small",
            )
        # Every panel starts at zero, so a bar twice as long really is twice the
        # quantity. Headroom on the right leaves space for the annotations.
        panel.set_xlim(0.0, max(values) * 1.35 if max(values) > 0 else 1.0)
        panel.set_title(title, fontsize="small")
        panel.grid(True, axis="x", alpha=0.25, linewidth=0.6)
        panel.set_axisbelow(True)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)

    panels[0].set_yticks(positions)
    panels[0].set_yticklabels(names)
    panels[0].set_ylim(-0.7, len(names) - 0.3)

    if len(uncertainties) == 1:
        figure.suptitle(
            f"{score_label} = UNC + MCB - DSC,  "
            f"UNC = {next(iter(uncertainties)):.4f} (irreducible)",
            fontsize="medium",
        )
    if axes is None:
        figure.tight_layout()
    return figure


def plot_mcb_dsc_plane(
    decompositions: Mapping[str, Mapping[str, float]],
    *,
    ax: Axes | None = None,
    contours: bool = True,
    n_contours: int = 6,
    annotate: bool = True,
) -> Axes:
    """Place forecasters on the discrimination-miscalibration plane.

    Each forecaster is a point at ``(DSC, MCB)``. Because ``UNC`` is a property
    of the data rather than of the forecaster, every method on one dataset shares
    it, and lines of constant score are straight lines of slope 1. Down and to
    the right is better: more discrimination, less miscalibration.

    This is the display to reach for when comparing several methods; the panels
    in :func:`plot_score_decomposition` are the ones for reading a single
    forecaster component by component.

    Args:
        decompositions: Mapping from forecaster name to a
            :func:`~calibre.score_decomposition` result.
        ax: Axes to draw on. A new figure is created when omitted.
        contours: Whether to draw lines of equal score.
        n_contours: How many such lines.
        annotate: Whether to label each point with its forecaster name.

    Returns:
        Axes: The axes drawn on.

    Raises:
        ValueError: If ``decompositions`` is empty or a decomposition is
            missing a component.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> from calibre import score_decomposition
        >>> from calibre.plots import plot_mcb_dsc_plane
        >>> rng = np.random.default_rng(0)
        >>> x = rng.uniform(0, 1, 500)
        >>> y = rng.binomial(1, x).astype(float)
        >>> ax = plot_mcb_dsc_plane({
        ...     "honest": score_decomposition(y, x),
        ...     "squashed": score_decomposition(y, 0.25 + 0.5 * x),
        ... })
        >>> ax.get_xlabel()
        'DSC (discrimination) -- more is better'
    """
    require_matplotlib()
    named = _as_mapping_of_decompositions(decompositions)
    if not named:
        raise ValueError("decompositions is empty; nothing to plot")

    axes = get_axes(ax, figsize=(5.5, 5.0))
    dsc = np.array([float(named[n]["discrimination"]) for n in named])
    mcb = np.array([float(named[n]["miscalibration"]) for n in named])
    unc = float(next(iter(named.values()))["uncertainty"])

    pad_x = max(float(np.ptp(dsc)), 1e-3) * 0.25 + 1e-4
    pad_y = max(float(np.ptp(mcb)), 1e-3) * 0.25 + 1e-4
    x_lo, x_hi = dsc.min() - pad_x, dsc.max() + pad_x
    y_lo, y_hi = max(0.0, mcb.min() - pad_y), mcb.max() + pad_y

    if contours:
        # score = UNC + MCB - DSC, so MCB = score - UNC + DSC: slope 1.
        scores = np.linspace(unc + y_lo - x_hi, unc + y_hi - x_lo, n_contours + 2)[1:-1]
        grid = np.array([x_lo, x_hi])
        for level in scores:
            axes.plot(
                grid,
                level - unc + grid,
                color=SEMANTIC["reference"],
                linewidth=0.7,
                linestyle=":",
                zorder=1,
                label="_calibre:isoscore",
            )

    axes.scatter(
        dsc,
        mcb,
        s=55,
        color=SEMANTIC["calibre"],
        zorder=3,
        label="_calibre:methods",
    )
    if annotate:
        for name, x_value, y_value in zip(named, dsc, mcb, strict=True):
            axes.annotate(
                name,
                xy=(float(x_value), float(y_value)),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize="small",
            )

    axes.set_xlim(x_lo, x_hi)
    axes.set_ylim(y_lo, y_hi)
    finalize(
        axes,
        xlabel="DSC (discrimination) -- more is better",
        ylabel="MCB (miscalibration) -- less is better",
        legend=False,
    )
    return axes
