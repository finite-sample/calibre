"""Plots for class-wise multiclass calibration."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ._deps import require_matplotlib
from ._style import SEMANTIC, finalize, get_axes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from calibre.evaluation import ReliabilityDiagram

__all__ = ["plot_classwise_reliability", "plot_miscalibration_profile"]


def plot_miscalibration_profile(
    profile: Mapping[str, Any],
    *,
    ax: Axes | None = None,
    class_names: Sequence[str] | None = None,
    highlight_worst: int = 3,
    show_reading: bool = True,
    reading_width: int = 72,
) -> Axes:
    """Show where multiclass miscalibration lives, and what to do about it.

    Per-class ``MCB`` as bars, with the worst classes picked out and the spread
    in the title. When ``show_reading`` is on, the profile's plain-language
    recommendation is printed beneath the axes.

    That caption is the point. calibre is the only Python calibration package
    that tells you *which* multiclass method your data needs, and picking wrong
    costs about a factor of six; making the reader fetch the string separately
    would waste the diagnostic.

    Args:
        profile: A :func:`~calibre.miscalibration_profile` result, with ``mcb``, ``spread``, ``worst_classes`` and ``reading``.
        ax: Axes to draw on. A new figure is created when omitted.
        class_names: Names for the classes. Defaults to their indices.
        highlight_worst: How many of the worst classes to draw in the accent colour.
        show_reading: Whether to print ``profile["reading"]`` below the axes.
        reading_width: Column width to wrap the reading at.

    Returns:
        Axes: The axes drawn on.

    Raises:
        ValueError: If ``profile`` is missing a key, or ``class_names`` has the wrong length.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> from calibre import miscalibration_profile
        >>> from calibre.plots import plot_miscalibration_profile
        >>> rng = np.random.default_rng(0)
        >>> truth = rng.dirichlet(np.ones(4), size=1500)
        >>> y = np.array([rng.choice(4, p=t) for t in truth])
        >>> ax = plot_miscalibration_profile(miscalibration_profile(truth, y))
        >>> ax.get_ylabel()
        'MCB (miscalibration)'
    """
    missing = [k for k in ("mcb", "spread", "worst_classes") if k not in profile]
    if missing:
        raise ValueError(
            f"profile is missing {missing}; expected the mapping returned by "
            "calibre.miscalibration_profile"
        )

    require_matplotlib()
    mcb = np.asarray(profile["mcb"], dtype=float).ravel()
    n_classes = mcb.size

    if class_names is None:
        labels = [str(i) for i in range(n_classes)]
    else:
        labels = list(class_names)
        if len(labels) != n_classes:
            raise ValueError(
                f"class_names has {len(labels)} entries but the profile covers "
                f"{n_classes} classes"
            )

    worst = set(np.asarray(profile["worst_classes"]).ravel()[:highlight_worst].tolist())
    colors = [
        SEMANTIC["highlight"] if i in worst else SEMANTIC["calibre"]
        for i in range(n_classes)
    ]

    axes = get_axes(ax, figsize=(max(5.0, 0.7 * n_classes + 2.0), 4.2))
    axes.bar(
        np.arange(n_classes),
        mcb,
        color=colors,
        width=0.7,
        zorder=2,
        label="_calibre:mcb",
    )
    axes.set_xticks(np.arange(n_classes))
    axes.set_xticklabels(labels)

    spread = float(profile["spread"])
    finalize(
        axes,
        xlabel="class",
        ylabel="MCB (miscalibration)",
        title=f"per-class miscalibration (spread {spread:.2f})",
        legend=False,
    )

    if show_reading and profile.get("reading"):
        axes.figure.text(
            0.02,
            -0.02,
            textwrap.fill(str(profile["reading"]), reading_width),
            ha="left",
            va="top",
            fontsize="small",
            wrap=True,
        )
    return axes


def plot_classwise_reliability(
    diagrams: Sequence[ReliabilityDiagram],
    *,
    axes: Sequence[Axes] | None = None,
    class_names: Sequence[str] | None = None,
    n_cols: int = 3,
    density: str = "none",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw one CORP reliability diagram per class, as small multiples.

    Args:
        diagrams: The list returned by :func:`~calibre.classwise_reliability`.
        axes: Existing axes to draw into, one per diagram. A new figure is created when omitted.
        class_names: Titles for the panels. Defaults to ``class 0``, ``class 1``, ...
        n_cols: Panels per row when creating a new figure.
        density: Passed to :func:`~calibre.plots.plot_reliability_diagram`. Defaults to ``"none"`` because appending a histogram panel to each cell of a grid distorts the layout.
        figsize: Size of the new figure. Defaults to 3 inches per panel.

    Returns:
        Figure: The figure holding the panels.

    Raises:
        ValueError: If ``diagrams`` is empty, or ``axes`` or ``class_names`` has the wrong length.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> from calibre import classwise_reliability
        >>> from calibre.plots import plot_classwise_reliability
        >>> rng = np.random.default_rng(0)
        >>> truth = rng.dirichlet(np.ones(3), size=900)
        >>> y = np.array([rng.choice(3, p=t) for t in truth])
        >>> fig = plot_classwise_reliability(classwise_reliability(truth, y))
        >>> len(fig.axes)
        3
    """
    from .reliability import plot_reliability_diagram

    if not diagrams:
        raise ValueError("diagrams is empty; nothing to plot")

    n = len(diagrams)
    if class_names is None:
        titles = [f"class {i}" for i in range(n)]
    else:
        titles = list(class_names)
        if len(titles) != n:
            raise ValueError(
                f"class_names has {len(titles)} entries but there are {n} diagrams"
            )

    owned = axes is None
    if axes is None:
        _, plt = require_matplotlib()
        n_cols = max(1, min(n_cols, n))
        n_rows = int(np.ceil(n / n_cols))
        size = figsize or (3.0 * n_cols, 3.0 * n_rows)
        figure, grid = plt.subplots(n_rows, n_cols, figsize=size, squeeze=False)
        panels = list(np.asarray(grid).ravel())
        for spare in panels[n:]:
            spare.set_visible(False)
        panels = panels[:n]
    else:
        panels = list(axes)
        if len(panels) != n:
            raise ValueError(
                f"axes has {len(panels)} entries but there are {n} diagrams"
            )
        # Axes.figure is a SubFigure when the caller nested one; the root figure
        # is what a caller means by "the figure this went on". Detached axes have
        # no figure at all, which is a caller error worth naming.
        root = panels[0].get_figure(root=True)
        if root is None:
            raise ValueError("the supplied axes are not attached to a figure")
        figure = root

    for panel, diagram, title in zip(panels, diagrams, titles, strict=True):
        plot_reliability_diagram(diagram, ax=panel, density=density)
        panel.set_title(title, fontsize="medium")

    # Only when we made the figure: a caller who supplied axes owns the layout.
    if owned:
        figure.tight_layout()
    return figure
