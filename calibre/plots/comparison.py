"""Side-by-side comparison of fitted calibrators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils import check_fitted
from ._deps import require_matplotlib
from ._style import SEMANTIC, add_diagonal, color_cycle, finalize, get_axes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from matplotlib.axes import Axes

    from calibre.evaluation import ReliabilityDiagram

__all__ = ["plot_calibrator_comparison"]


def _check_fitted(name: str, calibrator: Any) -> None:
    """Confirm a calibrator has been fitted.

    Args:
        name: Name to use in the error message.
        calibrator: The object to check.

    Raises:
        ValueError: If the calibrator has no ``transform``, or has one that
            reports it is not fitted.
    """
    if not hasattr(calibrator, "transform"):
        raise ValueError(f"{name!r} has no .transform(); expected a fitted calibrator")
    try:
        check_fitted(calibrator)
    except ValueError as exc:
        raise ValueError(
            f"{name!r} is not fitted: {exc}. Fit every calibrator before plotting "
            "it. This function deliberately never calls .fit() -- fitting a "
            "calibrator on the data you are about to display is the mistake that "
            "quietly ruins calibration."
        ) from exc


def plot_calibrator_comparison(
    calibrators: Mapping[str, Any],
    input_scores: np.ndarray,
    *,
    ax: Axes | None = None,
    reference: ReliabilityDiagram | None = None,
    n_grid: int = 500,
    diagonal: bool = True,
    annotate_distinct: bool = True,
) -> Axes:
    """Overlay the calibration maps that several fitted calibrators learned.

    Each calibrator is evaluated on a fine grid spanning the range of
    ``input_scores``, so
    what is drawn is the function itself rather than a scatter of its outputs.
    With ``annotate_distinct`` on, each legend entry also carries how many
    distinct values that calibrator produced on ``x`` -- so the comparison and
    the resolution cost land in one figure.

    Args:
        calibrators: Mapping from name to an **already fitted** calibrator
            exposing ``.transform``.
        input_scores: Scores used both for the grid's range and for the
            distinct-value counts.
        ax: Axes to draw on. A new figure is created when omitted.
        reference: Optional CORP diagram of the raw scores, drawn behind in
            grey as the empirical target the calibrators are trying to match.
        n_grid: Number of grid points.
        diagonal: Whether to draw the identity line, which is what "no
            recalibration" would look like.
        annotate_distinct: Whether to add distinct-value counts to the legend labels.

    Returns:
        Axes: The axes drawn on.

    Raises:
        ValueError: If ``calibrators`` is empty, ``input_scores`` is empty, or any
            calibrator is unfitted.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> from calibre import CenteredIsotonicCalibrator, IsotonicCalibrator
        >>> from calibre.plots import plot_calibrator_comparison
        >>> rng = np.random.default_rng(0)
        >>> scores = rng.uniform(0, 1, 600)
        >>> labels = rng.binomial(1, scores).astype(float)
        >>> fitted = {
        ...     "isotonic": IsotonicCalibrator().fit(scores, labels),
        ...     "centered": CenteredIsotonicCalibrator().fit(scores, labels),
        ... }
        >>> ax = plot_calibrator_comparison(fitted, scores)
        >>> ax.get_ylabel()
        'calibrated probability'
    """
    require_matplotlib()
    if not calibrators:
        raise ValueError("calibrators is empty; nothing to plot")

    scores = np.asarray(input_scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("input_scores is empty; nothing to plot")

    for name, calibrator in calibrators.items():
        _check_fitted(name, calibrator)

    axes = get_axes(ax, figsize=(6.0, 5.5))
    grid = np.linspace(float(scores.min()), float(scores.max()), n_grid)

    if diagonal:
        add_diagonal(axes)

    if reference is not None:
        axes.plot(
            np.asarray(reference.prediction_values, dtype=float),
            np.asarray(reference.event_probabilities, dtype=float),
            color=SEMANTIC["reference"],
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
            label="empirical (CORP)",
        )

    for (name, calibrator), color in zip(
        calibrators.items(), color_cycle(len(calibrators)), strict=True
    ):
        curve = np.asarray(calibrator.transform(grid), dtype=float)
        label = name
        if annotate_distinct:
            n_distinct = int(np.unique(np.round(calibrator.transform(scores), 6)).size)
            label = f"{name} ({n_distinct:,} distinct)"
        axes.plot(grid, curve, color=color, linewidth=1.7, zorder=3, label=label)

    finalize(
        axes,
        xlabel="input score",
        ylabel="calibrated probability",
        grid=False,
    )
    return axes
