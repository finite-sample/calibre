"""Plotting for calibration diagnostics.

matplotlib is an optional dependency::

    pip install 'calibre[plots]'

Importing this module does not import matplotlib; each function imports it when
first called, and raises an :class:`ImportError` naming the install command if it
is missing.

Conventions
-----------
**Plots draw; they do not compute.** Every function takes an already-computed
object -- a :class:`~calibre.evaluation.ReliabilityDiagram`, a
:func:`~calibre.score_decomposition` result, a bands mapping. Uncertainty bands
are a parameter and never an implicit flag, because
:func:`~calibre.consistency_bands` is a thousand PAV refits and must not fire
inside an innocuous-looking ``.plot()`` call. Nothing here ever calls ``.fit()``:
fitting a calibrator on the data you are about to display is the mistake that
quietly ruins calibration.

**Axes in, axes out.** Single-panel functions take ``ax=None`` and return the
:class:`~matplotlib.axes.Axes` they drew on, returning the very object you passed
when you passed one. Multi-panel functions take ``axes=None`` and return a
:class:`~matplotlib.figure.Figure`.

**No global state.** These functions never call ``plt.show()``, never touch
``rcParams``, and never reach for the current figure. Use
:func:`calibre.plots.style_context` if you want publication settings applied
temporarily.
"""

from __future__ import annotations

from ._style import PALETTE, SEMANTIC, color_cycle, style_context
from .comparison import plot_calibrator_comparison
from .decomposition import plot_mcb_dsc_plane, plot_score_decomposition
from .ece import plot_ece_bin_sensitivity
from .multiclass import plot_classwise_reliability, plot_miscalibration_profile
from .reliability import plot_reliability_diagram
from .resolution import plot_resolution_frontier, plot_resolution_loss

__all__ = [
    "PALETTE",
    "SEMANTIC",
    "color_cycle",
    "plot_calibrator_comparison",
    "plot_classwise_reliability",
    "plot_ece_bin_sensitivity",
    "plot_mcb_dsc_plane",
    "plot_miscalibration_profile",
    "plot_reliability_diagram",
    "plot_resolution_frontier",
    "plot_resolution_loss",
    "plot_score_decomposition",
    "style_context",
]
