"""Tests for the drawing contracts every calibre plot must honour.

Artists are located by label rather than by index. Indices shift whenever
drawing order changes, which makes index-based assertions fail for reasons that
have nothing to do with correctness; labels are a contract the plots promise to
keep.

There are no baseline-image comparisons here on purpose. The CI matrix spans
three operating systems and three Python versions, freetype and matplotlib minor
versions shift antialiasing, and a pixel diff reports that a pixel moved rather
than whether the picture still says what it claims. What the drawn artists
*mean* is asserted in ``test_plots_claims.py`` instead.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from calibre import (
    confidence_bands,
    consistency_bands,
    corp_reliability,
)
from calibre.plots import plot_reliability_diagram
from calibre.plots._style import artists


def find_artist(ax, label):
    """Return the single artist carrying ``label``.

    Parameters
    ----------
    ax
        Axes to scan.
    label
        Exact label, including the ``_calibre:`` prefix for internal artists.

    Returns
    -------
    Artist
        The matching artist.
    """
    matches = [a for a in artists(ax) if a.get_label() == label]
    assert len(matches) == 1, f"expected exactly one {label!r}, got {len(matches)}"
    return matches[0]


@pytest.fixture
def sample():
    """A moderately miscalibrated sample.

    Returns
    -------
    tuple of ndarray
        Forecasts and binary outcomes.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 400)
    y = rng.binomial(1, np.clip(x**1.4, 0, 1)).astype(float)
    return x, y


@pytest.fixture
def diagram(sample):
    """A fitted CORP diagram.

    Parameters
    ----------
    sample
        Forecasts and outcomes.

    Returns
    -------
    ReliabilityDiagram
        The fitted diagram.
    """
    x, y = sample
    return corp_reliability(x, y)


@pytest.fixture(autouse=True)
def _forbid_show(monkeypatch):
    """Fail loudly if any plot calls ``plt.show()``."""

    def _fail(*args, **kwargs):
        raise AssertionError("a calibre plot called plt.show()")

    monkeypatch.setattr(plt, "show", _fail)


def test_returns_axes(diagram):
    """A single-panel plot returns an Axes."""
    assert isinstance(plot_reliability_diagram(diagram, density="none"), Axes)


def test_returns_the_very_axes_passed_in(diagram):
    """When given an ax, the same object comes back."""
    _, ax = plt.subplots()
    assert plot_reliability_diagram(diagram, ax=ax, density="none") is ax


def test_supplying_an_ax_creates_no_new_figure(diagram):
    """Drawing onto caller-supplied axes must not open a figure."""
    _, ax = plt.subplots()
    before = set(plt.get_fignums())
    plot_reliability_diagram(diagram, ax=ax, density="none")
    assert set(plt.get_fignums()) == before


def test_rcparams_are_untouched(diagram):
    """No calibre plot may mutate global matplotlib state."""
    before = dict(mpl.rcParams)
    plot_reliability_diagram(diagram, density="hist")
    after = dict(mpl.rcParams)
    assert before == after


def test_the_drawn_curve_is_the_estimate(diagram):
    """The plotted line must be the CEP itself, not a smoothed cousin."""
    ax = plot_reliability_diagram(diagram, density="none")
    drawn = find_artist(ax, "_calibre:cep").get_xydata()
    np.testing.assert_allclose(drawn[:, 0], diagram.x)
    np.testing.assert_allclose(drawn[:, 1], diagram.cep)


def test_diagonal_is_the_unit_line(diagram):
    """The reference line runs from (0, 0) to (1, 1)."""
    ax = plot_reliability_diagram(diagram, density="none")
    np.testing.assert_allclose(
        find_artist(ax, "_calibre:diagonal").get_xydata(), [[0.0, 0.0], [1.0, 1.0]]
    )


def test_diagonal_can_be_suppressed(diagram):
    """``diagonal=False`` draws no reference line."""
    ax = plot_reliability_diagram(diagram, density="none", diagonal=False)
    assert [a for a in artists(ax) if a.get_label() == "_calibre:diagonal"] == []


def test_axes_are_square_and_unit(diagram):
    """A reliability diagram is only readable against a 1:1 diagonal."""
    ax = plot_reliability_diagram(diagram, density="none")
    assert ax.get_xlim() == (0.0, 1.0)
    assert ax.get_ylim() == (0.0, 1.0)
    assert ax.get_aspect() == 1.0


@pytest.mark.parametrize("maker", [consistency_bands, confidence_bands])
def test_band_polygon_spans_the_supplied_band(sample, diagram, maker):
    """The filled region must reach the band's own lower and upper edges."""
    x, y = sample
    band = maker(x, y, n_resamples=20, random_state=0)
    ax = plot_reliability_diagram(diagram, bands=band, density="none")
    vertices = find_artist(ax, "_calibre:band").get_paths()[0].vertices
    assert vertices[:, 1].min() == pytest.approx(band["lower"].min())
    assert vertices[:, 1].max() == pytest.approx(band["upper"].max())


def test_several_bands_can_be_nested(sample, diagram):
    """A sequence of bands draws one polygon each."""
    x, y = sample
    bands = [
        consistency_bands(x, y, level=level, n_resamples=20, random_state=0)
        for level in (0.5, 0.9)
    ]
    ax = plot_reliability_diagram(diagram, bands=bands, density="none")
    drawn = [a for a in artists(ax) if a.get_label() == "_calibre:band"]
    assert len(drawn) == 2


def test_malformed_band_is_rejected(diagram):
    """A band missing a key must say which one."""
    with pytest.raises(ValueError, match=r"missing \['upper'\]"):
        plot_reliability_diagram(
            diagram,
            bands={"x": np.array([0.0, 1.0]), "lower": np.array([0.0, 1.0])},
            density="none",
        )


def test_band_with_mismatched_lengths_is_rejected(diagram):
    """Ragged band arrays are a caller error, not a silent truncation."""
    with pytest.raises(ValueError, match="equal length"):
        plot_reliability_diagram(
            diagram,
            bands={
                "x": np.array([0.0, 0.5, 1.0]),
                "lower": np.array([0.0, 1.0]),
                "upper": np.array([0.0, 1.0]),
            },
            density="none",
        )


def test_histogram_density_adds_a_panel(diagram):
    """``density="hist"`` puts a marginal histogram below the diagram."""
    ax = plot_reliability_diagram(diagram, density="hist", density_bins=25)
    assert len(ax.figure.axes) == 2
    assert len(ax.figure.axes[-1].patches) == 25


def test_rug_density_stays_in_one_axes(diagram):
    """``density="rug"`` must not disturb a caller's grid geometry."""
    _, ax = plt.subplots()
    plot_reliability_diagram(diagram, ax=ax, density="rug")
    assert len(ax.figure.axes) == 1
    assert find_artist(ax, "_calibre:density") is not None


def test_no_density_draws_nothing_extra(diagram):
    """``density="none"`` leaves a single bare axes."""
    ax = plot_reliability_diagram(diagram, density="none")
    assert len(ax.figure.axes) == 1
    assert [a for a in artists(ax) if a.get_label() == "_calibre:density"] == []


def test_step_style_matches_the_pav_blocks(diagram):
    """``style="step"`` still draws the estimate, as a step function."""
    ax = plot_reliability_diagram(diagram, density="none", style="step")
    drawn = find_artist(ax, "_calibre:cep").get_xydata()
    np.testing.assert_allclose(drawn[:, 1], diagram.cep)


def test_label_becomes_a_legend_entry(diagram):
    """A public label is legible; the internal one is not."""
    ax = plot_reliability_diagram(diagram, density="none", label="my model")
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["my model"]


def test_unlabelled_plot_has_no_legend(diagram):
    """Internal ``_calibre:`` labels stay out of the legend."""
    ax = plot_reliability_diagram(diagram, density="none")
    _, labels = ax.get_legend_handles_labels()
    assert labels == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"density": "histogram"}, "density must be one of"),
        ({"style": "smooth"}, "style must be one of"),
    ],
)
def test_unknown_option_is_rejected(diagram, kwargs, match):
    """Typos in string options fail loudly rather than drawing something else."""
    with pytest.raises(ValueError, match=match):
        plot_reliability_diagram(diagram, **kwargs)


def test_diagram_method_matches_the_function(diagram):
    """``ReliabilityDiagram.plot`` is a delegate, not a second implementation."""
    ax = diagram.plot(density="none")
    np.testing.assert_allclose(
        find_artist(ax, "_calibre:cep").get_xydata()[:, 1], diagram.cep
    )
