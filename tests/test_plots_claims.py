"""Tests that each plot still says what it claims to say.

These are the substitute for baseline-image comparison. A pixel diff tells you a
pixel moved; these assert that the geometry actually drawn encodes the quantity
the docstring promises. If the resolution barcode ever stops meaning "one tick
per distinct value", or the decomposition panels stop carrying the exact
components, the test fails -- and it fails for the right reason.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.colors

from calibre import (
    CenteredIsotonicCalibrator,
    IsotonicCalibrator,
    miscalibration_profile,
    score_decomposition,
)

# The library's own wrapper: scipy returns a result object whose `statistic` is
# typed too loosely to read directly, which is why this helper exists.
from calibre.metrics import (
    _spearman,
    debiased_calibration_error,
    plugin_calibration_error,
)
from calibre.plots import (
    SEMANTIC,
    plot_ece_bin_sensitivity,
    plot_mcb_dsc_plane,
    plot_miscalibration_profile,
    plot_resolution_loss,
    plot_score_decomposition,
)
from calibre.plots._style import artists


def _bars(ax, label):
    """Return the bar rectangles drawn under ``label``.

    Parameters
    ----------
    ax
        Axes to scan.
    label
        Container label.

    Returns
    -------
    list
        The rectangles, in drawing order.
    """
    out = []
    for container in ax.containers:
        if container.get_label() == label:
            out.extend(container.patches)
    return out


@pytest.fixture
def calibrated():
    """Predictions that are calibrated by construction.

    Returns
    -------
    tuple of ndarray
        Labels and predictions. The true calibration error is exactly zero.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    y = rng.binomial(1, p).astype(float)
    return y, p


# --------------------------------------------------------------------------- #
# The resolution barcode: one tick per distinct output value, exactly.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", [3, 6])
def test_barcode_tick_count_equals_distinct_value_count(precision):
    """The whole claim of the plot, asserted exactly.

    If this drifts, the figure silently starts overstating or understating how
    much resolution a calibrator kept -- which is the package's headline number.
    """
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 900)
    labels = rng.binomial(1, scores).astype(float)
    outputs = {
        "isotonic": IsotonicCalibrator().fit(scores, labels).transform(scores),
        "centered": (
            CenteredIsotonicCalibrator().fit(scores, labels).transform(scores)
        ),
    }
    ax = plot_resolution_loss(outputs, scores, precision=precision)

    for name, values in outputs.items():
        drawn = [a for a in artists(ax) if a.get_label() == f"_calibre:ticks:{name}"]
        assert len(drawn) == 1
        n_ticks = len(drawn[0].get_segments())
        expected = int(np.unique(np.round(values, precision)).size)
        assert n_ticks == expected, f"{name}: {n_ticks} ticks for {expected} values"


def test_barcode_shows_isotonic_losing_resolution():
    """Isotonic must draw far fewer ticks than the resolution-preserving fit.

    Not a tuned threshold: the claim is a qualitative ordering that holds by
    construction, since centered isotonic interpolates between the same blocks
    isotonic flattens.
    """
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 1200)
    labels = rng.binomial(1, scores).astype(float)
    iso = IsotonicCalibrator().fit(scores, labels).transform(scores)
    cir = CenteredIsotonicCalibrator().fit(scores, labels).transform(scores)
    assert np.unique(iso).size < np.unique(cir).size


# --------------------------------------------------------------------------- #
# The score decomposition: panel geometry must carry the exact components.
# --------------------------------------------------------------------------- #


def test_decomposition_panels_carry_the_exact_components():
    """Bar lengths must be the components themselves, panel by panel.

    A decomposition drawn wrong is a wrong picture, and an offset of 0.003 in
    MCB -- which is the whole quantity on a well-calibrated model -- is
    invisible to the eye.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 2000)
    y = rng.binomial(1, x).astype(float)
    d = score_decomposition(x, y)

    fig = plot_score_decomposition(d)
    mcb_panel, dsc_panel, score_panel = fig.axes

    assert _bars(mcb_panel, "_calibre:mcb")[0].get_width() == pytest.approx(
        d["MCB"], abs=1e-12
    )
    assert _bars(dsc_panel, "_calibre:dsc")[0].get_width() == pytest.approx(
        d["DSC"], abs=1e-12
    )
    assert _bars(score_panel, "_calibre:mean_score")[0].get_width() == pytest.approx(
        d["mean_score"], abs=1e-12
    )


def test_decomposition_identity_holds_across_the_panels():
    """``UNC + MCB - DSC`` read off the drawn bars must be the drawn score."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 1500)
    y = rng.binomial(1, np.clip(x**1.3, 0, 1)).astype(float)
    d = score_decomposition(x, y)

    fig = plot_score_decomposition(d)
    mcb_panel, dsc_panel, score_panel = fig.axes
    mcb = _bars(mcb_panel, "_calibre:mcb")[0].get_width()
    dsc = _bars(dsc_panel, "_calibre:dsc")[0].get_width()
    score = _bars(score_panel, "_calibre:mean_score")[0].get_width()

    assert d["UNC"] + mcb - dsc == pytest.approx(score, abs=1e-12)


def test_every_decomposition_panel_starts_at_zero():
    """Bars are only comparable in length if their axis starts at zero.

    Separate scales per panel are what makes MCB visible next to a DSC a hundred
    times larger; a truncated axis on top of that would make the lengths lie.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 1200)
    y = rng.binomial(1, x).astype(float)
    fig = plot_score_decomposition(score_decomposition(x, y))
    for panel in fig.axes:
        assert panel.get_xlim()[0] == 0.0


def test_mcb_is_visible_next_to_a_much_larger_dsc():
    """The point of the separate-panel design, asserted.

    On a competent model DSC is around 0.10 and MCB around 0.001. Drawn on one
    shared axis the MCB bar would be under 1% of the width; here it must occupy
    a real fraction of its own panel.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 3000)
    y = rng.binomial(1, x).astype(float)
    d = score_decomposition(x, y)
    assert d["DSC"] > 20 * d["MCB"], "fixture no longer exercises the hard case"

    fig = plot_score_decomposition(d)
    mcb_panel = fig.axes[0]
    width = _bars(mcb_panel, "_calibre:mcb")[0].get_width()
    span = mcb_panel.get_xlim()[1] - mcb_panel.get_xlim()[0]
    assert width / span > 0.5


def test_decomposition_draws_one_row_per_forecaster_in_every_panel():
    """A mapping of decompositions produces a row each, in all three panels."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 1200)
    y = rng.binomial(1, x).astype(float)
    named = {
        "honest": score_decomposition(x, y),
        "squashed": score_decomposition(0.25 + 0.5 * x, y),
    }
    fig = plot_score_decomposition(named)
    mcb_panel, dsc_panel, score_panel = fig.axes

    assert len(_bars(mcb_panel, "_calibre:mcb")) == 2
    assert len(_bars(dsc_panel, "_calibre:dsc")) == 2
    assert len(_bars(score_panel, "_calibre:mean_score")) == 2
    assert [t.get_text() for t in mcb_panel.get_yticklabels()] == [
        "honest",
        "squashed",
    ]


def test_decomposition_reports_uncertainty_in_the_title():
    """UNC is constant across forecasters, so it is stated rather than drawn."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 1200)
    y = rng.binomial(1, x).astype(float)
    d = score_decomposition(x, y)
    fig = plot_score_decomposition(d)
    assert f"{d['UNC']:.4f}" in fig.get_suptitle()


def test_mcb_dsc_plane_places_each_method_at_its_components():
    """Scatter coordinates must be exactly ``(DSC, MCB)``."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 1500)
    y = rng.binomial(1, x).astype(float)
    named = {
        "honest": score_decomposition(x, y),
        "squashed": score_decomposition(0.25 + 0.5 * x, y),
    }
    ax = plot_mcb_dsc_plane(named)
    points = next(
        a for a in ax.collections if a.get_label() == "_calibre:methods"
    ).get_offsets()

    expected = np.array([[named[n]["DSC"], named[n]["MCB"]] for n in named])
    np.testing.assert_allclose(np.asarray(points), expected)


# --------------------------------------------------------------------------- #
# The ECE bin-sensitivity plot: it exists to argue that plugin ECE is biased.
# --------------------------------------------------------------------------- #


def test_plugin_curve_climbs_with_bin_count(calibrated):
    """On calibrated data the plugin estimator reports growing pure bias."""
    y, p = calibrated
    counts = list(range(2, 51))
    ax = plot_ece_bin_sensitivity(y, p, n_bins=counts, estimators=["plugin"])
    drawn = [line for line in ax.lines if line.get_label() == "plugin (uncorrected)"]
    values = drawn[0].get_xydata()[:, 1]

    assert _spearman(np.asarray(counts, dtype=float), values) > 0.9
    assert values[-1] > values[0]


def test_debiased_curve_does_not_climb(calibrated):
    """The correction removes the bin-count dependence the plugin has."""
    y, p = calibrated
    counts = list(range(2, 51))
    ax = plot_ece_bin_sensitivity(y, p, n_bins=counts, estimators=["debiased"])
    values = next(
        line for line in ax.lines if line.get_label() == "debiased"
    ).get_xydata()[:, 1]

    assert abs(_spearman(np.asarray(counts, dtype=float), values)) < 0.5


def test_debiased_stays_below_plugin_at_every_bin_count(calibrated):
    """The whole point of the correction, over the drawn range."""
    y, p = calibrated
    counts = list(range(2, 51))
    ax = plot_ece_bin_sensitivity(y, p, n_bins=counts)
    lines = {line.get_label(): line.get_xydata()[:, 1] for line in ax.lines}
    assert np.all(lines["debiased"] <= lines["plugin (uncorrected)"] + 1e-12)


def test_drawn_series_are_the_estimators_themselves(calibrated):
    """The plot must not smooth, rescale or otherwise reinterpret the numbers."""
    y, p = calibrated
    counts = [5, 10, 20]
    ax = plot_ece_bin_sensitivity(y, p, n_bins=counts, norm=2)
    lines = {line.get_label(): line.get_xydata()[:, 1] for line in ax.lines}

    np.testing.assert_allclose(
        lines["plugin (uncorrected)"],
        [plugin_calibration_error(y, p, b, 2) for b in counts],
    )
    np.testing.assert_allclose(
        lines["debiased"],
        [debiased_calibration_error(y, p, b) for b in counts],
    )


def test_sweep_line_reports_the_bin_count_it_chose(calibrated):
    """Half the sweep's answer is which bin count the data supported."""
    y, p = calibrated
    ax = plot_ece_bin_sensitivity(y, p, n_bins=[5, 10], estimators=["sweep"])
    labels = [line.get_label() for line in ax.lines]
    assert any(label.startswith("sweep (chose ") for label in labels)


def test_unknown_estimator_is_rejected(calibrated):
    """A typo must not silently draw fewer series than asked for."""
    y, p = calibrated
    with pytest.raises(ValueError, match="unknown estimators"):
        plot_ece_bin_sensitivity(y, p, estimators=["plugin", "magic"])


# --------------------------------------------------------------------------- #
# The multiclass profile: bar heights are the per-class MCB.
# --------------------------------------------------------------------------- #


def test_profile_bars_are_the_per_class_mcb():
    """Bar heights must be the diagnostic's own numbers."""
    rng = np.random.default_rng(0)
    truth = rng.dirichlet(np.ones(5) * 0.7, size=2000)
    y = np.array([rng.choice(5, p=t) for t in truth])
    skewed = truth ** np.linspace(0.6, 2.4, 5)
    scores = skewed / skewed.sum(axis=1, keepdims=True)

    profile = miscalibration_profile(scores, y)
    ax = plot_miscalibration_profile(profile)
    heights = [bar.get_height() for bar in _bars(ax, "_calibre:mcb")]
    np.testing.assert_allclose(heights, profile["mcb"])


def test_profile_highlights_the_worst_classes():
    """The accent colour must fall on the classes the profile named worst."""
    rng = np.random.default_rng(0)
    truth = rng.dirichlet(np.ones(5) * 0.7, size=2000)
    y = np.array([rng.choice(5, p=t) for t in truth])
    skewed = truth ** np.linspace(0.6, 2.4, 5)
    scores = skewed / skewed.sum(axis=1, keepdims=True)

    profile = miscalibration_profile(scores, y)
    ax = plot_miscalibration_profile(profile, highlight_worst=2)

    accented = {
        i
        for i, bar in enumerate(_bars(ax, "_calibre:mcb"))
        if bar.get_facecolor()[:3] == matplotlib.colors.to_rgb(SEMANTIC["highlight"])
    }
    assert accented == set(np.asarray(profile["worst_classes"])[:2].tolist())


def test_profile_reports_the_spread_in_the_title():
    """The spread is what decides the method, so it belongs on the figure."""
    rng = np.random.default_rng(0)
    truth = rng.dirichlet(np.ones(4), size=1200)
    y = np.array([rng.choice(4, p=t) for t in truth])
    profile = miscalibration_profile(truth, y)
    ax = plot_miscalibration_profile(profile)
    assert f"{profile['spread']:.2f}" in ax.get_title()
