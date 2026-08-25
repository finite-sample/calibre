"""Every plot on every awkward input.

Degenerate data is where plotting code fails in the ugliest way: an empty axes
that looks like a finding, a warning buried in a notebook, or a traceback from
deep inside matplotlib that says nothing about what the caller did wrong. Each
case here must produce *either* a drawn Axes *or* a ``ValueError`` naming the
problem -- never a bare exception, never a silent empty picture, and never a
warning.
"""

from __future__ import annotations

import warnings

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from calibre import (
    CenteredIsotonicCalibrator,
    IsotonicCalibrator,
    classwise_reliability,
    corp_reliability,
    miscalibration_profile,
    score_decomposition,
)
from calibre.plots import (
    plot_calibrator_comparison,
    plot_classwise_reliability,
    plot_ece_bin_sensitivity,
    plot_mcb_dsc_plane,
    plot_miscalibration_profile,
    plot_reliability_diagram,
    plot_resolution_frontier,
    plot_resolution_loss,
    plot_score_decomposition,
)

# (name, scores, labels). Every one of these turns up in real calibration work:
# rare events give all-zero labels, clipped or rounded model outputs give heavy
# ties and values sitting exactly on 0 and 1, and small validation splits give n
# small enough that a single bin is all the data supports.
CASES = [
    ("n_1", np.array([0.5]), np.array([1.0])),
    ("n_2", np.array([0.2, 0.8]), np.array([0.0, 1.0])),
    ("all_zero_labels", np.linspace(0.0, 1.0, 60), np.zeros(60)),
    ("all_one_labels", np.linspace(0.0, 1.0, 60), np.ones(60)),
    ("single_distinct_score", np.full(60, 0.3), np.tile([0.0, 1.0], 30)),
    (
        "heavy_ties",
        np.round(np.linspace(0.0, 1.0, 200), 1),
        np.tile([0.0, 1.0], 100),
    ),
    (
        "at_the_bounds",
        np.concatenate([np.zeros(30), np.ones(30)]),
        np.concatenate([np.zeros(30), np.ones(30)]),
    ),
    ("constant_half", np.full(40, 0.5), np.tile([0.0, 1.0], 20)),
]

IDS = [case[0] for case in CASES]


@pytest.fixture(autouse=True)
def _warnings_are_errors():
    """Treat any warning raised while drawing as a failure.

    A plot that warns is a plot that will spam a notebook, and the warning is
    usually a real geometry problem such as a singular axis range.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # matplotlib's open-figure warning is about the test loop, not the code
        # under test; conftest closes figures after each test.
        warnings.filterwarnings("ignore", message="More than 20 figures")
        yield


def _drew(result):
    """Assert that a plot returned something drawable.

    Parameters
    ----------
    result
        Whatever the plot function returned.

    Returns
    -------
    bool
        True when the result is an Axes or Figure.
    """
    return isinstance(result, Axes | Figure)


@pytest.mark.parametrize(("_name", "x", "y"), CASES, ids=IDS)
@pytest.mark.parametrize("density", ["hist", "rug", "none"])
def test_reliability_diagram_survives(_name, x, y, density):
    """A CORP diagram must draw for any binary sample."""
    assert _drew(plot_reliability_diagram(corp_reliability(x, y), density=density))


@pytest.mark.parametrize(("_name", "x", "y"), CASES, ids=IDS)
def test_score_decomposition_survives(_name, x, y):
    """The panels must draw whatever the decomposition returns."""
    assert _drew(plot_score_decomposition(score_decomposition(x, y)))


@pytest.mark.parametrize(("_name", "x", "y"), CASES, ids=IDS)
def test_mcb_dsc_plane_survives(_name, x, y):
    """A single point is a legitimate plane, with a degenerate range."""
    assert _drew(plot_mcb_dsc_plane({"only": score_decomposition(x, y)}))


@pytest.mark.parametrize(("_name", "x", "y"), CASES, ids=IDS)
def test_ece_bin_sensitivity_survives(_name, x, y):
    """Bin counts above the sample size must not raise."""
    assert _drew(plot_ece_bin_sensitivity(y, x, n_bins=[1, 2, 5, 40]))


@pytest.mark.parametrize(("_name", "x", "y"), CASES, ids=IDS)
def test_resolution_loss_survives(_name, x, y):
    """The barcode must draw even when every output is identical."""
    fitted = IsotonicCalibrator().fit(x, y)
    assert _drew(plot_resolution_loss({"isotonic": fitted.transform(x)}, x))


@pytest.mark.parametrize(("_name", "x", "y"), CASES, ids=IDS)
def test_calibrator_comparison_survives(_name, x, y):
    """Comparison must survive calibrators that collapse to a constant."""
    fitted = {
        "isotonic": IsotonicCalibrator().fit(x, y),
        "centered": CenteredIsotonicCalibrator().fit(x, y),
    }
    assert _drew(
        plot_calibrator_comparison(fitted, x, reference=corp_reliability(x, y))
    )


# --------------------------------------------------------------------------- #
# Empty and malformed input: a named ValueError, never a bare exception.
# --------------------------------------------------------------------------- #


def test_resolution_loss_rejects_empty_outputs():
    """Nothing to draw is a caller error worth naming."""
    with pytest.raises(ValueError, match="outputs is empty"):
        plot_resolution_loss({}, np.array([0.5]))


def test_resolution_loss_rejects_ragged_outputs():
    """Different-length outputs cannot be the same observations."""
    with pytest.raises(ValueError, match="same length"):
        plot_resolution_loss(
            {"a": np.array([0.1, 0.2]), "b": np.array([0.1, 0.2, 0.3])}
        )


def test_resolution_loss_rejects_mismatched_x():
    """``x`` must line up with the outputs it labels."""
    with pytest.raises(ValueError, match="but the outputs have"):
        plot_resolution_loss({"a": np.array([0.1, 0.2])}, np.array([0.1, 0.2, 0.3]))


def test_resolution_frontier_rejects_empty_results():
    """An empty frontier is a caller error."""
    with pytest.raises(ValueError, match="results is empty"):
        plot_resolution_frontier({})


def test_resolution_frontier_rejects_nonpositive_counts():
    """A log x-axis cannot show zero distinct values."""
    with pytest.raises(ValueError, match="must be positive"):
        plot_resolution_frontier({"broken": (0, 0.2)})


def test_score_decomposition_rejects_a_missing_component():
    """A mapping that is not a decomposition must say so."""
    with pytest.raises(ValueError, match="is missing"):
        plot_score_decomposition({"MCB": 0.1, "DSC": 0.2})


def test_mcb_dsc_plane_rejects_empty_input():
    """Nothing to place on the plane."""
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_mcb_dsc_plane({})


def test_comparison_rejects_empty_calibrators():
    """Nothing to compare."""
    with pytest.raises(ValueError, match="calibrators is empty"):
        plot_calibrator_comparison({}, np.array([0.5]))


def test_comparison_rejects_empty_x():
    """No grid to evaluate the calibrators on."""
    with pytest.raises(ValueError, match="x is empty"):
        plot_calibrator_comparison(
            {
                "iso": IsotonicCalibrator().fit(
                    np.array([0.1, 0.9]), np.array([0.0, 1.0])
                )
            },
            np.array([]),
        )


def test_comparison_refuses_an_unfitted_calibrator():
    """Fitting inside a plot is the mistake this package warns about.

    The error must name the offending calibrator and say why, rather than
    letting a NotFittedError surface from somewhere in sklearn.
    """
    with pytest.raises(ValueError, match="'iso' is not fitted"):
        plot_calibrator_comparison({"iso": IsotonicCalibrator()}, np.array([0.2, 0.8]))


def test_comparison_checks_fit_without_evaluating_an_arbitrary_score():
    """A fitted strict-bounds calibrator must not be rejected as unfitted."""
    x = np.array([0.7, 0.8, 0.9])
    y = np.array([0.0, 1.0, 1.0])
    calibrator = IsotonicCalibrator(out_of_bounds="raise").fit(x, y)

    ax = plot_calibrator_comparison({"iso": calibrator}, x)

    assert ax.lines


def test_comparison_refuses_an_object_that_cannot_transform():
    """Anything without ``.transform`` is not a calibrator."""
    with pytest.raises(ValueError, match=r"has no \.transform"):
        plot_calibrator_comparison({"nonsense": object()}, np.array([0.2, 0.8]))


def test_ece_rejects_empty_bin_list():
    """An empty sweep would draw an empty axes."""
    with pytest.raises(ValueError, match="n_bins is empty"):
        plot_ece_bin_sensitivity(np.array([0.0, 1.0]), np.array([0.2, 0.8]), n_bins=[])


def test_ece_rejects_a_bin_count_below_one():
    """Zero bins is not a binning."""
    with pytest.raises(ValueError, match="at least 1"):
        plot_ece_bin_sensitivity(
            np.array([0.0, 1.0]), np.array([0.2, 0.8]), n_bins=[0, 5]
        )


# --------------------------------------------------------------------------- #
# Multiclass.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_classes", [2, 3, 7])
def test_multiclass_plots_survive(n_classes):
    """Profile and small multiples must draw for any class count."""
    rng = np.random.default_rng(0)
    truth = rng.dirichlet(np.ones(n_classes), size=400)
    y = np.array([rng.choice(n_classes, p=t) for t in truth])

    assert _drew(plot_miscalibration_profile(miscalibration_profile(truth, y)))
    figure = plot_classwise_reliability(classwise_reliability(truth, y))
    assert isinstance(figure, Figure)
    assert len([a for a in figure.axes if a.get_visible()]) == n_classes


def test_profile_rejects_a_missing_key():
    """A mapping that is not a profile must say so."""
    with pytest.raises(ValueError, match="profile is missing"):
        plot_miscalibration_profile({"mcb": np.array([0.1, 0.2])})


def test_profile_rejects_wrong_length_class_names():
    """Mislabelled classes would be worse than unlabelled ones."""
    rng = np.random.default_rng(0)
    truth = rng.dirichlet(np.ones(3), size=300)
    y = np.array([rng.choice(3, p=t) for t in truth])
    with pytest.raises(ValueError, match="but the profile covers"):
        plot_miscalibration_profile(
            miscalibration_profile(truth, y), class_names=["a", "b"]
        )


def test_classwise_reliability_rejects_empty_input():
    """No diagrams, no panels."""
    with pytest.raises(ValueError, match="diagrams is empty"):
        plot_classwise_reliability([])


def test_classwise_reliability_rejects_wrong_axes_count():
    """One panel per class, or a clear error."""
    rng = np.random.default_rng(0)
    truth = rng.dirichlet(np.ones(3), size=300)
    y = np.array([rng.choice(3, p=t) for t in truth])
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="but there are 3 diagrams"):
        plot_classwise_reliability(classwise_reliability(truth, y), axes=list(axes))
