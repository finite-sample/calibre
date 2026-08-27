"""Tests for the CORP evaluation stack.

The decomposition's value is that its components are guaranteed, not merely
usually true: the identity is exact, and MCB and DSC are non-negative because the
PAV solution is optimal. So these assert exact algebra rather than tolerances,
and the numbers are pinned against R in ``tests/test_r_reference.py``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from calibre._core import weighted_pava
from calibre.evaluation import (
    confidence_bands,
    consistency_bands,
    corp_reliability,
    score_decomposition,
)

SCORES = ["brier", "log"]


def _calibrated(seed: int, n: int = 2000):
    """Generate forecasts that are calibrated by construction.

    Parameters
    ----------
    seed
        Random seed.
    n
        Number of observations.

    Returns
    -------
    tuple of ndarray
        Forecasts and outcomes.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    return x, rng.binomial(1, x).astype(float)


def _tied(seed: int, n: int = 600, decimals: int = 2):
    """Generate forecasts with heavy ties.

    Parameters
    ----------
    seed
        Random seed.
    n
        Number of observations.
    decimals
        Rounding applied to the forecasts.

    Returns
    -------
    tuple of ndarray
        Forecasts and outcomes.
    """
    rng = np.random.default_rng(seed)
    x = np.round(rng.uniform(0.0, 1.0, n), decimals)
    return x, rng.binomial(1, x).astype(float)


def _make_band(maker, x, y, **kwargs):
    """Call the one-input null band or the two-input confidence band."""
    if maker is consistency_bands:
        return maker(x, **kwargs)
    return maker(y, x, **kwargs)


# --------------------------------------------------------------------------- #
# The reliability diagram
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(5))
def test_cep_is_monotone(seed):
    """Conditional event probabilities must be non-decreasing.

    This is the regularisation that distinguishes CORP from binning-and-counting:
    a decreasing estimate is an artifact, and isotonicity rules it out.
    """
    x, y = _calibrated(seed, n=800)
    diagram = corp_reliability(y, x)
    assert np.all(np.diff(diagram.event_probabilities) >= -1e-12)


def test_cep_lies_in_the_unit_interval():
    """Recalibrated probabilities are probabilities."""
    x, y = _calibrated(1, n=800)
    diagram = corp_reliability(y, x)
    assert np.all(diagram.event_probabilities >= 0.0)
    assert np.all(diagram.event_probabilities <= 1.0)


def test_diagram_is_the_pav_fit_of_the_pooled_data():
    """The diagram is exactly weighted PAVA on tie-pooled outcomes.

    Stated as an identity rather than trusted, because every downstream number
    in this module depends on it.
    """
    x, y = _tied(2)
    diagram = corp_reliability(y, x)

    x_unique, inverse = np.unique(x, return_inverse=True)
    weight = np.bincount(inverse, minlength=x_unique.size).astype(float)
    y_mean = np.bincount(inverse, weights=y, minlength=x_unique.size) / weight

    np.testing.assert_allclose(
        diagram.event_probabilities, weighted_pava(y_mean, weight), atol=1e-12
    )
    np.testing.assert_allclose(diagram.prediction_values, x_unique)


def test_diagram_ignores_row_order():
    """Shuffling observations must not move the diagram."""
    x, y = _tied(3)
    first = corp_reliability(y, x)

    rng = np.random.default_rng(0)
    perm = rng.permutation(x.size)
    second = corp_reliability(y[perm], x[perm])

    np.testing.assert_allclose(first.prediction_values, second.prediction_values)
    np.testing.assert_allclose(
        first.event_probabilities, second.event_probabilities, atol=1e-12
    )


def test_single_distinct_forecast_gives_the_base_rate():
    """One forecast value can only be recalibrated to the observed frequency."""
    x = np.full(50, 0.3)
    y = np.concatenate([np.ones(20), np.zeros(30)])
    diagram = corp_reliability(y, x)
    assert diagram.event_probabilities.size == 1
    assert diagram.event_probabilities[0] == pytest.approx(0.4)


def test_diagram_rejects_negative_weights_even_when_ties_cancel_them():
    """Observation weights must be valid before tied forecasts are pooled."""
    x = np.array([0.2, 0.2, 0.8])
    y = np.array([0.0, 1.0, 1.0])
    w = np.array([2.0, -1.0, 1.0])

    with pytest.raises(ValueError, match="finite non-negative"):
        corp_reliability(y, x, sample_weight=w)


def test_diagram_rejects_unidentified_all_zero_weights():
    """A reliability diagram needs at least one weighted observation."""
    x, y = _calibrated(17, n=20)

    with pytest.raises(ValueError, match="at least one positive"):
        corp_reliability(y, x, sample_weight=np.zeros_like(y))


# --------------------------------------------------------------------------- #
# The score decomposition
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("seed", range(5))
def test_decomposition_identity_is_exact(score, seed):
    """mean_score == MCB - DSC + UNC, to floating point."""
    x, y = _calibrated(seed, n=1000)
    d = score_decomposition(y, x, score=score)
    assert d["mean_score"] == pytest.approx(
        d["miscalibration"] - d["discrimination"] + d["uncertainty"],
        abs=1e-12,
    )


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("seed", range(5))
def test_components_are_non_negative(score, seed):
    """MCB and DSC cannot be negative.

    Both follow from PAV optimality: no reordering of the forecasts scores better
    than the isotonic fit, and the isotonic fit is at least as good as the
    constant reference.
    """
    x, y = _calibrated(seed, n=1000)
    d = score_decomposition(y, x, score=score)
    assert d["miscalibration"] >= -1e-12
    assert d["discrimination"] >= -1e-12


@pytest.mark.parametrize("score", SCORES)
def test_recalibrated_forecasts_have_zero_miscalibration(score):
    """Recalibrating twice changes nothing, so MCB collapses to zero.

    PAV is idempotent, which is also why an in-sample MCB for an isotonic-family
    calibrator is uninformative -- see ``test_selection.py``.
    """
    x, y = _calibrated(4, n=1500)
    diagram = corp_reliability(y, x)
    recalibrated = np.interp(x, diagram.prediction_values, diagram.event_probabilities)
    assert score_decomposition(y, recalibrated, score=score)[
        "miscalibration"
    ] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("score", SCORES)
def test_miscalibration_detects_a_distorted_forecast(score):
    """Squashing forecasts toward the base rate must raise MCB."""
    x, y = _calibrated(5, n=3000)
    squashed = 0.5 + 0.4 * (x - 0.5)

    honest = score_decomposition(y, x, score=score)["miscalibration"]
    distorted = score_decomposition(y, squashed, score=score)["miscalibration"]
    assert distorted > honest


@pytest.mark.parametrize("score", SCORES)
def test_constant_forecast_has_no_discrimination(score):
    """A forecast that never varies cannot discriminate."""
    y = np.concatenate([np.ones(400), np.zeros(600)])
    x = np.full(y.size, 0.4)
    d = score_decomposition(y, x, score=score)
    assert d["discrimination"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("score", SCORES)
def test_uncertainty_ignores_the_forecast(score):
    """UNC depends on the outcomes alone, not on who is forecasting."""
    x, y = _calibrated(6, n=1000)
    rng = np.random.default_rng(7)
    useless = rng.uniform(0.0, 1.0, y.size)
    assert score_decomposition(y, x, score=score)["uncertainty"] == pytest.approx(
        score_decomposition(y, useless, score=score)["uncertainty"], abs=1e-12
    )


@pytest.mark.parametrize("score", SCORES)
def test_decomposition_survives_heavy_ties(score):
    """Tied forecasts must not break the identity.

    Ties have broken interpolation-based calibrators before, so anything
    built on interpolation gets checked against them.
    """
    x, y = _tied(8, decimals=1)
    d = score_decomposition(y, x, score=score)
    assert d["mean_score"] == pytest.approx(
        d["miscalibration"] - d["discrimination"] + d["uncertainty"],
        abs=1e-12,
    )
    assert d["miscalibration"] >= -1e-12


def test_mean_score_matches_a_direct_computation():
    """mean_score is the plain Brier score, not a transformed one."""
    x, y = _calibrated(9, n=500)
    d = score_decomposition(y, x, score="brier")
    assert d["mean_score"] == pytest.approx(float(np.mean((x - y) ** 2)), abs=1e-12)


def test_unknown_score_is_rejected():
    """Only proper scoring rules are accepted."""
    x, y = _calibrated(10, n=100)
    with pytest.raises(ValueError, match="proper scoring rule"):
        score_decomposition(y, x, score="ece")


def test_weights_replicate_duplication():
    """Weighting an observation by 2 matches listing it twice."""
    x, y = _tied(11, n=200)
    weighted = score_decomposition(y, x, sample_weight=np.full(x.size, 2.0))
    plain = score_decomposition(y, x)
    for key in plain:
        assert weighted[key] == pytest.approx(plain[key], abs=1e-12)


# --------------------------------------------------------------------------- #
# Uncertainty bands
# --------------------------------------------------------------------------- #


def test_consistency_bands_bracket_the_diagonal():
    """Under the calibration hypothesis the bands sit around the diagonal."""
    x, _ = _calibrated(12, n=800)
    band = consistency_bands(x, n_resamples=200, random_state=0)
    assert np.all(band["lower"] <= band["upper"] + 1e-12)
    # The diagonal is the hypothesised truth, so it should mostly be inside.
    grid = band["prediction_values"]
    inside = (band["lower"] <= grid) & (grid <= band["upper"])
    assert inside.mean() > 0.8


def test_consistency_band_api_names_the_probability_input_and_tuning_options():
    """The null band needs forecasts, not the observed labels it redraws."""
    signature = inspect.signature(consistency_bands)
    assert list(signature.parameters) == [
        "y_pred",
        "level",
        "n_resamples",
        "random_state",
    ]
    assert (
        signature.parameters["y_pred"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    for name in ("level", "n_resamples", "random_state"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_consistency_bands_use_descriptive_result_names():
    """Band coordinates use the same forecast terminology as CORP diagrams."""
    band = consistency_bands(
        np.array([0.1, 0.3, 0.7, 0.9]), n_resamples=20, random_state=0
    )
    assert set(band) == {"prediction_values", "lower", "upper"}


def test_consistency_bands_are_permutation_invariant_for_a_fixed_seed():
    """Row order cannot alter a seeded resampling distribution."""
    x = np.repeat(np.array([0.05, 0.2, 0.5, 0.8, 0.95]), 40)
    permutation = np.random.default_rng(42).permutation(x.size)
    original = consistency_bands(x, n_resamples=200, random_state=7)
    permuted = consistency_bands(x[permutation], n_resamples=200, random_state=7)
    for key in original:
        np.testing.assert_array_equal(original[key], permuted[key])


def test_consistency_bands_resample_the_empirical_forecast_distribution():
    """Pin the pair-bootstrap step used by the authors' R implementation.

    The rare high-forecast group has only five observations. Resampling forecast
    rows makes its effective group size vary; holding all forecast rows fixed
    instead gives a lower endpoint of 0.4 for this seed.
    """
    x = np.repeat(np.array([0.01, 0.05, 0.2, 0.8]), [180, 50, 15, 5])
    band = consistency_bands(x, n_resamples=2000, random_state=0)
    assert band["prediction_values"][-1] == 0.8
    assert band["lower"][-1] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(
    "y_pred",
    [
        np.array([]),
        np.array([[0.2, 0.8]]),
        np.array([0.2, np.nan]),
        np.array([-0.1, 0.8]),
        np.array([0.2, 1.1]),
        np.array(["low", "high"]),
    ],
    ids=["empty", "two-dimensional", "nonfinite", "below-zero", "above-one", "text"],
)
def test_consistency_bands_reject_invalid_probabilities(y_pred):
    """Malformed forecasts fail at the public boundary with the named contract."""
    with pytest.raises(ValueError, match="y_pred"):
        consistency_bands(y_pred, n_resamples=20)


@pytest.mark.parametrize("value", [1.5, True, "20"])
def test_consistency_bands_require_an_integer_resample_count(value):
    """A low-level array-shape error must not define the public contract."""
    with pytest.raises(ValueError, match=r"n_resamples.*integer"):
        consistency_bands(np.array([0.2, 0.8]), n_resamples=value)


@pytest.mark.parametrize("value", [-1, True, 1.5, "seed"])
def test_consistency_bands_validate_the_seed(value):
    """The documented seed domain is checked before NumPy is called."""
    with pytest.raises(ValueError, match="random_state"):
        consistency_bands(np.array([0.2, 0.8]), n_resamples=20, random_state=value)


@pytest.mark.parametrize(("forecast", "expected"), [(0.0, 0.0), (1.0, 1.0)])
def test_consistency_bands_are_degenerate_at_certain_forecasts(forecast, expected):
    """Bernoulli forecasts at the probability boundaries have no sampling noise."""
    band = consistency_bands(np.full(40, forecast), n_resamples=100, random_state=11)
    np.testing.assert_array_equal(band["prediction_values"], [forecast])
    np.testing.assert_array_equal(band["lower"], [expected])
    np.testing.assert_array_equal(band["upper"], [expected])


def test_consistency_bands_do_not_extrapolate_singleton_resamples():
    """A resample containing one forecast value cannot inform another value."""
    y_pred = np.array([0.0, 0.0, 1.0, 1.0])
    band = consistency_bands(y_pred, n_resamples=1000, random_state=0)

    np.testing.assert_array_equal(band["prediction_values"], [0.0, 1.0])
    np.testing.assert_array_equal(band["lower"], [0.0, 1.0])
    np.testing.assert_array_equal(band["upper"], [0.0, 1.0])


def test_confidence_bands_bracket_the_estimate():
    """Confidence bands cluster around the CORP estimate, not the diagonal."""
    x, y = _calibrated(13, n=800)
    diagram = corp_reliability(y, x)
    band = confidence_bands(y, x, n_resamples=200, random_state=0)
    inside = (band["lower"] <= diagram.event_probabilities) & (
        diagram.event_probabilities <= band["upper"]
    )
    assert inside.mean() > 0.8


def test_confidence_band_api_matches_evaluation_functions():
    """Outcomes precede forecasts and tuning arguments are keyword-only."""
    signature = inspect.signature(confidence_bands)
    assert list(signature.parameters) == [
        "y_true",
        "y_pred",
        "level",
        "n_resamples",
        "random_state",
    ]
    for name in ("y_true", "y_pred"):
        assert (
            signature.parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
    for name in ("level", "n_resamples", "random_state"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_confidence_bands_use_descriptive_result_names():
    """Confidence and consistency bands expose the same coordinate name."""
    x, y = _calibrated(130, n=100)
    band = confidence_bands(y, x, n_resamples=20, random_state=0)
    assert set(band) == {"prediction_values", "lower", "upper"}


def test_confidence_bands_are_permutation_invariant_for_a_fixed_seed():
    """Row order cannot alter a seeded pair-bootstrap distribution."""
    x, y = _tied(131, n=400, decimals=1)
    permutation = np.random.default_rng(42).permutation(x.size)
    original = confidence_bands(y, x, n_resamples=200, random_state=7)
    permuted = confidence_bands(
        y[permutation], x[permutation], n_resamples=200, random_state=7
    )
    for key in original:
        np.testing.assert_array_equal(original[key], permuted[key])


def test_confidence_bands_match_null_bands_for_an_identity_pav_fit():
    """The two procedures coincide when the estimated event rate is y_pred."""
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    x = np.repeat(levels, 100)
    y = np.concatenate(
        [
            np.r_[np.ones(round(100 * level)), np.zeros(round(100 * (1 - level)))]
            for level in levels
        ]
    )
    consistent = consistency_bands(x, n_resamples=200, random_state=7)
    confident = confidence_bands(y, x, n_resamples=200, random_state=7)
    for key in consistent:
        np.testing.assert_array_equal(consistent[key], confident[key])


def test_confidence_bands_correct_interior_boundary_plateaus():
    """Match reliabilitydiag's interpolation across interior PAV boundaries."""
    x = np.repeat(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 20)
    y = np.r_[np.zeros(60), np.tile([0.0, 1.0], 10), np.ones(20)]
    band = confidence_bands(y, x, n_resamples=500, random_state=17)

    np.testing.assert_array_equal(
        band["prediction_values"], np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    )
    for endpoint in (band["lower"], band["upper"]):
        assert endpoint[1] == pytest.approx(
            endpoint[0] + (endpoint[3] - endpoint[0]) / 3.0
        )
        assert endpoint[2] == pytest.approx(
            endpoint[0] + 2.0 * (endpoint[3] - endpoint[0]) / 3.0
        )


@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        (np.array([0.0, 0.5]), np.array([0.2, 0.8]), "y_true"),
        (np.array([0.0, 1.0]), np.array([-0.1, 0.8]), "y_pred"),
        (np.array([0.0]), np.array([0.2, 0.8]), "same shape"),
        (np.array([]), np.array([]), "must not be empty"),
    ],
    ids=["nonbinary-outcome", "invalid-probability", "shape", "empty"],
)
def test_confidence_bands_reject_invalid_evaluation_data(y_true, y_pred, match):
    """Malformed evaluation pairs fail at the public boundary."""
    with pytest.raises(ValueError, match=match):
        confidence_bands(y_true, y_pred, n_resamples=20)


@pytest.mark.parametrize(("outcome", "expected"), [(0.0, 0.0), (1.0, 1.0)])
def test_confidence_bands_handle_certain_fitted_event_rates(outcome, expected):
    """A constant fitted boundary probability has no bootstrap uncertainty."""
    band = confidence_bands(
        np.full(40, outcome),
        np.full(40, 0.2),
        n_resamples=100,
        random_state=11,
    )
    np.testing.assert_array_equal(band["prediction_values"], [0.2])
    np.testing.assert_array_equal(band["lower"], [expected])
    np.testing.assert_array_equal(band["upper"], [expected])


@pytest.mark.parametrize("maker", [consistency_bands, confidence_bands])
def test_bands_are_reproducible(maker):
    """The same seed must give the same bands."""
    x, y = _calibrated(14, n=400)
    first = _make_band(maker, x, y, n_resamples=50, random_state=3)
    second = _make_band(maker, x, y, n_resamples=50, random_state=3)
    np.testing.assert_allclose(first["lower"], second["lower"])
    np.testing.assert_allclose(first["upper"], second["upper"])


@pytest.mark.parametrize("maker", [consistency_bands, confidence_bands])
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"level": 0.0}, "level"),
        ({"level": 1.0}, "level"),
        ({"n_resamples": 1}, "n_resamples"),
    ],
)
def test_band_arguments_are_validated(maker, kwargs, match):
    """Nonsense band arguments raise rather than return nonsense."""
    x, y = _calibrated(15, n=100)
    with pytest.raises(ValueError, match=match):
        _make_band(maker, x, y, **kwargs)


def test_wider_level_gives_wider_bands():
    """A 99% band must contain the 50% band."""
    x, _ = _calibrated(16, n=600)
    narrow = consistency_bands(x, level=0.5, n_resamples=200, random_state=0)
    wide = consistency_bands(x, level=0.99, n_resamples=200, random_state=0)
    assert np.all(wide["lower"] <= narrow["lower"] + 1e-12)
    assert np.all(wide["upper"] >= narrow["upper"] - 1e-12)


def test_consistency_bands_are_not_a_simultaneous_envelope():
    """A cheap guard on the pointwise-versus-simultaneous distinction.

    The quantitative coverage study lives in ``tests/test_monte_carlo.py``, where
    the unit of replication is the dataset and the tolerance is a Monte Carlo
    standard error. This one pins only the qualitative fact the docstring warns
    about, at a cost of a second rather than half a minute: on calibrated data the
    diagram leaves a nominal band *somewhere* almost always, so the bands must not
    be read as an envelope.

    Replaces an assertion of ``0.5 <= covered / trials <= 1.0`` at a nominal 0.9,
    which a band covering half the time would have passed.
    """
    exits = 0
    trials = 12
    for seed in range(trials):
        x, y = _calibrated(1000 + seed, n=400)
        band = consistency_bands(x, level=0.9, n_resamples=100, random_state=seed)
        diagram = corp_reliability(y, x)
        inside = (band["lower"] <= diagram.event_probabilities) & (
            diagram.event_probabilities <= band["upper"]
        )
        exits += int(not inside.all())
    assert exits >= trials - 1, (
        f"only {exits}/{trials} calibrated samples left the band; if the bands "
        "have become simultaneous, the docstring warning is now wrong"
    )
