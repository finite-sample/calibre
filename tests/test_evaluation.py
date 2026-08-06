"""Tests for the CORP evaluation stack.

The decomposition's value is that its components are guaranteed, not merely
usually true: the identity is exact, and MCB and DSC are non-negative because the
PAV solution is optimal. So these assert exact algebra rather than tolerances,
and the numbers are pinned against R in ``tests/test_r_reference.py``.
"""

from __future__ import annotations

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
    diagram = corp_reliability(x, y)
    assert np.all(np.diff(diagram.cep) >= -1e-12)


def test_cep_lies_in_the_unit_interval():
    """Recalibrated probabilities are probabilities."""
    x, y = _calibrated(1, n=800)
    diagram = corp_reliability(x, y)
    assert np.all(diagram.cep >= 0.0)
    assert np.all(diagram.cep <= 1.0)


def test_diagram_is_the_pav_fit_of_the_pooled_data():
    """The diagram is exactly weighted PAVA on tie-pooled outcomes.

    Stated as an identity rather than trusted, because every downstream number
    in this module depends on it.
    """
    x, y = _tied(2)
    diagram = corp_reliability(x, y)

    x_unique, inverse = np.unique(x, return_inverse=True)
    weight = np.bincount(inverse, minlength=x_unique.size).astype(float)
    y_mean = np.bincount(inverse, weights=y, minlength=x_unique.size) / weight

    np.testing.assert_allclose(diagram.cep, weighted_pava(y_mean, weight), atol=1e-12)
    np.testing.assert_allclose(diagram.x, x_unique)


def test_diagram_ignores_row_order():
    """Shuffling observations must not move the diagram."""
    x, y = _tied(3)
    first = corp_reliability(x, y)

    rng = np.random.default_rng(0)
    perm = rng.permutation(x.size)
    second = corp_reliability(x[perm], y[perm])

    np.testing.assert_allclose(first.x, second.x)
    np.testing.assert_allclose(first.cep, second.cep, atol=1e-12)


def test_single_distinct_forecast_gives_the_base_rate():
    """One forecast value can only be recalibrated to the observed frequency."""
    x = np.full(50, 0.3)
    y = np.concatenate([np.ones(20), np.zeros(30)])
    diagram = corp_reliability(x, y)
    assert diagram.cep.size == 1
    assert diagram.cep[0] == pytest.approx(0.4)


def test_diagram_rejects_negative_weights_even_when_ties_cancel_them():
    """Observation weights must be valid before tied forecasts are pooled."""
    x = np.array([0.2, 0.2, 0.8])
    y = np.array([0.0, 1.0, 1.0])
    w = np.array([2.0, -1.0, 1.0])

    with pytest.raises(ValueError, match="finite non-negative"):
        corp_reliability(x, y, sample_weight=w)


def test_diagram_rejects_unidentified_all_zero_weights():
    """A reliability diagram needs at least one weighted observation."""
    x, y = _calibrated(17, n=20)

    with pytest.raises(ValueError, match="at least one positive"):
        corp_reliability(x, y, sample_weight=np.zeros_like(y))


# --------------------------------------------------------------------------- #
# The score decomposition
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("seed", range(5))
def test_decomposition_identity_is_exact(score, seed):
    """mean_score == MCB - DSC + UNC, to floating point."""
    x, y = _calibrated(seed, n=1000)
    d = score_decomposition(x, y, score=score)
    assert d["mean_score"] == pytest.approx(d["MCB"] - d["DSC"] + d["UNC"], abs=1e-12)


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("seed", range(5))
def test_components_are_non_negative(score, seed):
    """MCB and DSC cannot be negative.

    Both follow from PAV optimality: no reordering of the forecasts scores better
    than the isotonic fit, and the isotonic fit is at least as good as the
    constant reference.
    """
    x, y = _calibrated(seed, n=1000)
    d = score_decomposition(x, y, score=score)
    assert d["MCB"] >= -1e-12
    assert d["DSC"] >= -1e-12


@pytest.mark.parametrize("score", SCORES)
def test_recalibrated_forecasts_have_zero_miscalibration(score):
    """Recalibrating twice changes nothing, so MCB collapses to zero.

    PAV is idempotent, which is also why an in-sample MCB for an isotonic-family
    calibrator is uninformative -- see ``test_selection.py``.
    """
    x, y = _calibrated(4, n=1500)
    diagram = corp_reliability(x, y)
    recalibrated = np.interp(x, diagram.x, diagram.cep)
    assert score_decomposition(recalibrated, y, score=score)["MCB"] == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize("score", SCORES)
def test_miscalibration_detects_a_distorted_forecast(score):
    """Squashing forecasts toward the base rate must raise MCB."""
    x, y = _calibrated(5, n=3000)
    squashed = 0.5 + 0.4 * (x - 0.5)

    honest = score_decomposition(x, y, score=score)["MCB"]
    distorted = score_decomposition(squashed, y, score=score)["MCB"]
    assert distorted > honest


@pytest.mark.parametrize("score", SCORES)
def test_constant_forecast_has_no_discrimination(score):
    """A forecast that never varies cannot discriminate."""
    y = np.concatenate([np.ones(400), np.zeros(600)])
    x = np.full(y.size, 0.4)
    d = score_decomposition(x, y, score=score)
    assert d["DSC"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("score", SCORES)
def test_uncertainty_ignores_the_forecast(score):
    """UNC depends on the outcomes alone, not on who is forecasting."""
    x, y = _calibrated(6, n=1000)
    rng = np.random.default_rng(7)
    useless = rng.uniform(0.0, 1.0, y.size)
    assert score_decomposition(x, y, score=score)["UNC"] == pytest.approx(
        score_decomposition(useless, y, score=score)["UNC"], abs=1e-12
    )


@pytest.mark.parametrize("score", SCORES)
def test_decomposition_survives_heavy_ties(score):
    """Tied forecasts must not break the identity.

    Ties are what broke SmoothedIsotonicCalibrator before 0.7.1, so anything
    built on interpolation gets checked against them.
    """
    x, y = _tied(8, decimals=1)
    d = score_decomposition(x, y, score=score)
    assert d["mean_score"] == pytest.approx(d["MCB"] - d["DSC"] + d["UNC"], abs=1e-12)
    assert d["MCB"] >= -1e-12


def test_mean_score_matches_a_direct_computation():
    """mean_score is the plain Brier score, not a transformed one."""
    x, y = _calibrated(9, n=500)
    d = score_decomposition(x, y, score="brier")
    assert d["mean_score"] == pytest.approx(float(np.mean((x - y) ** 2)), abs=1e-12)


def test_unknown_score_is_rejected():
    """Only proper scoring rules are accepted."""
    x, y = _calibrated(10, n=100)
    with pytest.raises(ValueError, match="proper scoring rule"):
        score_decomposition(x, y, score="ece")


def test_weights_replicate_duplication():
    """Weighting an observation by 2 matches listing it twice."""
    x, y = _tied(11, n=200)
    weighted = score_decomposition(x, y, sample_weight=np.full(x.size, 2.0))
    plain = score_decomposition(x, y)
    for key in plain:
        assert weighted[key] == pytest.approx(plain[key], abs=1e-12)


# --------------------------------------------------------------------------- #
# Uncertainty bands
# --------------------------------------------------------------------------- #


def test_consistency_bands_bracket_the_diagonal():
    """Under the calibration hypothesis the bands sit around the diagonal."""
    x, y = _calibrated(12, n=800)
    band = consistency_bands(x, y, n_resamples=200, random_state=0)
    assert np.all(band["lower"] <= band["upper"] + 1e-12)
    # The diagonal is the hypothesised truth, so it should mostly be inside.
    inside = (band["lower"] <= band["x"]) & (band["x"] <= band["upper"])
    assert inside.mean() > 0.8


def test_confidence_bands_bracket_the_estimate():
    """Confidence bands cluster around the CORP estimate, not the diagonal."""
    x, y = _calibrated(13, n=800)
    diagram = corp_reliability(x, y)
    band = confidence_bands(x, y, n_resamples=200, random_state=0)
    inside = (band["lower"] <= diagram.cep) & (diagram.cep <= band["upper"])
    assert inside.mean() > 0.8


@pytest.mark.parametrize("maker", [consistency_bands, confidence_bands])
def test_bands_are_reproducible(maker):
    """The same seed must give the same bands."""
    x, y = _calibrated(14, n=400)
    first = maker(x, y, n_resamples=50, random_state=3)
    second = maker(x, y, n_resamples=50, random_state=3)
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
        maker(x, y, **kwargs)


def test_wider_level_gives_wider_bands():
    """A 99% band must contain the 50% band."""
    x, y = _calibrated(16, n=600)
    narrow = consistency_bands(x, y, level=0.5, n_resamples=200, random_state=0)
    wide = consistency_bands(x, y, level=0.99, n_resamples=200, random_state=0)
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
        band = consistency_bands(x, y, level=0.9, n_resamples=100, random_state=seed)
        diagram = corp_reliability(x, y)
        inside = (band["lower"] <= diagram.cep) & (diagram.cep <= band["upper"])
        exits += int(not inside.all())
    assert exits >= trials - 1, (
        f"only {exits}/{trials} calibrated samples left the band; if the bands "
        "have become simultaneous, the docstring warning is now wrong"
    )
