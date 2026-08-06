"""Tests for the multiclass evaluation surface.

Two things are worth stating up front, because they shape what is asserted here.

The class-wise decomposition is the binary one applied per column, so it inherits
exact guarantees rather than approximate ones: the identity holds to floating
point and the components are non-negative. Those are asserted as identities.

The regime diagnostic is a statistical claim, not an identity, so it is tested
over seeds and asserted as a separation between regimes rather than a threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibre.evaluation import score_decomposition
from calibre.multiclass import (
    TemperatureScaler,
    classwise_decomposition,
    classwise_ece,
    classwise_reliability,
    miscalibration_profile,
    top_label_ece,
)

SCORES = ["brier", "log"]


def _truth_and_labels(seed: int, n: int = 2000, J: int = 4):
    """Draw true class probabilities and labels from them.

    Parameters
    ----------
    seed
        Random seed.
    n
        Number of observations.
    J
        Number of classes.

    Returns
    -------
    tuple of ndarray
        True probability matrix and integer labels.
    """
    rng = np.random.default_rng(seed)
    truth = rng.dirichlet(np.ones(J) * 0.7, size=n)
    y = np.array([rng.choice(J, p=t) for t in truth])
    return truth, y


def _global_distortion(truth, power=2.2):
    """One sharpening applied to every class -- a temperature distortion."""
    s = truth**power
    return s / s.sum(axis=1, keepdims=True)


def _classwise_distortion(truth):
    """A different exponent per class, which no single temperature can undo."""
    powers = np.linspace(0.6, 2.4, truth.shape[1])
    s = truth ** powers[None, :]
    return s / s.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Class-wise decomposition
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("seed", range(4))
def test_identity_holds_for_every_class(score, seed):
    """mean_score == MCB - DSC + UNC, exactly, in each class."""
    truth, y = _truth_and_labels(seed)
    for part in classwise_decomposition(truth, y, score=score):
        assert part["mean_score"] == pytest.approx(
            part["MCB"] - part["DSC"] + part["UNC"], abs=1e-12
        )


@pytest.mark.parametrize("score", SCORES)
def test_components_are_non_negative_in_every_class(score):
    """PAV optimality gives MCB, DSC >= 0 per class, as in the binary case."""
    truth, y = _truth_and_labels(1)
    for part in classwise_decomposition(truth, y, score=score):
        assert part["MCB"] >= -1e-12
        assert part["DSC"] >= -1e-12


def test_two_class_case_matches_the_binary_function_exactly():
    """The multiclass path must reuse the binary one, not reimplement it."""
    truth, y = _truth_and_labels(2, J=2)
    parts = classwise_decomposition(truth, y)
    for k, part in enumerate(parts):
        direct = score_decomposition(truth[:, k], (y == k).astype(float))
        for key in direct:
            assert part[key] == pytest.approx(direct[key], abs=1e-15)


def test_decomposition_returns_one_entry_per_class():
    """Shape contract."""
    truth, y = _truth_and_labels(3, J=5)
    assert len(classwise_decomposition(truth, y)) == 5


# --------------------------------------------------------------------------- #
# The regime diagnostic
# --------------------------------------------------------------------------- #


def test_profile_separates_the_two_regimes():
    """The diagnostic must actually discriminate, or it is decoration.

    Asserted as a separation between regimes across seeds rather than against a
    fixed threshold: the threshold is a rule of thumb, the separation is the
    claim.
    """
    global_spread, classwise_spread = [], []
    for seed in range(6):
        truth, y = _truth_and_labels(seed, n=3000, J=5)
        global_spread.append(
            miscalibration_profile(_global_distortion(truth), y)["spread"]
        )
        classwise_spread.append(
            miscalibration_profile(_classwise_distortion(truth), y)["spread"]
        )

    assert max(global_spread) < min(classwise_spread), (
        f"regimes overlap: global {max(global_spread):.3f} vs "
        f"class-dependent {min(classwise_spread):.3f}"
    )


def test_profile_reading_matches_the_spread():
    """The prose must agree with the number it is derived from."""
    truth, y = _truth_and_labels(0, n=3000, J=5)

    uniform = miscalibration_profile(_global_distortion(truth), y)
    assert uniform["spread"] <= 0.25
    assert "evenly" in uniform["reading"]

    uneven = miscalibration_profile(_classwise_distortion(truth), y)
    assert uneven["spread"] > 0.25
    assert "per-class" in uneven["reading"]


def test_profile_names_the_worst_classes_in_order():
    """worst_classes must be ordered by descending miscalibration."""
    truth, y = _truth_and_labels(1, n=3000, J=5)
    profile = miscalibration_profile(_classwise_distortion(truth), y)
    ordered = profile["mcb"][profile["worst_classes"]]
    assert np.all(np.diff(ordered) <= 1e-12)


def test_profile_reports_no_miscalibration_when_there_is_none():
    """Perfectly calibrated input should not be talked into a method."""
    truth, y = _truth_and_labels(2, n=4000, J=3)
    profile = miscalibration_profile(truth, y)
    assert profile["mcb"].max() < 0.01


# --------------------------------------------------------------------------- #
# Class-wise and top-label error
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("estimator", ["debiased", "sweep"])
def test_calibrated_input_scores_low(estimator):
    """Both estimators should be near zero on calibrated predictions."""
    truth, y = _truth_and_labels(4, n=4000, J=3)
    assert classwise_ece(truth, y, estimator=estimator) < 0.05
    assert top_label_ece(truth, y, estimator=estimator) < 0.05


@pytest.mark.parametrize("estimator", ["debiased", "sweep"])
def test_distortion_raises_classwise_error(estimator):
    """A distorted model must score worse than the honest one."""
    truth, y = _truth_and_labels(5, n=4000, J=4)
    honest = classwise_ece(truth, y, estimator=estimator)
    distorted = classwise_ece(_classwise_distortion(truth), y, estimator=estimator)
    assert distorted > honest


@pytest.mark.parametrize("fn", [classwise_ece, top_label_ece])
def test_unknown_estimator_is_rejected(fn):
    """Only the two bias-aware estimators are offered."""
    truth, y = _truth_and_labels(6, n=200, J=3)
    with pytest.raises(ValueError, match="debiased"):
        fn(truth, y, estimator="plugin")


def test_classwise_reliability_is_monotone_per_class():
    """Each per-class diagram is a PAV fit, so it cannot decrease."""
    truth, y = _truth_and_labels(7, n=1500, J=4)
    diagrams = classwise_reliability(truth, y)
    assert len(diagrams) == 4
    for d in diagrams:
        assert np.all(np.diff(d.cep) >= -1e-12)


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(4))
def test_temperature_never_changes_the_predicted_class(seed):
    """The defining property: monotone in the logits, so argmax is fixed.

    Asserted exactly, on every row. This is what a user trades away per-class
    flexibility to obtain.
    """
    truth, y = _truth_and_labels(seed, n=1500, J=5)
    P = _classwise_distortion(truth)  # the regime it handles *badly*
    Q = TemperatureScaler().fit_transform(P, y)
    assert np.array_equal(Q.argmax(axis=1), P.argmax(axis=1))


def test_temperature_output_is_row_stochastic():
    """Rows must sum to 1 and stay non-negative."""
    truth, y = _truth_and_labels(0, n=1000, J=4)
    Q = TemperatureScaler().fit_transform(_global_distortion(truth), y)
    np.testing.assert_allclose(Q.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(Q >= 0.0)


def test_temperature_recovers_a_known_global_distortion():
    """Sharpening by a known power must be met with a temperature above 1."""
    truth, y = _truth_and_labels(1, n=4000, J=4)
    scaler = TemperatureScaler().fit(_global_distortion(truth, power=2.5), y)
    assert scaler.temperature_ > 1.5


def test_temperature_beats_per_class_error_in_its_own_regime():
    """On a globally distorted model it must get close to the truth.

    The measurement that justifies shipping it: error against the *known* true
    probabilities, not a proxy.
    """
    truth, y = _truth_and_labels(2, n=4000, J=4)
    P = _global_distortion(truth)
    Q = TemperatureScaler().fit_transform(P, y)
    assert np.abs(Q - truth).mean() < np.abs(P - truth).mean() / 4


def test_temperature_barely_helps_when_the_distortion_is_class_dependent():
    """Its ceiling, asserted rather than left as a caveat in prose."""
    truth, y = _truth_and_labels(3, n=4000, J=5)
    P = _classwise_distortion(truth)
    Q = TemperatureScaler().fit_transform(P, y)
    before, after = np.abs(P - truth).mean(), np.abs(Q - truth).mean()
    assert after > before / 2, (
        "temperature scaling should NOT be able to fix a class-dependent "
        f"distortion, but error fell from {before:.4f} to {after:.4f}"
    )


def test_temperature_of_one_is_the_identity():
    """A calibrated model needs no correction, so T should sit near 1."""
    truth, y = _truth_and_labels(4, n=4000, J=3)
    scaler = TemperatureScaler().fit(truth, y)
    assert scaler.temperature_ == pytest.approx(1.0, abs=0.15)


def test_transform_before_fit_raises():
    """Predicting before fitting must fail clearly."""
    with pytest.raises(AttributeError, match="not fitted"):
        TemperatureScaler().transform(np.array([[0.5, 0.5]]))


def test_transform_rejects_a_different_class_count():
    """Silently accepting the wrong width would return nonsense."""
    truth, y = _truth_and_labels(5, n=500, J=4)
    scaler = TemperatureScaler().fit(truth, y)
    with pytest.raises(ValueError, match="classes"):
        scaler.transform(np.full((10, 3), 1 / 3))


def test_invalid_bound_is_rejected():
    """A non-positive search bound has no meaning."""
    truth, y = _truth_and_labels(6, n=300, J=3)
    with pytest.raises(ValueError, match="max_log_temperature"):
        TemperatureScaler(max_log_temperature=0.0).fit(truth, y)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("P", "y", "match"),
    [
        (np.zeros((5,)), np.zeros(5), "2-D"),
        (np.empty((0, 3)), np.empty(0), "at least one sample"),
        (np.full((5, 3), 1 / 3), np.zeros(4), "rows"),
        (np.full((5, 3), 1 / 3), np.array([0, 1, 2, 3, 0]), "labels must lie"),
        (np.full((5, 3), 1 / 3), np.array([0, 1.9, 2, 1, 0]), "integers"),
        (np.full((5, 3), np.nan), np.zeros(5), "non-finite"),
        (np.array([[1.1, -0.1, 0.0]] * 5), np.zeros(5), "non-negative"),
        (np.full((5, 3), 0.2), np.zeros(5), "sum to 1"),
    ],
)
def test_malformed_input_is_rejected(P, y, match):
    """Bad probabilities and labels raise rather than produce a number."""
    with pytest.raises(ValueError, match=match):
        classwise_decomposition(P, y)


def test_temperature_transform_rejects_malformed_probability_rows():
    """Transform-time data must obey the same probability contract as fit data."""
    truth, y = _truth_and_labels(8, n=500, J=3)
    scaler = TemperatureScaler().fit(truth, y)

    with pytest.raises(ValueError, match="sum to 1"):
        scaler.transform(np.full((5, 3), 0.2))

    with pytest.raises(ValueError, match="non-negative"):
        scaler.transform(np.array([[1.1, -0.1, 0.0]] * 5))
