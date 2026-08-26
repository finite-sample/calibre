"""Tests for the shared cross-validation machinery.

The point of ``cross_val_calibrate`` is subtle enough to be worth stating: for an
isotonic-family calibrator, evaluating on the training data does not merely
flatter the model, it reports perfect calibration *by construction*, because the
calibrator and the CORP diagnostic are the same PAV projection and PAV is
idempotent. ``test_in_sample_miscalibration_is_structurally_zero`` pins that.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from calibre import (
    CenteredIsotonicCalibrator,
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SplineCalibrator,
)
from calibre.evaluation import score_decomposition
from calibre.selection import (
    cross_val_calibrate,
    make_folds,
    resolve_auto,
    select_by_cv,
)

AUTO_CALIBRATORS = [
    (NearlyIsotonicCalibrator, "lam"),
    (SplineCalibrator, "alpha"),
    (RelaxedPAVACalibrator, "epsilon"),
]


def _data(seed: int, n: int = 800):
    """Generate calibrated scores and outcomes.

    Parameters
    ----------
    seed
        Random seed.
    n
        Number of observations.

    Returns
    -------
    tuple of ndarray
        Scores and binary outcomes.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    return x, rng.binomial(1, x).astype(float)


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #


def test_folds_partition_the_data():
    """Every observation is validated exactly once."""
    x, y = _data(0, n=200)
    folds = make_folds(x, y, cv=5)
    validated = np.concatenate([v for _, v in folds])
    np.testing.assert_array_equal(np.sort(validated), np.arange(y.size))


def test_folds_never_share_rows_between_train_and_validation():
    """A row used for validation must not be in its own training split."""
    x, y = _data(1, n=200)
    for train, val in make_folds(x, y, cv=5):
        assert not set(train) & set(val)


def test_binary_folds_are_capped_by_the_rarer_class():
    """With three positives, five folds would leave folds with none."""
    y = np.zeros(200)
    y[:3] = 1.0
    x = np.linspace(0, 1, 200)
    assert len(make_folds(x, y, cv=5)) == 3


def test_binary_folds_reject_a_single_minority_observation():
    """No validation split can leave the sole minority row in training every time."""
    y = np.zeros(20)
    y[-1] = 1.0
    x = np.linspace(0, 1, y.size)
    with pytest.raises(ValueError, match="at least two observations from each class"):
        make_folds(x, y, cv=5)


def test_folds_reject_a_single_split():
    """cv=1 is not cross-validation."""
    x, y = _data(2, n=50)
    with pytest.raises(ValueError, match="cv must be at least 2"):
        make_folds(x, y, cv=1)


# --------------------------------------------------------------------------- #
# Out-of-fold calibration
# --------------------------------------------------------------------------- #


def test_out_of_fold_values_come_from_models_that_never_saw_them():
    """The leakage guarantee, checked by refitting each fold independently."""
    x, y = _data(3, n=600)
    oof = cross_val_calibrate(IsotonicCalibrator(), x, y, cv=5)

    for train, val in make_folds(x, y, cv=5, random_state=0):
        expected = IsotonicCalibrator().fit(x[train], y[train]).transform(x[val])
        np.testing.assert_allclose(oof[val], expected, atol=1e-12)


def test_out_of_fold_calibration_forwards_training_weights():
    """Each fold must fit the weighted estimator the caller requested."""
    x, y = _data(30, n=600)
    weights = np.linspace(0.2, 3.0, y.size)
    oof = cross_val_calibrate(IsotonicCalibrator(), x, y, sample_weight=weights, cv=5)

    for train, val in make_folds(x, y, cv=5, random_state=0):
        expected = (
            IsotonicCalibrator()
            .fit(x[train], y[train], sample_weight=weights[train])
            .transform(x[val])
        )
        np.testing.assert_allclose(oof[val], expected, atol=1e-12)


def test_out_of_fold_calibration_rejects_column_weights():
    """Cross-validation cannot silently flatten an invalid weight matrix."""
    x, y = _data(31, n=100)
    with pytest.raises(ValueError, match="sample_weight must be 1-dimensional"):
        cross_val_calibrate(
            IsotonicCalibrator(), x, y, sample_weight=np.ones((y.size, 1))
        )


def test_out_of_fold_covers_every_observation():
    """No NaN survives; folds partition the data."""
    x, y = _data(4, n=300)
    assert not np.isnan(cross_val_calibrate(IsotonicCalibrator(), x, y, cv=4)).any()


def test_cross_val_calibrate_leaves_the_estimator_unfitted():
    """The passed-in calibrator is cloned, not fitted in place."""
    x, y = _data(5, n=200)
    cal = IsotonicCalibrator()
    cross_val_calibrate(cal, x, y, cv=3)
    # Calibrators differ in which exception they raise when unfitted; the claim
    # here is only that the object was not fitted as a side effect.
    with pytest.raises((AttributeError, ValueError)):
        cal.transform(x)


def test_in_sample_miscalibration_is_structurally_zero():
    """In-sample MCB cannot detect miscalibration for an isotonic calibrator.

    Both the calibrator and the CORP recalibration are the same PAV projection,
    and PAV is idempotent, so scoring a fit on its own training data always
    reports MCB == 0 no matter how badly the model generalizes. The out-of-fold
    estimate is the only informative one, which is why cross_val_calibrate is a
    precondition for the evaluation stack rather than a refinement of it.
    """
    x, y = _data(6, n=1500)

    in_sample = IsotonicCalibrator().fit(x, y).transform(x)
    out_of_fold = cross_val_calibrate(IsotonicCalibrator(), x, y, cv=5)

    assert score_decomposition(in_sample, y)["MCB"] == pytest.approx(0.0, abs=1e-12)
    assert score_decomposition(out_of_fold, y)["MCB"] > 1e-4


@pytest.mark.parametrize(
    "calibrator", [IsotonicCalibrator(), CenteredIsotonicCalibrator()]
)
def test_out_of_fold_probabilities_stay_in_range(calibrator):
    """Out-of-fold output is still a probability."""
    x, y = _data(7, n=400)
    oof = cross_val_calibrate(calibrator, x, y, cv=4)
    assert np.all((oof >= 0.0) & (oof <= 1.0))


# --------------------------------------------------------------------------- #
# Grid search
# --------------------------------------------------------------------------- #


def test_select_by_cv_returns_a_member_of_the_grid():
    """Selection picks a candidate, it does not interpolate."""
    x, y = _data(8, n=400)
    grid = [0.1, 1.0, 10.0]
    best = select_by_cv(
        lambda **kw: NearlyIsotonicCalibrator(**kw), {"lam": grid}, x, y, cv=3
    )
    assert best["lam"] in grid


def test_select_by_cv_recovers_a_planted_optimum():
    """On data where heavy pooling is right, selection must choose it.

    The labels here are pure noise, so any structure the calibrator finds is
    overfitting and the strongest penalty should win.
    """
    rng = np.random.default_rng(9)
    x = rng.uniform(0.0, 1.0, 600)
    y = rng.binomial(1, 0.5, 600).astype(float)

    best = select_by_cv(
        lambda **kw: NearlyIsotonicCalibrator(**kw),
        {"lam": [0.01, 100.0]},
        x,
        y,
        cv=5,
    )
    assert best["lam"] == 100.0


def test_select_by_cv_passes_training_weights_to_fold_fits():
    """A weighted search must train each candidate with training-fold weights."""
    x, y = _data(20, n=80)
    weights = np.linspace(1.0, 3.0, y.size)
    seen = []

    class Recorder:
        def __init__(self, marker=0):
            self.marker = marker

        def fit(self, X, y, sample_weight=None):
            seen.append(None if sample_weight is None else np.asarray(sample_weight))
            return self

        def transform(self, X):
            return np.full_like(X, 0.5, dtype=float)

    select_by_cv(
        lambda **kw: Recorder(**kw), {"marker": [1]}, x, y, sample_weight=weights, cv=4
    )

    assert seen
    assert all(w is not None for w in seen)
    assert sorted(len(w) for w in seen) == sorted(
        len(train) for train, _ in make_folds(x, y, cv=4)
    )


def test_select_by_cv_does_not_invent_weights_when_none_are_supplied():
    """Unit weights are for scoring only; unsupported calibrators must not see them."""
    x, y = _data(21, n=80)
    seen = []

    class Recorder:
        def __init__(self, marker=0):
            self.marker = marker

        def fit(self, X, y, sample_weight=None):
            seen.append(sample_weight)
            return self

        def transform(self, X):
            return np.full_like(X, 0.5, dtype=float)

    select_by_cv(lambda **kw: Recorder(**kw), {"marker": [1]}, x, y, cv=4)

    assert seen
    assert all(w is None for w in seen)


@pytest.mark.parametrize(
    ("cls", "name"),
    [(SplineCalibrator, "alpha"), (RelaxedPAVACalibrator, "epsilon")],
)
def test_auto_selection_forwards_sample_weight(monkeypatch, cls, name):
    """Auto-parameter search must use the same weights as the final fit."""
    import calibre.selection as selection

    x, y = _data(22, n=80)
    weights = np.linspace(1.0, 4.0, y.size)
    captured = {}

    def fake_select_by_cv(*args, sample_weight=None, **kwargs):
        captured["sample_weight"] = sample_weight
        return {key: values[0] for key, values in args[1].items()}

    monkeypatch.setattr(selection, "select_by_cv", fake_select_by_cv)

    cls().fit(x, y, sample_weight=weights)

    np.testing.assert_allclose(captured["sample_weight"], weights)


@pytest.mark.parametrize("weighted", [False, True])
def test_nearly_isotonic_auto_cv_preserves_penalty_per_unit_weight(
    monkeypatch, weighted
):
    """Each fold must evaluate the full fit's effective regularization."""
    import calibre.calibrators.nearly_isotonic as nearly_module
    from calibre._core import aggregate_ties, weighted_pava

    x = np.linspace(0.0, 1.0, 80)
    y = (np.arange(80) % 3 == 0).astype(float)
    sample_weight = np.linspace(0.5, 3.0, y.size) if weighted else None
    explicit_weight = (
        np.ones_like(y) if sample_weight is None else np.asarray(sample_weight)
    )
    full_mass = float(np.sum(explicit_weight))

    _, y_mean, pooled_weight = aggregate_ties(x, y, sample_weight)
    isotonic = weighted_pava(y_mean, pooled_weight)
    residual = np.cumsum(pooled_weight * (y_mean - isotonic))[:-1]
    lam_max = max(0.0, float(np.max(residual, initial=0.0)))
    candidates = np.linspace(0.0, lam_max, NearlyIsotonicCalibrator.N_AUTO_LAMBDAS)
    folds = make_folds(x, y, cv=5, random_state=0)

    calls = []
    original = nearly_module.nearly_isotonic_path

    def recording_path(y, lam, sample_weight=None, return_path=False):
        weight = np.ones_like(y) if sample_weight is None else sample_weight
        calls.append((float(lam), float(np.sum(weight))))
        return original(y, lam, sample_weight, return_path)

    monkeypatch.setattr(nearly_module, "nearly_isotonic_path", recording_path)
    NearlyIsotonicCalibrator(cv=5, random_state=0).fit(
        x, y, sample_weight=sample_weight
    )

    n_cv_fits = len(candidates) * len(folds)
    assert len(calls) == n_cv_fits + 1
    for candidate_index, candidate in enumerate(candidates):
        for fold_index, (train, _) in enumerate(folds):
            observed_lam, observed_mass = calls[
                candidate_index * len(folds) + fold_index
            ]
            expected_mass = float(np.sum(explicit_weight[train]))
            assert observed_mass == pytest.approx(expected_mass)
            assert observed_lam / observed_mass == pytest.approx(candidate / full_mass)


@pytest.mark.parametrize(
    ("sample_weight", "match"),
    [
        (np.ones(79), "same shape"),
        (np.r_[np.ones(79), np.nan], "finite non-negative"),
        (np.r_[np.ones(79), -1.0], "finite non-negative"),
        (np.zeros(80), "at least one positive"),
    ],
)
def test_select_by_cv_rejects_malformed_sample_weights(sample_weight, match):
    """Malformed weights should fail before fitting candidates or scoring folds."""
    x, y = _data(23, n=80)

    with pytest.raises(ValueError, match=match):
        select_by_cv(
            lambda **kw: IsotonicCalibrator(**kw),
            {"out_of_bounds": ["clip"]},
            x,
            y,
            sample_weight=sample_weight,
            cv=4,
        )


def test_zero_weight_target_does_not_change_auto_scoring_or_selection():
    """A row carrying no mass cannot change the scoring domain or winner."""

    class Constant:
        def __init__(self, probability):
            self.probability = probability

        def fit(self, X, y, sample_weight=None):
            return self

        def transform(self, X):
            return np.full(len(X), self.probability)

    x = np.arange(9, dtype=float)
    y = np.array([0.0] * 6 + [1.0] * 2 + [2.0])
    weights = np.r_[np.ones(8), 0.0]
    grid = {"probability": [0.01, 0.5]}

    baseline = select_by_cv(
        Constant, grid, x[:8], y[:8], cv=2, scoring="auto", max_cv_samples=None
    )
    weighted = select_by_cv(
        Constant,
        grid,
        x,
        y,
        sample_weight=weights,
        cv=2,
        scoring="auto",
        max_cv_samples=None,
    )
    explicit_log = select_by_cv(
        Constant,
        grid,
        x,
        y,
        sample_weight=weights,
        cv=2,
        scoring="log_loss",
        max_cv_samples=None,
    )

    assert baseline == weighted == explicit_log == {"probability": 0.5}


def test_select_by_cv_rejects_calibration_error_as_a_criterion():
    """ECE is biased, so it is not offered as a selection criterion."""
    x, y = _data(10, n=200)
    with pytest.raises(ValueError, match="proper scoring rule"):
        select_by_cv(
            lambda **kw: NearlyIsotonicCalibrator(**kw),
            {"lam": [1.0]},
            x,
            y,
            cv=3,
            scoring="ece",
        )


def test_select_by_cv_rejects_an_empty_grid():
    """An empty grid has no answer."""
    x, y = _data(11, n=100)
    with pytest.raises(ValueError, match="at least one candidate"):
        select_by_cv(lambda **kw: NearlyIsotonicCalibrator(**kw), {}, x, y, cv=3)


@pytest.mark.parametrize("scoring", ["log_loss", "brier", "auto"])
def test_both_proper_scoring_rules_work(scoring):
    """Either proper rule or domain-aware selection may be used."""
    x, y = _data(12, n=300)
    best = select_by_cv(
        lambda **kw: NearlyIsotonicCalibrator(**kw),
        {"lam": [0.1, 10.0]},
        x,
        y,
        cv=3,
        scoring=scoring,
    )
    assert "lam" in best


def test_log_loss_rejects_targets_outside_its_domain():
    """Bernoulli log loss is not a scoring rule for unbounded targets."""
    x = np.linspace(0.0, 1.0, 30)
    y = np.linspace(-1.0, 2.0, 30)

    with pytest.raises(ValueError, match=r"log_loss.*targets in \[0, 1\]"):
        select_by_cv(
            lambda **kw: SplineCalibrator(link="identity", clip_output=False, **kw),
            {"alpha": [0.0, 1.0]},
            x,
            y,
            cv=3,
            scoring="log_loss",
        )

    assert "alpha" in select_by_cv(
        lambda **kw: SplineCalibrator(link="identity", clip_output=False, **kw),
        {"alpha": [0.0, 1.0]},
        x,
        y,
        cv=3,
        scoring="brier",
    )

    assert "alpha" in select_by_cv(
        lambda **kw: SplineCalibrator(link="identity", clip_output=False, **kw),
        {"alpha": [0.0, 1.0]},
        x,
        y,
        cv=3,
        scoring="auto",
    )


# --------------------------------------------------------------------------- #
# "auto" defaults on the calibrators
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("cls", "name"), AUTO_CALIBRATORS)
def test_auto_is_the_default(cls, name):
    """The pure bias-variance knobs default to selection, not to a guess."""
    assert cls().get_params()[name] == "auto"


@pytest.mark.parametrize(("cls", "name"), AUTO_CALIBRATORS)
def test_auto_records_its_choice_without_mutating_the_parameter(cls, name):
    """The resolved value lands on a trailing-underscore attribute.

    Writing it back onto the constructor argument would break get_params
    round-tripping and therefore sklearn.base.clone -- the 0.7.1 contract bug.
    """
    x, y = _data(13, n=400)
    cal = cls().fit(x, y)

    resolved = getattr(cal, f"{name}_")
    assert isinstance(resolved, float)
    assert cal.get_params()[name] == "auto"
    assert clone(cal).get_params() == cls().get_params()


@pytest.mark.parametrize(("cls", "name"), AUTO_CALIBRATORS)
def test_a_pinned_value_is_used_verbatim(cls, name):
    """Passing a number skips selection entirely."""
    x, y = _data(14, n=300)
    cal = cls(**{name: 0.05}).fit(x, y)
    assert getattr(cal, f"{name}_") == pytest.approx(0.05)


@pytest.mark.parametrize(("cls", "name"), AUTO_CALIBRATORS)
def test_a_bad_string_is_rejected(cls, name):
    """Only "auto" is a valid string."""
    x, y = _data(15, n=200)
    with pytest.raises(ValueError, match=f'{name} must be.*"auto"'):
        cls(**{name: "automatic"}).fit(x, y)


@pytest.mark.parametrize(("cls", "name"), AUTO_CALIBRATORS)
def test_a_negative_value_is_rejected(cls, name):
    """Negative penalties are meaningless."""
    x, y = _data(16, n=200)
    with pytest.raises(ValueError, match=f"{name} must be finite and non-negative"):
        cls(**{name: -1.0}).fit(x, y)


@pytest.mark.parametrize(("cls", "name"), AUTO_CALIBRATORS)
def test_auto_fits_are_deterministic(cls, name):
    """The same data must select the same value twice."""
    x, y = _data(17, n=400)
    first = getattr(cls().fit(x, y), f"{name}_")
    second = getattr(cls().fit(x, y), f"{name}_")
    assert first == second


def test_min_slope_pins_epsilon_rather_than_searching_against_it():
    """min_slope and epsilon conflict, so auto-epsilon defers to min_slope.

    Searching epsilon while min_slope is set would let selection contradict the
    caller's stated intent, and any non-zero choice would then raise.
    """
    x, y = _data(18, n=300)
    cal = RelaxedPAVACalibrator(min_slope=0.01).fit(x, y)
    assert cal.epsilon_ == 0.0


def test_explicit_epsilon_and_min_slope_still_conflict():
    """Setting both by hand remains an error."""
    x, y = _data(19, n=200)
    with pytest.raises(ValueError, match="opposite directions"):
        RelaxedPAVACalibrator(epsilon=0.05, min_slope=0.01).fit(x, y)


def test_resolve_auto_passes_numbers_through():
    """A number needs no data and no search."""
    empty = np.array([])
    assert resolve_auto(0.25, "alpha", [1.0], lambda **kw: None, empty, empty) == 0.25
