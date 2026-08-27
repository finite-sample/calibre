"""Reference, contract, and realistic tests for ``make_folds``."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from calibre import make_folds


def _assert_same_folds(
    actual: list[tuple[np.ndarray, np.ndarray]],
    expected: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Assert equality of corresponding train and validation indices."""
    assert len(actual) == len(expected)
    for (actual_train, actual_validation), (expected_train, expected_validation) in zip(
        actual, expected, strict=True
    ):
        np.testing.assert_array_equal(actual_train, expected_train)
        np.testing.assert_array_equal(actual_validation, expected_validation)


def test_binary_folds_match_stratified_kfold_reference():
    """Binary targets use scikit-learn's shuffled stratified splitter exactly."""
    scores = np.linspace(0.01, 0.99, 24)
    targets = np.array([0.0] * 16 + [1.0] * 8)

    actual = make_folds(scores, targets, cv=4, random_state=17)
    expected = list(
        StratifiedKFold(n_splits=4, shuffle=True, random_state=17).split(
            scores.reshape(-1, 1), targets
        )
    )
    _assert_same_folds(actual, expected)


def test_continuous_folds_match_kfold_reference_for_unbounded_targets():
    """Finite continuous calibration targets use ordinary shuffled K-fold."""
    scores = np.linspace(-3.0, 3.0, 15)
    targets = 5.0 + 2.0 * scores

    actual = make_folds(scores, targets, cv=3, random_state=9)
    expected = list(
        KFold(n_splits=3, shuffle=True, random_state=9).split(scores.reshape(-1, 1))
    )
    _assert_same_folds(actual, expected)


def test_binary_stratification_preserves_each_class_in_every_fold():
    """The supported fold count keeps both classes in train and validation sets."""
    scores = np.linspace(0.0, 1.0, 30)
    targets = np.array([0.0] * 24 + [1.0] * 6)

    folds = make_folds(scores, targets, cv=10, random_state=0)

    assert len(folds) == 6
    validated = np.concatenate([validation for _, validation in folds])
    np.testing.assert_array_equal(np.sort(validated), np.arange(targets.size))
    for train, validation in folds:
        assert set(targets[train]) == {0.0, 1.0}
        assert set(targets[validation]) == {0.0, 1.0}
        assert not np.intersect1d(train, validation).size


def test_integer_cv_above_sample_size_is_capped_at_sample_size():
    """The documented automatic reduction produces leave-one-out-sized folds."""
    scores = np.arange(4.0)
    targets = np.arange(4.0)
    folds = make_folds(scores, targets, cv=20)
    assert len(folds) == 4
    assert all(validation.size == 1 for _, validation in folds)


@pytest.mark.parametrize("cv", [True, False, 2.9, "3", np.inf, np.nan])
def test_cv_must_be_an_integer(cv):
    """Fold counts cannot be silently coerced or truncated."""
    scores = np.arange(6.0)
    targets = np.arange(6.0)
    with pytest.raises(ValueError, match="cv must be an integer"):
        make_folds(scores, targets, cv=cv)


def test_numpy_integer_cv_is_supported():
    """NumPy integer scalars satisfy the public integer contract."""
    scores = np.arange(6.0)
    targets = np.arange(6.0)
    assert len(make_folds(scores, targets, cv=np.int64(3))) == 3


def test_at_least_two_observations_are_required():
    """A one-row sample fails at the package boundary with a stable message."""
    with pytest.raises(ValueError, match="at least two observations"):
        make_folds(np.array([0.25]), np.array([0.0]), cv=2)


def test_public_names_match_estimators_and_options_are_keyword_only():
    """The splitting API matches estimators and separates optional controls."""
    signature = inspect.signature(make_folds)
    assert list(signature.parameters) == [
        "X",
        "y",
        "cv",
        "random_state",
    ]
    assert signature.parameters["cv"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["random_state"].kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        make_folds(np.arange(4.0), np.arange(4.0), 2)


def test_same_seed_is_bitwise_reproducible_and_different_seed_moves_rows():
    """The documented random state controls shuffled fold membership."""
    scores = np.linspace(0.0, 1.0, 50)
    targets = np.linspace(-2.0, 2.0, 50)
    first = make_folds(scores, targets, cv=5, random_state=123)
    repeated = make_folds(scores, targets, cv=5, random_state=123)
    changed = make_folds(scores, targets, cv=5, random_state=124)

    _assert_same_folds(first, repeated)
    assert any(
        not np.array_equal(first_validation, changed_validation)
        for (_, first_validation), (_, changed_validation) in zip(
            first, changed, strict=True
        )
    )
