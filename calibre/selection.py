"""Cross-validation shared by every calibrator.

Until 0.8.0 the only cross-validation in the package lived inside
``SplineCalibrator`` as a private method, and every other calibrator shipped a
fixed default for parameters that have no principled fixed value. ``lam=1.0`` and
``alpha=0.1`` are not neutral: they are bias-variance settings, and the right one
depends on the data. This module makes the machinery shared, so a calibrator can
select its own smoothing rather than have it guessed.

Two things live here:

``select_by_cv``
    Grid search returning the winning parameters, for calibrators to call during
    ``fit``.
``cross_val_calibrate``
    Out-of-fold calibrated probabilities, for *evaluating* a calibrator without
    scoring it on data it was fit on.

Selection scores on a strictly proper scoring rule -- log-loss by default, Brier
available. Not on expected calibration error: ECE is a biased estimator whose
value depends on the binning, so selecting on it optimizes binning artifacts.
Use :mod:`calibre.evaluation` to understand a fitted model.
"""

from __future__ import annotations

import logging
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

from .utils import check_arrays

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "cross_val_calibrate",
    "make_folds",
    "resolve_auto",
    "select_by_cv",
]

_LOG_EPS = np.finfo(float).eps


def _log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Pointwise log loss.

    Args:
        y: Targets in ``[0, 1]``.
        p: Predicted probabilities.

    Returns:
        ndarray: Per-observation loss.
    """
    p = np.clip(p, _LOG_EPS, 1.0 - _LOG_EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Pointwise squared error.

    Args:
        y: Targets.
        p: Predicted probabilities.

    Returns:
        ndarray: Per-observation loss.
    """
    return (p - y) ** 2


_SCORINGS = {"log_loss": _log_loss, "brier": _brier}


def _sample_weight_or_ones(
    y: np.ndarray, sample_weight: np.ndarray | None
) -> np.ndarray:
    """Validate observation weights and return an explicit array."""
    if sample_weight is None:
        return np.ones_like(y)
    w = np.asarray(sample_weight, dtype=float)
    if w.ndim != 1:
        raise ValueError("sample_weight must be 1-dimensional")
    if w.shape != y.shape:
        raise ValueError("sample_weight must have the same shape as y")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("sample_weight must contain finite non-negative values")
    if np.sum(w) <= 0.0:
        raise ValueError("sample_weight must contain at least one positive weight")
    return w


def _resolve_scoring(scoring: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Look up a selection criterion by name.

    Args:
        scoring: ``"log_loss"`` or ``"brier"``.

    Returns:
        callable: Pointwise loss, lower being better.

    Raises:
        ValueError: If the name is unknown. ECE is deliberately absent.
    """
    try:
        return _SCORINGS[scoring]
    except KeyError:
        raise ValueError(
            f"scoring must be one of {sorted(_SCORINGS)}, got {scoring!r}. "
            "Selection requires a strictly proper scoring rule; calibration "
            "error is biased and depends on its binning."
        ) from None


def make_folds(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    random_state: int | None = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build cross-validation folds, stratifying binary targets.

    Args:
        X: Uncalibrated scores.
        y: Targets: binary labels, or values in ``[0, 1]``.
        cv: Requested number of folds. Reduced when the data cannot support it.
        random_state: Seed for the shuffle.

    Returns:
        list of (ndarray, ndarray): ``(train_index, validation_index)`` pairs.

    Raises:
        ValueError: If ``cv`` is below 2.

    Notes:
        With binary targets the fold count is capped by the rarer class, so a rare
        positive appears in every training split rather than leaving a fold with no
        positives at all.

    Examples:
        >>> import numpy as np
        >>> from calibre.selection import make_folds
        >>> x = np.linspace(0, 1, 20)
        >>> y = (x > 0.5).astype(float)
        >>> len(make_folds(x, y, cv=4))
        4
    """
    X, y = check_arrays(X, y)
    if cv < 2:
        raise ValueError(f"cv must be at least 2, got {cv}")

    from sklearn.model_selection import KFold, StratifiedKFold

    is_binary = bool(np.all((y == 0.0) | (y == 1.0)))
    n_splits = cv
    if is_binary:
        counts = np.bincount(y.astype(int), minlength=2)
        present = counts[counts > 0]
        if present.size == 2 and present.min() < 2:
            raise ValueError(
                "binary cross-validation requires at least two observations "
                "from each class"
            )
        n_splits = min(n_splits, int(counts[counts > 0].min()))
    n_splits = int(max(2, min(n_splits, y.size)))

    if is_binary:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        return list(splitter.split(X.reshape(-1, 1), y))
    return list(
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
            X.reshape(-1, 1)
        )
    )


def _grid_points(param_grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into candidate dictionaries.

    Args:
        param_grid: Mapping from parameter name to candidate values.

    Returns:
        list of dict: One dictionary per combination, in a deterministic order.
    """
    names = list(param_grid)
    return [
        dict(zip(names, values, strict=True))
        for values in product(*param_grid.values())
    ]


def select_by_cv(
    factory: Callable[..., Any],
    param_grid: dict[str, Sequence[Any]],
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    cv: int = 5,
    scoring: str = "log_loss",
    max_cv_samples: int | None = 20_000,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Choose parameters by cross-validation on a proper scoring rule.

    The winning configuration is *not* fitted here. The caller refits it on all
    the data, which matters: keeping a fold's model would ship an estimator that
    had seen only ``(cv-1)/cv`` of the sample.

    Args:
        factory: Called with a candidate's keyword arguments, returning an
            unfitted calibrator.
        param_grid: Mapping from parameter name to candidate values.
        X: Uncalibrated scores.
        y: Targets.
        sample_weight: Non-negative per-observation weights.
        cv: Number of folds.
        scoring: ``"log_loss"`` (default), ``"brier"``, or ``"auto"``.
            Automatic scoring uses log loss for targets in ``[0, 1]`` and
            squared error otherwise.
        max_cv_samples: Subsample above this size before searching. Selection
            only has to rank candidates, so its cost is bounded; None
            disables.
        random_state: Seed for folds and subsampling.

    Returns:
        dict: The winning parameters, ready to splat into ``factory``.

    Raises:
        ValueError: If the grid is empty, ``scoring`` is unknown, log loss is
            requested for targets outside ``[0, 1]``, or every candidate failed.

    Examples:
        >>> import numpy as np
        >>> from calibre import NearlyIsotonicCalibrator
        >>> from calibre.selection import select_by_cv
        >>> rng = np.random.default_rng(0)
        >>> x = rng.uniform(0, 1, 300)
        >>> y = rng.binomial(1, x).astype(float)
        >>> best = select_by_cv(
        ...     lambda **kw: NearlyIsotonicCalibrator(**kw),
        ...     {"lam": [0.1, 1.0, 10.0]},
        ...     x,
        ...     y,
        ...     cv=3,
        ... )
        >>> sorted(best)
        ['lam']
    """
    X, y = check_arrays(X, y)
    # Checked before expanding: itertools.product() of no iterables yields one
    # empty tuple, so an empty grid would otherwise look like a single candidate
    # carrying the defaults rather than like the mistake it is.
    if not param_grid or any(len(v) == 0 for v in param_grid.values()):
        raise ValueError("param_grid must contain at least one candidate")
    candidates = _grid_points(param_grid)

    w = _sample_weight_or_ones(y, sample_weight)

    positive = w > 0.0
    X, y, w = X[positive], y[positive], w[positive]
    targets_are_probabilities = bool(np.all((y >= 0.0) & (y <= 1.0)))
    resolved_scoring = (
        ("log_loss" if targets_are_probabilities else "brier")
        if scoring == "auto"
        else scoring
    )
    loss = _resolve_scoring(resolved_scoring)
    if resolved_scoring == "log_loss" and not targets_are_probabilities:
        raise ValueError(
            'scoring="log_loss" requires targets in [0, 1]; use '
            'scoring="brier" for an unbounded identity-link target'
        )

    if max_cv_samples is not None and y.size > max_cv_samples:
        rng = np.random.default_rng(random_state)
        keep = rng.choice(y.size, size=max_cv_samples, replace=False)
        X, y, w = X[keep], y[keep], w[keep]

    folds = make_folds(X, y, cv=cv, random_state=random_state)

    best_score = np.inf
    best_params: dict[str, Any] | None = None
    for params in candidates:
        total = 0.0
        weight_total = 0.0
        failed = False
        for train_idx, val_idx in folds:
            try:
                model = factory(**params)
                if sample_weight is None:
                    model.fit(X[train_idx], y[train_idx])
                else:
                    model.fit(
                        X[train_idx],
                        y[train_idx],
                        sample_weight=w[train_idx],
                    )
                pred = model.transform(X[val_idx])
            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                logger.debug("fold failed at %s: %s", params, exc)
                failed = True
                break
            total += float(np.sum(w[val_idx] * loss(y[val_idx], pred)))
            weight_total += float(np.sum(w[val_idx]))

        if failed or weight_total <= 0.0:
            continue
        score = total / weight_total
        if score < best_score:
            best_score, best_params = score, params

    if best_params is None:
        raise ValueError(
            "every candidate failed to fit; check the data or narrow param_grid"
        )
    return best_params


def cross_val_calibrate(
    calibrator: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    cv: int = 5,
    random_state: int | None = 0,
) -> np.ndarray:
    """Return out-of-fold calibrated probabilities.

    Each value is produced by a calibrator that never saw that observation, so
    the result can be scored without the optimism of measuring a fit on its own
    training data. This is the honest input to
    :func:`calibre.evaluation.score_decomposition`.

    Args:
        calibrator: An unfitted calibrator. Cloned per fold, so the object
            passed in is left untouched.
        X: Uncalibrated scores.
        y: Targets.
        sample_weight: Non-negative per-observation weights used to fit each
            training fold. Validation rows retain their original positions,
            including rows with zero weight. Zero-weight rows do not affect fold
            stratification or fitting.
        cv: Number of folds.
        random_state: Seed for the folds.

    Returns:
        ndarray of shape (n_samples,): Out-of-fold calibrated probabilities,
            in the input's order.

    Raises:
        ValueError: If fewer than two observations have positive weight.
        RuntimeError: If the folds did not cover every observation, which would
            leave some rows with no out-of-fold prediction at all.

    Notes:
        A calibrator that selects its own parameters runs that selection inside each
        training fold, which makes this a nested cross-validation and keeps the
        reported performance honest for a tuned model.

    Examples:
        >>> import numpy as np
        >>> from calibre import IsotonicCalibrator
        >>> from calibre.selection import cross_val_calibrate
        >>> rng = np.random.default_rng(0)
        >>> x = rng.uniform(0, 1, 200)
        >>> y = rng.binomial(1, x).astype(float)
        >>> oof = cross_val_calibrate(IsotonicCalibrator(), x, y, cv=4)
        >>> oof.shape
        (200,)
    """
    from sklearn.base import clone

    X, y = check_arrays(X, y)
    w = _sample_weight_or_ones(y, sample_weight)
    if sample_weight is None:
        folds = make_folds(X, y, cv=cv, random_state=random_state)
    else:
        positive_idx = np.flatnonzero(w > 0.0)
        if positive_idx.size < 2:
            raise ValueError(
                "cross-validation requires at least two positive-weight observations"
            )
        effective_folds = make_folds(
            X[positive_idx], y[positive_idx], cv=cv, random_state=random_state
        )
        zero_chunks = np.array_split(np.flatnonzero(w == 0.0), len(effective_folds))
        folds = [
            (
                positive_idx[train_idx],
                np.concatenate((positive_idx[val_idx], zero_idx)),
            )
            for (train_idx, val_idx), zero_idx in zip(
                effective_folds, zero_chunks, strict=True
            )
        ]

    out = np.full(y.shape, np.nan, dtype=float)
    for train_idx, val_idx in folds:
        # Annotated because sklearn's clone() is overloaded for containers, so a
        # calibrator typed as Any resolves to a union without a fit method.
        model: Any = clone(calibrator)
        if sample_weight is None:
            model.fit(X[train_idx], y[train_idx])
        else:
            model.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
        out[val_idx] = model.transform(X[val_idx])

    if np.isnan(out).any():  # pragma: no cover - folds partition the data
        raise RuntimeError("folds did not cover every observation")
    return out


def resolve_auto(
    value: float | str,
    name: str,
    grid: Sequence[Any],
    factory: Callable[..., Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "log_loss",
    random_state: int | None = 0,
    minimum: float = 0.0,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Resolve one parameter that may be a number or ``"auto"``.

    The shared implementation behind every calibrator's ``"auto"`` default, so
    the selection rules live in one place rather than being restated per
    estimator.

    Args:
        value: The constructor argument: a number, or ``"auto"``.
        name: Parameter name, used in the grid and in error messages.
        grid: Candidate values searched when ``value`` is ``"auto"``.
        factory: Called with ``{name: candidate}`` to build an unfitted calibrator.
        X: Uncalibrated scores.
        y: Targets.
        cv: Number of folds.
        scoring: Selection criterion, a proper scoring rule.
        random_state: Seed for folds.
        minimum: Smallest permitted numeric value.
        sample_weight: Non-negative per-observation weights used during selection.

    Returns:
        float: The resolved value. Callers store it on a trailing-underscore
            attribute; writing it back onto the constructor argument would
            break ``get_params`` round-tripping and therefore ``clone``.

    Raises:
        ValueError: If ``value`` is a string other than ``"auto"``, or a
            number below ``minimum``.

    Examples:
        >>> import numpy as np
        >>> from calibre.selection import resolve_auto
        >>> resolve_auto(
        ...     0.5, "alpha", [0.1, 1.0], lambda **kw: None, np.array([]), np.array([])
        ... )
        0.5
    """
    # "non-negative" reads better than ">= 0.0" and is the wording these
    # calibrators have always used.
    limit = "non-negative" if minimum == 0.0 else f">= {minimum}"

    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f'{name} must be {limit} or "auto", got {value!r}')
        best = select_by_cv(
            factory,
            {name: list(grid)},
            X,
            y,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            sample_weight=sample_weight,
        )
        return float(best[name])

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be {limit} or "auto", got {value!r}') from exc
    if not np.isfinite(numeric) or numeric < minimum:
        raise ValueError(f"{name} must be finite and {limit}, got {value}")
    return numeric
