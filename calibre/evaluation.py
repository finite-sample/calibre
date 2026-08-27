r"""CORP evaluation: reliability diagrams and score decompositions.

A binned reliability diagram makes the analyst choose the bins, and the choice
changes the picture: Dimitriadis, Gneiting & Jordan show the same forecasts
looking well calibrated at 9 bins and badly calibrated at 10. CORP removes the
choice. The conditional event probabilities are estimated by isotonic regression
via the pool-adjacent-violators algorithm, which fixes the number and position of
the bins optimally and automatically, with no tuning parameter.

That estimator is :func:`calibre._core.weighted_pava`, which this package already
owns and pins against R. Everything here is built from it.

The score decomposition splits the mean score of any proper scoring rule into

.. math::
    \bar{S}_X = \underbrace{(\bar{S}_X - \bar{S}_C)}_{\mathrm{MCB}}
              - \underbrace{(\bar{S}_R - \bar{S}_C)}_{\mathrm{DSC}}
              + \underbrace{\bar{S}_R}_{\mathrm{UNC}}

where :math:`\bar{S}_C` is the mean score of the PAV-recalibrated forecasts and
:math:`\bar{S}_R` that of the constant reference forecast :math:`\bar{y}`.
Choosing those two specifically is what makes it the *CORP* decomposition, and it
is what guarantees ``MCB >= 0`` and ``DSC >= 0``.

Reference
---------
Dimitriadis, Gneiting & Jordan (2021), "Stable reliability diagrams for
probabilistic classifiers", *PNAS* 118(8), e2016191118. Pinned against that
paper's R implementation (``reliabilitydiag``) in ``tests/test_r_reference.py``.
"""

from __future__ import annotations

import warnings
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import DegenerateDataWarning
from scipy.stats import bootstrap as scipy_bootstrap

from ._core import PiecewiseLinear, StepFunction, aggregate_ties, weighted_pava
from .utils.validation import (
    _validate_binary_probability_metric_inputs,
    _validate_probability_vector,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "ReliabilityDiagram",
    "bootstrap_ci",
    "confidence_bands",
    "consistency_bands",
    "corp_reliability",
    "score_decomposition",
]

# Log score is unbounded at 0 and 1, so probabilities are clipped before taking
# logs. Matches sklearn's log_loss default.
_LOG_EPS = np.finfo(float).eps


def _brier(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pointwise Brier score ``(y_pred - y_true)**2``.

    Args:
        y_true: Binary outcomes.
        y_pred: Forecast probabilities.

    Returns:
        ndarray: Per-observation score.
    """
    return (y_pred - y_true) ** 2


def _log_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pointwise logarithmic score for binary probability forecasts.

    Args:
        y_true: Binary outcomes.
        y_pred: Forecast probabilities.

    Returns:
        ndarray: Per-observation score.
    """
    y_pred = np.clip(y_pred, _LOG_EPS, 1.0 - _LOG_EPS)
    return -y_true * np.log(y_pred) - (1.0 - y_true) * np.log1p(-y_pred)


_SCORES = {"brier": _brier, "log": _log_score}


def _resolve_score(
    score: str | Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Resolve a named or user-supplied pointwise scoring rule.

    Args:
        score: ``"brier"``, ``"log"``, or a vectorized scoring function
            accepting ``(y_true, y_pred)``.

    Returns:
        callable: Pointwise scoring function.

    Raises:
        ValueError: If ``score`` is neither a supported name nor callable.
    """
    if callable(score):
        return score
    if not isinstance(score, str):
        raise ValueError(
            f"score must be one of {sorted(_SCORES)} or a callable, got {score!r}"
        )
    try:
        return _SCORES[score]
    except KeyError:
        raise ValueError(
            f"score must be one of {sorted(_SCORES)} or a callable, got {score!r}. "
            "The decomposition requires a proper scoring rule."
        ) from None


def _score_values(
    score: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Evaluate and validate a vectorized pointwise scoring rule."""
    try:
        values = np.asarray(score(y_true, y_pred), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "score callable must return one finite numeric value per observation"
        ) from error
    if values.shape != y_true.shape or not np.all(np.isfinite(values)):
        raise ValueError(
            "score callable must return one finite numeric value per observation"
        )
    return values


class ReliabilityDiagram:
    """A fitted CORP reliability diagram.

    Attributes:
        prediction_values: The distinct forecast values, ascending.
        event_probabilities: PAV-recalibrated conditional event probability at each
            forecast value.
        prediction_weights: Evaluation mass carrying each forecast value.

    Notes:
        Points where the diagram is flat are the CORP bins: the PAV algorithm chose
        them, so no bin count needs to be supplied and none can be tuned to flatter
        the forecaster.
    """

    __slots__ = (
        "event_probabilities",
        "prediction_values",
        "prediction_weights",
    )

    def __init__(
        self,
        prediction_values: np.ndarray,
        event_probabilities: np.ndarray,
        prediction_weights: np.ndarray,
    ) -> None:
        """Store the diagram's three parallel arrays.

        Args:
            prediction_values: Forecast value at each bin.
            event_probabilities: Conditional event probability at each bin.
            prediction_weights: Evaluation mass behind each bin.
        """
        self.prediction_values = prediction_values
        self.event_probabilities = event_probabilities
        self.prediction_weights = prediction_weights

    def __call__(self, new_predictions: np.ndarray) -> np.ndarray:
        """Recalibrate new forecast values through the diagram.

        Args:
            new_predictions: Forecast values to recalibrate.

        Returns:
            ndarray: Recalibrated probabilities.
        """
        return self.as_function()(np.asarray(new_predictions, dtype=float).ravel())

    def as_function(self) -> PiecewiseLinear | StepFunction:
        """Return the recalibration map as a callable.

        Returns:
            PiecewiseLinear or StepFunction: Piecewise-linear interpolation of
                the diagram, matching the paper's display convention. A single
                distinct forecast value gives a step function, since there is
                nothing to interpolate between.
        """
        if self.prediction_values.size == 1:
            return StepFunction(self.prediction_values, self.event_probabilities)
        return PiecewiseLinear(self.prediction_values, self.event_probabilities)

    def plot(self, **kwargs: Any) -> Any:
        """Draw this diagram.

        Convenience wrapper around
        :func:`calibre.plots.plot_reliability_diagram`, which documents every
        keyword. Needs matplotlib: ``pip install 'calibre[plots]'``.

        Args:
            **kwargs: Passed straight through to
                :func:`~calibre.plots.plot_reliability_diagram` -- ``ax``,
                ``bands``, ``density``, ``style``, ``diagonal``, ``color``
                and ``label``.

        Returns:
            matplotlib.axes.Axes: The axes drawn on.

        Examples:
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from calibre import corp_reliability
            >>> rng = np.random.default_rng(0)
            >>> x = rng.uniform(0, 1, 200)
            >>> y = rng.binomial(1, x).astype(float)
            >>> ax = corp_reliability(y, x).plot(density="none")
            >>> ax.get_xlabel()
            'forecast probability'
        """
        # Imported here, not at module scope, so that matplotlib stays optional
        # and importing calibre.evaluation stays free.
        from .plots.reliability import plot_reliability_diagram

        return plot_reliability_diagram(self, **kwargs)

    def __repr__(self) -> str:
        """Return a short summary of the diagram."""
        return (
            f"ReliabilityDiagram(n_points={self.prediction_values.size}, "
            f"n_bins={np.unique(self.event_probabilities).size})"
        )


def corp_reliability(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> ReliabilityDiagram:
    """Estimate conditional event probabilities by isotonic regression.

    This is the CORP reliability diagram: the PAV-recalibrated forecast
    probabilities plotted against the original forecast values. Unlike a binned
    diagram it needs no bin count, because PAV determines the number and position
    of the flat segments itself.

    Args:
        y_true: Binary outcomes in ``{0, 1}`` from evaluation data not used to fit
            the forecaster.
        y_pred: Forecast probabilities in ``[0, 1]`` on the same evaluation data.
        sample_weight: Non-negative frequency or evaluation-mass weights. Zero-weight
            observations are ignored. Defaults to 1.

    Returns:
        ReliabilityDiagram: The fitted diagram.

    Examples:
        >>> import numpy as np
        >>> from calibre.evaluation import corp_reliability
        >>> x = np.array([0.2, 0.4, 0.6, 0.8])
        >>> y = np.array([0.0, 1.0, 0.0, 1.0])
        >>> diagram = corp_reliability(y, x)

        The middle pair violates monotonicity, so PAV pools it to its mean:

        >>> diagram.event_probabilities
        array([0. , 0.5, 0.5, 1. ])

    Notes:
        The unweighted estimator follows Dimitriadis, Gneiting & Jordan (2021) and
        is pinned against their ``reliabilitydiag`` R package. ``sample_weight`` is
        calibre's frequency-weight extension: integer weights give exactly the same
        diagram as literal row replication.

    See Also:
        score_decomposition : The score decomposition built on this estimate.
        calibre.CenteredIsotonicCalibrator : Recalibration, rather than diagnosis.
    """
    y_true, y_pred, weights = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight
    )

    # Tied forecasts are one point, carrying their pooled weight.
    prediction_values, y_mean, weights = aggregate_ties(y_pred, y_true, weights)
    event_probabilities = weighted_pava(y_mean, weights)
    return ReliabilityDiagram(prediction_values, event_probabilities, weights)


def score_decomposition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    score: str | Callable[[np.ndarray, np.ndarray], np.ndarray] = "brier",
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Decompose a mean score into miscalibration, discrimination and uncertainty.

    Returns the CORP decomposition ``mean_score = miscalibration - discrimination +
    uncertainty``, where the calibrated forecasts are the PAV-recalibrated
    probabilities and the reference forecast is the marginal event frequency.

    Miscalibration is what recalibration would save you, discrimination is what your
    forecasts buy over always predicting the base rate, and uncertainty is the
    difficulty of the problem, which no forecaster can change.

    Args:
        y_true: Binary outcomes in ``{0, 1}`` from evaluation data not used to fit
            the forecaster or calibrator.
        y_pred: Forecast probabilities in ``[0, 1]`` on the same evaluation data.
        score: Proper scoring rule: ``"brier"`` (default), ``"log"``, or a
            vectorized callable accepting ``(y_true, y_pred)`` and returning one
            finite loss per observation. The caller is responsible for ensuring a
            custom score is proper for binary probability forecasts.
        sample_weight: Non-negative frequency or evaluation-mass weights. Zero-weight
            observations are ignored. Defaults to 1.

    Returns:
        dict: ``mean_score``, ``miscalibration``, ``discrimination``, and
            ``uncertainty``. Miscalibration and discrimination are non-negative for
            proper scoring rules, guaranteed by the optimality of the PAV solution.

    Examples:
        >>> import numpy as np
        >>> from calibre.evaluation import score_decomposition
        >>> rng = np.random.default_rng(0)
        >>> x = rng.uniform(0, 1, 2000)
        >>> y = rng.binomial(1, x).astype(float)

        These forecasts are calibrated by construction, so miscalibration is small
        while discrimination is substantial:

        >>> d = score_decomposition(y, x)
        >>> bool(d["miscalibration"] < 0.01), bool(d["discrimination"] > 0.05)
        (True, True)

        The identity holds exactly:

        >>> reconstructed = (
        ...     d["miscalibration"]
        ...     - d["discrimination"]
        ...     + d["uncertainty"]
        ... )
        >>> bool(abs(d["mean_score"] - reconstructed) < 1e-12)
        True

    Warnings:
        Evaluate held-out or out-of-fold predictions. For an isotonic-family
        calibrator, in-sample miscalibration is zero by construction because the
        calibrator and this diagnostic use the same idempotent PAV projection.

    Notes:
        This follows Dimitriadis, Gneiting & Jordan (2021) and is pinned against
        their ``reliabilitydiag`` R package for the Brier score. Evaluation weights
        are calibre's extension; integer weights are equivalent to row replication.

    See Also:
        corp_reliability : The recalibration this decomposition is built on.
    """
    scoring = _resolve_score(score)
    y_true, y_pred, weights = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight
    )
    total = weights.sum()

    def mean_score(forecast: np.ndarray) -> float:
        values = _score_values(scoring, y_true, forecast)
        return float(np.sum(values * weights) / total)

    # Recalibrated forecasts, evaluated at each observation's own forecast value.
    diagram = corp_reliability(y_true, y_pred, sample_weight=weights)
    recalibrated = np.interp(
        y_pred, diagram.prediction_values, diagram.event_probabilities
    )

    # Reference forecast: the marginal event frequency, held constant.
    reference = float(np.sum(y_true * weights) / total)

    s_x = mean_score(y_pred)
    s_c = mean_score(recalibrated)
    s_r = mean_score(np.full_like(y_true, reference))

    return {
        "mean_score": s_x,
        "miscalibration": s_x - s_c,
        "discrimination": s_r - s_c,
        "uncertainty": s_r,
    }


def _band_from_draws(
    draws: np.ndarray, level: float, *, ignore_missing: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Take pointwise resampling percentiles as a band.

    Args:
        draws: Array of shape ``(n_resamples, n_points)``.
        level: Nominal coverage in ``(0, 1)``.
        ignore_missing: Ignore grid points outside a resampled forecast range.

    Returns:
        lower: Lower band.
        upper: Upper band.
    """
    tail = (1.0 - level) / 2.0
    quantile = np.nanquantile if ignore_missing else np.quantile
    lower = quantile(draws, tail, axis=0)
    upper = quantile(draws, 1.0 - tail, axis=0)
    return lower, upper


def _resample_bands(
    x: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray,
    level: float,
    n_resamples: int,
    random_state: int | None,
    *,
    resample_forecasts: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample forecasts when requested, draw outcomes, and refit the diagram.

    Args:
        x: Original forecast values, one per observation.
        probabilities: Success probability for each observation.
        grid: Forecast values at which to report the band.
        level: Nominal coverage in ``(0, 1)``.
        n_resamples: Number of resamples.
        random_state: Seed.
        resample_forecasts: Bootstrap forecast rows before drawing outcomes.

    Returns:
        lower: Lower band on ``grid``.
        upper: Upper band on ``grid``.

    Raises:
        RuntimeError: If no resample covers part of the original forecast support.
    """
    if resample_forecasts:
        order = np.argsort(x, kind="stable")
        x = x[order]
        probabilities = probabilities[order]

    rng = np.random.default_rng(random_state)
    fill = np.nan if resample_forecasts else 0.0
    draws = np.full((n_resamples, grid.size), fill, dtype=float)
    for i in range(n_resamples):
        if resample_forecasts:
            indices = rng.integers(0, x.size, size=x.size)
            x_star = x[indices]
            probabilities_star = probabilities[indices]
        else:
            x_star = x
            probabilities_star = probabilities
        y_star = rng.binomial(1, probabilities_star).astype(float)
        diagram = corp_reliability(y_star, x_star)
        if resample_forecasts:
            draws[i] = np.interp(
                grid,
                diagram.prediction_values,
                diagram.event_probabilities,
                left=np.nan,
                right=np.nan,
            )
        else:
            draws[i] = diagram.as_function()(grid)

    if resample_forecasts and np.any(np.all(np.isnan(draws), axis=0)):
        raise RuntimeError(
            "n_resamples produced no estimate at part of the forecast support; "
            "increase n_resamples"
        )
    return _band_from_draws(draws, level, ignore_missing=resample_forecasts)


def _validate_band_args(
    level: float, n_resamples: int, random_state: int | None
) -> None:
    """Check band arguments.

    Args:
        level: Nominal coverage.
        n_resamples: Number of resamples.
        random_state: Non-negative integer seed or ``None``.

    Raises:
        ValueError: If an argument is outside its documented domain.
    """
    if isinstance(level, (bool, np.bool_)) or not isinstance(level, Real):
        raise ValueError(f"level must be a real number in (0, 1), got {level!r}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if isinstance(n_resamples, (bool, np.bool_)) or not isinstance(
        n_resamples, Integral
    ):
        raise ValueError(f"n_resamples must be an integer, got {n_resamples!r}")
    if n_resamples < 2:
        raise ValueError(f"n_resamples must be at least 2, got {n_resamples}")
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, Integral)
        or random_state < 0
    ):
        raise ValueError(
            f"random_state must be a non-negative integer or None, got {random_state!r}"
        )


def consistency_bands(
    y_pred: np.ndarray,
    *,
    level: float = 0.9,
    n_resamples: int = 1000,
    random_state: int | None = 0,
) -> dict[str, np.ndarray]:
    """Pointwise null bands for a perfectly calibrated reliability diagram.

    Forecast rows are sampled with replacement from their empirical distribution.
    For each sample, outcomes are drawn as ``y* ~ Bernoulli(y_pred)`` and the CORP
    diagram is refit. Pointwise percentiles of those refits show how far the
    estimated diagram can wander under the calibration hypothesis.

    Use :func:`confidence_bands` instead for an interval around the estimate.

    Args:
        y_pred: Forecast probabilities in ``[0, 1]``. Observed outcomes are not
            needed because the calibration null supplies their distribution.
        level: Nominal coverage, default 0.9.
        n_resamples: Number of resamples, default 1000.
        random_state: Seed. Defaults to 0 so results are reproducible.

    Returns:
        dict: ``prediction_values`` (the original forecast support), ``lower`` and
            ``upper`` pointwise bounds.

    Notes:
        These are pointwise bands, not a simultaneous envelope or a global
        calibration test.

        The method follows Dimitriadis, Gneiting & Jordan (2021), "Stable
        reliability diagrams for probabilistic classifiers", and their
        ``reliabilitydiag`` R implementation. Only its resampling method is
        implemented here; the paper also derives asymptotic bands.

        Each resampled fit contributes only on its observed forecast support.
        This also applies to singleton fits, avoiding extrapolation to forecast
        values absent from that resample.
    """
    _validate_band_args(level, n_resamples, random_state)
    y_pred = _validate_probability_vector(y_pred)
    grid = np.unique(y_pred)
    lower, upper = _resample_bands(
        y_pred,
        y_pred,
        grid,
        level,
        n_resamples,
        random_state,
        resample_forecasts=True,
    )
    return {"prediction_values": grid, "lower": lower, "upper": upper}


def _correct_boundary_plateaus(
    endpoint: np.ndarray,
    prediction_values: np.ndarray,
    event_probabilities: np.ndarray,
) -> np.ndarray:
    """Interpolate a band endpoint across interior PAV estimates at zero or one."""
    if prediction_values.size == 1:
        return endpoint
    interior_boundary = (event_probabilities == 0.0) | (event_probabilities == 1.0)
    interior_boundary[[0, -1]] = False
    if not np.any(interior_boundary):
        return endpoint
    keep = ~interior_boundary
    return np.interp(prediction_values, prediction_values[keep], endpoint[keep])


def confidence_bands(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    level: float = 0.9,
    n_resamples: int = 1000,
    random_state: int | None = 0,
) -> dict[str, np.ndarray]:
    """Pointwise bootstrap intervals around a CORP reliability estimate.

    Forecast rows are sampled with replacement. Outcomes are then drawn from the
    PAV-recalibrated probabilities and a CORP diagram is refit. The returned
    bounds are pointwise percentiles of those refits, with the boundary correction
    used by the authors' reference implementation.

    Args:
        y_true: Binary outcomes in ``{0, 1}``.
        y_pred: Forecast probabilities in ``[0, 1]``.
        level: Nominal coverage, default 0.9.
        n_resamples: Number of resamples, default 1000.
        random_state: Seed. Defaults to 0 so results are reproducible.

    Returns:
        dict: ``prediction_values`` (the original forecast support), ``lower`` and
            ``upper`` pointwise bounds.

    Warnings:
        Evaluate held-out or out-of-fold predictions. These are pointwise
        bootstrap intervals, not a simultaneous envelope. Their finite-sample
        coverage can differ from ``level`` and should be validated for the
        intended data-generating setting.

    Notes:
        This implements the resampling confidence-region method of Dimitriadis,
        Gneiting & Jordan (2021), "Stable reliability diagrams for probabilistic
        classifiers", following their ``reliabilitydiag`` R implementation.
        Each resampled fit contributes only on its observed forecast support,
        including when that fit contains a single forecast value.
    """
    _validate_band_args(level, n_resamples, random_state)
    y_true, y_pred, _ = _validate_binary_probability_metric_inputs(y_true, y_pred, None)
    diagram = corp_reliability(y_true, y_pred)
    recalibrated = np.interp(
        y_pred, diagram.prediction_values, diagram.event_probabilities
    )
    grid = diagram.prediction_values
    lower, upper = _resample_bands(
        y_pred,
        recalibrated,
        grid,
        level,
        n_resamples,
        random_state,
        resample_forecasts=True,
    )
    lower = _correct_boundary_plateaus(lower, grid, diagram.event_probabilities)
    upper = _correct_boundary_plateaus(upper, grid, diagram.event_probabilities)
    return {"prediction_values": grid, "lower": lower, "upper": upper}


_CI_METHODS = ("bca", "basic", "percentile")
_BOOTSTRAP_BATCH_MEMORY = 8 * 1024**2


def _bootstrap_batch_size(n_observations: int, n_resamples: int) -> int:
    """Bound SciPy's resampling and jackknife index matrices near 8 MiB."""
    bytes_per_index_row = n_observations * np.dtype(np.intp).itemsize
    return min(n_resamples, max(1, _BOOTSTRAP_BATCH_MEMORY // bytes_per_index_row))


def _finite_metric_value(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Evaluate a metric and require a finite scalar result."""
    result = np.asarray(metric(y_true, y_pred))
    if result.ndim != 0:
        raise ValueError("metric must return a finite scalar")
    try:
        value = float(result)
    except (TypeError, ValueError) as error:
        raise ValueError("metric must return a finite scalar") from error
    if not np.isfinite(value):
        raise ValueError("metric must return a finite scalar")
    return value


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    level: float = 0.95,
    n_resamples: int = 1000,
    random_state: int | None = 0,
    method: str = "bca",
) -> dict[str, Any]:
    """Bootstrap an interval for a regular scalar evaluation metric.

    Observation pairs are sampled with replacement and passed to ``metric``.
    Interval construction delegates to :func:`scipy.stats.bootstrap` with paired
    resampling. BCa is the standard default; percentile and basic intervals are
    also available.

    Args:
        metric: Callable taking ``(y_true, y_pred)`` and returning a finite scalar.
        y_true: Binary outcomes in ``{0, 1}``.
        y_pred: Forecast probabilities in ``[0, 1]``.
        level: Nominal coverage in ``(0, 1)``. Defaults to 0.95.
        n_resamples: Number of bootstrap resamples. Defaults to 1000.
        random_state: Seed. Defaults to 0 so results are reproducible.
        method: ``"bca"`` (default), ``"basic"`` or ``"percentile"``.

    Returns:
        dict: ``estimate`` (the metric on the observed data), ``lower``,
            ``upper``, ``level``, ``n_resamples``, ``method`` and ``bias`` (the
            bootstrap mean minus the estimate).

    Raises:
        ValueError: If the data, bootstrap arguments, method, or a metric result is
            invalid.
        TypeError: If ``metric`` is not callable.
        RuntimeError: If the resampling distribution is degenerate or SciPy cannot
            define finite endpoints.

    Warnings:
        Rows must be independent held-out evaluation units. Resample clusters or
        time blocks instead when observations are dependent; this function does
        not implement those designs.

        Ordinary bootstrap intervals need a regular statistic. Calibration-error
        metrics are non-smooth at perfect calibration, and percentile, basic and
        BCa intervals need not attain nominal coverage there. Use this function for
        regular quantities such as held-out mean proper scores, not as a test that
        calibration error equals zero.

    Notes:
        BCa follows Efron (1987), "Better Bootstrap Confidence Intervals". The
        resampling and interval endpoints are computed by SciPy rather than a
        package-local implementation.

    Examples:
        >>> import numpy as np
        >>> from calibre.metrics import brier_score
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 500)
        >>> y = rng.binomial(1, p).astype(float)
        >>> ci = bootstrap_ci(brier_score, y, p, n_resamples=200)
        >>> bool(ci["lower"] <= ci["estimate"] <= ci["upper"])
        True
    """
    if not callable(metric):
        raise TypeError("metric must be callable")
    if method not in _CI_METHODS:
        raise ValueError(f"method must be one of {list(_CI_METHODS)}, got {method!r}")
    _validate_band_args(level, n_resamples, random_state)
    y_true, y_pred, _ = _validate_binary_probability_metric_inputs(y_true, y_pred, None)
    if y_true.size < 2:
        raise ValueError("y_true and y_pred must contain at least two observations")

    def statistic(true: np.ndarray, pred: np.ndarray) -> float:
        return _finite_metric_value(metric, true, pred)

    observed = statistic(y_true, y_pred)
    indices = np.arange(y_true.size)

    def indexed_statistic(row_indices: np.ndarray) -> float:
        selected = np.asarray(row_indices, dtype=np.intp)
        return statistic(y_true[selected], y_pred[selected])

    random_generator = np.random.default_rng(random_state)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DegenerateDataWarning)
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
            module=r"scipy\.stats\._resampling",
        )
        result = scipy_bootstrap(
            (indices,),
            indexed_statistic,
            vectorized=False,
            confidence_level=level,
            n_resamples=n_resamples,
            batch=_bootstrap_batch_size(y_true.size, n_resamples),
            method=method,
            rng=random_generator,
        )
    draws = np.asarray(result.bootstrap_distribution, dtype=float)
    if np.all(draws == draws[0]):
        raise RuntimeError(
            "bootstrap distribution is degenerate; uncertainty cannot be estimated"
        )
    lower = float(result.confidence_interval.low)
    upper = float(result.confidence_interval.high)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise RuntimeError("bootstrap interval is undefined for this metric and sample")

    return {
        "estimate": observed,
        "lower": lower,
        "upper": upper,
        "level": float(level),
        "n_resamples": int(n_resamples),
        "method": method,
        "bias": float(draws.mean() - observed),
    }
