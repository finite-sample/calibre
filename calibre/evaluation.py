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

import numpy as np

from ._core import PiecewiseLinear, StepFunction, aggregate_ties, weighted_pava
from .utils import check_arrays

__all__ = [
    "ReliabilityDiagram",
    "confidence_bands",
    "consistency_bands",
    "corp_reliability",
    "score_decomposition",
]

# Log score is unbounded at 0 and 1, so probabilities are clipped before taking
# logs. Matches sklearn's log_loss default.
_LOG_EPS = np.finfo(float).eps


def _brier(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pointwise Brier score ``(x - y)**2``.

    Parameters
    ----------
    x
        Forecast probabilities.
    y
        Binary outcomes.

    Returns
    -------
    ndarray
        Per-observation score.
    """
    return (x - y) ** 2


def _log_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pointwise logarithmic score ``-y log x - (1 - y) log(1 - x)``.

    Parameters
    ----------
    x
        Forecast probabilities.
    y
        Binary outcomes.

    Returns
    -------
    ndarray
        Per-observation score.
    """
    x = np.clip(x, _LOG_EPS, 1.0 - _LOG_EPS)
    return -y * np.log(x) - (1.0 - y) * np.log1p(-x)


_SCORES = {"brier": _brier, "log": _log_score}


def _resolve_score(score: str):
    """Look up a scoring rule by name.

    Parameters
    ----------
    score
        ``"brier"`` or ``"log"``.

    Returns
    -------
    callable
        Pointwise scoring function.

    Raises
    ------
    ValueError
        If the name is not a supported proper scoring rule.
    """
    try:
        return _SCORES[score]
    except KeyError:
        raise ValueError(
            f"score must be one of {sorted(_SCORES)}, got {score!r}. "
            "The decomposition requires a proper scoring rule."
        ) from None


class ReliabilityDiagram:
    """A fitted CORP reliability diagram.

    Attributes
    ----------
    x : ndarray
        The distinct forecast values, ascending.
    cep : ndarray
        PAV-recalibrated conditional event probability at each forecast value.
    weight : ndarray
        Number of observations carrying each forecast value.

    Notes
    -----
    Points where the diagram is flat are the CORP bins: the PAV algorithm chose
    them, so no bin count needs to be supplied and none can be tuned to flatter
    the forecaster.
    """

    __slots__ = ("cep", "weight", "x")

    def __init__(self, x: np.ndarray, cep: np.ndarray, weight: np.ndarray) -> None:
        self.x = x
        self.cep = cep
        self.weight = weight

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        """Recalibrate new forecast values through the diagram.

        Parameters
        ----------
        x_new
            Forecast values to recalibrate.

        Returns
        -------
        ndarray
            Recalibrated probabilities.
        """
        return self.as_function()(np.asarray(x_new, dtype=float).ravel())

    def as_function(self) -> PiecewiseLinear | StepFunction:
        """Return the recalibration map as a callable.

        Returns
        -------
        PiecewiseLinear or StepFunction
            Piecewise-linear interpolation of the diagram, matching the paper's
            display convention. A single distinct forecast value gives a step
            function, since there is nothing to interpolate between.
        """
        if self.x.size == 1:
            return StepFunction(self.x, self.cep)
        return PiecewiseLinear(self.x, self.cep)

    def __repr__(self) -> str:
        """Return a short summary of the diagram."""
        return (
            f"ReliabilityDiagram(n_points={self.x.size}, "
            f"n_bins={np.unique(self.cep).size})"
        )


def corp_reliability(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> ReliabilityDiagram:
    """Estimate conditional event probabilities by isotonic regression.

    This is the CORP reliability diagram: the PAV-recalibrated forecast
    probabilities plotted against the original forecast values. Unlike a binned
    diagram it needs no bin count, because PAV determines the number and position
    of the flat segments itself.

    Parameters
    ----------
    x
        Forecast probabilities.
    y
        Binary outcomes in ``{0, 1}``.
    sample_weight
        Non-negative per-observation weights. Defaults to 1.

    Returns
    -------
    ReliabilityDiagram
        The fitted diagram.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.evaluation import corp_reliability
    >>> x = np.array([0.2, 0.4, 0.6, 0.8])
    >>> y = np.array([0.0, 1.0, 0.0, 1.0])
    >>> diagram = corp_reliability(x, y)

    The middle pair violates monotonicity, so PAV pools it to its mean:

    >>> diagram.cep
    array([0. , 0.5, 0.5, 1. ])

    See Also
    --------
    score_decomposition : The score decomposition built on this estimate.
    calibre.CenteredIsotonicCalibrator : Recalibration, rather than diagnosis.
    """
    x, y = check_arrays(x, y)

    # Tied forecasts are one point, carrying the pooled weight. Skipping this is
    # what made SmoothedIsotonicCalibrator non-monotone before 0.7.1.
    x_unique, y_mean, weight = aggregate_ties(x, y, sample_weight)
    cep = weighted_pava(y_mean, weight)
    return ReliabilityDiagram(x_unique, cep, weight)


def score_decomposition(
    x: np.ndarray,
    y: np.ndarray,
    score: str = "brier",
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Decompose a mean score into miscalibration, discrimination and uncertainty.

    Returns the CORP decomposition ``mean_score = MCB - DSC + UNC``, where the
    calibrated forecasts are the PAV-recalibrated probabilities and the reference
    forecast is the marginal event frequency.

    Read it as: ``MCB`` is what recalibration would save you, ``DSC`` is what your
    forecasts buy over always predicting the base rate, and ``UNC`` is the
    difficulty of the problem, which no forecaster can change.

    Parameters
    ----------
    x
        Forecast probabilities.
    y
        Binary outcomes in ``{0, 1}``.
    score
        Proper scoring rule: ``"brier"`` (default) or ``"log"``.
    sample_weight
        Non-negative per-observation weights. Defaults to 1.

    Returns
    -------
    dict
        ``mean_score``, ``MCB``, ``DSC``, ``UNC``. ``MCB`` and ``DSC`` are
        non-negative, guaranteed by the optimality of the PAV solution.

    Raises
    ------
    ValueError
        If ``score`` is not a supported proper scoring rule.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.evaluation import score_decomposition
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 1, 2000)
    >>> y = rng.binomial(1, x).astype(float)

    These forecasts are calibrated by construction, so miscalibration is small
    while discrimination is substantial:

    >>> d = score_decomposition(x, y)
    >>> bool(d["MCB"] < 0.01), bool(d["DSC"] > 0.05)
    (True, True)

    The identity holds exactly:

    >>> bool(abs(d["mean_score"] - (d["MCB"] - d["DSC"] + d["UNC"])) < 1e-12)
    True

    See Also
    --------
    corp_reliability : The recalibration this decomposition is built on.
    """
    scoring = _resolve_score(score)
    x, y = check_arrays(x, y)
    w = (
        np.ones_like(y)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).ravel()
    )
    if w.shape != y.shape:
        raise ValueError(f"sample_weight shape {w.shape} does not match y {y.shape}")

    total = w.sum()
    if total <= 0.0:
        raise ValueError("sample_weight must include some positive weight")

    def mean_score(forecast: np.ndarray) -> float:
        return float(np.sum(scoring(forecast, y) * w) / total)

    # Recalibrated forecasts, evaluated at each observation's own forecast value.
    diagram = corp_reliability(x, y, w)
    recalibrated = np.interp(x, diagram.x, diagram.cep)

    # Reference forecast: the marginal event frequency, held constant.
    reference = float(np.sum(y * w) / total)

    s_x = mean_score(x)
    s_c = mean_score(recalibrated)
    s_r = mean_score(np.full_like(y, reference))

    return {
        "mean_score": s_x,
        "MCB": s_x - s_c,
        "DSC": s_r - s_c,
        "UNC": s_r,
    }


def _band_from_draws(draws: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Take pointwise resampling percentiles as a band.

    Parameters
    ----------
    draws
        Array of shape ``(n_resamples, n_points)``.
    level
        Nominal coverage in ``(0, 1)``.

    Returns
    -------
    lower : ndarray
        Lower band.
    upper : ndarray
        Upper band.
    """
    tail = (1.0 - level) / 2.0
    lower = np.quantile(draws, tail, axis=0)
    upper = np.quantile(draws, 1.0 - tail, axis=0)
    return lower, upper


def _resample_bands(
    x: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray,
    level: float,
    n_resamples: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw outcomes from ``probabilities`` and refit the diagram each time.

    Parameters
    ----------
    x
        Original forecast values, one per observation.
    probabilities
        Success probability for each observation.
    grid
        Forecast values at which to report the band.
    level
        Nominal coverage in ``(0, 1)``.
    n_resamples
        Number of resamples.
    random_state
        Seed.

    Returns
    -------
    lower : ndarray
        Lower band on ``grid``.
    upper : ndarray
        Upper band on ``grid``.
    """
    rng = np.random.default_rng(random_state)
    draws = np.empty((n_resamples, grid.size), dtype=float)
    for i in range(n_resamples):
        y_star = rng.binomial(1, probabilities).astype(float)
        diagram = corp_reliability(x, y_star)
        draws[i] = diagram.as_function()(grid)
    return _band_from_draws(draws, level)


def _validate_band_args(level: float, n_resamples: int) -> None:
    """Check band arguments.

    Parameters
    ----------
    level
        Nominal coverage.
    n_resamples
        Number of resamples.

    Raises
    ------
    ValueError
        If ``level`` is outside ``(0, 1)`` or ``n_resamples`` is below 2.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_resamples < 2:
        raise ValueError(f"n_resamples must be at least 2, got {n_resamples}")


def consistency_bands(
    x: np.ndarray,
    y: np.ndarray,
    level: float = 0.9,
    n_resamples: int = 1000,
    random_state: int | None = 0,
) -> dict[str, np.ndarray]:
    """Bands showing how a *calibrated* forecaster's diagram would scatter.

    Outcomes are redrawn as ``y* ~ Bernoulli(x)``, taking the original forecasts
    at face value, and the diagram is refit each time. The bands therefore sit
    around the diagonal and answer: if these forecasts were perfectly calibrated,
    how far from the diagonal would the estimate wander by chance alone? An
    observed diagram leaving the band is the analogue of a small p-value.

    Use :func:`confidence_bands` instead for an interval around the estimate.

    Parameters
    ----------
    x
        Forecast probabilities.
    y
        Binary outcomes in ``{0, 1}``. Used for the grid and for validation; the
        bands themselves are generated under the calibration hypothesis and do
        not depend on the observed outcomes.
    level
        Nominal coverage, default 0.9.
    n_resamples
        Number of resamples, default 1000.
    random_state
        Seed. Defaults to 0 so results are reproducible.

    Returns
    -------
    dict
        ``x`` (the grid), ``lower`` and ``upper``.

    Raises
    ------
    ValueError
        If ``level`` is outside ``(0, 1)`` or ``n_resamples`` is below 2.

    Notes
    -----
    Resampling only. The paper also derives asymptotic bands from isotonic
    regression theory (a Chernoff limit for continuous forecasts), which is not
    implemented here; at large sample sizes this function is the expensive
    option rather than the unavailable one.
    """
    _validate_band_args(level, n_resamples)
    x, y = check_arrays(x, y)
    grid = np.unique(x)
    lower, upper = _resample_bands(x, x, grid, level, n_resamples, random_state)
    return {"x": grid, "lower": lower, "upper": upper}


def confidence_bands(
    x: np.ndarray,
    y: np.ndarray,
    level: float = 0.9,
    n_resamples: int = 1000,
    random_state: int | None = 0,
) -> dict[str, np.ndarray]:
    """Bands around the estimated conditional event probabilities.

    Outcomes are redrawn from the PAV-recalibrated probabilities rather than the
    original forecasts, so the bands cluster around the CORP estimate and carry
    the usual frequentist reading: over repeated experiments, about ``level`` of
    such bands contain the true conditional event probability.

    Parameters
    ----------
    x
        Forecast probabilities.
    y
        Binary outcomes in ``{0, 1}``.
    level
        Nominal coverage, default 0.9.
    n_resamples
        Number of resamples, default 1000.
    random_state
        Seed. Defaults to 0 so results are reproducible.

    Returns
    -------
    dict
        ``x`` (the grid), ``lower`` and ``upper``.

    Raises
    ------
    ValueError
        If ``level`` is outside ``(0, 1)`` or ``n_resamples`` is below 2.

    Notes
    -----
    Resampling only; see :func:`consistency_bands`.
    """
    _validate_band_args(level, n_resamples)
    x, y = check_arrays(x, y)
    diagram = corp_reliability(x, y)
    recalibrated = np.interp(x, diagram.x, diagram.cep)
    grid = np.unique(x)
    lower, upper = _resample_bands(
        x, recalibrated, grid, level, n_resamples, random_state
    )
    return {"x": grid, "lower": lower, "upper": upper}
