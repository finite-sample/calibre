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

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import norm

from ._core import PiecewiseLinear, StepFunction, aggregate_ties, weighted_pava
from .utils import check_arrays

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


def _brier(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pointwise Brier score ``(x - y)**2``.

    Args:
        x: Forecast probabilities.
        y: Binary outcomes.

    Returns:
        ndarray: Per-observation score.
    """
    return (x - y) ** 2


def _log_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pointwise logarithmic score ``-y log x - (1 - y) log(1 - x)``.

    Args:
        x: Forecast probabilities.
        y: Binary outcomes.

    Returns:
        ndarray: Per-observation score.
    """
    x = np.clip(x, _LOG_EPS, 1.0 - _LOG_EPS)
    return -y * np.log(x) - (1.0 - y) * np.log1p(-x)


_SCORES = {"brier": _brier, "log": _log_score}


def _resolve_score(score: str):
    """Look up a scoring rule by name.

    Args:
        score: ``"brier"`` or ``"log"``.

    Returns:
        callable: Pointwise scoring function.

    Raises:
        ValueError: If the name is not a supported proper scoring rule.
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

    Attributes:
        x: The distinct forecast values, ascending.
        cep: PAV-recalibrated conditional event probability at each forecast value.
        weight: Number of observations carrying each forecast value.

    Notes:
        Points where the diagram is flat are the CORP bins: the PAV algorithm chose
        them, so no bin count needs to be supplied and none can be tuned to flatter
        the forecaster.
    """

    __slots__ = ("cep", "weight", "x")

    def __init__(self, x: np.ndarray, cep: np.ndarray, weight: np.ndarray) -> None:
        """Store the diagram's three parallel arrays.

        Args:
            x: Forecast value at each bin.
            cep: Conditional event probability at each bin.
            weight: Number of observations behind each bin.
        """
        self.x = x
        self.cep = cep
        self.weight = weight

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        """Recalibrate new forecast values through the diagram.

        Args:
            x_new: Forecast values to recalibrate.

        Returns:
            ndarray: Recalibrated probabilities.
        """
        return self.as_function()(np.asarray(x_new, dtype=float).ravel())

    def as_function(self) -> PiecewiseLinear | StepFunction:
        """Return the recalibration map as a callable.

        Returns:
            PiecewiseLinear or StepFunction: Piecewise-linear interpolation of
                the diagram, matching the paper's display convention. A single
                distinct forecast value gives a step function, since there is
                nothing to interpolate between.
        """
        if self.x.size == 1:
            return StepFunction(self.x, self.cep)
        return PiecewiseLinear(self.x, self.cep)

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
            >>> ax = corp_reliability(x, y).plot(density="none")
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

    Args:
        x: Forecast probabilities.
        y: Binary outcomes in ``{0, 1}``.
        sample_weight: Non-negative per-observation weights. Defaults to 1.

    Returns:
        ReliabilityDiagram: The fitted diagram.

    Examples:
        >>> import numpy as np
        >>> from calibre.evaluation import corp_reliability
        >>> x = np.array([0.2, 0.4, 0.6, 0.8])
        >>> y = np.array([0.0, 1.0, 0.0, 1.0])
        >>> diagram = corp_reliability(x, y)

        The middle pair violates monotonicity, so PAV pools it to its mean:

        >>> diagram.cep
        array([0. , 0.5, 0.5, 1. ])

    See Also:
        score_decomposition : The score decomposition built on this estimate.
        calibre.CenteredIsotonicCalibrator : Recalibration, rather than diagnosis.
    """
    x, y = check_arrays(x, y)

    # Tied forecasts are one point, carrying their pooled weight.
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

    Args:
        x: Forecast probabilities.
        y: Binary outcomes in ``{0, 1}``.
        score: Proper scoring rule: ``"brier"`` (default) or ``"log"``.
        sample_weight: Non-negative per-observation weights. Defaults to 1.

    Returns:
        dict: ``mean_score``, ``MCB``, ``DSC``, ``UNC``. ``MCB`` and ``DSC``
            are non-negative, guaranteed by the optimality of the PAV solution.

    Raises:
        ValueError: If ``score`` is not a supported proper scoring rule.

    Examples:
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

    See Also:
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

    Args:
        draws: Array of shape ``(n_resamples, n_points)``.
        level: Nominal coverage in ``(0, 1)``.

    Returns:
        lower: Lower band.
        upper: Upper band.
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

    Args:
        x: Original forecast values, one per observation.
        probabilities: Success probability for each observation.
        grid: Forecast values at which to report the band.
        level: Nominal coverage in ``(0, 1)``.
        n_resamples: Number of resamples.
        random_state: Seed.

    Returns:
        lower: Lower band on ``grid``.
        upper: Upper band on ``grid``.
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

    Args:
        level: Nominal coverage.
        n_resamples: Number of resamples.

    Raises:
        ValueError: If ``level`` is outside ``(0, 1)`` or ``n_resamples`` is below 2.
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

    Args:
        x: Forecast probabilities.
        y: Binary outcomes in ``{0, 1}``. Used for the grid and for
            validation; the bands themselves are generated under the
            calibration hypothesis and do not depend on the observed outcomes.
        level: Nominal coverage, default 0.9.
        n_resamples: Number of resamples, default 1000.
        random_state: Seed. Defaults to 0 so results are reproducible.

    Returns:
        dict: ``x`` (the grid), ``lower`` and ``upper``.

    Notes:
        **These are pointwise bands, not a simultaneous envelope.** Coverage holds at
        each forecast value separately. It does *not* hold at all of them at once, and
        the difference is not subtle: on perfectly calibrated data the observed
        diagram leaves a nominal 90% band *somewhere* on essentially every sample.
        Measured over 150 replications (``tests/test_monte_carlo.py``):

        ======  ====================  =======================
        ``n``   pointwise coverage    simultaneous coverage
        ======  ====================  =======================
        300     90.1%                 1.3%
        1200    89.6%                 0.0%
        4800    89.4%                 0.0%
        ======  ====================  =======================

        So "my curve stayed inside the band, therefore it is calibrated" is a test
        with a false-positive rate near one. Read the band at a forecast value you
        care about, or count excursions and compare that count against the nominal
        miss rate -- do not read it as an envelope.

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

    Args:
        x: Forecast probabilities.
        y: Binary outcomes in ``{0, 1}``.
        level: Nominal coverage, default 0.9.
        n_resamples: Number of resamples, default 1000.
        random_state: Seed. Defaults to 0 so results are reproducible.

    Returns:
        dict: ``x`` (the grid), ``lower`` and ``upper``.

    Notes:
        **Pointwise, not simultaneous**, exactly as for :func:`consistency_bands`;
        see the table there.

        **Coverage of the truth is below nominal on small samples.** These bands are
        centered on the PAV-recalibrated estimate, and isotonic regression is biased at
        finite sample size, so the band is centered slightly off the true conditional
        event probability curve. Coverage of that true curve, measured against a
        known data-generating process over 150 replications at a nominal 90%:

        ======  ==========================
        ``n``   coverage of the true curve
        ======  ==========================
        300     78.2%
        1200    86.9%
        4800    90.5%
        ======  ==========================

        The shortfall is a property of centring on an isotonic fit, not a defect in
        the resampling, and it vanishes as the centring bias does. Treat a 90% band
        on a few hundred observations as closer to an 80% one.

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


_CI_METHODS = ("bc", "bca", "basic", "percentile")


def _bias_correction(draws: np.ndarray, observed: float) -> float:
    r"""Estimate the median bias of the bootstrap distribution, on the z scale.

    Args:
        draws: Bootstrap draws of the statistic.
        observed: The statistic on the observed data.

    Returns:
        float: ``z0 = Phi^-1(fraction of draws below the observed value)``.
            Zero when the bootstrap distribution is centered on the estimate;
            strongly negative when the draws sit above it, which is the case
            this function exists for.

    Notes:
        Draws exactly equal to the estimate count as half, the usual tie correction
        for a statistic with an atom at a boundary. Without it this degenerates on
        precisely the data it matters most for: a floored estimator such as
        :func:`~calibre.debiased_calibration_error` returns exactly zero on well over
        half of well-calibrated samples, no resample can fall strictly below zero,
        and the interval collapses to ``[0, 0]`` -- zero width, asserting certainty
        rather than admitting ignorance.
    """
    below = float(np.mean(draws < observed) + 0.5 * np.mean(draws == observed))
    # Clamp away from 0 and 1, where the normal quantile is infinite. A draw
    # count of n_resamples cannot resolve tails finer than 1/(n_resamples + 1).
    eps = 1.0 / (draws.size + 1)
    below = min(max(below, eps), 1.0 - eps)
    return float(norm.ppf(below))


def _acceleration(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Estimate the acceleration constant by the jackknife.

    Args:
        metric: The statistic.
        y_true: Ground truth values.
        y_pred: Predicted probabilities.

    Returns:
        float: Efron's acceleration, from the skewness of the leave-one-out
            values. Costs ``n`` evaluations of ``metric``, which is why
            ``"bca"`` is offered rather than defaulted.
    """
    n = y_true.size
    keep = np.ones(n, dtype=bool)
    values = np.empty(n, dtype=float)
    for i in range(n):
        keep[i] = False
        values[i] = float(metric(y_true[keep], y_pred[keep]))
        keep[i] = True

    centered = values.mean() - values
    denominator = 6.0 * float(np.sum(centered**2)) ** 1.5
    if denominator == 0.0:
        return 0.0
    return float(np.sum(centered**3) / denominator)


def _interval(
    draws: np.ndarray,
    observed: float,
    level: float,
    method: str,
    acceleration: float,
) -> tuple[float, float]:
    """Turn bootstrap draws into an interval by the requested method.

    Args:
        draws: Bootstrap draws.
        observed: The statistic on the observed data.
        level: Nominal coverage.
        method: One of ``"percentile"``, ``"basic"``, ``"bc"``, ``"bca"``.
        acceleration: Acceleration constant; ignored unless ``method="bca"``.

    Returns:
        tuple of float: ``(lower, upper)``.
    """
    tail = (1.0 - level) / 2.0
    if method == "percentile":
        lower, upper = np.quantile(draws, [tail, 1.0 - tail])
        return float(lower), float(upper)

    if method == "basic":
        lower, upper = np.quantile(draws, [tail, 1.0 - tail])
        return float(2.0 * observed - upper), float(2.0 * observed - lower)

    z0 = _bias_correction(draws, observed)
    a = acceleration if method == "bca" else 0.0
    adjusted = []
    for z in (norm.ppf(tail), norm.ppf(1.0 - tail)):
        shifted = z0 + z
        adjusted.append(norm.cdf(z0 + shifted / (1.0 - a * shifted)))
    lower, upper = np.quantile(draws, np.clip(adjusted, 0.0, 1.0))
    return float(lower), float(upper)


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    level: float = 0.95,
    n_resamples: int = 1000,
    random_state: int | None = 0,
    method: str = "bc",
) -> dict[str, Any]:
    r"""Put a confidence interval on any calibration metric.

    Resamples observations with replacement, recomputes the metric on each
    resample, and turns the resulting draws into an interval. Works with any
    callable of ``(y_true, y_pred)``, so it covers the estimators in
    :mod:`calibre.metrics` and anything you write yourself.

    Calibration metrics are usually reported as bare numbers, which invites
    reading a difference of 0.002 between two models as real. On a few thousand
    observations the interval is often wider than that.

    Args:
        metric: Callable taking ``(y_true, y_pred)`` and returning a float.
        y_true: Ground truth values.
        y_pred: Predicted probabilities.
        level: Nominal coverage in ``(0, 1)``. Defaults to 0.95.
        n_resamples: Number of bootstrap resamples. Defaults to 1000.
        random_state: Seed. Defaults to 0 so results are reproducible.
        method: How to build the interval from the draws:

            - ``"bc"`` (default) -- bias-corrected percentile. Shifts the
              quantiles by how far the draws sit above the estimate. Costs
              nothing extra and, being a percentile method, can never return
              a bound outside the range of the statistic.
            - ``"bca"`` -- adds Efron's acceleration for skewness. Costs ``n``
              extra evaluations of ``metric`` for the jackknife, so it is
              offered rather than defaulted.
            - ``"basic"`` -- the reverse-percentile interval
              ``2*theta - quantiles``. Corrects the bias but routinely returns
              a **negative** lower bound for a non-negative statistic.
            - ``"percentile"`` -- the raw quantiles. Documented, and wrong
              here; see below.

    Returns:
        dict: ``estimate`` (the metric on the observed data), ``lower``,
            ``upper``, ``level``, ``n_resamples``, ``method``, ``bias`` (the
            bootstrap mean minus the estimate, so the distortion is visible
            rather than hidden), and ``degenerate`` (whether the interval
            collapsed to a point).

    Raises:
        ValueError: If ``level`` is outside ``(0, 1)``, ``n_resamples`` is
            below 2, the arrays disagree in length, or ``method`` is unknown.

    Notes:
        **Why the default is not the percentile interval.**

        The bootstrap resamples from the empirical measure, so ``E[F*] = F``. What
        happens to an estimator ``theta = g(F)`` is then decided entirely by the
        shape of ``g``:

        - ``g`` **linear** in ``F`` -- a plain mean, such as the Brier score --
          gives ``E[g(F*)] = g(F)`` exactly, by Jensen with equality.
        - ``g`` **convex** in ``F`` gives ``E[g(F*)] >= g(F)``, strictly. Every
          calibration error is convex: each bin's contribution is an absolute value
          of a linear functional of ``F``, and norms of those stay convex.

        The size of the gap is set by curvature at ``F``, which is unbounded at the
        kink ``||delta|| = 0`` and negligible far from it. **So the distortion is
        worst exactly when the model is well calibrated** -- the case the user most
        wants an honest answer for. Measured (see
        ``experiments/bootstrap_bias/investigate.py``), bootstrap mean over observed:

        ========================  ================  ==================
        statistic                 calibrated data   miscalibrated data
        ========================  ================  ==================
        Brier score (linear)      1.00x             1.00x
        plugin ECE (convex)       1.42x             1.01x
        smECE                     1.33x             1.04x
        ``MCB``                   1.52x             1.09x
        ========================  ================  ==================

        The plugin figure of 1.42 is the predicted ``sqrt(2)``: the observed value is
        ``||delta||`` for sampling noise ``delta``, while the resample gives
        ``||delta + eps||`` with ``eps`` of comparable variance, doubling the
        variance inside the norm. The effect **does not shrink with sample size** --
        measured at 1.43, 1.44, 1.44, 1.42 for ``n`` of 250, 1000, 4000 and 16000 --
        because both terms scale as ``1/sqrt(n)``. More data will not save you; a
        better interval will.

        Coverage of a true calibration error of exactly zero, at a nominal 95%,
        using :func:`~calibre.debiased_calibration_error` (whose estimand really is
        the true error):

        ==============  ========
        method          coverage
        ==============  ========
        ``percentile``  77%
        ``basic``       98%
        ``bc``          **95%**
        ==============  ========

        Hence the default, which is also **3.6 times tighter**: mean width 0.017
        against the percentile interval's 0.063. ``basic`` over-covers at 98% and
        returns negative lower bounds for a non-negative quantity.

        **One caveat on the default.** ``bc`` reads the bias off how many draws fall
        below the estimate, so it degenerates when the statistic has an atom at the
        estimate. :func:`~calibre.debiased_calibration_error` floors at zero and
        returns exactly zero on 59% of well-calibrated samples, and in 30% of those
        the interval collapses to ``[0, 0]``; the returned ``degenerate`` flag says
        when that happened. No such collapse occurs for the plugin error, smECE, the
        Brier score or ``MCB``, none of which are censored. If you want an interval
        that stays non-degenerate near zero, measure with
        :func:`~calibre.smooth_calibration_error`, which never floors.

        **A separate point about plugin estimators.** The uncorrected binned error is
        biased, so its estimand is ``E[plugin] > 0`` rather than the true error. An
        interval for it *correctly* excludes zero, and no choice of interval method
        changes that. If you want an interval that can cover zero, measure with an
        estimator that targets zero -- :func:`~calibre.debiased_calibration_error` or
        :func:`~calibre.smooth_calibration_error`.

        **``MCB`` and ``DSC`` carry an extra problem.** They are functionals of an
        isotonic fit, and a resample leaves only about 63% of rows distinct
        (measured: 0.630 against a theoretical 0.632), so PAV overfits the
        duplicates. Their inflation tracks effective sample size rather than
        convexity alone: subsampling without replacement gives ``MCB`` of 0.0155,
        0.0088 and 0.0056 at ``m`` of 200, 500 and 1000 against 0.0036 observed at
        ``n = 2000``. Prefer :func:`consistency_bands` or :func:`confidence_bands`
        there, which resample *outcomes* rather than rows.

    Examples:
        >>> import numpy as np
        >>> from calibre.metrics import debiased_calibration_error
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 2000)
        >>> y = rng.binomial(1, p).astype(float)
        >>> ci = bootstrap_ci(debiased_calibration_error, y, p, n_resamples=200)

        The data are calibrated by construction, so the interval should reach zero:

        >>> bool(ci["lower"] <= 0.001)
        True

        The reported bias is how far the resampling pushed the statistic up:

        >>> bool(ci["bias"] > 0.0)
        True

        A percentile interval on the same draws sits higher:

        >>> percentile = bootstrap_ci(
        ...     debiased_calibration_error, y, p, n_resamples=200, method="percentile"
        ... )
        >>> bool(percentile["lower"] >= ci["lower"])
        True
    """
    if method not in _CI_METHODS:
        raise ValueError(f"method must be one of {list(_CI_METHODS)}, got {method!r}")
    _validate_band_args(level, n_resamples)
    y_true, y_pred = check_arrays(y_true, y_pred)

    n = y_true.size
    rng = np.random.default_rng(random_state)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        index = rng.integers(0, n, size=n)
        draws[i] = float(metric(y_true[index], y_pred[index]))

    observed = float(metric(y_true, y_pred))
    acceleration = _acceleration(metric, y_true, y_pred) if method == "bca" else 0.0
    lower, upper = _interval(draws, observed, level, method, acceleration)

    return {
        "estimate": observed,
        "lower": lower,
        "upper": upper,
        "level": float(level),
        "n_resamples": int(n_resamples),
        "method": method,
        "bias": float(draws.mean() - observed),
        # True when the interval has collapsed to a point, which happens when a
        # censored statistic sits on its boundary and most resamples do too. The
        # reading is "no detectable miscalibration at this resolution", but a
        # zero-width interval should never pass for a measurement without being
        # named as one.
        "degenerate": bool(upper - lower < 1e-12),
    }
