"""Evaluation metrics for calibration."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, rand_score

from .utils.validation import _validate_binary_probability_metric_inputs


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two arrays, as a plain float.

    Wrapped because ``scipy.stats.spearmanr`` returns a result object whose
    ``correlation`` attribute type checkers cannot see, and whose tuple form is
    typed too loosely to convert directly.

    Args:
        a: First array.
        b: Second array.

    Returns:
        float: The correlation coefficient, or NaN if either input is constant.
    """
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    result = spearmanr(a, b)
    coefficient = getattr(result, "statistic", None)
    if coefficient is None:  # pragma: no cover - older scipy
        coefficient = getattr(result, "correlation", None)
    if coefficient is None:  # pragma: no cover - unexpected scipy shape
        coefficient = next(iter(result))  # type: ignore[call-overload]
    return float(coefficient)  # type: ignore[arg-type]


def _validate_rank_vector(
    values: np.ndarray,
    *,
    name: str,
    expected_size: int | None = None,
    expected_name: str = "y_true",
) -> np.ndarray:
    """Validate one numeric vector used by a rank diagnostic."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if expected_size is not None and array.size != expected_size:
        raise ValueError(f"{name} must have the same length as {expected_name}")
    try:
        array = array.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def mean_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    r"""Calculate the absolute calibration-in-the-large error.

    .. math::
        \left|
        \frac{\sum_i w_i \hat p_i}{\sum_i w_i}
        - \frac{\sum_i w_i y_i}{\sum_i w_i}
        \right|

    This measures whether predictions are right on average. It can be zero when
    positive and negative errors cancel, even for completely reversed predictions,
    so use it with a proper score and a calibration curve rather than as a complete
    calibration assessment.

    Args:
        y_true: Ground truth values (0 or 1 for binary classification).
        y_pred: Predicted probabilities in ``[0, 1]``.
        sample_weight: Non-negative evaluation weights. Zero-weight observations
            are ignored. A common rescaling does not change the result.

    Returns:
        float: Absolute difference between the mean prediction and the base rate.

    References:
        Van Calster, B., Nieboer, D., Vergouwe, Y., De Cock, B., Pencina, M. J.,
        & Steyerberg, E. W. (2016). A calibration hierarchy for risk models was
        defined: from utopia to empirical data. *Journal of Clinical Epidemiology*,
        74, 167--176. https://doi.org/10.1016/j.jclinepi.2015.12.005

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
        >>> round(mean_calibration_error(y_true, y_pred), 4)   # mean 0.54 vs base 0.6
        0.06

        A perfectly calibrated predictor scores zero, however unsharp it is:

        >>> y = np.array([0, 0, 1, 1])
        >>> mean_calibration_error(y, np.full(4, 0.5))
        0.0
    """
    true, pred, weight = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight
    )
    return float(
        abs(np.average(pred, weights=weight) - np.average(true, weights=weight))
    )


def _weighted_equal_mass_bins(
    y_pred: np.ndarray, sample_weight: np.ndarray, n_bins: int
) -> tuple[np.ndarray, int]:
    """Assign tied predictions to weighted, approximately equal-mass bins."""
    values, inverse = np.unique(y_pred, return_inverse=True)
    value_weight = np.bincount(inverse, weights=sample_weight, minlength=values.size)
    cumulative_fraction = np.cumsum(value_weight) / np.sum(value_weight)
    targets = np.arange(1, n_bins, dtype=float) / n_bins
    edges = np.searchsorted(cumulative_fraction, targets, side="left") + 1
    edges = np.unique(edges[(edges > 0) & (edges < values.size)])
    value_bin = np.searchsorted(edges, np.arange(values.size), side="right")
    return value_bin[inverse], edges.size + 1


@overload
def root_mean_squared_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = ...,
    strategy: Literal["uniform", "quantile"] = ...,
    sample_weight: np.ndarray | None = ...,
    return_details: Literal[False] = ...,
) -> float: ...


@overload
def root_mean_squared_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = ...,
    strategy: Literal["uniform", "quantile"] = ...,
    sample_weight: np.ndarray | None = ...,
    return_details: Literal[True],
) -> dict[str, float | np.ndarray]: ...


def root_mean_squared_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    sample_weight: np.ndarray | None = None,
    return_details: bool = False,
) -> float | dict[str, float | np.ndarray]:
    r"""Calculate the mass-weighted root mean squared calibration error.

    .. math::
        \operatorname{RMSCE} = \left[\sum_k \frac{w_k}{\sum_j w_j}
        (\bar p_k - \bar y_k)^2\right]^{1/2}

    Unlike an unweighted average across occupied bins, this definition gives a
    bin influence proportional to its evaluation-sample mass. Uniform bins divide
    the probability domain ``[0, 1]`` into equal-width intervals. Quantile bins are
    approximately equal in weighted mass and never split tied predictions.

    This is a plugin estimator: its value depends on the binning choice and is
    biased upward by finite-sample variation in bin outcome rates. Use the same
    held-out evaluation observations for comparisons between prediction systems;
    do not evaluate a calibrator on the observations used to fit it.

    Args:
        y_true: Binary outcomes in ``{0, 1}``.
        y_pred: Predicted probabilities in ``[0, 1]``.
        n_bins: Requested number of bins, capped at the number of positive-weight
            observations. Quantile binning can use fewer bins when ties prevent a
            split.
        strategy: ``"uniform"`` for fixed-width probability bins or ``"quantile"``
            for weighted equal-mass, tie-preserving bins.
        sample_weight: Non-negative evaluation weights. Zero-weight observations
            are ignored. A common rescaling does not change the result.
        return_details: Return the score and occupied-bin summaries when true.

    Returns:
        float: RMSCE when ``return_details`` is false.
        dict: RMSCE, counts, weights, prediction ranges, and weighted bin means
        when ``return_details`` is true.

    Raises:
        ValueError: If inputs are empty, multidimensional, non-finite, inconsistent
            in length, outside their documented domains, or carry invalid weights;
            if ``n_bins`` is not a positive integer; or if ``strategy`` is unknown.
        TypeError: If ``return_details`` is not boolean.

    References:
        Guo, Pleiss, Sun & Weinberger (2017), "On Calibration of Modern Neural
        Networks", ICML, PMLR 70:1321--1330. Kumar, Liang & Ma (2019), "Verified
        Uncertainty Calibration", NeurIPS 32. The formula matches the ``norm="l2"``
        calibration error implemented by TorchMetrics.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.25, 0.25, 0.75, 0.75])
        >>> root_mean_squared_calibration_error(y_true, y_pred, n_bins=2)
        0.25
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ):
        raise ValueError("n_bins must be a positive integer")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError(f"Unknown binning strategy: {strategy}")
    if not isinstance(return_details, (bool, np.bool_)):
        raise TypeError("return_details must be boolean")

    true, pred, weight = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight
    )

    n_bins = min(int(n_bins), true.size)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_id = np.searchsorted(edges[1:-1], pred, side="right")
        n_used = n_bins
    else:
        bin_id, n_used = _weighted_equal_mass_bins(pred, weight, n_bins)

    counts = np.bincount(bin_id, minlength=n_used)
    bin_weight = np.bincount(bin_id, weights=weight, minlength=n_used)
    weighted_pred = np.bincount(bin_id, weights=weight * pred, minlength=n_used)
    weighted_true = np.bincount(bin_id, weights=weight * true, minlength=n_used)
    occupied = bin_weight > 0.0
    counts = counts[occupied]
    bin_weight = bin_weight[occupied]
    mean_pred = weighted_pred[occupied] / bin_weight
    mean_true = weighted_true[occupied] / bin_weight
    gaps = mean_pred - mean_true
    rmsce = float(np.sqrt(np.sum(bin_weight * gaps**2) / np.sum(bin_weight)))

    if not return_details:
        return rmsce

    occupied_ids = np.flatnonzero(occupied)
    score_min = np.array([np.min(pred[bin_id == k]) for k in occupied_ids])
    score_max = np.array([np.max(pred[bin_id == k]) for k in occupied_ids])
    return {
        "root_mean_squared_calibration_error": rmsce,
        "bin_counts": counts,
        "bin_weights": bin_weight,
        "bin_score_minimums": score_min,
        "bin_score_maximums": score_max,
        "bin_prediction_means": mean_pred,
        "bin_event_rates": mean_true,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> float:
    r"""Calculate uniform-bin expected calibration error (ECE).

    .. math::
        \operatorname{ECE} = \sum_k \frac{w_k}{\sum_j w_j}
        \left|\bar p_k - \bar y_k\right|

    The probability interval ``[0, 1]`` is divided into equal-width bins, and
    each occupied bin's absolute calibration gap is weighted by its evaluation
    mass. This is the conventional plugin ECE estimator, matching the
    ``norm="l1"`` definition in TorchMetrics.

    ECE is sensitive to the bin count and is not a consistent measure of all
    calibration departures: overprediction and underprediction inside one bin
    can cancel exactly. Finite-sample variation in bin outcome rates also biases
    the score upward. Use :func:`smooth_calibration_error`,
    :func:`debiased_calibration_error`, or :func:`sweep_calibration_error` when a
    less arbitrary headline estimate is required.

    Args:
        y_true: Binary outcomes in ``{0, 1}``.
        y_pred: Predicted probabilities in ``[0, 1]``.
        n_bins: Requested number of equal-width bins, capped at the number of
            positive-weight observations.
        sample_weight: Non-negative evaluation weights. Zero-weight observations
            are ignored. A common rescaling does not change the result.

    Returns:
        float: Sample-mass-weighted absolute calibration gap.

    Raises:
        ValueError: If inputs are empty, multidimensional, non-finite, inconsistent
            in length, outside their documented domains, or carry invalid weights;
            or if ``n_bins`` is not a positive integer.

    References:
        Naeini, Cooper & Hauskrecht (2015), "Obtaining Well Calibrated
        Probabilities Using Bayesian Binning", AAAI 29. Guo, Pleiss, Sun &
        Weinberger (2017), "On Calibration of Modern Neural Networks", ICML,
        PMLR 70:1321--1330. Roelofs, Cain, Shlens & Mozer (2022), "Mitigating
        Bias in Calibration Error Estimation", AISTATS, PMLR 151:4036--4054.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> round(expected_calibration_error(y_true, y_pred, n_bins=2), 2)
        0.29
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ):
        raise ValueError("n_bins must be a positive integer")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")

    true, pred, weight = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight
    )
    n_bins = min(int(n_bins), true.size)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.searchsorted(edges[1:-1], pred, side="right")

    bin_weight = np.bincount(bin_id, weights=weight, minlength=n_bins)
    weighted_pred = np.bincount(bin_id, weights=weight * pred, minlength=n_bins)
    weighted_true = np.bincount(bin_id, weights=weight * true, minlength=n_bins)
    occupied = bin_weight > 0.0
    mean_pred = weighted_pred[occupied] / bin_weight[occupied]
    mean_true = weighted_true[occupied] / bin_weight[occupied]
    ece = np.sum(
        bin_weight[occupied] / np.sum(bin_weight) * np.abs(mean_pred - mean_true)
    )
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> float:
    r"""Calculate uniform-bin maximum calibration error (MCE).

    .. math::
        \operatorname{MCE} = \max_k \left|\bar p_k - \bar y_k\right|

    The probability interval ``[0, 1]`` is divided into equal-width bins. MCE is
    the largest absolute gap between the weighted mean prediction and weighted
    event rate among the occupied bins. This is the conventional infinity-norm
    plugin estimator, matching the ``norm="max"`` definition in TorchMetrics.

    MCE gives the same influence to every occupied bin regardless of its
    evaluation mass: one low-mass bin can determine the entire score. It is also
    sensitive to the bin count, can hide equal and opposite errors within a bin,
    and selects the largest finite-sample fluctuation. Treat it as a worst-bin
    diagnostic rather than a standalone estimate of population calibration. Use
    held-out observations and inspect the bin counts in a reliability diagram.

    Args:
        y_true: Binary outcomes in ``{0, 1}``.
        y_pred: Predicted probabilities in ``[0, 1]``.
        n_bins: Requested number of equal-width bins, capped at the number of
            positive-weight observations.
        sample_weight: Non-negative evaluation weights used to calculate each
            bin's means. Zero-weight observations are ignored. A common rescaling
            does not change the result.

    Returns:
        float: Largest occupied-bin absolute calibration gap.

    Raises:
        ValueError: If inputs are empty, multidimensional, non-finite, inconsistent
            in length, outside their documented domains, or carry invalid weights;
            or if ``n_bins`` is not a positive integer.

    References:
        Naeini, Cooper & Hauskrecht (2015), "Obtaining Well Calibrated
        Probabilities Using Bayesian Binning", AAAI 29. Guo, Pleiss, Sun &
        Weinberger (2017), "On Calibration of Modern Neural Networks", ICML,
        PMLR 70:1321--1330. Roelofs, Cain, Shlens & Mozer (2022), "Mitigating
        Bias in Calibration Error Estimation", AISTATS, PMLR 151:4036--4054.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> round(maximum_calibration_error(y_true, y_pred, n_bins=2), 4)
        0.3167
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ):
        raise ValueError("n_bins must be a positive integer")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")

    true, pred, weight = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight
    )
    n_bins = min(int(n_bins), true.size)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.searchsorted(edges[1:-1], pred, side="right")

    bin_weight = np.bincount(bin_id, weights=weight, minlength=n_bins)
    weighted_pred = np.bincount(bin_id, weights=weight * pred, minlength=n_bins)
    weighted_true = np.bincount(bin_id, weights=weight * true, minlength=n_bins)
    occupied = bin_weight > 0.0
    mean_pred = weighted_pred[occupied] / bin_weight[occupied]
    mean_true = weighted_true[occupied] / bin_weight[occupied]
    return float(np.max(np.abs(mean_pred - mean_true)))


def brier_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    r"""Calculate the binary Brier score.

    .. math::
        \operatorname{BS} = \frac{\sum_i w_i (p_i-y_i)^2}{\sum_i w_i}

    The Brier score is a strictly proper scoring rule: in expectation, reporting
    the true event probability uniquely minimises it. Lower is better, with zero
    for perfect binary probability forecasts and one for forecasts that are wrong
    with certainty.

    Brier score measures overall probabilistic performance, not calibration alone.
    Its Murphy decomposition separates miscalibration, discrimination, and outcome
    uncertainty. Compare systems on the same held-out observations; do not evaluate
    a calibrator on the observations used to fit it.

    Args:
        y_true: Binary outcomes in ``{0, 1}``.
        y_pred: Predicted probabilities in ``[0, 1]``.
        sample_weight: Non-negative evaluation weights. Zero-weight observations
            are ignored. A common rescaling does not change the result.

    Returns:
        float: Weighted mean squared probability error.

    References:
        Brier (1950), "Verification of Forecasts Expressed in Terms of
        Probability", Monthly Weather Review 78:1--3. Murphy (1973), "A New
        Vector Partition of the Probability Score", Journal of Applied
        Meteorology 12:595--600. The calculation delegates to
        :func:`sklearn.metrics.brier_score_loss`.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
        >>> brier_score(y_true, y_pred)
        0.098
    """
    true, pred, weight = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight
    )
    return float(brier_score_loss(true, pred, sample_weight=weight, pos_label=1))


def correlation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    input_scores: np.ndarray | None = None,
    original_predictions: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate descriptive Spearman rank correlations.

    These correlations diagnose monotonic association and rank preservation; they
    do not measure probability calibration. In particular, every strictly
    increasing transformation of a prediction vector has the same Spearman
    correlation even when the transformed probabilities are badly miscalibrated.

    A correlation is undefined when either member of a pair is constant. The
    corresponding value is ``NaN`` without emitting SciPy's constant-input warning.
    Weighted Spearman correlation is intentionally unsupported because SciPy has no
    standard weighted definition.

    Args:
        y_true: Observed outcomes or continuous targets.
        y_pred: Predicted or calibrated values on the same observations.
        input_scores: One-dimensional scores supplied to the calibrator.
        original_predictions: Predictions before calibration.

    Returns:
        dict: Spearman correlations to the outcomes and any supplied comparison
        vectors.

    References:
        Spearman (1904), "The Proof and Measurement of Association between Two
        Things", American Journal of Psychology 15:72--101. The calculation
        delegates to :func:`scipy.stats.spearmanr`.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
        >>> original = np.array([0.1, 0.6, 0.9, 0.3, 0.5])
        >>> corr = correlation_metrics(
        ...     y_true, y_pred, original_predictions=original
        ... )
        >>> sorted(corr)
        ['spearman_corr_to_original_predictions', 'spearman_corr_to_y_true']
        >>> round(float(corr["spearman_corr_to_y_true"]), 4)
        0.866
        >>> round(float(corr["spearman_corr_to_original_predictions"]), 4)
        1.0
    """
    true = _validate_rank_vector(y_true, name="y_true")
    pred = _validate_rank_vector(y_pred, name="y_pred", expected_size=true.size)

    results = {"spearman_corr_to_y_true": _spearman(true, pred)}

    if original_predictions is not None:
        original = _validate_rank_vector(
            original_predictions,
            name="original_predictions",
            expected_size=true.size,
        )
        results["spearman_corr_to_original_predictions"] = _spearman(original, pred)

    if input_scores is not None:
        scores = _validate_rank_vector(
            input_scores, name="input_scores", expected_size=true.size
        )
        results["spearman_corr_to_input_scores"] = _spearman(scores, pred)

    return results


def unique_value_counts(
    predictions: np.ndarray,
    *,
    original_predictions: np.ndarray | None = None,
) -> dict:
    """Count exact distinct values before and after calibration.

    This is a structural granularity diagnostic, not a measure of calibration or
    statistical resolution. Counts are exact: the function does not silently merge
    nearby floating-point values. Apply domain-specific rounding explicitly before
    calling if operationally equivalent values should be grouped.

    When original predictions are supplied, both vectors must describe the same
    observations. The ratio is the calibrated count divided by the original count;
    values below one indicate that calibration produced fewer distinct outputs.

    Args:
        predictions: Predicted or calibrated values.
        original_predictions: Predictions before calibration on the same rows.

    Returns:
        dict: Exact distinct-value counts and, when available, their ratio.

    References:
        Counting delegates to :func:`numpy.unique`. Statistical resolution should
        instead be evaluated through a proper-score decomposition such as
        :func:`brier_decomposition`.

    Examples:
        >>> import numpy as np
        >>> predictions = np.array([0.2, 0.7, 0.7, 0.2, 0.7])
        >>> original = np.array([0.1, 0.6, 0.9, 0.2, 0.5])
        >>> counts = unique_value_counts(
        ...     predictions, original_predictions=original
        ... )
        >>> counts["n_unique_predictions"], counts["n_unique_original_predictions"]
        (2, 5)
        >>> counts["unique_prediction_ratio"]
        0.4
    """
    values = _validate_rank_vector(predictions, name="predictions")

    results: dict[str, int | float] = {
        "n_unique_predictions": int(np.unique(values).size)
    }

    if original_predictions is not None:
        original = _validate_rank_vector(
            original_predictions,
            name="original_predictions",
            expected_size=values.size,
            expected_name="predictions",
        )
        results["n_unique_original_predictions"] = int(np.unique(original).size)
        results["unique_prediction_ratio"] = float(
            results["n_unique_predictions"]
        ) / int(results["n_unique_original_predictions"])

    return results


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute occupied-bin summaries for a binary reliability diagram.

    Uniform bins divide the probability domain ``[0, 1]`` into equal-width
    intervals. Quantile bins have approximately equal evaluation mass and never
    split tied predictions. Empty bins are omitted: returning them as zero-valued
    points would make missing data look like observed perfect predictions.

    The bin means use ``sample_weight`` when supplied. ``bin_counts`` remains the
    number of positive-weight observations in each occupied bin, so it describes
    empirical support rather than weighted mass. Use held-out evaluation data;
    computing this curve on the observations used to fit a calibrator gives an
    optimistic picture of its performance.

    Args:
        y_true: Binary outcomes in ``{0, 1}``.
        y_pred: Predicted probabilities in ``[0, 1]``.
        n_bins: Requested number of bins, capped at the number of positive-weight
            observations. Quantile binning can use fewer bins when ties prevent a
            split.
        strategy: ``"uniform"`` for fixed-width probability bins or ``"quantile"``
            for weighted equal-mass, tie-preserving bins.
        sample_weight: Non-negative evaluation weights. Zero-weight observations
            are ignored. A common rescaling changes neither the bin assignments nor
            the bin means, including with quantile binning.

    Returns:
        prob_true: Weighted event rate in each occupied bin.
        prob_pred: Weighted mean predicted probability in each occupied bin.
        bin_counts: Number of positive-weight observations in each occupied bin.

    Raises:
        ValueError: If inputs are empty, multidimensional, non-finite, inconsistent
            in length, outside their documented domains, or carry invalid weights;
            if ``n_bins`` is not a positive integer; or if ``strategy`` is unknown.

    References:
        Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities with
        Supervised Learning", ICML, section 4. Guo, Pleiss, Sun & Weinberger
        (2017), "On Calibration of Modern Neural Networks", ICML, PMLR
        70:1321--1330. The unweighted API and occupied-bin convention follow
        :func:`sklearn.calibration.calibration_curve`.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        >>> prob_true, prob_pred, counts = calibration_curve(
        ...     y_true, y_pred, n_bins=2
        ... )
        >>> prob_true
        array([0., 1.])
        >>> prob_pred
        array([0.15, 0.85])
        >>> counts
        array([2, 2])
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ):
        raise ValueError("n_bins must be a positive integer")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    true, pred, weight = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight
    )
    n_bins = min(int(n_bins), true.size)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_id = np.searchsorted(edges[1:-1], pred, side="right")
        n_used = n_bins
    else:
        bin_id, n_used = _weighted_equal_mass_bins(pred, weight, n_bins)

    bin_counts = np.bincount(bin_id, minlength=n_used)
    bin_weight = np.bincount(bin_id, weights=weight, minlength=n_used)
    weighted_true = np.bincount(bin_id, weights=weight * true, minlength=n_used)
    weighted_pred = np.bincount(bin_id, weights=weight * pred, minlength=n_used)
    occupied = bin_weight > 0.0

    prob_true = weighted_true[occupied] / bin_weight[occupied]
    prob_pred = weighted_pred[occupied] / bin_weight[occupied]
    return prob_true, prob_pred, bin_counts[occupied]


def tie_preservation_score(
    original_predictions: np.ndarray, calibrated_predictions: np.ndarray
) -> float:
    """Measure pairwise agreement between original and calibrated tie partitions.

    Each distinct prediction value defines a group. The returned Rand index is the
    fraction of observation pairs that are either tied in both vectors or distinct
    in both vectors. It is one when calibration leaves every pair's tie status
    unchanged and zero when every pair's status changes.

    Equality is exact, matching :func:`unique_value_counts`. This is a structural
    diagnostic, not a calibration or forecast-quality metric: prediction vectors
    with the same tie partition can have very different proper scores.

    Args:
        original_predictions: Predictions before calibration.
        calibrated_predictions: Calibrated predictions on the same rows.

    Returns:
        float: Rand index between zero and one.

    References:
        Rand (1971), "Objective Criteria for the Evaluation of Clustering
        Methods", Journal of the American Statistical Association 66(336),
        846--850. The implementation delegates to
        :func:`sklearn.metrics.rand_score`.

    Examples:
        >>> import numpy as np
        >>> original = np.array([0.1, 0.2, 0.3, 0.4])
        >>> calibrated = np.array([0.1, 0.1, 0.3, 0.4])
        >>> tie_preservation_score(original, calibrated)
        0.8333333333333334
    """
    original = _validate_rank_vector(original_predictions, name="original_predictions")
    calibrated = _validate_rank_vector(
        calibrated_predictions,
        name="calibrated_predictions",
        expected_size=original.size,
        expected_name="original_predictions",
    )
    _, original_partition = np.unique(original, return_inverse=True)
    _, calibrated_partition = np.unique(calibrated, return_inverse=True)
    return float(rand_score(original_partition, calibrated_partition))


def _equal_mass_bins(y_pred: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:
    """Assign each prediction to an approximately equal-mass bin.

    Args:
        y_pred: Predicted probabilities.
        n_bins: Requested number of bins. Fewer are returned when ties prevent it.

    Returns:
        bin_id: Bin index per observation.
        n_used: Number of bins actually produced.

    Notes:
        Equal-mass rather than equal-width because Roelofs et al. (2022) measure
        consistently smaller bias for equal-mass binning, a point they note is "not
        well appreciated in the literature" -- equal width is the common practice,
        including in the debiased estimator's original presentation.

        Bin edges are snapped outward to the end of each run of tied predictions, so
        identical scores always share a bin. Splitting a tie group would compare a
        bin's mean prediction against a mean label drawn from an arbitrary subset of
        observations carrying that same prediction, which measures the sort order
        rather than calibration. Clipped or rounded scores make this common: a
        forecast clipped into [0, 1] can put hundreds of observations on a single
        value. The cost is that bins are only approximately equal in mass, and that
        heavily tied data supports fewer bins than requested.
    """
    order = np.argsort(y_pred, kind="mergesort")
    sorted_pred = y_pred[order]
    n = y_pred.size

    # Ideal rank cut points. A cut is moved only when a run of tied predictions
    # straddles it, and then forward to that run's end; an unconditional snap
    # would shift every cut by one even on data with no ties at all.
    ideal = (np.arange(1, n_bins) * n) // n_bins
    ideal = ideal[(ideal > 0) & (ideal < n)]
    straddles = sorted_pred[ideal - 1] == sorted_pred[ideal]
    snapped = np.where(
        straddles,
        np.searchsorted(sorted_pred, sorted_pred[ideal], side="right"),
        ideal,
    )
    edges = np.unique(snapped)
    edges = edges[(edges > 0) & (edges < n)]

    bin_id = np.empty(n, dtype=int)
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [n]])
    for k, (lo, hi) in enumerate(zip(starts, stops, strict=True)):
        bin_id[order[lo:hi]] = k
    return bin_id, len(starts)


def _bin_summaries(
    y_true: np.ndarray, y_pred: np.ndarray, bin_id: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin counts, mean prediction and mean label.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted probabilities.
        bin_id: Bin index per observation.
        n_bins: Number of bins.

    Returns:
        counts: Observations per bin.
        mean_pred: Mean prediction per bin.
        mean_true: Mean label per bin.
    """
    counts = np.bincount(bin_id, minlength=n_bins).astype(float)
    safe = np.where(counts > 0, counts, 1.0)
    mean_pred = np.bincount(bin_id, weights=y_pred, minlength=n_bins) / safe
    mean_true = np.bincount(bin_id, weights=y_true, minlength=n_bins) / safe
    return counts, mean_pred, mean_true


def plugin_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = 15,
    norm: int = 2,
    sample_weight: np.ndarray | None = None,
) -> float:
    r"""Calculate the uncorrected :math:`\ell_p` binned calibration error.

    .. math::
        \widehat{\mathrm{CE}}_p = \left[ \sum_k \frac{W_k}{W}
        \left| \bar{f}_k - \bar{y}_k \right|^p \right]^{1/p}

    Here :math:`W_k` is the total evaluation weight in bin :math:`k`; with unit
    weights, :math:`W_k/W = n_k/n`.

    This is the plain plugin estimator: the quantity
    :func:`debiased_calibration_error` corrects and
    :func:`sweep_calibration_error` chooses a bin count for. It exists so those
    three can be compared on equal terms.

    That comparison is otherwise a trap. :func:`expected_calibration_error` is
    :math:`\ell_1` on **uniform-width** bins, while
    :func:`debiased_calibration_error` and :func:`sweep_calibration_error` default
    to :math:`\ell_2` on **equal-mass** bins. This function takes both the norm and
    bin count as arguments and uses the same equal-mass, tie-safe binning as the
    bias-aware estimators, so callers can compare matched quantities.

    Args:
        y_true: Ground truth values (0 or 1).
        y_pred: Predicted probabilities.
        n_bins: Number of equal-mass bins. Fewer are used when ties prevent it.
        norm: Norm. 1 gives the familiar weighted mean absolute gap; 2 matches
            :func:`debiased_calibration_error`.
        sample_weight: Non-negative evaluation weights. Zero-weight rows are
            excluded before validation and binning.

    Returns:
        float: The uncorrected calibration error. Biased upward, and
            increasingly so as ``n_bins`` grows.

    Raises:
        ValueError: If inputs are empty, multidimensional, non-finite, inconsistent
            in length, outside their documented domains, or carry invalid weights;
            or if ``n_bins`` or ``norm`` is not a positive integer.

    References:
        Kumar, Liang & Ma (2019), "Verified Uncertainty Calibration", NeurIPS
        32. The formula follows the authors' ``plugin_ce`` reference
        implementation; the 15-bin default follows their
        ``lower_bound_scaling_ce`` interface. Roelofs, Cain, Shlens & Mozer
        (2022), "Mitigating Bias in Calibration Error Estimation", AISTATS,
        PMLR 151:4036--4054.

    See Also:
        debiased_calibration_error : The same quantity at ``norm=2``, bias-corrected.
        sweep_calibration_error : Chooses ``n_bins`` rather than fixing it.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p_hat = rng.uniform(0, 1, 4000)
        >>> y = rng.binomial(1, p_hat).astype(float)

        These are calibrated by construction, so the true error is zero and whatever
        the plugin reports is bias -- which grows with the bin count:

        >>> coarse = plugin_calibration_error(y, p_hat, n_bins=5)
        >>> fine = plugin_calibration_error(y, p_hat, n_bins=50)
        >>> bool(fine > coarse)
        True

        Debiasing removes it:

        >>> bool(debiased_calibration_error(y, p_hat, n_bins=50) < fine)
        True
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ):
        raise ValueError("n_bins must be a positive integer")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if isinstance(norm, (bool, np.bool_)) or not isinstance(norm, (int, np.integer)):
        raise ValueError("norm must be a positive integer")
    if norm < 1:
        raise ValueError(f"norm must be at least 1, got {norm}")

    true, pred, weight = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight
    )
    n_bins = min(int(n_bins), true.size)
    bin_id, n_used = _weighted_equal_mass_bins(pred, weight, n_bins)
    bin_weight = np.bincount(bin_id, weights=weight, minlength=n_used)
    weighted_pred = np.bincount(bin_id, weights=weight * pred, minlength=n_used)
    weighted_true = np.bincount(bin_id, weights=weight * true, minlength=n_used)
    occupied = bin_weight > 0.0
    mean_pred = weighted_pred[occupied] / bin_weight[occupied]
    mean_true = weighted_true[occupied] / bin_weight[occupied]
    gaps = np.abs(mean_pred - mean_true) ** int(norm)
    total = float(np.sum(bin_weight[occupied] * gaps) / np.sum(bin_weight))
    return float(total ** (1.0 / norm))


def debiased_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = 15,
    squared: bool = False,
) -> float:
    r"""Calculate the debiased :math:`\ell_2` calibration error.

    The plugin binned estimator mistakes sampling noise in each bin's outcome
    rate for miscalibration. This estimator subtracts that Bernoulli-noise term
    from the squared gap before aggregating bins. It removes this finite-sample
    component of bias; it does not remove the approximation error caused by
    representing a continuous calibration curve with finitely many bins.

    .. math::
        \widehat{\mathrm{CE}}^2 = \sum_k \frac{n_k}{n}
        \left[ (\bar{f}_k - \bar{y}_k)^2
             - \frac{\bar{y}_k (1 - \bar{y}_k)}{n_k - 1} \right]

    Args:
        y_true: Ground truth values (0 or 1).
        y_pred: Predicted probabilities.
        n_bins: Requested number of equal-mass bins, capped at the sample size.
            Tied predictions are never split, so fewer bins may be used. The
            conventional default of 15 follows the experiments in Roelofs et al.;
            it is not universally optimal.
        squared: Return the estimate of the **squared** error instead, without
            the square root or the floor at zero. This is the quantity the
            correction actually makes unbiased, and it may legitimately come
            out negative -- see Notes.

    Returns:
        float: Debiased calibration error. Floored at zero: the correction can
            drive the sum negative on well-calibrated data, which is evidence
            of no detectable miscalibration rather than of negative error.
            With ``squared=True`` the unfloored sum is returned instead.

    Raises:
        ValueError: If inputs are empty, multidimensional, non-finite, inconsistent
            in length, or outside their documented domains; if ``n_bins`` is not a
            positive integer; or if ``squared`` is not boolean.

    Notes:
        This is the :math:`\ell_2` error, so it is not comparable in magnitude to
        :func:`expected_calibration_error`, which is :math:`\ell_1`.

        **The correction operates on the squared scale, not the root scale.** On
        well-calibrated samples, the corrected squared estimate fluctuates around
        zero and can legitimately be negative. The default floor and square root
        produce a reportable non-negative error but reintroduce upward bias near
        zero. For averaging or model comparison, use ``squared=True`` and aggregate
        on that scale.

        ``sample_weight`` is deliberately unsupported. The cited correction is
        derived for individual independent observations; treating arbitrary weights
        as replication or importance weights would change that statistical claim.

    References:
        Bröcker (2012), "Estimating Reliability and Resolution of Probability
        Forecasts through Decomposition of the Empirical Score"; Ferro & Fricker
        (2012), "A Bias-Corrected Decomposition of the Brier Score". The formula
        follows Kumar, Liang & Ma's ``unbiased_square_ce`` and ``unbiased_l2_ce``
        reference implementation for "Verified Uncertainty Calibration" (NeurIPS
        2019). Equal-mass binning follows the empirical recommendation of Roelofs,
        Cain, Shlens & Mozer (2022), "Mitigating Bias in Calibration Error
        Estimation".

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 4000)
        >>> y = rng.binomial(1, p).astype(float)

        These predictions are calibrated, so the matched :math:`\ell_2` plugin
        estimator reports finite-sample noise that the correction largely removes:

        >>> plugin = plugin_calibration_error(y, p, n_bins=15, norm=2)
        >>> debiased = debiased_calibration_error(y, p, n_bins=15)
        >>> bool(debiased < plugin)
        True

    See Also:
        sweep_calibration_error : Chooses the bin count instead of fixing it.
        calibre.evaluation.score_decomposition : Avoids binning altogether.
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ):
        raise ValueError("n_bins must be a positive integer")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if not isinstance(squared, (bool, np.bool_)):
        raise ValueError("squared must be boolean")

    true, pred, weight = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight=None
    )
    n_bins = min(int(n_bins), true.size)
    bin_id, n_used = _weighted_equal_mass_bins(pred, weight, n_bins)
    counts = np.bincount(bin_id, minlength=n_used).astype(float)
    mean_pred = np.bincount(bin_id, weights=pred, minlength=n_used) / counts
    mean_true = np.bincount(bin_id, weights=true, minlength=n_used) / counts

    # A bin holding one observation has no within-bin variance estimate, so its
    # plugin term is pure noise with nothing to subtract. It contributes zero
    # rather than an uncorrectable term, matching the reference implementation
    # accompanying Kumar et al. (2019).
    correctable = counts > 1
    per_bin = np.zeros_like(counts)
    variance = (
        mean_true[correctable]
        * (1.0 - mean_true[correctable])
        / (counts[correctable] - 1.0)
    )
    per_bin[correctable] = (
        mean_pred[correctable] - mean_true[correctable]
    ) ** 2 - variance

    total = float(np.sum(counts / true.size * per_bin))
    if squared:
        return total
    return float(np.sqrt(max(total, 0.0)))


@overload
def sweep_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    norm: int = ...,
    return_n_bins: Literal[False] = ...,
) -> float: ...


@overload
def sweep_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    norm: int = ...,
    return_n_bins: Literal[True],
) -> tuple[float, int]: ...


def sweep_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    norm: int = 2,
    return_n_bins: bool = False,
) -> float | tuple[float, int]:
    r"""Calculate the monotonic sweep calibration error (``ECE_sweep``).

    Fixing the bin count is the weak point of binned calibration error: too few
    bins hide miscalibration, too many measure noise, and the best choice depends
    on the sample size and the score distribution. This estimator chooses instead.

    The method assumes a non-decreasing population calibration curve, as is often
    expected for likelihood-trained models. Bins are added while the observed bin
    heights stay monotone, and the sweep stops at the largest bin count for which
    they do. Under that assumption, non-monotonicity signals that the bins have
    become fine enough to read sampling noise. For a genuinely nonmonotone
    calibration curve, the method can stop too early and understate error.

    Args:
        y_true: Ground truth values (0 or 1).
        y_pred: Predicted probabilities.
        norm: Positive integer norm. The default is 2, matching the paper's
            experiments and the authors' reference implementation. Set to 1 for
            the weighted mean absolute gap.
        return_n_bins: Also return the bin count the sweep settled on. That
            number is half of what the estimator has to say -- it is the
            sweep's answer to "how fine can these data support?" -- and
            reporting only the error hides it.

    Returns:
        float or tuple of (float, int): Binned calibration error at the
            selected bin count, and that bin count when ``return_n_bins`` is
            True. The count is the number of bins actually occupied, which
            ties can hold below the number the sweep reached.

    Raises:
        ValueError: If inputs are empty, multidimensional, non-finite, inconsistent
            in length, or outside their documented domains; if ``norm`` is not a
            positive integer; or if ``return_n_bins`` is not boolean.

    Notes:
        This function evaluates held-out predictions; it does not fit a calibrator.
        The selected bin count and error are properties of this evaluation sample.

        ``sample_weight`` is deliberately unsupported. The cited algorithm defines
        equal mass as equal numbers of independent observations; arbitrary weights
        require a separate definition of both bin mass and monotonicity.

    References:
        Roelofs, Cain, Shlens & Mozer (2022), "Mitigating Bias in Calibration Error
        Estimation", AISTATS. The implementation follows Algorithm 1 and the
        authors' Google Research ``em_monotonic_sweep`` reference, with the
        additional guarantee that tied predictions are never split.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 4000)
        >>> y = rng.binomial(1, p).astype(float)
        >>> float(sweep_calibration_error(y, p)) < 0.05
        True

    See Also:
        debiased_calibration_error : Fixes the bin count and corrects the bias.
        calibre.evaluation.score_decomposition : Lets isotonic regression bin.
    """
    if isinstance(norm, (bool, np.bool_)) or not isinstance(norm, (int, np.integer)):
        raise ValueError("norm must be a positive integer")
    if norm < 1:
        raise ValueError(f"norm must be at least 1, got {norm}")
    if not isinstance(return_n_bins, (bool, np.bool_)):
        raise ValueError("return_n_bins must be boolean")

    true, pred, _ = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight=None
    )
    n = true.size
    _, inverse = np.unique(pred, return_inverse=True)
    value_counts = np.bincount(inverse)
    cumulative_fraction = np.cumsum(value_counts) / n
    value_indices = np.arange(value_counts.size)

    def error_at(n_bins: int) -> tuple[float, bool, int]:
        targets = np.arange(1, n_bins, dtype=float) / n_bins
        edges = np.searchsorted(cumulative_fraction, targets, side="left") + 1
        edges = np.unique(edges[(edges > 0) & (edges < value_counts.size)])
        value_bin = np.searchsorted(edges, value_indices, side="right")
        bin_id = value_bin[inverse]
        n_used = edges.size + 1
        counts = np.bincount(bin_id, minlength=n_used).astype(float)
        mean_pred = np.bincount(bin_id, weights=pred, minlength=n_used) / counts
        mean_true = np.bincount(bin_id, weights=true, minlength=n_used) / counts
        monotone = bool(np.all(np.diff(mean_true) >= 0.0))
        gaps = np.abs(mean_pred - mean_true) ** int(norm)
        error = float(np.sum(counts / n * gaps) ** (1.0 / norm))
        return error, monotone, int(n_used)

    # b = 2 is guaranteed monotone only in the sense that the sweep needs a
    # starting point; if even it is not, one bin is all the data supports.
    best, _, best_bins = error_at(1)
    for n_bins in range(2, n + 1):
        error, monotone, occupied = error_at(n_bins)
        if not monotone:
            break
        best, best_bins = error, occupied
    return (best, best_bins) if return_n_bins else best


def _reflect_and_convolve(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve on the unit interval with reflecting boundaries.

    Args:
        values: Gridded mass, length ``m``.
        kernel: Kernel evaluated on the same grid, length ``m``.

    Returns:
        ndarray: The smoothed grid, length ``m``.

    Notes:
        Predictions pile up at 0 and 1 -- a clipped or confident model puts real mass
        exactly on the bounds -- and a plain convolution would let that mass leak off
        the ends, understating error precisely where models are most overconfident.
        Reflecting the grid at both ends keeps it inside.
    """
    m = values.size
    extended = np.concatenate([np.flip(values)[:-1], values, np.flip(values)[1:]])
    return np.convolve(extended, kernel, "valid")[m // 2 : m // 2 + m]


_SMECE_MIN_SIGMA = 1e-3
_SMECE_EVALUATION_POINTS = 200
_SMECE_GRID_WIDTHS = 20.0
_SMECE_DENSITY_FLOOR = 1e-4
_SMECE_BISECTION_STEPS = 10


def _gaussian_kernel(sigma: float, n_points: int) -> np.ndarray:
    """Evaluate a Gaussian kernel of width ``sigma`` on a unit grid.

    Args:
        sigma: Kernel bandwidth.
        n_points: Grid size.

    Returns:
        ndarray: Kernel values, centered on the grid.
    """
    t = np.linspace(0.0, 1.0, n_points)
    return np.exp(-((t - 0.5) ** 2) / (2.0 * sigma**2)) / (np.sqrt(2.0 * np.pi) * sigma)


def _spread_to_grid(y_pred: np.ndarray, values: np.ndarray, m: int) -> np.ndarray:
    """Bin ``values`` onto a regular grid, splitting each linearly between neighbours.

    Args:
        y_pred: Positions in ``[0, 1]``.
        values: Mass carried by each position.
        m: Grid size.

    Returns:
        ndarray: Gridded mass, length ``m``.
    """
    grid = np.zeros(m)
    scaled = y_pred * (m - 1)
    lower = scaled.astype(int).clip(0, m - 2)
    frac = scaled - lower
    np.add.at(grid, lower, (1.0 - frac) * values)
    np.add.at(grid, lower + 1, frac * values)
    return grid


def _interpolate_grid(t: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate a gridded function at ``t``.

    Args:
        t: Evaluation points in ``[0, 1]``.
        grid: Function values on a regular grid over ``[0, 1]``.

    Returns:
        ndarray: Interpolated values.
    """
    n = grid.size
    index = (t * (n - 1)).astype(int).clip(0, n - 2)
    residual = t * (n - 1) - index
    return grid[index] * (1.0 - residual) + grid[index + 1] * residual


def _smooth_at(
    y_pred: np.ndarray, values: np.ndarray, t: np.ndarray, sigma: float
) -> np.ndarray:
    """Kernel-smooth ``values`` located at ``y_pred``, evaluated at ``t``.

    Args:
        y_pred: Positions in ``[0, 1]``.
        values: Mass carried by each position.
        t: Evaluation points.
        sigma: Kernel bandwidth.

    Returns:
        ndarray: Smoothed values at ``t``.
    """
    m = max(2000, round(_SMECE_GRID_WIDTHS / sigma)) // 2 + 1
    gridded = _spread_to_grid(y_pred, values, m)
    smoothed = _reflect_and_convolve(gridded, _gaussian_kernel(sigma, m))
    return _interpolate_grid(t, smoothed)


@overload
def smooth_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    bandwidth: float | None = ...,
    return_bandwidth: Literal[False] = ...,
) -> float: ...


@overload
def smooth_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    bandwidth: float | None = ...,
    return_bandwidth: Literal[True],
) -> tuple[float, float]: ...


def smooth_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    bandwidth: float | None = None,
    return_bandwidth: bool = False,
) -> float | tuple[float, float]:
    r"""Calculate the smooth calibration error (smECE).

    Binned calibration error has no consistent limit: refine the bins and the
    estimate keeps climbing on data that is perfectly calibrated. smECE replaces
    the bins with a Gaussian kernel and, by default, chooses its own bandwidth.
    The default estimator therefore has no bin-count or bandwidth choice for the
    caller to tune.

    .. math::
        \mathrm{smECE}_\sigma(f, y) =
        \frac{\int \left| (K_\sigma \star \nu)(t) \right| \, dt}
             {\int (K_\sigma \star \rho)(t) \, dt},
        \qquad
        \nu = \sum_i (f_i - y_i)\, \delta_{f_i},
        \quad \rho = \sum_i \delta_{f_i}

    The bandwidth is the fixed point :math:`\sigma = \mathrm{smECE}_\sigma`, found
    by bisection. Below that width the kernel is resolving noise; above it, it is
    smoothing away real miscalibration.

    This is a *consistent* calibration measure in the sense of Blasiok, Gopalan,
    Hu and Nakkiran (2023): it is bounded above and below by polynomial functions
    of the true distance to the nearest perfectly calibrated predictor. Binned ECE
    is not, which is why it can report a large error for a predictor that is
    almost calibrated and a small one for a predictor that is not.

    Evaluate smECE on held-out predictions. The automatic bandwidth is selected
    from that evaluation sample as part of the estimator; it is not a calibrator
    fit and does not require a separate tuning split. To assess recalibration,
    pair smECE with held-out proper scores and resolution or discrimination
    diagnostics rather than selecting a model from calibration error alone.

    Args:
        y_true: Binary outcomes in ``{0, 1}``.
        y_pred: Predicted probabilities in ``[0, 1]``.
        bandwidth: Kernel bandwidth, at least ``0.001``. When None, the fixed point
            above is used, which is the recommended behavior and what makes the
            estimator hyperparameter-free. A fixed bandwidth is useful when its
            scale has a substantive interpretation, with weaker theoretical
            guarantees than the automatic choice.
        return_bandwidth: Also return the bandwidth used. Worth reporting: it is
            an interpretable scale, roughly the resolution at which
            miscalibration is detectable.

    Returns:
        float or tuple of (float, float): The smooth calibration error, and
            the bandwidth when ``return_bandwidth``.

    Raises:
        ValueError: If inputs are empty, multidimensional, non-finite,
            inconsistent in shape, outside their documented domains, or
            non-numeric; or if ``bandwidth`` is not a finite number at least
            ``0.001``.
        TypeError: If ``return_bandwidth`` is not boolean.

    See Also:
        - :func:`debiased_calibration_error` -- bias-corrected, but still needs a
          bin count.
        - :func:`calibre.evaluation.score_decomposition` -- avoids binning by using
          isotonic regression, and decomposes the score rather than summarizing
          the error.

    References:
        Błasiok & Nakkiran (2024), "Smooth ECE: Principled Reliability Diagrams
        via Kernel Smoothing", ICLR. Błasiok, Gopalan, Hu & Nakkiran (2023), "A
        Unifying Theory of Distance from Calibration", STOC. The numerical
        discretization and bandwidth search follow Błasiok and Nakkiran's Apple
        ``relplot`` reference implementation:
        https://github.com/apple/ml-calibration. See ``THIRD_PARTY_NOTICES.md``.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 2000)
        >>> y = rng.binomial(1, p).astype(float)

        Calibrated by construction, so the error is near zero:

        >>> bool(smooth_calibration_error(y, p) < 0.03)
        True

        An overconfident predictor is caught:

        >>> squashed = np.clip(2.0 * (p - 0.5) + 0.5, 0, 1)
        >>> bool(smooth_calibration_error(y, squashed) > 0.05)
        True

        Unlike a binned estimator there is no bin count to justify; the bandwidth is
        chosen by the data:

        >>> error, width = smooth_calibration_error(y, p, return_bandwidth=True)
        >>> bool(0.0 < width <= 1.0)
        True
    """
    if not isinstance(return_bandwidth, (bool, np.bool_)):
        raise TypeError("return_bandwidth must be boolean")
    if bandwidth is not None:
        if isinstance(bandwidth, (bool, np.bool_)) or not isinstance(
            bandwidth, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"bandwidth must be a finite number at least {_SMECE_MIN_SIGMA}"
            )
        bandwidth = float(bandwidth)
        if not np.isfinite(bandwidth) or bandwidth < _SMECE_MIN_SIGMA:
            raise ValueError(
                f"bandwidth must be a finite number at least {_SMECE_MIN_SIGMA}"
            )

    y_true, y_pred, _ = _validate_binary_probability_metric_inputs(
        y_true, y_pred, sample_weight=None
    )

    residual = y_pred - y_true

    def error_at(width: float) -> float:
        n_eval = max(round(10.0 / width), _SMECE_EVALUATION_POINTS)
        t = np.linspace(0.0, 1.0, n_eval)
        smoothed = _smooth_at(y_pred, residual, t, width)
        # The reference floor keeps the ratio finite where the kernel reaches a
        # region holding no predictions at all.
        density = (
            _smooth_at(y_pred, np.ones_like(residual), t, width) + _SMECE_DENSITY_FLOOR
        )
        return float(np.sum(np.abs(smoothed)) / np.sum(density))

    if bandwidth is not None:
        value = error_at(bandwidth)
        return (value, float(bandwidth)) if return_bandwidth else value

    # Bisect for the fixed point sigma = smECE_sigma. `resolved(w)` is True once
    # the kernel is at least as wide as the error it measures, so the smallest
    # such width is the self-consistent one.
    def resolved(width: float) -> bool:
        return width < _SMECE_MIN_SIGMA or width < error_at(width)

    width = 1.0
    if not resolved(width):
        low, high = 1.0, 0.0
        for _ in range(_SMECE_BISECTION_STEPS):
            mid = (low + high) / 2.0
            if resolved(mid):
                high = mid
            else:
                low = mid
        width = low

    value = error_at(width)
    return (value, float(width)) if return_bandwidth else value


# Declared at the end of the module so that every public name above is already
# defined. Declared near the top, this list silently omitted the two bias-aware
# estimators from ``from calibre.metrics import *``.
__all__ = [
    "brier_score",
    "calibration_curve",
    "correlation_metrics",
    "debiased_calibration_error",
    "expected_calibration_error",
    "maximum_calibration_error",
    "mean_calibration_error",
    "plugin_calibration_error",
    "root_mean_squared_calibration_error",
    "smooth_calibration_error",
    "sweep_calibration_error",
    "tie_preservation_score",
    "unique_value_counts",
]
