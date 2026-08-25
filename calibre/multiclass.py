r"""Multiclass calibration: measuring it, and the one method that earns its place.

There is no single best multiclass calibration method. There are two regimes, and
they have different winners. Measured against known true probabilities over 12
seeds on 5 classes:

===========================  ============  ============  ==========
miscalibration               temperature   per-class     winner
===========================  ============  ============  ==========
global (one distortion)      **0.0025**    0.0165        temperature
class-dependent              0.0849        **0.0173**    per-class
class-dependent + shift      0.0276        **0.0176**    per-class
===========================  ============  ============  ==========

Temperature scaling has one parameter applied to every class, so it cannot express
a fix that differs by class; when the distortion *is* global it is exactly right
and wins by a factor of six. It is also argmax-preserving by construction, so it
can never change accuracy -- per-class calibration gained 6.5 accuracy points in
the class-dependent regime, because reordering is precisely what fixes
differently-distorted classes.

So the useful thing to ship is not another calibrator but a way to tell which
regime you are in. :func:`miscalibration_profile` does that: the spread of
per-class miscalibration is ~0.13 when the distortion is global and 0.38-0.92
when it is class-dependent.

Everything here is built from the binary machinery in :mod:`calibre.evaluation`
and :mod:`calibre.metrics`, applied one class at a time. Only *class-wise*
calibration is targeted. Canonical calibration -- requiring the whole probability
vector to be correct -- is infeasible to verify beyond four or five classes and is
deliberately out of scope.
"""

from __future__ import annotations

import numpy as np

from .evaluation import ReliabilityDiagram, corp_reliability, score_decomposition
from .metrics import debiased_calibration_error, sweep_calibration_error

__all__ = [
    "TemperatureScaler",
    "classwise_decomposition",
    "classwise_ece",
    "classwise_reliability",
    "miscalibration_profile",
    "top_label_ece",
]

_EPS = np.finfo(float).eps


def _check_probability_matrix(
    P: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """Validate a matrix of class probabilities.

    Args:
        P: Predicted probabilities, shape ``(n_samples, n_classes)``.
        n_classes: Expected number of columns, when validating data at transform time.

    Returns:
        ndarray: Validated probabilities as float.

    Raises:
        ValueError: If ``P`` is not 2-D, has the wrong width, contains
            non-finite or negative values, or has rows that do not sum to one.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D (n_samples, n_classes), got shape {P.shape}")
    if P.shape[0] == 0 or P.shape[1] == 0:
        raise ValueError("P must contain at least one sample and one class")
    if n_classes is not None and P.shape[1] != n_classes:
        raise ValueError(
            f"P has {P.shape[1]} classes but the scaler was fit on {n_classes}"
        )
    if not np.all(np.isfinite(P)):
        raise ValueError("P contains non-finite values")
    if np.any(P < 0.0):
        raise ValueError("P must contain non-negative probabilities")

    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-8, atol=1e-8):
        raise ValueError("rows of P must sum to 1")
    return P


def _check_matrix(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a probability matrix and its integer labels.

    Args:
        P: Predicted probabilities, shape ``(n_samples, n_classes)``.
        y: Integer class labels in ``[0, n_classes)``.

    Returns:
        P: Validated probabilities as float.
        y: Validated labels as int.

    Raises:
        ValueError: If ``P`` is not 2-D, the lengths disagree, labels are
            non-integer or outside the class range, or ``P`` is not a valid
            probability matrix.
    """
    P = _check_probability_matrix(P)
    y_raw = np.asarray(y)
    if y_raw.ndim == 0:
        raise ValueError("y must be a 1-D array of integer class labels")

    try:
        y_float = np.asarray(y_raw, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be finite integers") from exc
    if P.shape[0] != y_float.shape[0]:
        raise ValueError(
            f"P has {P.shape[0]} rows but y has {y_float.shape[0]} entries"
        )
    if not np.all(np.isfinite(y_float)):
        raise ValueError("labels must be finite integers")
    if not np.all(y_float == np.floor(y_float)):
        raise ValueError("labels must be integers")

    y = y_float.astype(int, copy=False)
    n_classes = P.shape[1]
    if y.size and (y.min() < 0 or y.max() >= n_classes):
        raise ValueError(
            f"labels must lie in [0, {n_classes}), got [{y.min()}, {y.max()}]"
        )
    return P, y


def classwise_decomposition(
    P: np.ndarray, y: np.ndarray, score: str = "brier"
) -> list[dict[str, float]]:
    """Decompose the score of each class one-vs-rest.

    Runs the CORP decomposition (:func:`calibre.evaluation.score_decomposition`)
    separately on each column, treating class ``k`` against all others. Each
    entry therefore carries the same guarantees as the binary case: the identity
    ``mean_score = MCB - DSC + UNC`` is exact, and ``MCB`` and ``DSC`` are
    non-negative.

    Args:
        P: Predicted probabilities, shape ``(n_samples, n_classes)``.
        y: Integer class labels.
        score: Proper scoring rule: ``"brier"`` (default) or ``"log"``.

    Returns:
        list of dict: One decomposition per class, in class order. Each has
            ``mean_score``, ``MCB``, ``DSC``, ``UNC``.

    Notes:
        This is *class-wise* calibration, the standard relaxation of the multiclass
        problem. It says nothing about whether whole probability vectors are jointly
        calibrated.

    Examples:
        >>> import numpy as np
        >>> from calibre.multiclass import classwise_decomposition
        >>> rng = np.random.default_rng(0)
        >>> truth = rng.dirichlet(np.ones(3), size=1500)
        >>> y = np.array([rng.choice(3, p=t) for t in truth])
        >>> parts = classwise_decomposition(truth, y)
        >>> len(parts)
        3

        The identity holds for every class:

        >>> all(
        ...     abs(d["mean_score"] - (d["MCB"] - d["DSC"] + d["UNC"])) < 1e-12
        ...     for d in parts
        ... )
        True

    See Also:
        miscalibration_profile : Reads these to say which calibration method to use.
    """
    P, y = _check_matrix(P, y)
    return [
        score_decomposition(P[:, k], (y == k).astype(float), score=score)
        for k in range(P.shape[1])
    ]


def miscalibration_profile(P: np.ndarray, y: np.ndarray) -> dict:
    """Report how miscalibration is distributed across classes.

    This is the diagnostic worth running before choosing a multiclass method. If
    every class is miscalibrated by about the same amount, a single global
    correction such as temperature scaling can capture it. If the miscalibration
    is concentrated in particular classes, no one-parameter method can express
    the fix and per-class calibration is needed.

    Args:
        P: Predicted probabilities, shape ``(n_samples, n_classes)``.
        y: Integer class labels.

    Returns:
        dict: ``mcb`` (per-class miscalibration), ``spread`` (coefficient of
            variation of ``mcb``), ``worst_classes`` (indices ordered by
            descending ``mcb``), and ``reading`` (a plain-language
            interpretation).

    Notes:
        Calibrated on synthetic data where the true regime is known, ``spread`` is
        about 0.13 when the distortion is global and 0.38-0.92 when it is
        class-dependent. The 0.25 threshold used for ``reading`` sits between those,
        but it is a rule of thumb from one study design, not a test with a
        calibrated false-positive rate. Treat a borderline value as "try both".

        Measure this on **out-of-fold** predictions. Miscalibration estimated on the
        data a calibrator was fit to is not merely optimistic; for an
        isotonic-family calibrator it is identically zero.

    Examples:
        >>> import numpy as np
        >>> from calibre.multiclass import miscalibration_profile
        >>> rng = np.random.default_rng(0)
        >>> truth = rng.dirichlet(np.ones(4) * 0.7, size=3000)
        >>> y = np.array([rng.choice(4, p=t) for t in truth])

        Distort each class differently -- no single temperature can undo this:

        >>> skewed = truth ** np.array([0.6, 1.2, 1.8, 2.4])
        >>> P = skewed / skewed.sum(axis=1, keepdims=True)
        >>> profile = miscalibration_profile(P, y)
        >>> bool(profile["spread"] > 0.25)
        True
        >>> "per-class" in profile["reading"]
        True

    See Also:
        TemperatureScaler : The method to reach for when the spread is small.
    """
    parts = classwise_decomposition(P, y)
    mcb = np.array([d["MCB"] for d in parts], dtype=float)

    mean = float(mcb.mean())
    spread = float(mcb.std() / mean) if mean > _EPS else 0.0
    worst = np.argsort(mcb)[::-1]

    if mean <= 1e-6:
        reading = (
            "No appreciable miscalibration in any class; calibration is unlikely "
            "to help."
        )
    elif spread <= 0.25:
        reading = (
            f"Miscalibration is spread evenly across classes (spread {spread:.2f}). "
            "A single global correction such as TemperatureScaler is likely to be "
            "enough, and it cannot hurt accuracy."
        )
    else:
        listed = ", ".join(str(int(k)) for k in worst[:3])
        reading = (
            f"Miscalibration is concentrated in classes {listed} "
            f"(spread {spread:.2f}). A one-parameter method applies the same "
            "correction to every class and cannot express this; per-class "
            "calibration is likely to help."
        )

    return {
        "mcb": mcb,
        "spread": spread,
        "worst_classes": worst,
        "reading": reading,
    }


def classwise_ece(
    P: np.ndarray, y: np.ndarray, n_bins: int = 15, estimator: str = "debiased"
) -> float:
    """Average one-vs-rest calibration error across classes.

    Args:
        P: Predicted probabilities, shape ``(n_samples, n_classes)``.
        y: Integer class labels.
        n_bins: Bins per class, used by the ``"debiased"`` estimator.
        estimator: ``"debiased"`` (default) subtracts the per-bin Bernoulli
            variance; ``"sweep"`` chooses the bin count by monotonicity
            instead.

    Returns:
        float: Mean per-class calibration error.

    Raises:
        ValueError: If ``estimator`` is unknown.

    Notes:
        Built on the bias-aware estimators in :mod:`calibre.metrics`, so the plugin
        bias that grows with the bin count is corrected, and no bin edge ever splits
        a group of tied predictions.

    Examples:
        >>> import numpy as np
        >>> from calibre.multiclass import classwise_ece
        >>> rng = np.random.default_rng(0)
        >>> truth = rng.dirichlet(np.ones(3), size=2000)
        >>> y = np.array([rng.choice(3, p=t) for t in truth])
        >>> bool(classwise_ece(truth, y) < 0.05)
        True
    """
    P, y = _check_matrix(P, y)
    if estimator == "debiased":
        per_class = [
            debiased_calibration_error((y == k).astype(float), P[:, k], n_bins=n_bins)
            for k in range(P.shape[1])
        ]
    elif estimator == "sweep":
        per_class = [
            sweep_calibration_error((y == k).astype(float), P[:, k])
            for k in range(P.shape[1])
        ]
    else:
        raise ValueError(f'estimator must be "debiased" or "sweep", got {estimator!r}')
    return float(np.mean(per_class))


def top_label_ece(
    P: np.ndarray, y: np.ndarray, n_bins: int = 15, estimator: str = "debiased"
) -> float:
    """Calibration error of the predicted class's confidence.

    Asks whether, among the cases predicted with confidence ``c``, the model is
    right about ``c`` of the time. This is the weakest and most commonly reported
    notion of multiclass calibration; a model can score perfectly here while
    being badly miscalibrated on every non-predicted class.

    Args:
        P: Predicted probabilities, shape ``(n_samples, n_classes)``.
        y: Integer class labels.
        n_bins: Bins, used by the ``"debiased"`` estimator.
        estimator: ``"debiased"`` (default) or ``"sweep"``.

    Returns:
        float: Calibration error of the top-label confidence.

    Raises:
        ValueError: If ``estimator`` is unknown.

    Examples:
        >>> import numpy as np
        >>> from calibre.multiclass import top_label_ece
        >>> rng = np.random.default_rng(0)
        >>> truth = rng.dirichlet(np.ones(3), size=2000)
        >>> y = np.array([rng.choice(3, p=t) for t in truth])
        >>> bool(top_label_ece(truth, y) < 0.05)
        True

    See Also:
        classwise_ece : The stronger notion, averaging over every class.
    """
    P, y = _check_matrix(P, y)
    confidence = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    if estimator == "debiased":
        return debiased_calibration_error(correct, confidence, n_bins=n_bins)
    if estimator == "sweep":
        return sweep_calibration_error(correct, confidence)
    raise ValueError(f'estimator must be "debiased" or "sweep", got {estimator!r}')


def classwise_reliability(P: np.ndarray, y: np.ndarray) -> list[ReliabilityDiagram]:
    """Build a CORP reliability diagram for each class, one-vs-rest.

    Args:
        P: Predicted probabilities, shape ``(n_samples, n_classes)``.
        y: Integer class labels.

    Returns:
        list of ReliabilityDiagram: One diagram per class, in class order. No
            bin count to choose.

    Examples:
        >>> import numpy as np
        >>> from calibre.multiclass import classwise_reliability
        >>> rng = np.random.default_rng(0)
        >>> truth = rng.dirichlet(np.ones(3), size=1000)
        >>> y = np.array([rng.choice(3, p=t) for t in truth])
        >>> diagrams = classwise_reliability(truth, y)
        >>> len(diagrams)
        3
        >>> all(np.all(np.diff(d.cep) >= -1e-12) for d in diagrams)
        True
    """
    P, y = _check_matrix(P, y)
    return [
        corp_reliability(P[:, k], (y == k).astype(float)) for k in range(P.shape[1])
    ]


class TemperatureScaler:
    """Divide the logits by one fitted constant.

    The strongest method available when miscalibration is global: on synthetic
    data with a single global distortion it beat per-class calibration by a
    factor of six, winning all 12 seeds. It has one parameter, so it cannot
    overfit a calibration set, and it is monotone in the logits, so **the
    predicted class never changes** -- accuracy is exactly preserved.

    That guarantee is also its ceiling. Because the same temperature is applied
    to every class, it cannot correct a distortion that differs by class; in that
    regime it barely improved on doing nothing. Run
    :func:`miscalibration_profile` first.

    Args:
        max_log_temperature: The search runs over ``log(T)`` in ``[-b, b]``.
            Widen only if the fitted temperature lands on a bound.

    Attributes:
        temperature_: Fitted temperature. Above 1 softens the predictions,
            below 1 sharpens.
        n_features_in_: Number of classes seen during fit.

    Notes:
        Temperature scaling preserves the class ordering *within each row*, but not
        the ordering of people *within a class*: the softmax denominator makes each
        calibrated probability depend on the whole row. Measured at 49.6% of adjacent
        within-class pairs inverted -- worse than one-vs-rest calibration followed by
        normalization. If you rank people by their probability of a given class,
        that reordering is real and no standard calibration metric reveals it.

    Examples:
        >>> import numpy as np
        >>> from calibre.multiclass import TemperatureScaler
        >>> rng = np.random.default_rng(0)
        >>> truth = rng.dirichlet(np.ones(4) * 0.7, size=2000)
        >>> y = np.array([rng.choice(4, p=t) for t in truth])

        An overconfident model, sharpened globally:

        >>> sharp = truth ** 2.2
        >>> P = sharp / sharp.sum(axis=1, keepdims=True)
        >>> scaler = TemperatureScaler().fit(P, y)

        It recovers a temperature above 1, softening the predictions back:

        >>> bool(scaler.temperature_ > 1.5)
        True

        And the predicted class is untouched, by construction:

        >>> Q = scaler.transform(P)
        >>> bool(np.all(Q.argmax(axis=1) == P.argmax(axis=1)))
        True

    See Also:
        miscalibration_profile : Tells you whether this method suits your data.
    """

    def __init__(self, max_log_temperature: float = 3.0) -> None:
        self.max_log_temperature = max_log_temperature

    def _logits(self, P: np.ndarray) -> np.ndarray:
        """Return log-probabilities, floored away from zero.

        Args:
            P: Probability matrix.

        Returns:
            ndarray: Logits.
        """
        return np.log(np.clip(P, _EPS, None))

    def _softmax(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Softmax of ``logits / temperature``, computed stably.

        Args:
            logits: Log-probabilities.
            temperature: Divisor.

        Returns:
            ndarray: Row-stochastic probabilities.
        """
        z = logits / temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def fit(self, P: np.ndarray, y: np.ndarray) -> TemperatureScaler:
        """Fit the temperature by minimising negative log-likelihood.

        Args:
            P: Predicted probabilities from a held-out calibration set.
            y: Integer class labels.

        Returns:
            TemperatureScaler: self.

        Raises:
            ValueError: If ``max_log_temperature`` is not positive, or the
                inputs are malformed.
        """
        from scipy.optimize import minimize_scalar

        if self.max_log_temperature <= 0:
            raise ValueError(
                f"max_log_temperature must be positive, got {self.max_log_temperature}"
            )
        P, y = _check_matrix(P, y)
        logits = self._logits(P)
        rows = np.arange(y.size)

        def nll(log_temperature: float) -> float:
            probs = self._softmax(logits, float(np.exp(log_temperature)))
            return float(-np.mean(np.log(np.clip(probs[rows, y], _EPS, None))))

        bound = float(self.max_log_temperature)
        best = minimize_scalar(nll, bounds=(-bound, bound), method="bounded")
        self.temperature_ = float(np.exp(best.x))
        self.n_features_in_ = P.shape[1]
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature.

        Args:
            P: Predicted probabilities.

        Returns:
            ndarray: Calibrated probabilities, rows summing to 1. Every row's
                argmax is unchanged.

        Raises:
            AttributeError: If called before :meth:`fit`.
        """
        if not hasattr(self, "temperature_"):
            raise AttributeError(
                f"{type(self).__name__} is not fitted yet. Call fit() first."
            )
        P = _check_probability_matrix(P, n_classes=self.n_features_in_)
        return self._softmax(self._logits(P), self.temperature_)

    def fit_transform(self, P: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit on ``(P, y)`` and return the calibrated probabilities.

        Args:
            P: Predicted probabilities.
            y: Integer class labels.

        Returns:
            ndarray: Calibrated probabilities.
        """
        return self.fit(P, y).transform(P)

    def __repr__(self) -> str:
        """Return a short description, including the fitted temperature."""
        if hasattr(self, "temperature_"):
            return f"TemperatureScaler(temperature_={self.temperature_:.4f})"
        return "TemperatureScaler()"
