"""The calibrators under comparison, plus the baselines they answer to.

Every one is constructed at **library defaults**. Tuning calibre's methods
against an untuned scikit-learn isotonic baseline would decide the comparison by
construction rather than by measurement, so ``config.CALIBRATOR_DEFAULTS_ONLY``
records the rule and this module keeps it.

The one asymmetry worth naming: :class:`~calibre.SplineCalibrator` and the
``"auto"`` defaults of the relaxed and regularized calibrators select their own
hyperparameters by internal cross-validation. That is a real advantage over a
fixed-hyperparameter competitor, and it is paid for in the fit time this
benchmark also records.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

__all__ = ["METHODS", "available", "calibrate", "netcal_available"]

# Name -> the family it belongs to, used only for grouping in the figures.
METHODS: dict[str, str] = {
    "uncalibrated": "none",
    "sklearn_isotonic": "sklearn",
    "sklearn_platt": "sklearn",
    "sklearn_temperature": "sklearn",
    "calibre_isotonic": "calibre",
    "calibre_centered": "calibre",
    "calibre_spline": "calibre",
    "calibre_relaxed_pava": "calibre",
    "calibre_regularized": "calibre",
    "calibre_nearly_isotonic": "calibre",
    "netcal_beta": "netcal",
    "netcal_bbq": "netcal",
}

_NETCAL = frozenset({"netcal_beta", "netcal_bbq"})
_SLOW = frozenset({"netcal_bbq"})


def netcal_available() -> bool:
    """Whether ``netcal`` can be imported.

    Deliberately optional and off by default. netcal drags its own numerics
    stack, and the moment it lags a Python release a hard dependency would make
    the whole harness un-runnable -- which silently stops the benchmark being
    re-run, the failure this design exists to prevent.

    Returns
    -------
    bool
        True when netcal is importable.
    """
    return importlib.util.find_spec("netcal") is not None


def available(include_netcal: bool = False, include_slow: bool = False) -> list[str]:
    """List the methods that can run in this environment.

    Parameters
    ----------
    include_netcal
        Include the netcal comparators, if netcal is installed.
    include_slow
        Include methods that take minutes on tens of thousands of rows.

    Returns
    -------
    list of str
        Method names, in a stable order.
    """
    out = []
    for name in METHODS:
        if name in _NETCAL and not (include_netcal and netcal_available()):
            continue
        if name in _SLOW and not include_slow:
            continue
        out.append(name)
    return out


def _build(name: str) -> Any:
    """Construct one calibre calibrator at its library defaults.

    Parameters
    ----------
    name
        Method name.

    Returns
    -------
    object
        An unfitted calibrator.

    Raises
    ------
    ValueError
        If the name is not a calibre method.
    """
    from calibre import (
        CenteredIsotonicCalibrator,
        IsotonicCalibrator,
        NearlyIsotonicCalibrator,
        RegularizedIsotonicCalibrator,
        RelaxedPAVACalibrator,
        SplineCalibrator,
    )

    builders = {
        "calibre_isotonic": IsotonicCalibrator,
        "calibre_centered": CenteredIsotonicCalibrator,
        "calibre_spline": SplineCalibrator,
        "calibre_relaxed_pava": RelaxedPAVACalibrator,
        "calibre_regularized": RegularizedIsotonicCalibrator,
        "calibre_nearly_isotonic": NearlyIsotonicCalibrator,
    }
    if name not in builders:
        raise ValueError(f"{name!r} is not a calibre method")
    return builders[name]()


def _sklearn_isotonic(fit_scores, fit_labels, test_scores):
    """scikit-learn's isotonic regression, the baseline everything answers to.

    Parameters
    ----------
    fit_scores
        Out-of-fold model scores.
    fit_labels
        Labels for those scores.
    test_scores
        Scores to calibrate.

    Returns
    -------
    ndarray
        Calibrated test probabilities.
    """
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(fit_scores, fit_labels)
    return np.asarray(model.predict(test_scores), dtype=float)


def _sklearn_platt(fit_scores, fit_labels, test_scores):
    """Platt scaling: a logistic fit on the log-odds of the score.

    Fitted on the logit rather than the raw probability, which is what
    ``CalibratedClassifierCV(method="sigmoid")`` does internally and what makes
    it a *scaling* rather than an arbitrary logistic regression.

    Parameters
    ----------
    fit_scores
        Out-of-fold model scores.
    fit_labels
        Labels for those scores.
    test_scores
        Scores to calibrate.

    Returns
    -------
    ndarray
        Calibrated test probabilities.
    """
    from sklearn.linear_model import LogisticRegression

    eps = np.finfo(float).eps

    def logit(p):
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p) - np.log1p(-p)

    model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    model.fit(logit(fit_scores).reshape(-1, 1), fit_labels)
    return np.asarray(
        model.predict_proba(logit(test_scores).reshape(-1, 1))[:, 1], dtype=float
    )


def _sklearn_temperature(fit_scores, fit_labels, test_scores):
    """One-parameter temperature scaling on the log-odds.

    scikit-learn 1.8 added ``method="temperature"`` to
    :class:`~sklearn.calibration.CalibratedClassifierCV`. That wrapper wants an
    estimator rather than scores, so the same one-parameter model is fitted
    directly here: a logistic fit on the logit with the intercept held at zero.

    Parameters
    ----------
    fit_scores
        Out-of-fold model scores.
    fit_labels
        Labels for those scores.
    test_scores
        Scores to calibrate.

    Returns
    -------
    ndarray
        Calibrated test probabilities.
    """
    from scipy.optimize import minimize_scalar

    eps = np.finfo(float).eps

    def logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p) - np.log1p(-p)

    z_fit = logit(fit_scores)

    def negative_log_likelihood(log_temperature: float) -> float:
        scaled = 1.0 / (1.0 + np.exp(-z_fit / np.exp(log_temperature)))
        scaled = np.clip(scaled, eps, 1 - eps)
        return float(
            -np.mean(fit_labels * np.log(scaled) + (1 - fit_labels) * np.log1p(-scaled))
        )

    best = minimize_scalar(
        negative_log_likelihood, bounds=(-3.0, 3.0), method="bounded"
    )
    temperature = float(np.exp(best.x))
    return np.asarray(
        1.0 / (1.0 + np.exp(-logit(test_scores) / temperature)), dtype=float
    )


def _netcal(name: str, fit_scores, fit_labels, test_scores):
    """A netcal comparator, imported only when actually used.

    Parameters
    ----------
    name
        ``"netcal_beta"`` or ``"netcal_bbq"``.
    fit_scores
        Out-of-fold model scores.
    fit_labels
        Labels for those scores.
    test_scores
        Scores to calibrate.

    Returns
    -------
    ndarray
        Calibrated test probabilities.
    """
    if name == "netcal_beta":
        from netcal.scaling import BetaCalibration as Model
    else:
        from netcal.binning import BBQ

        Model = BBQ

    model = Model()
    model.fit(np.asarray(fit_scores, dtype=float), np.asarray(fit_labels, dtype=int))
    return np.asarray(
        model.transform(np.asarray(test_scores, dtype=float)), dtype=float
    )


def calibrate(name: str, fit_scores, fit_labels, test_scores):
    """Fit one method on the out-of-fold scores and apply it to the test scores.

    Parameters
    ----------
    name
        Method name from :data:`METHODS`.
    fit_scores
        Out-of-fold model scores from the fitting half.
    fit_labels
        Labels for those scores.
    test_scores
        Scores from the held-out half, to be calibrated.

    Returns
    -------
    ndarray
        Calibrated probabilities for ``test_scores``.

    Raises
    ------
    ValueError
        If the name is unknown.
    """
    fit_scores = np.asarray(fit_scores, dtype=float)
    fit_labels = np.asarray(fit_labels, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)

    if name == "uncalibrated":
        return test_scores
    if name == "sklearn_isotonic":
        return _sklearn_isotonic(fit_scores, fit_labels, test_scores)
    if name == "sklearn_platt":
        return _sklearn_platt(fit_scores, fit_labels, test_scores)
    if name == "sklearn_temperature":
        return _sklearn_temperature(fit_scores, fit_labels, test_scores)
    if name in _NETCAL:
        return _netcal(name, fit_scores, fit_labels, test_scores)
    if name.startswith("calibre_"):
        model = _build(name)
        model.fit(fit_scores, fit_labels)
        return np.asarray(model.transform(test_scores), dtype=float)
    raise ValueError(f"unknown method {name!r}")
