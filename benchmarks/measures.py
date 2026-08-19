"""What gets recorded for one calibrated test set.

Deliberately several numbers rather than one. A composite score is where a thumb
goes on the scale, so score and resolution stay on separate axes and the reader
does the trading off.
"""

from __future__ import annotations

import numpy as np

__all__ = ["COLUMNS", "evaluate"]

# Declared here so `aggregate.py` can assert the schema rather than discover it.
COLUMNS: tuple[str, ...] = (
    "brier",
    "log_loss",
    "mcb",
    "dsc",
    "unc",
    "mcb_log",
    "dsc_log",
    "unc_log",
    "plugin_ece",
    "debiased_ece",
    "sweep_ece",
    "sweep_bins",
    "smece",
    "smece_sigma",
    "n_distinct",
    "distinct_ratio",
    "spearman_to_raw",
    "auc",
    "auc_raw",
    "tie_preservation",
    "true_error",
)


def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean logarithmic score, clipped away from the singularities.

    Args:
        y_true: Binary outcomes.
        y_pred: Predicted probabilities.

    Returns:
        float: Mean log loss.
    """
    eps = np.finfo(float).eps
    p = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log1p(-p)))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_scores: np.ndarray,
    p_true: np.ndarray | None,
    n_bins: int,
) -> dict[str, float]:
    """Measure one calibrated test set every way the benchmark reports.

    Args:
        y_true: Binary outcomes on the held-out half.
        y_pred: Calibrated probabilities.
        raw_scores: The uncalibrated model scores for the same rows, so the
            effect of calibration on ranking can be measured.
        p_true: True event probabilities when the dataset is synthetic, else
            None. When present this gives ``true_error``, which is much the
            strongest evidence available -- error against the truth rather
            than against a noisy label.
        n_bins: Bin count for the fixed-bin estimators.

    Returns:
        dict: One value per entry in :data:`COLUMNS`.
    """
    from sklearn.metrics import roc_auc_score

    from calibre import score_decomposition
    from calibre.metrics import (
        _spearman,
        brier_score,
        debiased_calibration_error,
        plugin_calibration_error,
        smooth_calibration_error,
        sweep_calibration_error,
        tie_preservation_score,
    )

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    raw_scores = np.asarray(raw_scores, dtype=float)

    brier = score_decomposition(y_pred, y_true, score="brier")
    logs = score_decomposition(y_pred, y_true, score="log")
    sweep, sweep_bins = sweep_calibration_error(y_true, y_pred, return_n_bins=True)
    smece, sigma = smooth_calibration_error(y_true, y_pred, return_sigma=True)

    n = y_true.size
    n_distinct = int(np.unique(np.round(y_pred, 6)).size)

    # A calibrator is meant to be monotone, so AUC should be unchanged. Recording
    # both makes any reordering visible instead of implicit.
    def safe_auc(scores: np.ndarray) -> float:
        if np.unique(y_true).size < 2:
            return float("nan")
        return float(roc_auc_score(y_true, scores))

    return {
        "brier": brier_score(y_true, y_pred),
        "log_loss": _log_loss(y_true, y_pred),
        "mcb": float(brier["MCB"]),
        "dsc": float(brier["DSC"]),
        "unc": float(brier["UNC"]),
        "mcb_log": float(logs["MCB"]),
        "dsc_log": float(logs["DSC"]),
        "unc_log": float(logs["UNC"]),
        "plugin_ece": plugin_calibration_error(y_true, y_pred, n_bins, 2),
        "debiased_ece": debiased_calibration_error(y_true, y_pred, n_bins),
        "sweep_ece": float(sweep),
        "sweep_bins": float(sweep_bins),
        "smece": float(smece),
        "smece_sigma": float(sigma),
        "n_distinct": float(n_distinct),
        "distinct_ratio": float(n_distinct / n) if n else float("nan"),
        "spearman_to_raw": _spearman(raw_scores, y_pred),
        "auc": safe_auc(y_pred),
        "auc_raw": safe_auc(raw_scores),
        "tie_preservation": tie_preservation_score(raw_scores, y_pred)
        if n <= 4000
        else float("nan"),
        # Error against the truth, available only where the truth is known.
        "true_error": float(np.mean(np.abs(y_pred - np.asarray(p_true, dtype=float))))
        if p_true is not None
        else float("nan"),
    }
