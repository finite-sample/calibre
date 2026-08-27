"""One benchmark cell: a dataset, a model, and a seed.

This module holds the fairness controls, so they are in one place and visible in
one diff:

1. **The test split is touched exactly once**, at the end, to score. Nothing is
   selected, tuned or inspected on it.
2. **Calibrators fit on out-of-fold model scores.** A model's scores on its own
   training rows are already too good, so a calibrator fitted there learns the
   wrong correction -- the mistake calibre's own README leads with.
3. **Every calibrator in a cell sees identical inputs.** The out-of-fold scores
   and the test scores are computed once and shared, so the calibrator is the
   only thing that varies within a cell. Anything else would confound the
   comparison with resampling noise.
4. **A self-check that would catch us cheating.** ``calibre_isotonic`` must
   reproduce ``sklearn_isotonic`` to 1e-12. If calibre's wrapper ever diverges,
   this fails loudly rather than letting calibre look spuriously better.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from . import config, datasets, measures, methods, models

__all__ = ["ISOTONIC_AGREEMENT_TOLERANCE", "run_cell"]

# The wrapper is a thin pass-through to sklearn, so agreement should be to
# floating-point noise. A looser tolerance here would defeat the purpose.
ISOTONIC_AGREEMENT_TOLERANCE = 1e-12


def _scores_for_cell(
    dataset: datasets.Dataset, model_name: str, seed: int
) -> tuple[np.ndarray, ...]:
    """Produce the shared inputs every calibrator in this cell will see.

    Args:
        dataset: The dataset.
        model_name: Base model name, or ``"identity"`` for a synthetic design
            where the score *is* the construction and there is no model to fit.
        seed: Seed for the split and the model.

    Returns:
        tuple of ndarray: ``(fit_scores, fit_labels, test_scores, test_labels,
        test_p_true)``. ``test_p_true`` is an empty array when the truth is
        unknown.
    """
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_predict,
        train_test_split,
    )

    y = np.asarray(dataset.y, dtype=float)
    stratify = y if np.unique(y).size > 1 else None

    if dataset.kind == "synthetic":
        # The "model" is the identity: these designs construct the reported score
        # directly, which isolates the calibrator from any question of how well a
        # classifier happened to fit.
        scores = np.asarray(dataset.X, dtype=float).ravel()
        p_true = (
            np.asarray(dataset.p_true, dtype=float)
            if dataset.p_true is not None
            else None
        )
        index = np.arange(scores.size)
        fit_idx, test_idx = train_test_split(
            index,
            test_size=config.TEST_SIZE,
            random_state=seed,
            stratify=stratify,
        )
        return (
            scores[fit_idx],
            y[fit_idx],
            scores[test_idx],
            y[test_idx],
            p_true[test_idx] if p_true is not None else np.empty(0),
        )

    X = dataset.X
    index = np.arange(y.size)
    fit_idx, test_idx = train_test_split(
        index, test_size=config.TEST_SIZE, random_state=seed, stratify=stratify
    )
    X_fit = X.iloc[fit_idx] if hasattr(X, "iloc") else X[fit_idx]
    X_test = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
    y_fit, y_test = y[fit_idx], y[test_idx]

    folds = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    # Out-of-fold: every score comes from a model that never saw that row.
    oof = cross_val_predict(
        models.build(model_name, seed), X_fit, y_fit, cv=folds, method="predict_proba"
    )[:, 1]

    # The model that scores the test half is fitted on all of the fitting half,
    # which is what a practitioner would deploy.
    deployed = models.build(model_name, seed).fit(X_fit, y_fit)
    test_scores = np.asarray(deployed.predict_proba(X_test)[:, 1], dtype=float)

    return oof, y_fit, test_scores, y_test, np.empty(0)


def run_cell(
    dataset_name: str,
    model_name: str,
    seed: int,
    method_names: list[str],
    n_bins: int = config.N_BINS,
) -> list[dict[str, Any]]:
    """Evaluate every method on one (dataset, model, seed) cell.

    Args:
        dataset_name: Dataset to load.
        model_name: Base model, or ``"identity"`` for synthetic designs.
        seed: Seed for the dataset, the split and the model.
        method_names: Methods to compare.
        n_bins: Bin count for the fixed-bin estimators.

    Returns:
        list of dict: One row per method, carrying the cell's identifiers,
        every measure, and the fit and transform times.

    Raises:
        AssertionError: If ``calibre_isotonic`` and ``sklearn_isotonic``
            disagree by more than :data:`ISOTONIC_AGREEMENT_TOLERANCE`.
    """
    dataset = datasets.load(dataset_name, seed)
    fit_scores, fit_labels, test_scores, test_labels, test_p_true = _scores_for_cell(
        dataset, model_name, seed
    )
    p_true = test_p_true if test_p_true.size else None

    rows: list[dict[str, Any]] = []
    calibrated_cache: dict[str, np.ndarray] = {}

    for name in method_names:
        started = time.perf_counter()
        calibrated = methods.calibrate(name, fit_scores, fit_labels, test_scores)
        elapsed = time.perf_counter() - started
        calibrated_cache[name] = calibrated

        row: dict[str, Any] = {
            "dataset": dataset_name,
            "kind": dataset.kind,
            "model": model_name,
            "seed": seed,
            "method": name,
            "family": methods.METHODS[name],
            "n_fit": int(fit_scores.size),
            "n_test": int(test_scores.size),
            "base_rate": float(np.mean(test_labels)),
            "n_distinct_input": int(np.unique(np.round(test_scores, 6)).size),
            "seconds": float(elapsed),
        }
        row.update(
            measures.evaluate(test_labels, calibrated, test_scores, p_true, n_bins)
        )
        rows.append(row)

    # The self-check. calibre's IsotonicCalibrator is a thin wrapper over
    # sklearn's, so any divergence is a bug, and a benchmark that hid it would be
    # reporting calibre's advantage over its own baseline.
    if {"calibre_isotonic", "sklearn_isotonic"} <= calibrated_cache.keys():
        gap = float(
            np.max(
                np.abs(
                    calibrated_cache["calibre_isotonic"]
                    - calibrated_cache["sklearn_isotonic"]
                )
            )
        )
        if gap > ISOTONIC_AGREEMENT_TOLERANCE:
            # Raised rather than asserted: `python -O` strips assert, and this
            # is the check the benchmark's credibility rests on.
            raise AssertionError(
                f"calibre_isotonic and sklearn_isotonic differ by {gap:.3e} on "
                f"{dataset_name}/{model_name}/seed={seed}; calibre's wrapper has "
                "diverged from the baseline it is measured against"
            )

    return rows
