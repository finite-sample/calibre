"""Datasets for the benchmark.

Two kinds. **Real** datasets say whether the methods help on data anyone would
actually meet. **Synthetic** ones carry a known true probability for every row,
so error can be measured against the truth rather than against a noisy label --
much the strongest evidence available, and the reason the multiclass results in
``calibre.multiclass`` are stated as confidently as they are.

The synthetic set deliberately includes a regime where calibre's monotone methods
are expected to lose (``nonmonotone``). A benchmark with no losses is a marketing
document.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SYNTHETIC", "Dataset", "load", "names"]


@dataclass(frozen=True)
class Dataset:
    """One benchmark dataset.

    Attributes:
        name: Identifier used in the results table.
        X: Features, shape ``(n_samples, n_features)``.
        y: Binary labels.
        p_true: True event probability per row, when it is known. None for real data.
        kind: ``"real"`` or ``"synthetic"``.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    p_true: np.ndarray | None
    kind: str


# --------------------------------------------------------------------------- #
# Synthetic: the model's *scores* are constructed directly, so `X` is the score
# itself and the "model" is the identity. This isolates the calibrator from any
# question of how well a classifier fits.
# --------------------------------------------------------------------------- #


def _logistic(z: np.ndarray) -> np.ndarray:
    """Standard logistic function.

    Args:
        z: Log-odds.

    Returns:
        ndarray: Probabilities.
    """
    return 1.0 / (1.0 + np.exp(-z))


def _from_scores(
    name: str, scores: np.ndarray, p_true: np.ndarray, rng: np.random.Generator
) -> Dataset:
    """Draw labels from the true probabilities and package the result.

    Args:
        name: Dataset name.
        scores: The (miscalibrated) scores a model would report.
        p_true: True event probabilities.
        rng: Random generator.

    Returns:
        Dataset: The assembled dataset.
    """
    y = rng.binomial(1, p_true).astype(float)
    return Dataset(name, scores.reshape(-1, 1), y, p_true, "synthetic")


def _overconfident(n: int, rng: np.random.Generator) -> Dataset:
    """Log-odds inflated by 1.8: the textbook overconfident model."""
    z = rng.normal(0, 2, n)
    return _from_scores("overconfident", _logistic(1.8 * z), _logistic(z), rng)


def _underconfident(n: int, rng: np.random.Generator) -> Dataset:
    """Log-odds shrunk by 0.6, as heavy regularisation produces."""
    z = rng.normal(0, 2, n)
    return _from_scores("underconfident", _logistic(0.6 * z), _logistic(z), rng)


def _prior_shift(n: int, rng: np.random.Generator) -> Dataset:
    """A constant log-odds offset, as a changed base rate produces."""
    z = rng.normal(0, 2, n)
    return _from_scores("prior_shift", _logistic(z + 0.8), _logistic(z), rng)


def _heavy_tie(n: int, rng: np.random.Generator) -> Dataset:
    """Scores rounded to two decimals, as a vote-fraction model produces.

    Ties are where ``aggregate_ties`` and the granularity claim actually bite,
    and where a calibrator that splits tied scores across bins misbehaves.
    """
    z = rng.normal(0, 2, n)
    return _from_scores("heavy_tie", np.round(_logistic(1.8 * z), 2), _logistic(z), rng)


def _rare_event(n: int, rng: np.random.Generator) -> Dataset:
    """A 1% base rate, where the interesting region is a sliver near zero."""
    z = rng.normal(-4.6, 1.5, n)
    return _from_scores("rare_event", _logistic(1.6 * z), _logistic(z), rng)


def _small_n(n: int, rng: np.random.Generator) -> Dataset:
    """Overconfident, but with far too little data to fit a curve on.

    scikit-learn's own documentation warns that isotonic regression overfits
    below about a thousand calibration samples.
    """
    z = rng.normal(0, 2, min(n, 300))
    return _from_scores("small_n", _logistic(1.8 * z), _logistic(z), rng)


def _nonmonotone(n: int, rng: np.random.Generator) -> Dataset:
    """A genuinely non-monotone calibration curve.

    Included **because every monotone method in this package should lose here.**
    The true probability dips in the middle of the score range, which no monotone
    calibrator can express and which ``NearlyIsotonicCalibrator`` exists for.
    """
    scores = rng.uniform(0, 1, n)
    p_true = np.clip(scores + 0.25 * np.sin(2.0 * np.pi * scores), 0.01, 0.99)
    return _from_scores("nonmonotone", scores, p_true, rng)


SYNTHETIC = {
    "overconfident": _overconfident,
    "underconfident": _underconfident,
    "prior_shift": _prior_shift,
    "heavy_tie": _heavy_tie,
    "rare_event": _rare_event,
    "small_n": _small_n,
    "nonmonotone": _nonmonotone,
}

_SYNTHETIC_N = 4000

# --------------------------------------------------------------------------- #
# Real data. Only breast_cancer ships with scikit-learn; the rest need a fetch
# and are cached under benchmarks/.cache (gitignored).
# --------------------------------------------------------------------------- #

_OPENML = {
    "credit_g": ("credit-g", 1),
    "spambase": ("spambase", 1),
    "adult": ("adult", 2),
    "bank_marketing": ("bank-marketing", 1),
}

REAL = ("breast_cancer", *_OPENML, "covtype_bin")


def _cache_dir():
    """Return the fetch cache directory, creating it if needed.

    Returns:
        pathlib.Path: The cache directory.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent / ".cache"
    path.mkdir(exist_ok=True)
    return path


def _load_real(name: str) -> Dataset:
    """Load one real dataset.

    Args:
        name: Dataset name.

    Returns:
        Dataset: Features and binary labels, with ``p_true`` None.

    Raises:
        ValueError: If the name is not a known real dataset.
    """
    from sklearn.datasets import fetch_covtype, fetch_openml, load_breast_cancer

    if name == "breast_cancer":
        data = load_breast_cancer()
        return Dataset(name, data.data, data.target.astype(float), None, "real")

    if name == "covtype_bin":
        data = fetch_covtype(data_home=str(_cache_dir()))
        # Class 2 against the rest, subsampled: the full set is 581k rows and the
        # benchmark is about calibration, not about throughput.
        rng = np.random.default_rng(0)
        index = rng.choice(data.data.shape[0], size=50_000, replace=False)
        y = (data.target[index] == 2).astype(float)
        return Dataset(name, data.data[index], y, None, "real")

    if name in _OPENML:
        openml_name, version = _OPENML[name]
        data = fetch_openml(
            openml_name,
            version=version,
            as_frame=True,
            data_home=str(_cache_dir()),
        )
        target = data.target
        # OpenML targets arrive as strings with dataset-specific level names; the
        # positive class is whichever level is rarer, which is the convention for
        # all four of these.
        levels, counts = np.unique(np.asarray(target), return_counts=True)
        positive = levels[np.argmin(counts)]
        y = (np.asarray(target) == positive).astype(float)
        return Dataset(name, data.data, y, None, "real")

    raise ValueError(f"unknown real dataset {name!r}")


def names(include_remote: bool = False, include_large: bool = False) -> list[str]:
    """List available dataset names.

    Args:
        include_remote: Include datasets that need a network fetch.
        include_large: Include the large datasets.

    Returns:
        list of str: Dataset names.
    """
    from .config import LARGE_DATASETS, REMOTE_DATASETS

    out = ["breast_cancer", *SYNTHETIC]
    if include_remote:
        for name in REAL:
            if name == "breast_cancer":
                continue
            if name in LARGE_DATASETS and not include_large:
                continue
            if name in REMOTE_DATASETS:
                out.append(name)
    return out


def load(name: str, seed: int) -> Dataset:
    """Load or generate one dataset.

    Args:
        name: Dataset name.
        seed: Seed, used only by the synthetic generators. Real datasets are fixed, and the seed varies their train/test split instead.

    Returns:
        Dataset: The dataset.

    Raises:
        ValueError: If the name is unknown.
    """
    if name in SYNTHETIC:
        return SYNTHETIC[name](_SYNTHETIC_N, np.random.default_rng(seed))
    if name in REAL:
        return _load_real(name)
    raise ValueError(f"unknown dataset {name!r}")
