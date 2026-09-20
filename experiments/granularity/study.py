"""Run the finite-population experiment and retain every fitted prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata

from benchmarks import methods
from benchmarks.run import _environment

ROOT = Path(__file__).resolve().parent
DESIGNS = ("linear", "curved", "steps", "constant", "nonmonotone")
SIZES = (200, 1000, 5000)
SEEDS = tuple(range(30))
METHODS = (
    "uncalibrated",
    "sklearn_isotonic",
    "calibre_centered",
    "calibre_spline",
    "sklearn_platt",
    "sklearn_temperature",
    "calibre_nearly_isotonic",
)
LABELS = (
    "Original score",
    "Isotonic",
    "Centered isotonic",
    "I-spline",
    "Logistic (log-odds)",
    "Temperature",
    "Nearly isotonic",
)
PRECISIONS = ("exact", "six_decimals")
THRESHOLDS = np.linspace(0, 1, 501)
CAPACITIES = np.arange(1, 100) / 100
N_POPULATION = 1000
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260919
METRICS = (
    "pooling_loss",
    "miscalibration",
    "total_error",
    "expected_brier",
    "pooled_pairs",
    "reversed_pairs",
    "distinct",
    "spearman",
)


def population(design: str) -> tuple[np.ndarray, np.ndarray]:
    """Return all equally weighted reported scores and true risks."""
    s = (np.arange(N_POPULATION) + 0.5) / N_POPULATION
    risks = {
        "linear": s,
        "curved": 0.05 + 0.9 * s**2,
        "steps": np.select([s < 1 / 3, s < 2 / 3], [0.2, 0.5], default=0.8),
        "constant": np.full(s.size, 0.3),
        "nonmonotone": 0.5 + 0.35 * np.sin(2 * np.pi * s),
    }
    return expit(1.8 * logit(s)), risks[design]


def conditional_risk(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute oracle risk conditional on exact output in the full population."""
    _, inverse, counts = np.unique(q, return_inverse=True, return_counts=True)
    return (np.bincount(inverse, weights=p) / counts)[inverse]


def _validate(p: np.ndarray, q: np.ndarray) -> None:
    """Reject invalid population arrays before computing oracle quantities."""
    if p.ndim != 1 or q.shape != p.shape or p.size < 2:
        raise ValueError(
            "equal one-dimensional arrays with at least two types required"
        )
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("population risks and forecasts must be finite")
    if np.any((p < 0) | (p > 1) | (q < 0) | (q > 1)):
        raise ValueError("population risks and forecasts must lie in [0, 1]")


def losses(p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Decompose squared probability error for an equally weighted population."""
    _validate(p, q)
    r = conditional_risk(p, q)
    pool = float(np.mean((p - r) ** 2))
    calibration = float(np.mean((r - q) ** 2))
    total = float(np.mean((p - q) ** 2))
    if not np.isclose(total, pool + calibration, rtol=0, atol=1e-12):
        raise ArithmeticError("population decomposition failed")
    return {
        "pooling_loss": pool,
        "miscalibration": calibration,
        "total_error": total,
        "expected_brier": total + float(np.mean(p * (1 - p))),
    }


def ordering(q: np.ndarray) -> dict[str, float]:
    """Count ties and reversals relative to the strictly increasing input order."""
    _, counts = np.unique(q, return_counts=True)
    denominator = q.size * (q.size - 1) / 2
    reversed_count = sum(np.count_nonzero(q[i + 1 :] < x) for i, x in enumerate(q))
    ranks = rankdata(q)
    correlation = (
        float(np.corrcoef(np.arange(q.size), ranks)[0, 1])
        if counts.size > 1
        else float("nan")
    )
    return {
        "pooled_pairs": float(np.sum(counts * (counts - 1) / 2) / denominator),
        "reversed_pairs": float(reversed_count / denominator),
        "distinct": float(counts.size),
        "spearman": correlation,
    }


def regret_curves(
    p: np.ndarray, q: np.ndarray, thresholds: np.ndarray = THRESHOLDS
) -> np.ndarray:
    """Return information-loss and implemented-rule regret at each threshold."""
    _validate(p, q)
    r = conditional_risk(p, q)
    margin = p[:, None] - thresholds
    oracle = np.maximum(margin, 0).mean(axis=0)
    information = oracle - np.maximum(r[:, None] - thresholds, 0).mean(axis=0)
    implemented = oracle - (margin * (q[:, None] > thresholds)).mean(axis=0)
    if min(information.min(), implemented.min()) < -1e-12:
        raise ArithmeticError("oracle decision dominated")
    return np.stack([information, implemented])


def capacity_values(
    p: np.ndarray, q: np.ndarray, capacities: np.ndarray = CAPACITIES
) -> np.ndarray:
    """Expected successes per 1,000 with random and original-score tie-breaking.

    Inputs are ordered by strictly increasing original score. Fractional boundary
    selections are integrated exactly, including the random tie lottery.
    """
    _validate(p, q)
    order = np.lexsort((-np.arange(q.size), -q))
    sorted_q, sorted_p = q[order], p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_q)) + 1]
    ends = np.r_[starts[1:], q.size]
    random_p = np.repeat(
        np.add.reduceat(sorted_p, starts) / (ends - starts), ends - starts
    )
    boundaries = np.arange(q.size + 1)
    return np.stack(
        [
            np.interp(capacities * q.size, boundaries, np.r_[0, np.cumsum(values)])
            * 1000
            / q.size
            for values in (random_p, sorted_p)
        ]
    )


def configuration() -> dict:
    """Serialize the fixed design before looking at its results."""
    return {
        "designs": DESIGNS,
        "sizes": SIZES,
        "seeds": SEEDS,
        "methods": METHODS,
        "precisions": PRECISIONS,
        "population_size": N_POPULATION,
        "thresholds": THRESHOLDS.tolist(),
        "capacities": CAPACITIES.tolist(),
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "risk_functions": {
            "linear": "s",
            "curved": "0.05+0.9*s**2",
            "steps": "0.2,0.5,0.8 on equal thirds",
            "constant": "0.3",
            "nonmonotone": "0.5+0.35*sin(2*pi*s)",
        },
    }


def run_cell(cell: tuple[str, int, int]) -> tuple[tuple, np.ndarray]:
    """Fit every method to an identical calibration sample in one replicate."""
    design, n, seed = cell
    scores, p = population(design)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, n, DESIGNS.index(design)])
    )
    indices = rng.integers(0, scores.size, n)
    labels = rng.binomial(1, p[indices])
    predictions = np.stack(
        [methods.calibrate(name, scores[indices], labels, scores) for name in METHODS]
    )
    for q in predictions:
        _validate(p, q)
    return cell, predictions


def evaluate(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute every metric and decision curve from retained predictions."""
    shape = (len(DESIGNS), len(SIZES), len(SEEDS), len(METHODS), len(PRECISIONS))
    if predictions.shape != (*shape[:4], N_POPULATION):
        raise ValueError("incomplete prediction grid")
    metrics = np.empty((*shape, len(METRICS)))
    curves = np.empty((*shape, 2, THRESHOLDS.size))
    capacity = np.empty((*shape, 2, CAPACITIES.size))
    for d, design in enumerate(DESIGNS):
        _, p = population(design)
        for n, seed, m in np.ndindex(shape[1:4]):
            for precision in range(len(PRECISIONS)):
                q = predictions[d, n, seed, m]
                if precision:
                    q = np.round(q, 6)
                row = losses(p, q) | ordering(q)
                idx = (d, n, seed, m, precision)
                metrics[idx] = [row[key] for key in METRICS]
                curves[idx] = regret_curves(p, q)
                capacity[idx] = capacity_values(p, q)
    return metrics, curves, capacity


def main() -> None:
    """Run the fixed simulation or recompute evaluations from saved predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out / "predictions.npz"
    if args.evaluate_only:
        saved = json.loads((args.out / "config.json").read_text())
        if saved != json.loads(json.dumps(configuration())):
            raise ValueError("saved configuration differs; refit the experiment")
        predictions = np.load(prediction_path)["predictions"]
    else:
        environment = _environment()
        sources = [ROOT / "study.py", ROOT / "DESIGN.md"]
        environment["experiment_sources"] = {
            str(path.relative_to(ROOT.parent.parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
        }
        (args.out / "environment.json").write_text(
            json.dumps(environment, indent=2) + "\n"
        )
        (args.out / "config.json").write_text(
            json.dumps(configuration(), indent=2) + "\n"
        )
        predictions = np.full((5, 3, 30, 7, N_POPULATION), np.nan)
        cells = [(d, n, s) for d in DESIGNS for n in SIZES for s in SEEDS]
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for count, (cell, fitted) in enumerate(executor.map(run_cell, cells), 1):
                d, n, s = cell
                predictions[DESIGNS.index(d), SIZES.index(n), SEEDS.index(s)] = fitted
                if count % 10 == 0:
                    print(f"Fitted {count}/{len(cells)} cells", flush=True)
        np.savez_compressed(prediction_path, predictions=predictions)
    metrics, curves, capacity = evaluate(predictions)
    np.savez_compressed(
        args.out / "evaluation.npz", metrics=metrics, curves=curves, capacity=capacity
    )
    manifest = {
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "evaluation_sha256": hashlib.sha256(
            (args.out / "evaluation.npz").read_bytes()
        ).hexdigest(),
        "configuration": configuration(),
        "evaluation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "dimensions": ["design", "size", "seed", "method", "precision"],
        "metric_names": METRICS,
        "curve_names": ["information_regret", "implemented_regret"],
        "capacity_names": ["random_ties", "original_score_ties"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved complete experiment to {args.out}")


if __name__ == "__main__":
    main()
