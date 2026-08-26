"""Turn ``results/raw.csv`` into the two tables the docs page reads.

::

    python -m benchmarks.aggregate

Writes ``summary.csv`` (levels, for the table) and ``paired.csv`` (differences
against the baseline, for the claim).

Why paired differences
----------------------
Seed variance dwarfs the effect being measured. On the quick grid the spread in
held-out Brier across seeds is an order of magnitude larger than the gap between
calibrators, so a table of means-of-levels invites reading noise as a result.
Differencing *within* a seed removes the dataset draw, the model fit and the
split, leaving only the calibrator -- which is the only thing that varied.

The bootstrap interval here resamples **seeds**, not rows: the seed is the unit
of replication, and pooling rows would multiply the apparent sample size while
adding almost no information.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config, measures

RESULTS = Path(__file__).resolve().parent / "results"

# Reported as levels. Everything else is a difference against the baseline.
LEVEL_COLUMNS = (
    "brier",
    "log_loss",
    "mcb",
    "dsc",
    "smece",
    "debiased_ece",
    "n_distinct",
    "distinct_ratio",
    "auc",
    "true_error",
    "seconds",
)


def _read(path: Path) -> list[dict[str, str]]:
    """Read a results CSV.

    Args:
        path: File to read.

    Returns:
        list of dict: Rows.

    Raises:
        FileNotFoundError: If the file is missing, with the command that produces it.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run `python -m benchmarks.run` first"
        )
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _number(value: str) -> float:
    """Parse a CSV cell as a float, mapping blanks and NaN alike.

    Args:
        value: Cell contents.

    Returns:
        float: The value, or NaN.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def check_completeness(rows: list[dict[str, str]], expected_seeds: int) -> None:
    """Refuse to summarize a cell that is missing seeds.

    A dataset that errored on half its seeds would otherwise be averaged over
    whatever survived and reported alongside the rest, which is how a benchmark
    quietly starts flattering whichever method happened not to crash.

    Args:
        rows: Raw rows.
        expected_seeds: Seeds each (dataset, model, method) should have.

    Raises:
        ValueError: If any cell is incomplete, naming the worst offenders.
    """
    counts: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        counts[(row["dataset"], row["model"], row["method"])].add(row["seed"])

    short = {
        key: len(seeds) for key, seeds in counts.items() if len(seeds) != expected_seeds
    }
    if short:
        worst = sorted(short.items(), key=lambda kv: kv[1])[:5]
        listing = "; ".join(f"{'/'.join(k)} has {v}" for k, v in worst)
        raise ValueError(
            f"{len(short)} cells do not have {expected_seeds} seeds ({listing}). "
            "Summarising an incomplete grid would hide whichever configuration "
            "failed; re-run those cells or narrow the grid deliberately."
        )


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Mean each level column over seeds, per dataset/model/method.

    Args:
        rows: Raw rows.

    Returns:
        list of dict: Summary rows.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["method"])].append(row)

    out = []
    for (dataset, model, method), group in sorted(grouped.items()):
        record: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "method": method,
            "family": group[0]["family"],
            "kind": group[0]["kind"],
            "n_seeds": len(group),
        }
        for column in LEVEL_COLUMNS:
            values = [_number(r[column]) for r in group]
            finite = [v for v in values if np.isfinite(v)]
            record[column] = round(statistics.fmean(finite), 6) if finite else ""
        out.append(record)
    return out


def _paired_bootstrap(
    differences: list[float], n_resamples: int, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval for a mean difference, resampling seeds.

    Args:
        differences: One per-seed difference.
        n_resamples: Bootstrap resamples.
        seed: Generator seed.

    Returns:
        tuple of float: ``(lower, upper)`` at 95%.
    """
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array(
        [
            values[rng.integers(0, values.size, values.size)].mean()
            for _ in range(n_resamples)
        ]
    )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pair_against_baseline(
    rows: list[dict[str, str]], baseline: str, n_resamples: int
) -> list[dict[str, object]]:
    """Difference every method against the baseline, within seed.

    Args:
        rows: Raw rows.
        baseline: Method every other is compared against.
        n_resamples: Bootstrap resamples.

    Returns:
        list of dict: One row per dataset/model/method, with the mean difference,
        its interval and the win count.
    """
    indexed: dict[tuple[str, str, str, str], dict[str, str]] = {
        (r["dataset"], r["model"], r["seed"], r["method"]): r for r in rows
    }
    combos: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for dataset, model, seed, method in indexed:
        combos[(dataset, model, method)].append(seed)

    out = []
    for (dataset, model, method), seeds in sorted(combos.items()):
        if method == baseline:
            continue
        deltas_brier, deltas_mcb, distinct_ratio = [], [], []
        for seed in seeds:
            base = indexed.get((dataset, model, seed, baseline))
            here = indexed.get((dataset, model, seed, method))
            if base is None or here is None:
                continue
            # Signed so that positive means the method beat the baseline.
            deltas_brier.append(_number(base["brier"]) - _number(here["brier"]))
            deltas_mcb.append(_number(base["mcb"]) - _number(here["mcb"]))
            denominator = _number(base["n_distinct"])
            if denominator > 0:
                distinct_ratio.append(_number(here["n_distinct"]) / denominator)

        if not deltas_brier:
            continue
        low, high = _paired_bootstrap(deltas_brier, n_resamples)
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "baseline": baseline,
                "n_seeds": len(deltas_brier),
                "delta_brier": round(statistics.fmean(deltas_brier), 6),
                "delta_brier_lo": round(low, 6),
                "delta_brier_hi": round(high, 6),
                # The honest reading: an interval spanning zero is shown spanning
                # zero rather than being reported as a win.
                "beats_baseline": bool(low > 0.0),
                "wins": sum(1 for d in deltas_brier if d > 0),
                "delta_mcb": round(statistics.fmean(deltas_mcb), 6),
                "distinct_ratio_vs_baseline": round(statistics.fmean(distinct_ratio), 3)
                if distinct_ratio
                else "",
            }
        )
    return out


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV.

    Args:
        path: Destination.
        rows: Rows to write.
    """
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main(argv: list[str] | None = None) -> int:
    """Aggregate the raw results.

    Args:
        argv: Command line.

    Returns:
        int: Exit status.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RESULTS / "raw.csv")
    parser.add_argument(
        "--skip-completeness",
        action="store_true",
        help="summarize a partial grid; for exploration only",
    )
    args = parser.parse_args(argv)

    rows = _read(args.raw)
    missing = set(measures.COLUMNS) - set(rows[0])
    if missing:
        raise ValueError(f"{args.raw} is missing columns {sorted(missing)}")

    if not args.skip_completeness:
        expected = len({r["seed"] for r in rows})
        check_completeness(rows, expected)

    _write(RESULTS / "summary.csv", summarize(rows))
    _write(
        RESULTS / "paired.csv",
        pair_against_baseline(rows, config.BASELINE, config.N_BOOTSTRAP),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
