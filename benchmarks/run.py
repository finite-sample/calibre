"""Run the benchmark grid and write ``results/raw.csv``.

::

    python -m benchmarks.run --quick        # offline and fast; what CI runs
    python -m benchmarks.run                # the committed grid
    python -m benchmarks.run --include-remote --include-large

Rows are independent and individually seeded, so the output is identical at any
``--n-jobs``. Nothing here reads the committed results; nothing downstream reruns
the grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

from . import config, datasets, measures, methods, models, protocol

RESULTS = Path(__file__).resolve().parent / "results"

# Identifiers written before the measures, so the CSV reads left to right from
# "which cell is this" to "what happened".
KEY_COLUMNS = (
    "dataset",
    "kind",
    "model",
    "seed",
    "method",
    "family",
    "n_fit",
    "n_test",
    "base_rate",
    "n_distinct_input",
    "seconds",
)
FIELDNAMES = (*KEY_COLUMNS, *measures.COLUMNS)


def _git_sha() -> str:
    """Return the current commit, or ``"unknown"`` outside a checkout.

    Returns:
        str: Short SHA.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _environment() -> dict[str, Any]:
    """Record what produced these numbers.

    Returns:
        dict: Versions, platform and commit.
    """
    import numpy as np
    import scipy
    import sklearn

    import calibre

    return {
        "calibre": calibre.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "netcal_available": methods.netcal_available(),
    }


def _cells(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    """Enumerate the (dataset, model, seed) grid.

    Synthetic designs construct the score directly, so they take the identity
    "model" rather than being crossed with three classifiers that would have
    nothing to do.

    Args:
        args: Parsed command line.

    Returns:
        list of tuple: The cells to run.
    """
    if args.quick:
        names = list(config.QUICK_DATASETS)
        model_names = list(config.QUICK_MODELS)
        seeds = list(config.QUICK_SEEDS)
    else:
        names = datasets.names(
            include_remote=args.include_remote, include_large=args.include_large
        )
        model_names = list(models.MODELS)
        seeds = list(config.SEEDS)

    cells: list[tuple[str, str, int]] = []
    for name in names:
        if name in datasets.SYNTHETIC:
            cells.extend((name, "identity", seed) for seed in seeds)
        else:
            cells.extend(
                (name, model, seed) for model, seed in product(model_names, seeds)
            )
    return cells


def main(argv: list[str] | None = None) -> int:
    """Run the grid.

    Args:
        argv: Command line, defaulting to :data:`sys.argv`.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small offline subset")
    parser.add_argument("--include-remote", action="store_true", help="fetch OpenML")
    parser.add_argument("--include-large", action="store_true", help="large datasets")
    parser.add_argument("--include-netcal", action="store_true", help="add netcal")
    parser.add_argument("--include-slow", action="store_true", help="add BBQ/ENIR")
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel workers")
    parser.add_argument("--out", type=Path, default=None, help="output CSV")
    args = parser.parse_args(argv)

    method_names = methods.available(
        include_netcal=args.include_netcal, include_slow=args.include_slow
    )
    cells = _cells(args)
    print(
        f"{len(cells)} cells x {len(method_names)} methods "
        f"= {len(cells) * len(method_names)} rows"
    )
    if args.include_netcal and not methods.netcal_available():
        print("  netcal requested but not installed; continuing without it")

    def one(cell: tuple[str, str, int]) -> list[dict[str, Any]]:
        dataset_name, model_name, seed = cell
        return protocol.run_cell(dataset_name, model_name, seed, method_names)

    if args.n_jobs == 1:
        rows: list[dict[str, Any]] = []
        for i, cell in enumerate(cells, 1):
            rows.extend(one(cell))
            print(f"  [{i}/{len(cells)}] {cell[0]}/{cell[1]}/seed={cell[2]}")
    else:
        from joblib import Parallel, delayed

        batches = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(one)(cell) for cell in cells
        )
        rows = [row for batch in batches for row in batch]

    # Canonical order, so a re-run produces a readable diff rather than a
    # reshuffled file.
    rows.sort(key=lambda r: (r["dataset"], r["model"], r["seed"], r["method"]))

    RESULTS.mkdir(exist_ok=True)
    out = args.out or (RESULTS / ("quick.csv" if args.quick else "raw.csv"))
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    (RESULTS / "environment.json").write_text(json.dumps(_environment(), indent=1))
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
