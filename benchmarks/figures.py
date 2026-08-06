"""Draw the benchmark's figures, through ``calibre.plots``.

::

    python -m benchmarks.figures

Every figure goes through the package's own plotting API rather than raw
matplotlib. That is deliberate: the benchmark dogfoods ``calibre.plots``, so a
regression in the plotting layer breaks the benchmark build instead of silently
producing a wrong picture in the documentation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = Path(__file__).resolve().parents[1] / "docs" / "source" / "_static" / "bench"


def _read(name: str) -> list[dict[str, str]]:
    """Read one aggregated CSV.

    Parameters
    ----------
    name
        File name inside ``results/``.

    Returns
    -------
    list of dict
        Rows.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    """
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run `python -m benchmarks.aggregate` first"
        )
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _float(value: str) -> float:
    """Parse a cell, mapping blanks to NaN.

    Parameters
    ----------
    value
        Cell contents.

    Returns
    -------
    float
        The value or NaN.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _save(figure, stem: str) -> None:
    """Write one figure as both SVG and PNG.

    The docs use the SVG. The README needs the PNG: it is rendered by GitHub and
    by PyPI, and neither handles remote SVG reliably.

    Parameters
    ----------
    figure
        Figure to write.
    stem
        File name without extension.
    """
    figure.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.png", dpi=140, bbox_inches="tight")


def frontier(summary: list[dict[str, str]], dataset: str, model: str) -> None:
    """Held-out score against distinct values retained.

    The headline figure: it answers the obvious objection to the resolution
    claim, which is that the extra distinct values might be noise. If they were,
    the methods keeping them would sit higher on the score axis. They do not.

    Parameters
    ----------
    summary
        Rows from ``summary.csv``.
    dataset
        Dataset to draw.
    model
        Model to draw.
    """
    import matplotlib.pyplot as plt

    from calibre.plots import plot_resolution_frontier, style_context

    rows = [r for r in summary if r["dataset"] == dataset and r["model"] == model]
    results = {
        r["method"]: (round(_float(r["n_distinct"])), _float(r["brier"]))
        for r in rows
        if _float(r["n_distinct"]) > 0
    }
    if not results:
        return

    with style_context():
        ax = plot_resolution_frontier(
            results,
            highlight=[m for m in results if m.startswith("calibre_")],
            score_label="held-out Brier score",
        )
        ax.set_title(f"{dataset} / {model}")
        _save(ax.figure, "resolution_frontier")
        plt.close(ax.figure)


def paired_deltas(paired: list[dict[str, str]], dataset: str, model: str) -> None:
    """Per-seed Brier differences against the baseline, with intervals.

    The honesty-critical figure. Levels would let seed variance masquerade as an
    effect; differencing within seed removes it, and an interval that spans zero
    is drawn spanning zero.

    Parameters
    ----------
    paired
        Rows from ``paired.csv``.
    dataset
        Dataset to draw.
    model
        Model to draw.
    """
    import matplotlib.pyplot as plt

    from calibre.plots import SEMANTIC, style_context

    rows = [r for r in paired if r["dataset"] == dataset and r["model"] == model]
    if not rows:
        return
    rows.sort(key=lambda r: _float(r["delta_brier"]))

    with style_context():
        figure, ax = plt.subplots(figsize=(7.0, 0.42 * len(rows) + 1.8))
        positions = np.arange(len(rows))
        for position, row in zip(positions, rows, strict=True):
            low, high = _float(row["delta_brier_lo"]), _float(row["delta_brier_hi"])
            beats = row["beats_baseline"] == "True"
            colour = SEMANTIC["calibre"] if beats else SEMANTIC["reference"]
            ax.plot([low, high], [position, position], color=colour, linewidth=2.0)
            ax.plot(
                [_float(row["delta_brier"])],
                [position],
                marker="o",
                markersize=6,
                color=colour,
            )
        ax.axvline(0.0, color=SEMANTIC["score"], linewidth=1.0, linestyle="--")
        ax.set_yticks(positions)
        ax.set_yticklabels([r["method"] for r in rows])
        ax.set_xlabel("Brier improvement over sklearn_isotonic (paired, per seed)")
        ax.set_title(f"{dataset} / {model}   (right of the line is better)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        _save(figure, "brier_deltas")
        plt.close(figure)


def resolution_barcode(dataset: str, model: str, seed: int) -> None:
    """The collapse barcode, recomputed on one cell.

    Parameters
    ----------
    dataset
        Dataset to draw.
    model
        Model to draw.
    seed
        Seed to draw.
    """
    import matplotlib.pyplot as plt

    from calibre.plots import plot_resolution_loss, style_context

    from . import datasets as datasets_module
    from . import methods, protocol

    dataset_obj = datasets_module.load(dataset, seed)
    fit_scores, fit_labels, test_scores, _, _ = protocol._scores_for_cell(
        dataset_obj, model, seed
    )
    wanted = [
        "sklearn_isotonic",
        "calibre_centered",
        "calibre_spline",
        "calibre_regularized",
        "calibre_relaxed_pava",
    ]
    outputs = {
        name: methods.calibrate(name, fit_scores, fit_labels, test_scores)
        for name in wanted
    }
    with style_context():
        ax = plot_resolution_loss(outputs, test_scores)
        ax.set_title(f"what each calibrator did to resolution ({dataset}/{model})")
        _save(ax.figure, "resolution_loss")
        plt.close(ax.figure)


def decomposition(summary: list[dict[str, str]], dataset: str, model: str) -> None:
    """Forecasters on the discrimination-miscalibration plane.

    Parameters
    ----------
    summary
        Rows from ``summary.csv``.
    dataset
        Dataset to draw.
    model
        Model to draw.
    """
    import matplotlib.pyplot as plt

    from calibre.plots import plot_mcb_dsc_plane, style_context

    rows = [r for r in summary if r["dataset"] == dataset and r["model"] == model]
    decompositions = {}
    for row in rows:
        mcb, dsc = _float(row["mcb"]), _float(row["dsc"])
        if not (np.isfinite(mcb) and np.isfinite(dsc)):
            continue
        brier = _float(row["brier"])
        decompositions[row["method"]] = {
            "MCB": mcb,
            "DSC": dsc,
            "UNC": brier - mcb + dsc,
            "mean_score": brier,
        }
    if len(decompositions) < 2:
        return

    with style_context():
        ax = plot_mcb_dsc_plane(decompositions)
        ax.set_title(f"{dataset} / {model}")
        _save(ax.figure, "mcb_dsc_plane")
        plt.close(ax.figure)


def main(argv: list[str] | None = None) -> int:
    """Draw every figure.

    Parameters
    ----------
    argv
        Command line.

    Returns
    -------
    int
        Exit status.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="overconfident")
    parser.add_argument("--model", default="identity")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = _read("summary.csv")
    paired = _read("paired.csv")

    frontier(summary, args.dataset, args.model)
    paired_deltas(paired, args.dataset, args.model)
    decomposition(summary, args.dataset, args.model)
    resolution_barcode(args.dataset, args.model, args.seed)
    print(f"wrote figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
