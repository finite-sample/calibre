"""Generate manuscript exhibits from the complete offline benchmark.

Run ``python -m benchmarks.paper`` after ``python -m benchmarks.aggregate``.
The paper build reads committed results and does not rerun the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from . import aggregate, config, datasets, methods, models, protocol

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
OUTPUT = ROOT / "ms" / "generated"
LABELS = {
    "uncalibrated": "Uncalibrated",
    "sklearn_isotonic": "Isotonic",
    "sklearn_platt": "Logistic (log-odds)",
    "sklearn_temperature": "Temperature",
    "calibre_isotonic": "Calibre isotonic",
    "calibre_centered": "Centered isotonic",
    "calibre_spline": "I-spline",
    "calibre_nearly_isotonic": "Nearly isotonic",
}
ORDER = tuple(LABELS)
VISIBLE = tuple(m for m in ORDER if m != "calibre_isotonic")
METRICS = ("brier", "mcb", "log_loss", "auc", "n_distinct", "true_error", "seconds")


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read benchmark records without changing numeric precision."""
    with path.open() as handle:
        return list(csv.DictReader(handle))


def validate_grid(rows: list[dict[str, str]]) -> None:
    """Reject incomplete, duplicated, or invalid manuscript evidence."""
    expected = {
        (dataset, model, str(seed), method)
        for dataset in datasets.names()
        for model in (["identity"] if dataset in datasets.SYNTHETIC else models.MODELS)
        for seed, method in product(config.SEEDS, ORDER)
    }
    keys = [(r["dataset"], r["model"], r["seed"], r["method"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Paper requires the complete, unique offline benchmark grid")
    for row in rows:
        required = (*METRICS, "dsc", "unc", "n_fit", "n_test")
        for metric in required:
            if metric == "true_error" and row["dataset"] not in datasets.SYNTHETIC:
                continue
            if not np.isfinite(float(row[metric])):
                raise ValueError(
                    f"Non-finite {metric} for {row['dataset']}/{row['method']}"
                )
        brier, mcb, dsc, unc = (float(row[k]) for k in ("brier", "mcb", "dsc", "unc"))
        if not np.isclose(brier, mcb - dsc + unc, atol=1e-12, rtol=0):
            raise ValueError("Brier decomposition does not reconcile")
        if not 0 <= float(row["auc"]) <= 1:
            raise ValueError("AUC outside [0, 1]")
        if not 1 <= float(row["n_distinct"]) <= float(row["n_test"]):
            raise ValueError("Distinct count outside evaluation sample size")


def tex_escape(value: str) -> str:
    """Escape identifiers placed in generated LaTeX."""
    return value.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def format_value(metric: str, value: object) -> str:
    """Use one precision per quantity and an explicit missing-value mark."""
    if value == "" or not np.isfinite(float(value)):
        return "--"
    number = float(value)
    if metric == "n_distinct":
        return f"{number:.0f}"
    if metric == "seconds":
        return r"$<0.001$" if number < 0.001 else f"{number:.3f}"
    return f"{number:.4f}"


def write_tables(summary: list[dict], output: Path) -> None:
    """Write the headline and complete results as table bodies."""
    indexed = {(r["dataset"], r["model"], r["method"]): r for r in summary}
    cells = sorted({(r["dataset"], r["model"]) for r in summary})
    blocks = []
    for dataset, model in cells:
        blocks.append(
            r"\multicolumn{8}{l}{\textit{"
            + tex_escape(f"{dataset} / {model}")
            + r"}} \\*"
        )
        for method in ORDER:
            row = indexed[dataset, model, method]
            blocks.append(
                " & ".join(
                    [LABELS[method], *[format_value(k, row[k]) for k in METRICS]]
                )
                + (r" \\" if method == ORDER[-1] else r" \\*")
            )
        blocks.append(r"\addlinespace")
    (output / "complete.tex").write_text("\n".join(blocks) + "\n")
    lines = []
    for method in VISIBLE:
        row = indexed["overconfident", "identity", method]
        columns = ("brier", "mcb", "n_distinct", "true_error")
        lines.append(
            " & ".join([LABELS[method], *[format_value(k, row[k]) for k in columns]])
            + r" \\"
        )
    (output / "overconfident.tex").write_text("\n".join(lines) + "\n")


def write_numbers(
    rows: list[dict], summary: list[dict], paired: list[dict], output: Path
) -> None:
    """Generate counts and every empirical number quoted in the manuscript."""
    counts = {
        "BenchRows": len(rows),
        "BenchSeeds": len(config.SEEDS),
        "BenchMethods": len(ORDER),
        "BenchCells": len({(r["dataset"], r["model"]) for r in rows}),
        "BenchSynthetic": len(datasets.SYNTHETIC),
        "BenchTestPercent": round(100 * config.TEST_SIZE),
        "BenchFolds": config.CV_FOLDS,
        "BenchBootstrap": config.N_BOOTSTRAP,
        "BenchBins": config.N_BINS,
    }
    numbers = {k: str(v) for k, v in counts.items()}
    for method, prefix in (
        ("sklearn_isotonic", "Iso"),
        ("calibre_centered", "Cir"),
        ("calibre_spline", "Spline"),
        ("sklearn_temperature", "Temp"),
    ):
        row = next(
            r
            for r in summary
            if (r["dataset"], r["method"]) == ("overconfident", method)
        )
        for metric, suffix in (
            ("brier", "Brier"),
            ("n_distinct", "Distinct"),
            ("true_error", "Error"),
        ):
            numbers[prefix + suffix] = format_value(metric, row[metric])
    row = next(
        r
        for r in paired
        if (r["dataset"], r["model"], r["method"])
        == ("breast_cancer", "logreg", "uncalibrated")
    )
    for key, suffix in (
        ("delta_brier", "Gain"),
        ("delta_brier_lo", "Low"),
        ("delta_brier_hi", "High"),
    ):
        numbers["Uncal" + suffix] = f"{float(row[key]):.4f}"
    (output / "numbers.tex").write_text(
        "".join(
            f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in numbers.items()
        )
    )


def paired_figure(paired: list[dict]):
    """Show every paired comparison on a shared Brier-difference scale."""
    import matplotlib.pyplot as plt

    cells = sorted({(r["dataset"], r["model"]) for r in paired})
    figure, axes = plt.subplots(
        5, 2, figsize=(7.1, 8.0), sharex=True, sharey=True, layout="constrained"
    )
    wanted = [m for m in VISIBLE if m != config.BASELINE]
    lookup = {(r["dataset"], r["model"], r["method"]): r for r in paired}
    for ax, (dataset, model) in zip(axes.flat, cells, strict=True):
        for y, method in enumerate(wanted):
            row = lookup[dataset, model, method]
            low, mean, high = (
                1000 * float(row[k])
                for k in ("delta_brier_lo", "delta_brier", "delta_brier_hi")
            )
            ax.plot([low, high], [y, y], color="#555555", lw=1.1)
            ax.plot(mean, y, "o", color="#176485", ms=3.5)
        ax.axvline(0, color="0.45", lw=0.7, ls=":")
        ax.set_title(f"{dataset.replace('_', ' ')} / {model}", fontsize=9, loc="left")
        ax.set_yticks(range(len(wanted)), [LABELS[m] for m in wanted], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].invert_yaxis()
    figure.supxlabel(
        "Brier improvement over isotonic (x1,000); right is better", fontsize=10
    )
    return figure


def granularity_figure(summary: list[dict]):
    """Plot score against distinct-value fractions without combining them."""
    import matplotlib.pyplot as plt

    cells = [
        ("overconfident", "identity"),
        ("heavy_tie", "identity"),
        ("small_n", "identity"),
        ("nonmonotone", "identity"),
    ]
    figure, axes = plt.subplots(
        2, 2, figsize=(7.1, 5.0), sharex=True, sharey=True, layout="constrained"
    )
    codes = {m: str(i + 1) for i, m in enumerate(VISIBLE)}
    for ax, (dataset, model) in zip(axes.flat, cells, strict=True):
        points = []
        for row in summary:
            if (row["dataset"], row["model"]) != (dataset, model) or row[
                "method"
            ] not in VISIBLE:
                continue
            x, y = float(row["distinct_ratio"]), float(row["brier"])
            ax.plot(x, y, "o", color="#176485", ms=4)
            points.append((x, y, codes[row["method"]]))
        for left in (True, False):
            group = sorted(
                (p for p in points if (p[0] < 0.5) == left), key=lambda p: p[1]
            )
            last = 0.145
            for x, y, code in group:
                label_y = max(y, last + 0.009)
                ax.annotate(
                    code,
                    (x, y),
                    xytext=(0.24 if left else 0.65, label_y),
                    fontsize=9,
                    ha="center",
                    va="center",
                    arrowprops={"arrowstyle": "-", "color": "0.6", "lw": 0.6},
                )
                last = label_y
        ax.set_title(dataset.replace("_", " "), fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.04, 1.08)
        ax.set_ylim(0.145, 0.265)
    figure.supxlabel("Mean distinct-value fraction", fontsize=10)
    figure.supylabel("Mean held-out Brier score", fontsize=10)
    return figure


def map_figure():
    """Illustrate maps fitted to the existing overconfidence design, seed zero."""
    import matplotlib.pyplot as plt

    from calibre.plots import plot_calibrator_comparison

    data = datasets.load("overconfident", 0)
    X, y, _, _, _ = protocol._scores_for_cell(data, "identity", 0)  # noqa: SLF001
    fitted = {
        LABELS[name]: methods._build(name).fit(X, y)  # noqa: SLF001
        for name in ("calibre_isotonic", "calibre_centered", "calibre_spline")
    }
    figure, ax = plt.subplots(figsize=(6.7, 3.5), layout="constrained")
    # The plotting API accepts fitted estimators and a score grid.
    plot_calibrator_comparison(fitted, X, ax=ax, annotate_distinct=False)
    styles = {"Centered isotonic": "-.", "I-spline": ":"}
    for line in ax.lines:
        if line.get_label() in styles:
            line.set_linestyle(styles[line.get_label()])
    grid = np.linspace(0.0001, 0.9999, 500)
    truth = 1 / (1 + np.exp(-np.log(grid / (1 - grid)) / 1.8))
    ax.plot(
        grid,
        truth,
        color="black",
        ls="--",
        lw=1.4,
        label="Known conditional probability",
    )
    ax.set_xlabel("Input probability score")
    ax.set_ylabel("Mapped probability")
    ax.legend(fontsize=8)
    return figure


def main(argv: list[str] | None = None) -> int:
    """Generate tables, figures, and a manifest without fitting the grid again."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    rows = read_rows(args.results / "raw.csv")
    validate_grid(rows)
    summary = aggregate.summarize(rows)
    paired = aggregate.pair_against_baseline(rows, config.BASELINE, config.N_BOOTSTRAP)
    args.out.mkdir(parents=True, exist_ok=True)
    write_tables(summary, args.out)
    write_numbers(rows, summary, paired, args.out)
    with mpl.rc_context({"font.size": 9, "pdf.fonttype": 42}):
        for name, figure in (
            ("paired", paired_figure(paired)),
            ("granularity", granularity_figure(summary)),
            ("maps", map_figure()),
        ):
            figure.savefig(
                args.out / f"{name}.pdf",
                metadata={"CreationDate": None, "ModDate": None},
            )
            plt.close(figure)
    environment = json.loads((args.results / "environment.json").read_text())
    manifest = {
        "raw_sha256": hashlib.sha256(
            (args.results / "raw.csv").read_bytes()
        ).hexdigest(),
        "environment": environment,
        "illustration_seed": 0,
        "interval": "95% percentile bootstrap over paired seed differences",
        "n_bootstrap": config.N_BOOTSTRAP,
        "difference": "isotonic minus method",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manuscript exhibits to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
