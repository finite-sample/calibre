"""Generate the granularity study exhibits and machine-readable summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from . import study

TITLES = (
    "Smooth linear risk",
    "Smooth curved risk",
    "Genuinely flat regions",
    "Constant risk",
    "Nonmonotone risk",
)
COLORS = ("#555555", "#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7A6515")
STYLES = (":", "-", "-", "--", "-.", ":", (0, (5, 2, 1, 2)))
THEME = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
}


def bootstrap_weights() -> np.ndarray:
    """Use identical seed resamples across methods and outcome contrasts."""
    rng = np.random.default_rng(study.BOOTSTRAP_SEED)
    indices = rng.integers(0, len(study.SEEDS), (study.N_BOOTSTRAP, len(study.SEEDS)))
    return np.stack(
        [np.bincount(row, minlength=len(study.SEEDS)) for row in indices]
    ) / len(study.SEEDS)


def interval(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mean and percentile interval over resampled calibration replicates."""
    flat = values.reshape(len(study.SEEDS), -1)
    draws = weights @ flat
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
    shape = values.shape[1:]
    return values.mean(axis=0), lower.reshape(shape), upper.reshape(shape)


def load_results(path: Path) -> dict[str, np.ndarray]:
    """Reject stale or incomplete results before generating any exhibit."""
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["configuration"] != json.loads(json.dumps(study.configuration())):
        raise ValueError("result configuration does not match study")
    for filename, key in (
        ("predictions.npz", "prediction_sha256"),
        ("evaluation.npz", "evaluation_sha256"),
    ):
        if hashlib.sha256((path / filename).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f"result hash mismatch: {filename}")
    if (
        hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest()
        != manifest["evaluation_source_sha256"]
    ):
        raise ValueError("evaluation source changed; run --evaluate-only")
    arrays = dict(np.load(path / "evaluation.npz"))
    base = (
        len(study.DESIGNS),
        len(study.SIZES),
        len(study.SEEDS),
        len(study.METHODS),
        len(study.PRECISIONS),
    )
    expected = {
        "metrics": (*base, len(study.METRICS)),
        "curves": (*base, 2, len(study.THRESHOLDS)),
        "capacity": (*base, 2, len(study.CAPACITIES)),
    }
    for name, array in arrays.items():
        if array.shape != expected[name]:
            raise ValueError(f"incomplete {name} grid")
        required = array[..., :-1] if name == "metrics" else array
        if not np.isfinite(required).all():
            raise ValueError(f"nonfinite {name}")
    return arrays


def save_figure(fig, path: Path) -> None:
    """Export a vector figure and close its graphics resources."""
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def toy_figure():
    """Show identical pooling counts with different amounts of lost information."""
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), layout="constrained")
    for i, d in enumerate((0, 0.05, 0.15)):
        p = np.array([0.3 - d, 0.3 + d])
        q = np.full(2, 0.3)
        axes[0].plot([i, i], p, color=COLORS[2], lw=2)
        axes[0].scatter([i, i], p, color=COLORS[2], s=20)
        axes[0].scatter(
            i, 0.3, facecolors="white", edgecolors="black", marker="s", s=30, zorder=3
        )
        loss = study.regret_curves(p, q)[0]
        axes[1].plot(
            study.THRESHOLDS,
            loss * 1000,
            color=COLORS[i + 1],
            ls=STYLES[i + 1],
            label=f"d = {d:g}",
        )
    axes[0].set(
        xticks=[0, 1, 2],
        xticklabels=["0", "0.05", "0.15"],
        xlabel="Risk separation d",
        ylabel="Event probability",
        ylim=(0.1, 0.5),
        title="True and pooled risks",
    )
    axes[1].set(
        xlim=(0.1, 0.5),
        xlabel="Action cost t",
        ylabel="Payoff lost per 1,000",
        title="Payoff lost through pooling",
    )
    axes[1].legend(frameon=False)
    p = np.full(1000, 0.3)
    q = np.linspace(0.2, 0.4, 1000)
    row = study.losses(p, q)
    axes[2].bar(
        [0, 1],
        [1000 * row["pooling_loss"], 1000 * row["total_error"]],
        color=["#777777", COLORS[2]],
        width=0.5,
    )
    axes[2].set(
        xticks=[0, 1],
        xticklabels=["Information\nlost", "Total\nerror"],
        ylabel="Squared probability error x 1,000",
        title="Artificial distinctions",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def decomposition_figure(metrics: np.ndarray, design: int, precision: int):
    """Compare pooling and probability-label error across sample sizes."""
    fig, axes = plt.subplots(
        1, 3, figsize=(7.1, 2.8), sharex=True, sharey=True, layout="constrained"
    )
    y = np.arange(len(study.METHODS))
    for n, ax in enumerate(axes):
        means = metrics[design, n, :, :, precision, :3].mean(axis=0) * 1000
        ax.barh(y, means[:, 0], color=COLORS[2], label="Information lost")
        ax.barh(
            y, means[:, 1], left=means[:, 0], color="#BDBDBD", label="Miscalibration"
        )
        ax.set(
            title=f"Calibration n = {study.SIZES[n]:,}",
            yticks=y,
            yticklabels=study.LABELS,
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.supxlabel(
        "Mean squared probability error x 1,000 (sum of both components)", fontsize=9
    )
    return fig


def threshold_figure(
    curves: np.ndarray, design: int, precision: int, weights: np.ndarray
):
    """Show paired decision regret relative to isotonic at n=1,000."""
    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 2.8), sharex=True, sharey=True, layout="constrained"
    )
    values = curves[design, 1, :, :, precision]
    for kind, ax in enumerate(axes):
        for m in range(len(study.METHODS)):
            if m == 1:
                continue
            mean, low, high = interval(
                (values[:, m, kind] - values[:, 1, kind]) * 1000, weights
            )
            ax.plot(
                study.THRESHOLDS,
                mean,
                color=COLORS[m],
                ls=STYLES[m],
                lw=1.1,
                label=study.LABELS[m],
            )
            ax.fill_between(
                study.THRESHOLDS, low, high, color=COLORS[m], alpha=0.10, lw=0
            )
        ax.axhline(0, color="black", lw=0.6)
        ax.set(
            title=("Information-loss difference", "Implemented-regret difference")[
                kind
            ],
            xlabel="Action cost t",
            xlim=(0, 1),
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Extra payoff lost per 1,000 vs isotonic")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    return fig


def capacity_figure(capacity: np.ndarray, precision: int, weights: np.ndarray):
    """Compare payoff changes against original-score allocation at every capacity."""
    fig, axes = plt.subplots(
        5, 2, figsize=(7.1, 8.6), sharex=True, sharey="row", layout="constrained"
    )
    for d, design in enumerate(study.DESIGNS):
        scores, p = study.population(design)
        baseline = study.capacity_values(p, scores)[0]
        oracle = study.capacity_values(p, p)[0] - baseline
        for policy, ax in enumerate(axes[d]):
            for m in range(1, len(study.METHODS)):
                mean, low, high = interval(
                    capacity[d, 1, :, m, precision, policy] - baseline, weights
                )
                ax.plot(
                    study.CAPACITIES * 100,
                    mean,
                    color=COLORS[m],
                    ls=STYLES[m],
                    lw=1,
                    label=study.LABELS[m],
                )
                ax.fill_between(
                    study.CAPACITIES * 100, low, high, color=COLORS[m], alpha=0.10, lw=0
                )
            ax.plot(
                study.CAPACITIES * 100,
                oracle,
                color="#777777",
                ls=(0, (2, 3)),
                lw=1,
                label="Oracle risk ranking",
            )
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(TITLES[d], loc="left", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].text(
        0.5,
        1.3,
        "Random boundary ties",
        transform=axes[0, 0].transAxes,
        ha="center",
        fontsize=10,
    )
    axes[0, 1].text(
        0.5,
        1.3,
        "Original-score boundary ties",
        transform=axes[0, 1].transAxes,
        ha="center",
        fontsize=10,
    )
    for ax in axes[-1]:
        ax.set_xlabel("Population selected (%)")
    for ax in axes[3]:
        ax.set_ylim(-0.5, 0.5)
    fig.supylabel(
        "Extra expected successes per 1,000 vs original-score ranking", fontsize=9
    )
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    return fig


def summaries(arrays: dict, out: Path, weights: np.ndarray) -> None:
    """Write all means, seed records, and paired confidence intervals."""
    metrics = arrays["metrics"]
    fields = ["design", "n_fit", "seed", "method", "precision", *study.METRICS]
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d, n, s, m, precision in np.ndindex(metrics.shape[:5]):
            row = dict(zip(study.METRICS, metrics[d, n, s, m, precision], strict=True))
            if not np.isfinite(row["spearman"]):
                row["spearman"] = ""
            writer.writerow(
                dict(
                    design=study.DESIGNS[d],
                    n_fit=study.SIZES[n],
                    seed=study.SEEDS[s],
                    method=study.METHODS[m],
                    precision=study.PRECISIONS[precision],
                    **row,
                )
            )
    fields = [
        "design",
        "n_fit",
        "method",
        "precision",
        "metric",
        "mean",
        "difference_vs_isotonic",
        "lower",
        "upper",
    ]
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d, n, m, precision, k in np.ndindex(5, 3, 7, 2, len(study.METRICS)):
            values = metrics[d, n, :, m, precision, k]
            baseline = metrics[d, n, :, 1, precision, k]
            valid = np.isfinite(values).all() and np.isfinite(baseline).all()
            difference, low, high = (
                interval(values - baseline, weights) if valid else ("", "", "")
            )
            writer.writerow(
                {
                    "design": study.DESIGNS[d],
                    "n_fit": study.SIZES[n],
                    "method": study.METHODS[m],
                    "precision": study.PRECISIONS[precision],
                    "metric": study.METRICS[k],
                    "mean": values.mean() if np.isfinite(values).all() else "",
                    "difference_vs_isotonic": difference,
                    "lower": low,
                    "upper": high,
                }
            )
    table = []
    for precision in range(2):
        for d, _design in enumerate(study.DESIGNS):
            for n, size in enumerate(study.SIZES):
                table.append(
                    r"\multicolumn{7}{l}{\textit{"
                    + f"{TITLES[d]}, n={size:,}, "
                    + ("exact" if precision == 0 else "six decimals")
                    + r"}}\\*"
                )
                for m, label in enumerate(study.LABELS):
                    values = metrics[d, n, :, m, precision]
                    mean = values.mean(axis=0)
                    delta, low, high = interval(
                        values[:, 2] - metrics[d, n, :, 1, precision, 2], weights
                    )
                    nums = [f"{1000 * x:.3f}" for x in mean[:3]]
                    tail = (
                        f"{1000 * delta:.3f} & "
                        f"[{1000 * low:.3f}, {1000 * high:.3f}] & {mean[6]:.0f}"
                    )
                    table.append(
                        label
                        + " & "
                        + " & ".join(nums)
                        + " & "
                        + tail
                        + (r"\\*" if m < 6 else r"\\\addlinespace")
                    )
    (out / "complete.tex").write_text("\n".join(table) + "\n")
    # All prose quantities are generated from the same stored per-seed results.
    macros = []
    for key, m in [("Centered", 2), ("Spline", 3), ("Temperature", 5)]:
        vals = metrics[0, 1, :, m, 0, 2] - metrics[0, 1, :, 1, 0, 2]
        mean, low, high = interval(vals * 1000, weights)
        macros.append(
            "\\newcommand{" + chr(92) + key + "Difference}{" + f"{mean:.3f}" + "}"
        )
        macros.append(
            "\\newcommand{"
            + chr(92)
            + key
            + "Interval}{"
            + f"[{low:.3f}, {high:.3f}]"
            + "}"
        )
    (out / "numbers.tex").write_text("\n".join(macros) + "\n")


def main() -> None:
    """Build all exhibits from saved results without refitting calibrators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=study.ROOT / "results")
    parser.add_argument("--out", type=Path, default=study.ROOT / "build")
    args = parser.parse_args()
    arrays = load_results(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    weights = bootstrap_weights()
    summaries(arrays, args.out, weights)
    with plt.rc_context(THEME):
        save_figure(toy_figure(), args.out / "mechanism.pdf")
        for precision in range(2):
            for d, design in enumerate(study.DESIGNS):
                save_figure(
                    decomposition_figure(arrays["metrics"], d, precision),
                    args.out / f"decomposition_{design}_{precision}.pdf",
                )
                save_figure(
                    threshold_figure(arrays["curves"], d, precision, weights),
                    args.out / f"threshold_{design}_{precision}.pdf",
                )
            save_figure(
                capacity_figure(arrays["capacity"], precision, weights),
                args.out / f"capacity_{precision}.pdf",
            )
    records = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.out.glob("*"))
        if p.suffix in {".pdf", ".csv", ".tex"} and p.name != "note.pdf"
    }
    (args.out / "report_manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "artifacts": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote figures and complete summaries to {args.out}")


if __name__ == "__main__":
    main()
