"""
Visualization tools for calibration diagnostics.

This module provides plotting functions to visualize plateau analysis results
and compare different calibration methods.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib() -> None:
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_plateau_diagnostics(
    results: dict[str, Any],
    X: np.ndarray | None = None,
    y_calibrated: np.ndarray | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Plot comprehensive plateau diagnostic results.

    Parameters
    ----------
    results
        Results from IsotonicDiagnostics.analyze().
    X
        Input features for plotting calibration curve.
    y_calibrated
        Calibrated predictions for plotting.
    figsize
        Figure size.

    Returns
    -------
    The created figure.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre import IsotonicDiagnostics, plot_plateau_diagnostics
    >>> X = np.linspace(0, 1, 100)
    >>> y = np.random.binomial(1, X, 100)
    >>> diagnostics = IsotonicDiagnostics()
    >>> results = diagnostics.analyze(X, y)
    >>> fig = plot_plateau_diagnostics(results, X)
    """
    _check_matplotlib()

    if results["n_plateaus"] == 0:
        # Simple plot for no plateaus case
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No plateaus detected",
            ha="center",
            va="center",
            fontsize=16,
            transform=ax.transAxes,
        )
        ax.set_title("Plateau Diagnostic Results")
        return fig

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Plateau Diagnostic Analysis", fontsize=16)

    # Plot 1: Calibration curve with plateau highlighting
    ax1 = axes[0, 0]
    if X is not None and y_calibrated is not None:
        ax1.plot(X, y_calibrated, "b-", linewidth=2, label="Calibrated")

        # Highlight plateaus
        colors = ["red", "orange", "yellow", "green", "purple"]
        for i, plateau in enumerate(results["plateaus"]):
            start_idx, end_idx = plateau["indices"]
            x_range = plateau["x_range"]
            value = plateau["value"]

            color = colors[i % len(colors)]
            ax1.axhspan(
                value - 0.01,
                value + 0.01,
                xmin=(x_range[0] - X.min()) / (X.max() - X.min()),
                xmax=(x_range[1] - X.min()) / (X.max() - X.min()),
                alpha=0.3,
                color=color,
                label=f"Plateau {i + 1} ({plateau['classification']})",
            )

        ax1.set_xlabel("Input Score")
        ax1.set_ylabel("Calibrated Probability")
        ax1.set_title("Calibration Curve with Plateaus")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(
            0.5,
            0.5,
            "Calibration curve\nnot available",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title("Calibration Curve")

    # Plot 2: Classification summary (pie chart)
    ax2 = axes[0, 1]
    counts = results["classification_counts"]
    labels = []
    sizes = []
    colors_pie = []

    if counts["supported"] > 0:
        labels.append(f"Supported ({counts['supported']})")
        sizes.append(counts["supported"])
        colors_pie.append("green")

    if counts["limited_data"] > 0:
        labels.append(f"Limited-data ({counts['limited_data']})")
        sizes.append(counts["limited_data"])
        colors_pie.append("red")

    if counts["inconclusive"] > 0:
        labels.append(f"Inconclusive ({counts['inconclusive']})")
        sizes.append(counts["inconclusive"])
        colors_pie.append("orange")

    if sizes:
        ax2.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%")
    else:
        ax2.text(
            0.5,
            0.5,
            "No plateaus\nto classify",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    ax2.set_title("Plateau Classifications")

    # Plot 3: Tie stability scores
    ax3 = axes[1, 0]
    tie_stabilities = [
        p["tie_stability"]
        for p in results["plateaus"]
        if p["tie_stability"] is not None
    ]

    if tie_stabilities:
        ax3.bar(
            range(len(tie_stabilities)),
            tie_stabilities,
            color=[
                "green" if ts > 0.7 else "red" if ts < 0.3 else "orange"
                for ts in tie_stabilities
            ],
        )
        ax3.set_xlabel("Plateau ID")
        ax3.set_ylabel("Tie Stability")
        ax3.set_title("Bootstrap Tie Stability")
        ax3.set_xticks(range(len(tie_stabilities)))
        ax3.set_xticklabels([f"P{i + 1}" for i in range(len(tie_stabilities))])
        ax3.axhline(
            y=0.7, color="green", linestyle="--", alpha=0.5, label="High stability"
        )
        ax3.axhline(
            y=0.3, color="red", linestyle="--", alpha=0.5, label="Low stability"
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "Tie stability\nnot available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("Bootstrap Tie Stability")

    # Plot 4: Conditional AUC scores
    ax4 = axes[1, 1]
    conditional_aucs = []
    auc_cis = []
    valid_plateaus = []

    for p in results["plateaus"]:
        if p["conditional_auc"] is not None:
            conditional_aucs.append(p["conditional_auc"])
            auc_cis.append(p["conditional_auc_ci"])
            valid_plateaus.append(p["plateau_id"] + 1)

    if conditional_aucs:
        ax4.bar(
            range(len(conditional_aucs)),
            conditional_aucs,
            color=[
                "red" if auc > 0.6 else "green" if auc < 0.55 else "orange"
                for auc in conditional_aucs
            ],
        )

        # Add error bars if CI available
        if any(ci is not None for ci in auc_cis):
            yerr_lower = []
            yerr_upper = []
            for auc, ci in zip(conditional_aucs, auc_cis, strict=True):
                if ci is not None:
                    yerr_lower.append(auc - ci[0])
                    yerr_upper.append(ci[1] - auc)
                else:
                    yerr_lower.append(0)
                    yerr_upper.append(0)

            ax4.errorbar(
                range(len(conditional_aucs)),
                conditional_aucs,
                yerr=[yerr_lower, yerr_upper],
                fmt="none",
                ecolor="black",
                capsize=3,
            )

        ax4.set_xlabel("Plateau ID")
        ax4.set_ylabel("Conditional AUC")
        ax4.set_title("Conditional AUC Among Tied Pairs")
        ax4.set_xticks(range(len(conditional_aucs)))
        ax4.set_xticklabels([f"P{pid}" for pid in valid_plateaus])
        ax4.axhline(
            y=0.5, color="green", linestyle="--", alpha=0.5, label="Random (0.5)"
        )
        ax4.axhline(
            y=0.6, color="red", linestyle="--", alpha=0.5, label="High discrimination"
        )
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5,
            0.5,
            "Conditional AUC\nnot available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Conditional AUC")

    plt.tight_layout()
    return fig


def plot_stability_heatmap(
    stability_matrix: np.ndarray,
    plateau_labels: list[str] | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Plot a heatmap of tie stability across bootstrap resamples.

    Parameters
    ----------
    stability_matrix
        Binary matrix indicating whether ties were preserved.
    plateau_labels
        Labels for plateaus.
    figsize
        Figure size.

    Returns
    -------
    The created figure.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(stability_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xlabel("Bootstrap Sample")
    ax.set_ylabel("Plateau")
    ax.set_title("Tie Stability Across Bootstrap Resamples")

    if plateau_labels:
        ax.set_yticks(range(len(plateau_labels)))
        ax.set_yticklabels(plateau_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Tie Preserved")

    plt.tight_layout()
    return fig


def plot_progressive_sampling(
    sample_sizes: list[int],
    diversities: list[float],
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Plot diversity vs sample size curve.

    Parameters
    ----------
    sample_sizes
        Sample sizes tested.
    diversities
        Corresponding diversity values.
    figsize
        Figure size.

    Returns
    -------
    The created figure.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(sample_sizes, diversities, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Calibration Diversity")
    ax.set_title("Diversity vs Sample Size")
    ax.grid(True, alpha=0.3)

    # Add trend interpretation
    if len(sample_sizes) > 1:
        slope = (diversities[-1] - diversities[0]) / (
            sample_sizes[-1] - sample_sizes[0]
        )
        if slope > 0.001:
            ax.text(
                0.05,
                0.95,
                "Increasing diversity\n(suggests limited-data flattening)",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox={"boxstyle": "round", "facecolor": "orange", "alpha": 0.7},
            )
        elif slope < -0.001:
            ax.text(
                0.05,
                0.95,
                "Decreasing diversity\n(unusual pattern)",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox={"boxstyle": "round", "facecolor": "red", "alpha": 0.7},
            )
        else:
            ax.text(
                0.05,
                0.95,
                "Stable diversity\n(suggests genuine flatness)",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox={"boxstyle": "round", "facecolor": "green", "alpha": 0.7},
            )

    plt.tight_layout()
    return fig


def plot_calibration_comparison(
    X: np.ndarray,
    y_true: np.ndarray,
    calibrators: dict[str, Any],
    figsize: tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Compare calibration curves from different methods.

    Parameters
    ----------
    X
        Input features.
    y_true
        True target values.
    calibrators
        Dictionary mapping method names to fitted calibrator objects.
    figsize
        Figure size.

    Returns
    -------
    The created figure.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.isotonic import IsotonicRegression
    >>> from calibre import NearlyIsotonicRegression
    >>> X = np.linspace(0, 1, 100)
    >>> y = np.random.binomial(1, X, 100)
    >>> iso = IsotonicRegression().fit(X, y)
    >>> nearly_iso = NearlyIsotonicRegression().fit(X, y)
    >>> calibrators = {'Isotonic': iso, 'Nearly Isotonic': nearly_iso}
    >>> fig = plot_calibration_comparison(X, y, calibrators)
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Calibration Method Comparison", fontsize=16)

    # Sort X for plotting
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]

    colors = plt.cm.get_cmap("Set1")(np.linspace(0, 1, len(calibrators)))

    # Plot 1: Calibration curves
    ax1 = axes[0, 0]
    for i, (name, calibrator) in enumerate(calibrators.items()):
        try:
            y_cal = calibrator.transform(X_sorted)
            ax1.plot(X_sorted, y_cal, color=colors[i], linewidth=2, label=name)
        except Exception as e:
            print(f"Warning: Could not plot {name}: {e}")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.set_xlabel("Input Score")
    ax1.set_ylabel("Calibrated Probability")
    ax1.set_title("Calibration Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Number of unique values (diversity)
    ax2 = axes[0, 1]
    method_names = []
    diversities = []

    for name, calibrator in calibrators.items():
        try:
            y_cal = calibrator.transform(X)
            diversity = len(np.unique(y_cal)) / len(y_cal)
            method_names.append(name)
            diversities.append(diversity)
        except Exception:
            continue

    if diversities:
        ax2.bar(
            range(len(diversities)), diversities, color=colors[: len(diversities)]
        )
        ax2.set_xlabel("Method")
        ax2.set_ylabel("Diversity Index")
        ax2.set_title("Calibration Diversity")
        ax2.set_xticks(range(len(method_names)))
        ax2.set_xticklabels(method_names, rotation=45)
        ax2.grid(True, alpha=0.3)

    # Plot 3: Calibration error comparison
    ax3 = axes[1, 0]
    from .metrics import mean_calibration_error

    method_names_error = []
    errors = []

    for name, calibrator in calibrators.items():
        try:
            y_cal = calibrator.transform(X)
            error = mean_calibration_error(y_true, y_cal)
            method_names_error.append(name)
            errors.append(error)
        except Exception:
            continue

    if errors:
        ax3.bar(range(len(errors)), errors, color=colors[: len(errors)])
        ax3.set_xlabel("Method")
        ax3.set_ylabel("Mean Calibration Error")
        ax3.set_title("Calibration Error Comparison")
        ax3.set_xticks(range(len(method_names_error)))
        ax3.set_xticklabels(method_names_error, rotation=45)
        ax3.grid(True, alpha=0.3)

    # Plot 4: Efficient frontier (error vs diversity)
    ax4 = axes[1, 1]
    if diversities and errors and len(diversities) == len(errors):
        ax4.scatter(
            diversities,
            errors,
            c=range(len(diversities)),
            cmap="viridis",
            s=100,
            alpha=0.7,
        )

        for i, name in enumerate(method_names):
            if i < len(diversities) and i < len(errors):
                ax4.annotate(
                    name,
                    (diversities[i], errors[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        ax4.set_xlabel("Diversity Index")
        ax4.set_ylabel("Mean Calibration Error")
        ax4.set_title("Efficient Frontier: Error vs Diversity")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5,
            0.5,
            "Insufficient data\nfor comparison",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Error vs Diversity")

    plt.tight_layout()
    return fig


def plot_mdd_analysis(
    results: dict[str, Any], figsize: tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Plot minimum detectable difference analysis for plateaus.

    Parameters
    ----------
    results
        Results from IsotonicDiagnostics.analyze().
    figsize
        Figure size.

    Returns
    -------
    The created figure.
    """
    _check_matplotlib()

    if results["n_plateaus"] == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No plateaus detected",
            ha="center",
            va="center",
            fontsize=16,
            transform=ax.transAxes,
        )
        ax.set_title("Minimum Detectable Difference Analysis")
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Minimum Detectable Difference Analysis", fontsize=14)

    mdd_left = [p["mdd_left"] for p in results["plateaus"] if p["mdd_left"] is not None]
    mdd_right = [
        p["mdd_right"] for p in results["plateaus"] if p["mdd_right"] is not None
    ]

    # Filter out infinite values for plotting
    mdd_left_finite = [mdd for mdd in mdd_left if not np.isinf(mdd)]
    mdd_right_finite = [mdd for mdd in mdd_right if not np.isinf(mdd)]

    # Plot left boundary MDDs
    if mdd_left_finite:
        ax1.bar(range(len(mdd_left_finite)), mdd_left_finite, alpha=0.7, color="blue")
        ax1.set_xlabel("Plateau ID")
        ax1.set_ylabel("MDD")
        ax1.set_title("Left Boundary MDD")
        ax1.set_xticks(range(len(mdd_left_finite)))
        ax1.set_xticklabels([f"P{i + 1}" for i in range(len(mdd_left_finite))])
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(
            0.5,
            0.5,
            "No finite\nMDD values",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title("Left Boundary MDD")

    # Plot right boundary MDDs
    if mdd_right_finite:
        ax2.bar(range(len(mdd_right_finite)), mdd_right_finite, alpha=0.7, color="red")
        ax2.set_xlabel("Plateau ID")
        ax2.set_ylabel("MDD")
        ax2.set_title("Right Boundary MDD")
        ax2.set_xticks(range(len(mdd_right_finite)))
        ax2.set_xticklabels([f"P{i + 1}" for i in range(len(mdd_right_finite))])
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No finite\nMDD values",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Right Boundary MDD")

    plt.tight_layout()
    return fig
