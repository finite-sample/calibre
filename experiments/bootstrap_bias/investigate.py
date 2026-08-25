"""Why bootstrap confidence intervals for calibration errors are wrong.

Run with::

    uv run python experiments/bootstrap_bias/investigate.py

Prints only; writes nothing. Every experiment states a prediction up front and
reports PASS or FAIL against it. The point is to *falsify* the hypothesis if it
is wrong, so failures are printed as failures.

The hypothesis under test
-------------------------
The bootstrap resamples from the empirical measure, so ``E[F*] = F``. For an
estimator ``theta = g(F)``:

* ``g`` linear in ``F``  ->  ``E[g(F*)] = g(F)`` exactly.
* ``g`` convex in ``F``  ->  ``E[g(F*)] >= g(F)`` by Jensen, strictly.

Calibration errors are convex (a norm of a linear functional); proper scoring
rules are linear (a plain mean). The relative gap is governed by curvature at
``F``, which is unbounded at the kink ``||delta|| = 0`` and negligible far from
it -- hence the gap is worst exactly when the model is well calibrated.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from calibre.evaluation import score_decomposition
from calibre.metrics import (
    _bin_summaries,
    _equal_mass_bins,
    debiased_calibration_error,
    plugin_calibration_error,
    smooth_calibration_error,
)

SQRT2 = float(np.sqrt(2.0))
RESULTS: list[tuple[str, str, bool]] = []


def record(name: str, detail: str, passed: bool) -> None:
    """Record and print one prediction's outcome.

    Parameters
    ----------
    name
        Experiment label.
    detail
        What was measured.
    passed
        Whether the prediction held.
    """
    RESULTS.append((name, detail, passed))
    print(f"    {'PASS' if passed else 'FAIL'}  {detail}")


def calibrated(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Perfectly calibrated forecasts: the true calibration error is exactly 0.

    Parameters
    ----------
    n
        Sample size.
    rng
        Random generator.

    Returns:
    -------
    tuple of ndarray
        ``(y_true, y_pred)``.
    """
    p = rng.uniform(0, 1, n)
    return rng.binomial(1, p).astype(float), p


def distorted(
    n: int, a: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts stretched about 0.5 by ``a``; ``a = 1`` is calibrated.

    Parameters
    ----------
    n
        Sample size.
    a
        Distortion strength.
    rng
        Random generator.

    Returns:
    -------
    tuple of ndarray
        ``(y_true, y_pred)``.
    """
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p).astype(float)
    return y, np.clip(a * (p - 0.5) + 0.5, 0.0, 1.0)


def boot_draws(metric, y, p, n_resamples, rng):
    """Bootstrap draws of ``metric`` by resampling observations.

    Parameters
    ----------
    metric
        Callable of ``(y_true, y_pred)``.
    y
        Labels.
    p
        Predictions.
    n_resamples
        Number of resamples.
    rng
        Random generator.

    Returns:
    -------
    ndarray
        The draws.
    """
    n = y.size
    out = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        out[i] = metric(y[idx], p[idx])
    return out


def ratio(metric, y, p, n_resamples, rng) -> tuple[float, float]:
    """Observed value and bootstrap mean.

    Parameters
    ----------
    metric
        Callable of ``(y_true, y_pred)``.
    y
        Labels.
    p
        Predictions.
    n_resamples
        Number of resamples.
    rng
        Random generator.

    Returns:
    -------
    tuple of float
        ``(observed, bootstrap_mean)``.
    """
    return float(metric(y, p)), float(boot_draws(metric, y, p, n_resamples, rng).mean())


# --------------------------------------------------------------------------- #


def e1_signed_versus_absolute() -> None:
    """The decisive control: same data, same bootstrap, linear vs convex.

    ``mean(p) - mean(y)`` is linear in the empirical measure.
    ``|mean(p) - mean(y)|`` is the same thing inside an absolute value, hence
    convex. If the hypothesis is right the linear one must show no shift and the
    convex one must.

    Reported as a standardised difference rather than a ratio, because the signed
    statistic is approximately zero on calibrated data and a ratio would be 0/0.
    """
    print("\nE1  signed (linear) vs absolute (convex) mean calibration error")
    print("    prediction: |z| < 0.1 for signed; z clearly positive for absolute")
    rng = np.random.default_rng(0)
    y, p = calibrated(2000, rng)

    def signed(t, q):
        return float(np.mean(q) - np.mean(t))

    def absolute(t, q):
        return float(abs(np.mean(q) - np.mean(t)))

    for label, metric in (("signed  ", signed), ("absolute", absolute)):
        draws = boot_draws(metric, y, p, 2000, np.random.default_rng(1))
        observed = metric(y, p)
        z = (draws.mean() - observed) / draws.std(ddof=1)
        print(
            f"    {label}  observed {observed:+.6f}  boot mean {draws.mean():+.6f}"
            f"  z {z:+.3f}"
        )
        if label.strip() == "signed":
            record(
                "E1", f"linear functional unshifted (z={z:+.3f})", bool(abs(z) < 0.1)
            )
        else:
            record("E1", f"convex functional shifted up (z={z:+.3f})", bool(z > 0.3))


def _plugin_frozen(p_ref, n_bins, norm_p):
    """Build a plugin estimator whose bin edges are frozen from a reference sample.

    Parameters
    ----------
    p_ref
        Reference predictions, whose bin edges are reused.
    n_bins
        Requested bin count.
    norm_p
        Norm.

    Returns:
    -------
    callable
        A metric of ``(y_true, y_pred)`` using the frozen edges.
    """
    bin_id_ref, n_used = _equal_mass_bins(p_ref, n_bins)
    order = np.argsort(p_ref, kind="mergesort")
    edges = p_ref[order][np.flatnonzero(np.diff(bin_id_ref[order]) != 0) + 1]

    def metric(t, q):
        bin_id = np.searchsorted(edges, q, side="right")
        counts, mean_pred, mean_true = _bin_summaries(t, q, bin_id, n_used)
        occupied = counts > 0
        gaps = np.abs(mean_pred[occupied] - mean_true[occupied]) ** norm_p
        total = float(np.sum(counts[occupied] / t.size * gaps))
        return float(total ** (1.0 / norm_p))

    return metric


def e2_sqrt2(n_datasets: int = 200, n: int = 2000, n_resamples: int = 150) -> None:
    """The quantitative prediction: a factor of sqrt(2) at true error zero.

    On calibrated data the observed value is ``||delta||`` where ``delta`` is pure
    sampling noise. The bootstrap statistic is ``||delta + eps||`` with ``eps`` of
    comparable variance, so the norm should grow by about ``sqrt(2)``.

    Parameters
    ----------
    n_datasets
        Independent datasets to average over.
    n
        Observations per dataset.
    n_resamples
        Bootstrap resamples per dataset.
    """
    print(f"\nE2  plugin ECE inflation at true error 0 (n={n}, {n_datasets} datasets)")
    print(f"    prediction: ratio ~ sqrt(2) = {SQRT2:.3f}")
    for norm_p in (1, 2):
        obs_all, boot_all = [], []
        rng = np.random.default_rng(10 + norm_p)
        for _ in range(n_datasets):
            y, p = calibrated(n, rng)
            o, b = ratio(
                lambda t, q, _p=norm_p: plugin_calibration_error(t, q, 15, _p),
                y,
                p,
                n_resamples,
                rng,
            )
            obs_all.append(o)
            boot_all.append(b)
        r = float(np.mean(boot_all) / np.mean(obs_all))
        print(
            f"    L{norm_p}  observed {np.mean(obs_all):.5f}  "
            f"boot mean {np.mean(boot_all):.5f}  ratio {r:.3f}"
        )
        record("E2", f"L{norm_p} ratio near sqrt(2) (got {r:.3f})", 1.25 < r < 1.55)


def e2b_frozen_edges(n_datasets: int = 150, n: int = 2000, n_resamples: int = 150):
    """Freeze the bin edges, to separate re-binning from convexity.

    ``_equal_mass_bins`` recomputes edges on every resample, so part of the
    inflation could be re-binning rather than the convexity of the norm.

    Parameters
    ----------
    n_datasets
        Independent datasets.
    n
        Observations per dataset.
    n_resamples
        Resamples per dataset.
    """
    print("\nE2b frozen bin edges (isolates re-binning from convexity)")
    print("    prediction: still inflated; convexity is not an artifact of re-binning")
    rng = np.random.default_rng(20)
    obs_all, boot_all = [], []
    for _ in range(n_datasets):
        y, p = calibrated(n, rng)
        metric = _plugin_frozen(p, 15, 2)
        o, b = ratio(metric, y, p, n_resamples, rng)
        obs_all.append(o)
        boot_all.append(b)
    r = float(np.mean(boot_all) / np.mean(obs_all))
    print(
        f"    L2 frozen  observed {np.mean(obs_all):.5f}  "
        f"boot mean {np.mean(boot_all):.5f}  ratio {r:.3f}"
    )
    record("E2b", f"inflation survives frozen edges (ratio {r:.3f})", r > 1.2)


def e3_sample_size(n_resamples: int = 120, n_datasets: int = 80) -> None:
    """The inflation is not a small-sample artifact.

    Both the observed value and the added noise scale as ``1/sqrt(n)``, so their
    ratio should be roughly constant. If instead it shrank with ``n``, the effect
    would be ordinary finite-sample noise rather than a structural property.

    Parameters
    ----------
    n_resamples
        Resamples per dataset.
    n_datasets
        Datasets per sample size.
    """
    print("\nE3  does the inflation shrink with n?")
    print("    prediction: ratio roughly constant in n")
    ratios = []
    for n in (250, 1000, 4000, 16000):
        rng = np.random.default_rng(30 + n)
        obs_all, boot_all = [], []
        for _ in range(n_datasets):
            y, p = calibrated(n, rng)
            o, b = ratio(
                lambda t, q: plugin_calibration_error(t, q, 15, 2),
                y,
                p,
                n_resamples,
                rng,
            )
            obs_all.append(o)
            boot_all.append(b)
        r = float(np.mean(boot_all) / np.mean(obs_all))
        ratios.append(r)
        print(f"    n={n:>6}  ratio {r:.3f}")
    spread = max(ratios) - min(ratios)
    record(
        "E3",
        f"ratio stable across a 64x range of n (spread {spread:.3f})",
        spread < 0.25 and min(ratios) > 1.2,
    )


def e4_curvature(n_datasets: int = 80, n: int = 4000, n_resamples: int = 120) -> None:
    """The gap tracks curvature: large at the kink, vanishing away from it.

    This is the core of the explanation. If the ratio did not decay as the true
    error grows, the mechanism would not be Jensen at the kink of a norm.

    Parameters
    ----------
    n_datasets
        Datasets per distortion level.
    n
        Observations per dataset.
    n_resamples
        Resamples per dataset.
    """
    print("\nE4  does the inflation decay as true miscalibration grows?")
    print("    prediction: ratio falls from ~1.4 toward 1.0")
    ratios = []
    for a in (1.0, 1.1, 1.25, 1.5, 2.0, 3.0):
        rng = np.random.default_rng(40)
        obs_all, boot_all = [], []
        for _ in range(n_datasets):
            y, p = distorted(n, a, rng)
            o, b = ratio(
                lambda t, q: plugin_calibration_error(t, q, 15, 2),
                y,
                p,
                n_resamples,
                rng,
            )
            obs_all.append(o)
            boot_all.append(b)
        r = float(np.mean(boot_all) / np.mean(obs_all))
        ratios.append(r)
        print(f"    a={a:<4}  observed {np.mean(obs_all):.5f}  ratio {r:.3f}")
    record(
        "E4",
        f"ratio decays with distortion ({ratios[0]:.3f} -> {ratios[-1]:.3f})",
        ratios[0] > ratios[-1] + 0.2 and ratios[-1] < 1.15,
    )


def _debiased_total(y_true, y_pred, n_bins=15) -> float:
    """The debiased sum *before* the square root and the floor.

    Parameters
    ----------
    y_true
        Labels.
    y_pred
        Predictions.
    n_bins
        Bin count.

    Returns:
    -------
    float
        The signed total.
    """
    n_bins = min(n_bins, len(y_true))
    bin_id, n_used = _equal_mass_bins(y_pred, n_bins)
    counts, mean_pred, mean_true = _bin_summaries(y_true, y_pred, bin_id, n_used)
    correctable = counts > 1
    per_bin = np.zeros_like(counts)
    variance = (
        mean_true[correctable]
        * (1.0 - mean_true[correctable])
        / (counts[correctable] - 1.0)
    )
    per_bin[correctable] = (
        mean_pred[correctable] - mean_true[correctable]
    ) ** 2 - variance
    return float(np.sum(counts / len(y_true) * per_bin))


def e5_debiased_floor(n: int = 2000, n_resamples: int = 400) -> None:
    """Which part of the debiased estimator amplifies: the floor, or the sqrt?

    ``debiased_calibration_error`` returns ``sqrt(max(total, 0))``. ``sqrt`` is
    *concave*, so Jensen applied to it pushes downward -- the amplification must
    come from its unbounded derivative at 0 and from the one-sided floor. Testing
    the raw ``total`` separates the two.

    Parameters
    ----------
    n
        Observations.
    n_resamples
        Resamples.
    """
    print("\nE5  debiased ECE: is it the floor, the sqrt, or convexity?")
    print("    prediction: raw total ~1.0 would mean sqrt/floor, not convexity")
    rng = np.random.default_rng(50)
    y, p = calibrated(n, rng)

    variants = {
        "raw total        ": _debiased_total,
        "sign*sqrt(|t|)   ": lambda t, q: float(
            np.sign(_debiased_total(t, q)) * np.sqrt(abs(_debiased_total(t, q)))
        ),
        "sqrt(max(t,0))   ": lambda t, q: debiased_calibration_error(t, q, 15),
    }
    for label, metric in variants.items():
        draws = boot_draws(metric, y, p, n_resamples, np.random.default_rng(51))
        observed = metric(y, p)
        shown = (
            f"ratio {draws.mean() / observed:+.3f}"
            if abs(observed) > 1e-12
            else "ratio n/a (observed ~ 0)"
        )
        print(
            f"    {label} observed {observed:+.6f}  boot mean {draws.mean():+.6f}"
            f"  {shown}"
        )

    floor_hits = float(
        np.mean(
            boot_draws(_debiased_total, y, p, n_resamples, np.random.default_rng(52))
            <= 0.0
        )
    )
    print(f"    bootstrap draws hitting the floor (total <= 0): {floor_hits:.1%}")

    rng2 = np.random.default_rng(53)
    totals = []
    for _ in range(150):
        y_new, p_new = calibrated(n, rng2)
        totals.append(_debiased_total(y_new, p_new))
    negatives = float(np.mean(np.asarray(totals) <= 0.0))
    print(f"    independent calibrated datasets with total <= 0: {negatives:.1%}")
    record(
        "E5",
        f"the floor binds often on calibrated data ({floor_hits:.0%} of draws)",
        floor_hits > 0.05,
    )


def e6_mcb(n: int = 2000, n_resamples: int = 300) -> None:
    """MCB: pure convexity, or loss of effective sample size?

    ``MCB = S(x) - inf_g S(g)`` over monotone ``g``, i.e. a supremum of
    functionals each linear in the empirical measure -- hence convex, so Jensen
    applies with no kink required. The competing story is that a resample holds
    only ~63% distinct rows and PAV overfits the duplicates.

    Subsampling *without* replacement at smaller ``m`` separates them: the
    overfitting story predicts inflation that tracks ``m``.

    Parameters
    ----------
    n
        Observations.
    n_resamples
        Resamples.
    """
    print("\nE6  MCB: convexity vs loss of effective sample size")
    rng = np.random.default_rng(60)
    y, p = calibrated(n, rng)

    def mcb(t, q):
        return float(score_decomposition(q, t)["MCB"])

    observed = mcb(y, p)
    draws = boot_draws(mcb, y, p, n_resamples, np.random.default_rng(61))
    print(
        f"    n-out-of-n bootstrap   observed {observed:.5f}  "
        f"boot mean {draws.mean():.5f}  ratio {draws.mean() / observed:.3f}"
    )

    distinct = np.mean(
        [
            np.unique(np.random.default_rng(62 + i).integers(0, n, n)).size / n
            for i in range(20)
        ]
    )
    print(f"    distinct rows in a resample: {distinct:.3f} (theory 0.632)")

    sub_rng = np.random.default_rng(63)
    for m in (n // 10, n // 4, n // 2):
        vals = []
        for _ in range(120):
            idx = sub_rng.choice(n, m, replace=False)
            vals.append(mcb(y[idx], p[idx]))
        print(
            f"    subsample m={m:>5} (no replacement)  mean MCB {np.mean(vals):.5f}"
            f"  vs observed {observed:.5f}"
        )

    fresh = np.random.default_rng(64)

    def mean_mcb(size: int, repeats: int = 60) -> float:
        vals = []
        for _ in range(repeats):
            y_new, p_new = calibrated(size, fresh)
            vals.append(mcb(y_new, p_new))
        return float(np.mean(vals))

    at_full = mean_mcb(n)
    at_632 = mean_mcb(int(0.632 * n))
    print(
        f"    fresh data: MCB at n={n} is {at_full:.5f}; "
        f"at n={int(0.632 * n)} it is {at_632:.5f}"
    )
    record(
        "E6",
        f"MCB inflates under the naive bootstrap (ratio {draws.mean() / observed:.2f})",
        bool(draws.mean() > observed),
    )


def intervals(draws: np.ndarray, observed: float, level: float = 0.95) -> dict:
    """Percentile, basic and bias-corrected intervals from the same draws.

    Parameters
    ----------
    draws
        Bootstrap draws.
    observed
        The observed statistic.
    level
        Nominal coverage.

    Returns:
    -------
    dict
        Method name to ``(lower, upper)``.
    """
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])

    # Bias-corrected: shift the quantiles by how far the draws sit above theta.
    # Ties count as half. Without that correction a floored estimator sitting
    # exactly at zero has no draw strictly below it, z0 pins to its clamp, and
    # the interval collapses to [0, 0]. Must match calibre.evaluation exactly,
    # or this experiment does not measure what the library ships.
    below = float(np.mean(draws < observed) + 0.5 * np.mean(draws == observed))
    below = min(max(below, 1.0 / (len(draws) + 1)), 1.0 - 1.0 / (len(draws) + 1))
    z0 = norm.ppf(below)
    z_lo, z_hi = norm.ppf(tail), norm.ppf(1.0 - tail)
    a_lo = norm.cdf(2 * z0 + z_lo)
    a_hi = norm.cdf(2 * z0 + z_hi)
    bc = tuple(np.quantile(draws, [a_lo, a_hi]))

    return {
        "percentile": (float(lo), float(hi)),
        "basic": (float(2 * observed - hi), float(2 * observed - lo)),
        "bc": (float(bc[0]), float(bc[1])),
    }


def e7_intervals(n: int = 2000, n_resamples: int = 600) -> None:
    """What each interval method actually reports.

    Parameters
    ----------
    n
        Observations.
    n_resamples
        Resamples.
    """
    print("\nE7  interval methods on calibrated data (true error 0)")
    rng = np.random.default_rng(70)
    y, p = calibrated(n, rng)
    for label, metric in (
        ("plugin L2", lambda t, q: plugin_calibration_error(t, q, 15, 2)),
        ("smECE    ", smooth_calibration_error),
        ("debiased ", lambda t, q: debiased_calibration_error(t, q, 15)),
    ):
        draws = boot_draws(metric, y, p, n_resamples, np.random.default_rng(71))
        observed = float(metric(y, p))
        got = intervals(draws, observed)
        print(f"    {label}  observed {observed:.5f}")
        for name, (lo, hi) in got.items():
            covers = "covers 0" if lo <= 0.0 <= hi else "EXCLUDES 0"
            print(f"        {name:<11} [{lo:+.5f}, {hi:+.5f}]  {covers}")
        record(
            "E7",
            f"{label.strip()}: bc lower bound is not negative",
            got["bc"][0] >= 0.0,
        )


def e8_coverage(n_datasets: int = 200, n: int = 1500, n_resamples: int = 200) -> None:
    """The number that actually matters: coverage of the true value.

    Two estimands are separated deliberately. The plugin estimator is biased, so
    its estimand is *not* zero and an interval that excludes zero is behaving
    correctly. The debiased estimator targets the true error, which here is
    exactly zero, so its interval should cover zero.

    Parameters
    ----------
    n_datasets
        Datasets.
    n
        Observations per dataset.
    n_resamples
        Resamples per dataset.
    """
    print(f"\nE8  coverage of the TRUE error (= 0), {n_datasets} datasets, n={n}")
    print("    note: plugin's estimand is not 0, so excluding 0 is correct for it")
    print("    width is reported too: coverage bought with a zero-width interval")
    print("    is not coverage, it is an assertion of certainty")
    for label, metric in (
        ("plugin L2", lambda t, q: plugin_calibration_error(t, q, 15, 2)),
        ("debiased ", lambda t, q: debiased_calibration_error(t, q, 15)),
    ):
        hits = {"percentile": 0, "basic": 0, "bc": 0}
        widths: dict[str, list[float]] = {"percentile": [], "basic": [], "bc": []}
        degenerate = dict.fromkeys(hits, 0)
        rng = np.random.default_rng(80)
        for _ in range(n_datasets):
            y, p = calibrated(n, rng)
            draws = boot_draws(metric, y, p, n_resamples, rng)
            observed = float(metric(y, p))
            for name, (lo, hi) in intervals(draws, observed).items():
                if lo <= 0.0 <= hi:
                    hits[name] += 1
                widths[name].append(hi - lo)
                if hi - lo < 1e-12:
                    degenerate[name] += 1
        print(f"    {label}")
        for name in hits:
            print(
                f"        {name:<11} coverage {hits[name] / n_datasets:>4.0%}"
                f"   mean width {np.mean(widths[name]):.5f}"
                f"   zero-width {degenerate[name] / n_datasets:.0%}"
            )
        if label.strip() == "debiased":
            record(
                "E8",
                f"debiased: basic/bc cover 0 more often than percentile "
                f"({hits['bc'] / n_datasets:.0%} / {hits['basic'] / n_datasets:.0%} "
                f"vs {hits['percentile'] / n_datasets:.0%})",
                bool(max(hits["bc"], hits["basic"]) >= hits["percentile"]),
            )


def e9_smece_bandwidth(n: int = 2000, n_resamples: int = 250) -> None:
    """smECE re-selects its bandwidth per resample; does that drive the inflation?

    Parameters
    ----------
    n
        Observations.
    n_resamples
        Resamples.
    """
    print("\nE9  smECE: convexity of |.| vs bandwidth re-selection")
    rng = np.random.default_rng(90)
    y, p = calibrated(n, rng)
    _, sigma = smooth_calibration_error(y, p, return_sigma=True)
    print(f"    bandwidth chosen on the observed data: {sigma:.5f}")

    auto_o, auto_b = ratio(
        smooth_calibration_error, y, p, n_resamples, np.random.default_rng(91)
    )
    fixed_o, fixed_b = ratio(
        lambda t, q: smooth_calibration_error(t, q, sigma=sigma),
        y,
        p,
        n_resamples,
        np.random.default_rng(91),
    )
    print(
        f"    auto sigma   observed {auto_o:.5f}  boot {auto_b:.5f}  "
        f"ratio {auto_b / auto_o:.3f}"
    )
    print(
        f"    fixed sigma  observed {fixed_o:.5f}  boot {fixed_b:.5f}  "
        f"ratio {fixed_b / fixed_o:.3f}"
    )
    record(
        "E9",
        f"inflation survives a fixed bandwidth (ratio {fixed_b / fixed_o:.3f})",
        fixed_b / fixed_o > 1.05,
    )


def main() -> None:
    """Run every experiment and print a summary."""
    print("=" * 74)
    print("Why bootstrap CIs for calibration errors are wrong")
    print("=" * 74)

    e1_signed_versus_absolute()
    e2_sqrt2()
    e2b_frozen_edges()
    e3_sample_size()
    e4_curvature()
    e5_debiased_floor()
    e6_mcb()
    e7_intervals()
    e8_coverage()
    e9_smece_bandwidth()

    print("\n" + "=" * 74)
    failed = [r for r in RESULTS if not r[2]]
    for name, detail, passed in RESULTS:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<4} {detail}")
    print("=" * 74)
    print(f"{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print("\nFAILED PREDICTIONS -- the hypothesis needs revising:")
        for name, detail, _ in failed:
            print(f"  {name}: {detail}")


if __name__ == "__main__":
    main()
