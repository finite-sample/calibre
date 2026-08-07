r"""Simulation tests: unbiasedness, coverage, size and power.

The rest of the suite establishes that the code computes what a reference
implementation computes, and that algebraic identities hold. Neither asks whether
the estimators are any *good*. These tests do, in the standard way: draw from a
data-generating process whose truth is known in closed form, repeat, and check
that the estimator lands on the truth and that a nominal interval covers it at
the nominal rate.

Every tolerance is a **Monte Carlo standard error**, never a chosen number:
``sd/sqrt(R)`` for a mean, ``sqrt(c(1-c)/R)`` for a proportion. Tightening an
assertion therefore means raising the replication count, not editing a constant,
and a failure message reports how many standard errors away the estimate landed.

Replication counts are sized so the effects asserted are several standard errors
wide, which keeps this file fast enough to run on every commit. The deep versions,
with an order of magnitude more replications, live in
``experiments/simulation_study/``.

See ``tests/simulation.py`` for the designs and their closed forms.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibre import (
    CenteredIsotonicCalibrator,
    IsotonicCalibrator,
    RelaxedPAVACalibrator,
    bootstrap_ci,
    confidence_bands,
    consistency_bands,
    corp_reliability,
    score_decomposition,
)
from calibre.metrics import (
    brier_score,
    debiased_calibration_error,
    plugin_calibration_error,
    smooth_calibration_error,
)
from tests.simulation import (
    DESIGNS,
    assert_biased_upward,
    assert_coverage,
    assert_unbiased,
)

# Sized so the asserted effects are several MC standard errors wide.
R_FAST = 200
N_FAST = 1500


def replicate(design, n, r, seed=0):
    """Draw ``r`` independent datasets from a design.

    Parameters
    ----------
    design
        The data-generating process.
    n
        Observations per dataset.
    r
        Number of replications.
    seed
        Base seed; each replication gets its own generator.

    Yields
    ------
    tuple of ndarray
        ``(y, x, p_true)`` per replication.
    """
    for i in range(r):
        yield design.sample(n, np.random.default_rng(seed + i))


# --------------------------------------------------------------------------- #
# The closed forms themselves. Everything below depends on these being right,
# so they are checked against a large simulation before being trusted as targets.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(DESIGNS))
def test_population_identity_holds_exactly(name):
    """``MCB - DSC + UNC = mean score`` must hold in population, not just in sample.

    This is a statement about the closed forms in ``tests/simulation.py``, not
    about any estimator. If it fails, every target in this file is wrong.
    """
    design = DESIGNS[name]
    assert design.mcb - design.dsc + design.unc == pytest.approx(
        design.brier, abs=1e-12
    )


@pytest.mark.parametrize("name", sorted(DESIGNS))
def test_the_link_is_a_bijection(name):
    """The closed forms need ``x`` to determine ``p``; clipping would break that.

    If the link were not invertible, ``E[y | x]`` would be an average over a set
    of ``p`` values rather than a single one, and every population quantity here
    would be quietly wrong.
    """
    design = DESIGNS[name]
    _, x, p = design.sample(4000, np.random.default_rng(0))
    np.testing.assert_allclose(design.inverse(x), p, atol=1e-9)
    np.testing.assert_allclose(design.true_cep(x), p, atol=1e-9)


@pytest.mark.parametrize("name", sorted(DESIGNS))
def test_quadrature_targets_match_a_large_simulation(name):
    """The quadrature must agree with brute force, or the targets are fiction."""
    design = DESIGNS[name]
    rng = np.random.default_rng(12345)
    p = design.draw_p(400_000, rng)
    x = design.link(p)

    assert float(p.mean()) == pytest.approx(design.p_bar, abs=0.003)
    assert float(p.var()) == pytest.approx(design.dsc, abs=0.003)
    assert float(((x - p) ** 2).mean()) == pytest.approx(design.mcb, abs=0.003)
    assert float(np.abs(x - p).mean()) == pytest.approx(design.ce_l1, abs=0.003)


# --------------------------------------------------------------------------- #
# The decomposition against its population values.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["calibrated", "overconfident", "discrete"])
def test_uncertainty_matches_its_exact_finite_sample_expectation(name):
    """``UNC`` has a known finite-sample bias, so the target is exact, not asymptotic.

    ``UNC`` is estimated by ``y_bar (1 - y_bar)``, and ``E[y_bar(1-y_bar)]``
    equals ``p_bar(1-p_bar)(1 - 1/n)`` exactly, because ``Var(y_bar)`` is
    ``p_bar(1-p_bar)/n`` however ``p`` is distributed. That makes this the
    sharpest check available on the decomposition: not "close to" but "equal to",
    within Monte Carlo error.
    """
    design = DESIGNS[name]
    values = np.array(
        [
            score_decomposition(x, y)["UNC"]
            for y, x, _ in replicate(design, N_FAST, R_FAST)
        ]
    )
    assert_unbiased(
        values,
        design.expected_unc_at(N_FAST),
        label=f"UNC on {name}",
    )


@pytest.mark.parametrize("name", ["calibrated", "overconfident", "prior_shift"])
def test_discrimination_converges_to_the_variance_of_the_truth(name):
    """``DSC`` estimates ``Var(p)``; the bias must shrink as ``n`` grows.

    Not asserted as unbiased at any single ``n``: ``DSC`` is a plug-in built on a
    PAV fit to the same data, so it carries a finite-sample bias of its own.
    """
    design = DESIGNS[name]
    errors = []
    for n in (400, 6400):
        values = np.array(
            [
                score_decomposition(x, y)["DSC"]
                for y, x, _ in replicate(design, n, 80, seed=n)
            ]
        )
        errors.append(abs(float(values.mean()) - design.dsc))
    assert errors[1] < errors[0], (
        f"DSC on {name}: bias did not shrink with n "
        f"({errors[0]:.5f} at n=400 vs {errors[1]:.5f} at n=6400)"
    )


@pytest.mark.parametrize("name", ["overconfident", "prior_shift", "discrete"])
def test_miscalibration_converges_to_its_population_value(name):
    """``MCB`` estimates ``E[(x - p)^2]``; the bias must shrink as ``n`` grows.

    ``MCB`` is upward-biased at finite ``n`` because the recalibrated forecasts
    are fitted to the same data the score is computed on -- the same effect that
    makes the naive bootstrap inconsistent for it.
    """
    design = DESIGNS[name]
    errors = []
    for n in (400, 6400):
        values = np.array(
            [
                score_decomposition(x, y)["MCB"]
                for y, x, _ in replicate(design, n, 80, seed=n)
            ]
        )
        errors.append(abs(float(values.mean()) - design.mcb))
    assert errors[1] < errors[0], (
        f"MCB on {name}: bias did not shrink with n "
        f"({errors[0]:.5f} at n=400 vs {errors[1]:.5f} at n=6400)"
    )


def test_the_brier_score_is_unbiased_for_its_population_value():
    """The control: a plain mean, so it must land on the truth at any ``n``.

    Everything else in this file is a nonlinear functional with a finite-sample
    bias. This one is not, and if it ever failed the designs themselves would be
    suspect rather than the estimator.
    """
    for name in ("calibrated", "overconfident", "rare_event"):
        design = DESIGNS[name]
        values = np.array(
            [brier_score(y, x) for y, x, _ in replicate(design, N_FAST, R_FAST)]
        )
        assert_unbiased(values, design.brier, label=f"Brier on {name}")


# --------------------------------------------------------------------------- #
# The discriminating pair: one estimator must be unbiased where the other is not.
# --------------------------------------------------------------------------- #


def test_the_debiased_correction_is_unbiased_on_the_squared_scale():
    """The correction works exactly -- on the scale it actually corrects.

    ``debiased_calibration_error`` subtracts the per-bin Bernoulli variance from
    the squared gap. That makes the *sum* an unbiased estimate of the squared
    calibration error, which is zero here. Asserted with ``squared=True``, before
    the square root and the floor.
    """
    design = DESIGNS["calibrated"]
    values = np.array(
        [
            debiased_calibration_error(y, x, 15, squared=True)
            for y, x, _ in replicate(design, N_FAST, R_FAST)
        ]
    )
    assert_unbiased(values, 0.0, label="debiased squared error on calibrated data")


def test_the_unbiased_squared_estimate_is_negative_about_half_the_time():
    """Which is what an unbiased estimate of zero must do.

    Also the reason the floor exists, and the reason the floor introduces bias:
    it discards this half.
    """
    design = DESIGNS["calibrated"]
    values = np.array(
        [
            debiased_calibration_error(y, x, 15, squared=True)
            for y, x, _ in replicate(design, N_FAST, R_FAST)
        ]
    )
    fraction = float(np.mean(values < 0.0))
    se = np.sqrt(0.25 / values.size)
    assert abs(fraction - 0.5) <= 4.0 * se, (
        f"unbiased squared estimate was negative {fraction:.1%} of the time, "
        f"expected about 50% (se {se:.3f}, R {values.size})"
    )


def test_the_floor_makes_the_reported_error_biased_upward():
    """The cost of reporting a non-negative number, quantified.

    The sum is unbiased; ``sqrt(max(sum, 0))`` is not, because the floor throws
    away the negative half. No amount of data removes it -- this is a property of
    the transform, not a finite-sample effect. Documented in the estimator's own
    Notes, and asserted here so the documentation cannot quietly become false.
    """
    design = DESIGNS["calibrated"]
    reported = np.array(
        [
            debiased_calibration_error(y, x, 15)
            for y, x, _ in replicate(design, N_FAST, R_FAST)
        ]
    )
    assert_biased_upward(
        reported, 0.0, label="reported debiased error on calibrated data"
    )


def test_plugin_error_is_detectably_biased_where_the_truth_is_zero():
    """And the uncorrected estimator must fail the same check.

    This is the justification for the debiased estimator existing, and it was
    asserted nowhere before. If the plugin ever became unbiased, the correction
    would be pointless and the documentation would be wrong.
    """
    design = DESIGNS["calibrated"]
    values = np.array(
        [
            plugin_calibration_error(y, x, 15, 2)
            for y, x, _ in replicate(design, N_FAST, R_FAST)
        ]
    )
    assert_biased_upward(values, 0.0, label="plugin error on calibrated data")


def test_the_plugin_bias_grows_with_the_bin_count():
    """More bins means fewer observations per bin, hence more noise mistaken for error.

    The reason a bare ECE number without its bin count is not interpretable.
    """
    design = DESIGNS["calibrated"]
    means = []
    for n_bins in (5, 50):
        values = np.array(
            [
                plugin_calibration_error(y, x, n_bins, 2)
                for y, x, _ in replicate(design, N_FAST, 100, seed=n_bins)
            ]
        )
        means.append(float(values.mean()))
    assert means[1] > means[0] * 1.5, (
        f"plugin bias did not grow with bins: {means[0]:.5f} at 5 bins vs "
        f"{means[1]:.5f} at 50"
    )


# --------------------------------------------------------------------------- #
# The cross-check between two independently written parts of the package.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["overconfident", "prior_shift"])
def test_the_two_routes_to_the_l2_error_agree(name):
    """``sqrt(MCB)`` and the debiased l2 error estimate the same population value.

    ``calibre.evaluation`` reaches it through a PAV fit with no bins;
    ``calibre.metrics`` reaches it through equal-mass bins with a variance
    correction. Nothing in the package previously required them to agree, and
    they are the only two estimators of the same quantity it ships.

    Asserted as convergence rather than equality at one ``n``: both carry
    finite-sample bias, in opposite directions -- MCB upward from fitting, the
    binned error downward from averaging within bins.
    """
    design = DESIGNS[name]
    gaps = []
    for n in (500, 8000):
        pairs = [
            (
                np.sqrt(score_decomposition(x, y)["MCB"]),
                debiased_calibration_error(y, x, 15),
            )
            for y, x, _ in replicate(design, n, 60, seed=n)
        ]
        route_a = float(np.mean([a for a, _ in pairs]))
        route_b = float(np.mean([b for _, b in pairs]))
        gaps.append(abs(route_a - route_b))
    assert gaps[1] < gaps[0], (
        f"the two routes to the l2 error on {name} did not converge: "
        f"gap {gaps[0]:.5f} at n=500 vs {gaps[1]:.5f} at n=8000"
    )


@pytest.mark.parametrize("name", ["overconfident", "prior_shift", "discrete"])
def test_sqrt_mcb_converges_to_the_true_l2_error(name):
    """The population identity ``ce_l2 = sqrt(MCB)``, checked through the estimator."""
    design = DESIGNS[name]
    errors = []
    for n in (500, 8000):
        values = np.array(
            [
                np.sqrt(score_decomposition(x, y)["MCB"])
                for y, x, _ in replicate(design, n, 60, seed=n)
            ]
        )
        errors.append(abs(float(values.mean()) - design.ce_l2))
    assert errors[1] < errors[0], (
        f"sqrt(MCB) on {name} did not converge to the true l2 error "
        f"({errors[0]:.5f} at n=500 vs {errors[1]:.5f} at n=8000)"
    )


# --------------------------------------------------------------------------- #
# smECE has no closed form, but it is a consistent measure by construction.
# --------------------------------------------------------------------------- #


def test_smece_shrinks_toward_zero_on_calibrated_data():
    """A consistent measure of distance from calibration must vanish at zero distance."""
    means = []
    design = DESIGNS["calibrated"]
    for n in (500, 8000):
        values = np.array(
            [
                smooth_calibration_error(y, x)
                for y, x, _ in replicate(design, n, 60, seed=n)
            ]
        )
        means.append(float(values.mean()))
    assert means[1] < means[0] * 0.6, (
        f"smECE did not shrink with n on calibrated data: "
        f"{means[0]:.5f} at n=500 vs {means[1]:.5f} at n=8000"
    )


def test_smece_orders_designs_by_their_true_error():
    """The measure must rank forecasters the way the truth does.

    A calibration measure that cannot order a calibrated forecaster below a
    miscalibrated one is worse than useless, however well it is estimated.
    """
    ordered = ["calibrated", "overconfident", "prior_shift"]
    truths = [DESIGNS[name].ce_l1 for name in ordered]
    assert truths == sorted(truths), "fixture no longer orders the designs"

    means = []
    for name in ordered:
        values = np.array(
            [
                smooth_calibration_error(y, x)
                for y, x, _ in replicate(DESIGNS[name], 3000, 40, seed=7)
            ]
        )
        means.append(float(values.mean()))
    assert means == sorted(means), (
        f"smECE ordered the designs {means}, but their true errors are {truths}"
    )


# --------------------------------------------------------------------------- #
# Coverage of intervals.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["percentile", "basic", "bc"])
def test_bootstrap_intervals_cover_the_brier_score(method):
    """The control for interval coverage: a linear, unbiased statistic.

    Whatever the method, an interval for a plain mean must achieve its nominal
    level. A method failing here is broken outright, independently of the
    convexity problem that afflicts calibration errors.
    """
    design = DESIGNS["overconfident"]
    hits = 0
    replications = 120
    for y, x, _ in replicate(design, 800, replications, seed=500):
        ci = bootstrap_ci(brier_score, y, x, level=0.9, n_resamples=120, method=method)
        hits += int(ci["lower"] <= design.brier <= ci["upper"])
    assert_coverage(
        hits, replications, 0.9, label=f"Brier interval, method={method}", n_se=3.5
    )


# --------------------------------------------------------------------------- #
# Coverage of bands, and the size of the test they imply.
# --------------------------------------------------------------------------- #


def _band_coverage(design, n, replications, level, kind, seed):
    """Per-dataset pointwise coverage fractions, and whether each covered fully.

    The unit of replication is the dataset, not the grid point: points within one
    dataset are strongly dependent, so treating them as independent would
    understate the Monte Carlo error several-fold.

    Parameters
    ----------
    design
        The data-generating process.
    n
        Observations per dataset.
    replications
        Number of datasets.
    level
        Nominal level.
    kind
        ``"consistency"`` compares the observed diagram against the band under
        the calibration null; ``"confidence"`` compares the *true* conditional
        event probability curve against the band.
    seed
        Base seed.

    Returns
    -------
    tuple of ndarray
        Per-dataset pointwise coverage, and per-dataset simultaneous coverage.
    """
    pointwise = []
    simultaneous = []
    for i, (y, x, _) in enumerate(replicate(design, n, replications, seed=seed)):
        if kind == "consistency":
            band = consistency_bands(x, y, level=level, n_resamples=100, random_state=i)
            target = corp_reliability(x, y).cep
        else:
            band = confidence_bands(x, y, level=level, n_resamples=100, random_state=i)
            target = design.true_cep(band["x"])
        inside = (band["lower"] <= target) & (target <= band["upper"])
        pointwise.append(float(inside.mean()))
        simultaneous.append(bool(inside.all()))
    return np.array(pointwise), np.array(simultaneous)


def test_consistency_bands_have_nominal_pointwise_coverage():
    """The claim the bands actually support, asserted with a usable tolerance.

    Replaces an assertion of ``0.5 <= covered / trials <= 1.0`` at a nominal 0.9,
    which a band covering half the time would have passed.
    """
    level = 0.9
    pointwise, _ = _band_coverage(
        DESIGNS["calibrated"], 400, 60, level, "consistency", seed=2000
    )
    se = float(pointwise.std(ddof=1) / np.sqrt(pointwise.size))
    deviation = abs(float(pointwise.mean()) - level) / se
    assert deviation <= 3.0, (
        f"consistency bands: pointwise coverage {pointwise.mean():.1%} vs nominal "
        f"{level:.0%} is {deviation:.1f} MC standard errors away "
        f"(se {se:.4f}, R {pointwise.size})"
    )


def test_the_bands_are_pointwise_and_not_simultaneous():
    """A band that holds at each point separately does not hold at all points.

    Measured on perfectly calibrated data, the observed diagram leaves a nominal
    90% consistency band *somewhere* on essentially every replication. So reading
    the bands as an envelope -- "my curve stayed inside, therefore calibrated" --
    is a test with a false-positive rate near one. The docstrings say so; this
    keeps them honest.
    """
    _, simultaneous = _band_coverage(
        DESIGNS["calibrated"], 400, 60, 0.9, "consistency", seed=2100
    )
    size = 1.0 - float(simultaneous.mean())
    assert size > 0.8, (
        f"the 'exits anywhere' reading rejected only {size:.0%} of calibrated "
        "samples; if the bands have become simultaneous, the documented warning "
        "is now wrong and should be removed"
    )


def test_confidence_bands_approach_nominal_coverage_of_the_truth():
    """These bands are centred on an isotonic fit, which is biased at small n.

    Coverage of the *true* conditional event probability curve is therefore below
    nominal on small samples and improves as the centring bias shrinks. Asserted
    as convergence rather than as a level, because the small-sample shortfall is a
    genuine property rather than a defect.
    """
    level = 0.9
    coverages = []
    for n in (300, 4800):
        pointwise, _ = _band_coverage(
            DESIGNS["overconfident"], n, 40, level, "confidence", seed=2200 + n
        )
        coverages.append(float(pointwise.mean()))
    assert coverages[1] > coverages[0], (
        f"confidence band coverage of the truth did not improve with n: "
        f"{coverages[0]:.1%} at n=300 vs {coverages[1]:.1%} at n=4800"
    )
    assert coverages[1] == pytest.approx(level, abs=0.06), (
        f"confidence band coverage at n=4800 was {coverages[1]:.1%}, "
        f"expected close to {level:.0%}"
    )


# --------------------------------------------------------------------------- #
# Power: a diagnostic that cannot detect miscalibration is worse than none.
# --------------------------------------------------------------------------- #


def test_band_excursions_are_rarer_under_the_null_than_under_distortion():
    """Size against power, using the fraction of the curve outside the band.

    Under the calibration null that fraction is the nominal miss rate. Under a
    distorted forecaster it must be clearly larger, or the bands carry no
    information about calibration at all.
    """
    level = 0.9
    null, _ = _band_coverage(
        DESIGNS["calibrated"], 800, 40, level, "consistency", seed=2300
    )
    alt, _ = _band_coverage(
        DESIGNS["overconfident"], 800, 40, level, "consistency", seed=2300
    )
    null_miss = 1.0 - null
    alt_miss = 1.0 - alt
    se = float(
        np.sqrt(
            null_miss.var(ddof=1) / null_miss.size
            + alt_miss.var(ddof=1) / alt_miss.size
        )
    )
    gap = float(alt_miss.mean() - null_miss.mean())
    assert gap > 3.0 * se, (
        f"excursion rate under distortion ({alt_miss.mean():.1%}) is only "
        f"{gap / se:.1f} MC standard errors above the null rate "
        f"({null_miss.mean():.1%})"
    )


def test_power_grows_with_sample_size():
    """More data must make a miscalibrated forecaster easier to detect."""
    level = 0.9
    misses = []
    for n in (300, 4800):
        pointwise, _ = _band_coverage(
            DESIGNS["overconfident"], n, 40, level, "consistency", seed=2400 + n
        )
        misses.append(1.0 - float(pointwise.mean()))
    assert misses[1] > misses[0], (
        f"excursion rate did not grow with n on a miscalibrated forecaster: "
        f"{misses[0]:.1%} at n=300 vs {misses[1]:.1%} at n=4800"
    )


# --------------------------------------------------------------------------- #
# Calibrator consistency: does a calibrator recover the *right* function?
# --------------------------------------------------------------------------- #


CALIBRATORS = {
    "isotonic": IsotonicCalibrator,
    "centered": CenteredIsotonicCalibrator,
    "relaxed_pava": RelaxedPAVACalibrator,
}


def _fitted_map_error(design, calibrator_cls, n, seed):
    """L2 distance between a fitted calibration map and the true inverse link.

    Parameters
    ----------
    design
        The data-generating process.
    calibrator_cls
        Calibrator to fit.
    n
        Training size.
    seed
        Seed.

    Returns
    -------
    float
        Root mean squared deviation from the truth, over a held-out sample.
    """
    rng = np.random.default_rng(seed)
    y, x, _ = design.sample(n, rng)
    fitted = calibrator_cls().fit(x, y)

    # Evaluated on a fresh draw, so this measures the map rather than the fit.
    _, x_new, p_new = design.sample(4000, np.random.default_rng(seed + 10_000))
    predicted = np.asarray(fitted.transform(x_new), dtype=float)
    return float(np.sqrt(np.mean((predicted - p_new) ** 2)))


@pytest.mark.parametrize("name", sorted(CALIBRATORS))
def test_the_fitted_map_converges_to_the_true_inverse_link(name):
    """A calibrator must recover the right function, not merely a monotone one.

    Every other test of these calibrators checks a property -- bounded, monotone,
    granular. None asks whether the curve is *correct*. Here the truth is known,
    so the distance to it can be measured, and it must shrink with data.
    """
    design = DESIGNS["overconfident"]
    errors = [
        float(
            np.mean(
                [
                    _fitted_map_error(design, CALIBRATORS[name], n, seed=3000 + i)
                    for i in range(12)
                ]
            )
        )
        for n in (250, 4000)
    ]
    assert errors[1] < errors[0], (
        f"{name}: distance to the true calibration map did not shrink with n "
        f"({errors[0]:.4f} at n=250 vs {errors[1]:.4f} at n=4000)"
    )


@pytest.mark.parametrize("name", sorted(CALIBRATORS))
def test_calibration_helps_a_miscalibrated_model(name):
    """The claim the package is built on, with an error bar on it.

    Paired within replication -- both scores come from the same held-out draw --
    so the standard error is of the *difference*, which is far smaller than that
    of either score.
    """
    design = DESIGNS["overconfident"]
    gains = []
    for i in range(30):
        rng = np.random.default_rng(4000 + i)
        y_fit, x_fit, _ = design.sample(2000, rng)
        y_test, x_test, _ = design.sample(2000, rng)
        fitted = CALIBRATORS[name]().fit(x_fit, y_fit)
        calibrated = np.asarray(fitted.transform(x_test), dtype=float)
        gains.append(brier_score(y_test, x_test) - brier_score(y_test, calibrated))
    gains = np.array(gains)
    se = float(gains.std(ddof=1) / np.sqrt(gains.size))
    assert float(gains.mean()) > 3.0 * se, (
        f"{name}: mean Brier gain {gains.mean():.5f} is only "
        f"{gains.mean() / se:.1f} MC standard errors above zero "
        f"(se {se:.5f}, R {gains.size})"
    )


@pytest.mark.parametrize("name", sorted(CALIBRATORS))
def test_calibrating_an_already_calibrated_model_costs_little_and_less_with_n(name):
    """There is nothing to fix, so any loss is pure fitting cost.

    Worth quantifying rather than assuming: this is the price of applying the
    package's own recommendation when it was not needed, and it must vanish as
    data accumulates.
    """
    design = DESIGNS["calibrated"]
    costs = []
    for n in (250, 4000):
        losses = []
        for i in range(12):
            rng = np.random.default_rng(5000 + i)
            y_fit, x_fit, _ = design.sample(n, rng)
            y_test, x_test, _ = design.sample(4000, rng)
            fitted = CALIBRATORS[name]().fit(x_fit, y_fit)
            calibrated = np.asarray(fitted.transform(x_test), dtype=float)
            losses.append(brier_score(y_test, calibrated) - brier_score(y_test, x_test))
        costs.append(float(np.mean(losses)))
    assert costs[1] < costs[0], (
        f"{name}: the cost of needless calibration did not shrink with n "
        f"({costs[0]:+.5f} at n=250 vs {costs[1]:+.5f} at n=4000)"
    )
    assert costs[1] < 0.01, (
        f"{name}: calibrating already-calibrated scores still cost "
        f"{costs[1]:+.5f} of Brier at n=4000"
    )
