"""Tests for the monotone spline machinery and the calibrators built on it.

The central claim under test is monotonicity. `SplineCalibrator` advertised it for
several releases while a B-spline basis with non-negative coefficients cannot
deliver it, and nothing in the suite ever checked -- so every assertion here that
mentions monotonicity is a regression test for that gap.
"""

from __future__ import annotations

import numpy as np
import pytest

# Configurations swept by the monotonicity tests. Small and large knot counts,
# every supported degree, and both knot-placement strategies.
BASIS_CONFIGS = [
    (5, 1, "uniform"),
    (5, 3, "quantile"),
    (10, 2, "uniform"),
    (10, 3, "quantile"),
    (20, 3, "quantile"),
]

LINKS = ["logit", "identity"]

# Every exported calibrator.
ALL_CALIBRATORS = [
    "CDIIsotonicCalibrator",
    "CenteredIsotonicCalibrator",
    "IsotonicCalibrator",
    "NearlyIsotonicCalibrator",
    "RegularizedIsotonicCalibrator",
    "RelaxedPAVACalibrator",
    "SmoothedIsotonicCalibrator",
    "SplineCalibrator",
]

# NearlyIsotonicCalibrator is deliberately absent: it penalises monotonicity
# violations rather than forbidding them, so a violation there is the estimator
# working, not failing.
MONOTONE_CALIBRATORS = [c for c in ALL_CALIBRATORS if c != "NearlyIsotonicCalibrator"]


def _dataset(seed: int, n: int = 500, shape: str = "logistic"):
    """Generate a miscalibrated score/label pair.

    Parameters
    ----------
    seed
        Random seed.
    n
        Number of observations.
    shape
        Which true calibration curve to use.

    Returns
    -------
    x : ndarray
        Uncalibrated scores.
    y : ndarray
        Binary labels.
    """
    rng = np.random.default_rng(seed)
    if shape == "logistic":
        z = rng.normal(0, 2, n)
        p = 1.0 / (1.0 + np.exp(-z))
        x = 1.0 / (1.0 + np.exp(-1.8 * z))  # overconfident
    elif shape == "step":
        x = rng.random(n)
        p = np.where(x < 0.5, 0.1, 0.9)
    elif shape == "flat":
        x = rng.random(n)
        p = np.full(n, 0.3)
    else:  # concave
        x = rng.random(n)
        p = np.sqrt(x)
    y = (rng.random(n) < p).astype(float)
    return x, y


# --------------------------------------------------------------------------- #
# The basis itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("n_knots", "degree", "knots"), BASIS_CONFIGS)
def test_ispline_basis_columns_are_monotone(n_knots, degree, knots):
    """Every I-spline basis column must be non-decreasing.

    This is the property that makes non-negative coefficients sufficient for a
    monotone fit. A raw B-spline basis fails it -- its columns are bumps.
    """
    from calibre._core import monotone_spline_basis

    basis = monotone_spline_basis(n_knots=n_knots, degree=degree, knots=knots)
    x = np.linspace(0.0, 1.0, 600)
    basis.fit(x)
    M = basis.design(x)

    diffs = np.diff(M, axis=0)
    assert np.all(diffs >= -1e-10), (
        f"basis column not monotone; worst decrease {diffs.min():.3e}"
    )


@pytest.mark.parametrize(("n_knots", "degree", "knots"), BASIS_CONFIGS)
def test_ispline_basis_is_monotone_outside_the_knot_range(n_knots, degree, knots):
    """Monotonicity must also hold where the basis is extrapolated."""
    from calibre._core import monotone_spline_basis

    basis = monotone_spline_basis(n_knots=n_knots, degree=degree, knots=knots)
    basis.fit(np.linspace(0.2, 0.8, 400))
    M = basis.design(np.linspace(-0.5, 1.5, 800))

    diffs = np.diff(M, axis=0)
    assert np.all(diffs >= -1e-10), (
        f"basis not monotone under extrapolation; worst {diffs.min():.3e}"
    )


def test_ispline_basis_drops_the_constant_column():
    """The constant column is replaced by an explicit intercept.

    The full I-spline basis has ``I_0 = sum_j B_j = 1`` by partition of unity.
    Keeping it alongside an intercept would be exactly collinear.
    """
    from calibre._core import monotone_spline_basis

    basis = monotone_spline_basis(n_knots=8, degree=3, knots="uniform")
    x = np.linspace(0.0, 1.0, 300)
    basis.fit(x)
    M = basis.design(x)

    for j in range(M.shape[1]):
        assert not np.allclose(M[:, j], M[0, j]), (
            f"column {j} is constant; the partition-of-unity column was not dropped"
        )


# --------------------------------------------------------------------------- #
# Monotonicity of the fitted calibrators -- the regression test for A1
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize(("n_knots", "degree", "knots"), BASIS_CONFIGS)
def test_spline_calibrator_is_monotone(link, n_knots, degree, knots):
    """The fitted calibration curve must have exactly zero violations.

    Not "few" violations, not "within tolerance" -- monotonicity here is
    structural (non-negative coefficients on a monotone basis), so any violation
    beyond floating-point noise means the construction is wrong.
    """
    from calibre import SplineCalibrator

    x, y = _dataset(0, n=600)
    cal = SplineCalibrator(
        n_knots=n_knots, degree=degree, knots=knots, link=link, alpha=0.1
    )
    cal.fit(x, y)

    grid = np.linspace(x.min(), x.max(), 2000)
    fitted = cal.transform(grid)
    violations = int(np.sum(np.diff(fitted) < -1e-10))
    assert violations == 0, f"{violations} monotonicity violations"


@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize("seed", range(12))
def test_spline_calibrator_is_monotone_across_datasets(link, seed):
    """Zero violations across many random datasets and shapes."""
    from calibre import SplineCalibrator

    shape = ["logistic", "step", "flat", "concave"][seed % 4]
    x, y = _dataset(seed, n=400, shape=shape)
    cal = SplineCalibrator(link=link, alpha=0.1).fit(x, y)

    grid = np.linspace(-0.2, 1.2, 1500)
    fitted = cal.transform(grid)
    assert np.all(np.diff(fitted) >= -1e-10), f"seed={seed} shape={shape}: not monotone"


@pytest.mark.parametrize("alpha", [0.0, 1e-6, 0.01, 1.0, 100.0])
def test_regularized_calibrator_is_monotone(alpha):
    """The regularized variant shares the basis, so it inherits the guarantee."""
    from calibre import RegularizedIsotonicCalibrator

    x, y = _dataset(1, n=600)
    cal = RegularizedIsotonicCalibrator(alpha=alpha).fit(x, y)

    grid = np.linspace(x.min(), x.max(), 2000)
    fitted = cal.transform(grid)
    assert np.all(np.diff(fitted) >= -1e-10), f"alpha={alpha}: not monotone"


# --------------------------------------------------------------------------- #
# Regression test for A2: stored basis and coefficients must correspond
# --------------------------------------------------------------------------- #


def test_stored_basis_and_coefficients_are_consistent():
    """Coefficient count must match the *stored* basis's design width.

    The previous implementation refit one mutable transformer inside its CV loop
    and stored a reference to it, so the retained knots came from the last fold
    while the coefficients came from the best fold. Nothing detected the mismatch
    because the shapes still lined up -- so this checks the design produced by the
    stored basis, which is the object actually used at predict time.
    """
    from calibre import SplineCalibrator

    x, y = _dataset(2, n=500)
    cal = SplineCalibrator(n_knots=10, degree=3).fit(x, y)

    design = cal.basis_.design(x)
    assert design.shape[1] == cal.coef_.size, (
        f"stored basis produces {design.shape[1]} columns but there are "
        f"{cal.coef_.size} coefficients"
    )


def test_fit_is_deterministic_given_random_state():
    """Same random_state must give an identical curve."""
    from calibre import SplineCalibrator

    x, y = _dataset(3, n=500)
    grid = np.linspace(0, 1, 200)

    a = SplineCalibrator(random_state=0).fit(x, y).transform(grid)
    b = SplineCalibrator(random_state=0).fit(x, y).transform(grid)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_final_model_is_refit_on_all_data():
    """After CV the model must be refit on the full sample, not kept from a fold.

    Selecting a single fold's fitted model discards 1/cv of the data and picks on
    validation noise. The refit model must therefore differ from every per-fold
    fit on the same split.
    """
    from sklearn.model_selection import StratifiedKFold

    from calibre import SplineCalibrator

    x, y = _dataset(4, n=400)
    cal = SplineCalibrator(n_knots=8, degree=3, alpha=0.1, cv=4, random_state=0)
    cal.fit(x, y)

    grid = np.linspace(0.05, 0.95, 100)
    full = cal.transform(grid)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    for train_idx, _ in skf.split(x.reshape(-1, 1), y):
        fold = SplineCalibrator(
            n_knots=8, degree=3, alpha=0.1, cv=4, random_state=0
        ).fit(x[train_idx], y[train_idx])
        assert not np.allclose(full, fold.transform(grid), atol=1e-12), (
            "fitted curve matches a single-fold fit, so it was not refit on all data"
        )


def test_cross_validation_selects_alpha():
    """With alpha=None the class must actually choose, and record the choice."""
    from calibre import SplineCalibrator

    x, y = _dataset(5, n=800)
    cal = SplineCalibrator(alpha=None, cv=4, random_state=0).fit(x, y)

    assert hasattr(cal, "alpha_"), "the selected alpha must be recorded as alpha_"
    assert cal.alpha_ >= 0
    # An explicitly supplied alpha must be honoured verbatim.
    fixed = SplineCalibrator(alpha=3.0, random_state=0).fit(x, y)
    assert fixed.alpha_ == 3.0


# --------------------------------------------------------------------------- #
# Optimality, against an independent solver
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [0.0, 0.05, 1.0])
def test_identity_link_attains_the_constrained_optimum(alpha):
    """The identity-link solve must match CVXPY on the same problem.

    CVXPY is reliable at this size, so it is a genuine independent check that the
    bounded least-squares formulation reaches the constrained optimum rather than
    merely producing a feasible point.
    """
    import cvxpy as cp

    from calibre._core import fit_monotone_spline, monotone_spline_basis

    x, y = _dataset(6, n=300)
    basis = monotone_spline_basis(n_knots=10, degree=3, knots="quantile")
    basis.fit(x)
    M = basis.design(x)
    w = np.ones_like(y)
    p = M.shape[1]

    intercept, delta = fit_monotone_spline(
        M, y, sample_weight=w, alpha=alpha, link="identity"
    )

    theta = cp.Variable(1)
    d = cp.Variable(p)
    eta = theta + M @ d
    obj = cp.sum(cp.multiply(w, cp.square(eta - y)))
    if alpha > 0 and p > 1:
        obj = obj + alpha * cp.sum_squares(cp.diff(d))
    prob = cp.Problem(cp.Minimize(obj), [d >= 0])
    prob.solve(solver=cp.CLARABEL)

    def objective(b0, dd):
        e = b0 + M @ dd
        val = float(np.sum(w * (e - y) ** 2))
        if alpha > 0 and p > 1:
            val += alpha * float(np.sum(np.diff(dd) ** 2))
        return val

    ours = objective(intercept, delta)
    ref = objective(float(theta.value[0]), np.asarray(d.value))
    assert ours <= ref + 1e-6 * max(1.0, abs(ref)), (
        f"ours={ours:.8f} worse than CVXPY={ref:.8f}"
    )


def test_links_agree_closely_on_log_loss():
    """Both links must land in the same place, since both are near-optimal.

    The tempting assertion -- "fitting the proper score wins on the proper score"
    -- is empirically false here, and worth recording rather than asserting away.
    Measured over 30 held-out splits with each link choosing its own ``alpha`` by
    cross-validation, ``identity`` won on log-loss 20 times out of 30 (mean 0.46494
    against 0.46542). The gap is ~0.1%, i.e. both links recover essentially the
    same monotone curve.

    A shared ``alpha`` would not even be a fair comparison: logit coefficients are
    in log-odds units, so the same penalty is far stronger on that scale.

    ``logit`` remains the default for reasons other than accuracy -- it returns
    probabilities in the open interval with no clipping, which matters whenever the
    output is fed to a log-loss-based downstream step.
    """
    from sklearn.metrics import log_loss

    from calibre import SplineCalibrator

    x, y = _dataset(7, n=1500)
    eps = 1e-6
    scores = {}
    for link in LINKS:
        f = SplineCalibrator(link=link, random_state=0).fit_transform(x, y)
        scores[link] = log_loss(y, np.clip(f, eps, 1 - eps))

    best = min(scores.values())
    for link, value in scores.items():
        assert value <= best * 1.05, (
            f"{link} is more than 5% worse than the better link: {scores}"
        )


# --------------------------------------------------------------------------- #
# Boundary behaviour, ranges, and scale
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("link", LINKS)
def test_extrapolation_is_constant(link):
    """Outside the fitted range the curve must flatten, not diverge."""
    from calibre import SplineCalibrator

    x, y = _dataset(8, n=400)
    cal = SplineCalibrator(link=link, alpha=0.1).fit(x, y)

    lo = cal.transform(np.array([x.min() - 5.0, x.min() - 1.0, x.min()]))
    hi = cal.transform(np.array([x.max(), x.max() + 1.0, x.max() + 5.0]))
    np.testing.assert_allclose(lo, lo[-1], atol=1e-9)
    np.testing.assert_allclose(hi, hi[0], atol=1e-9)


def test_logit_link_needs_no_clipping():
    """The logistic path returns probabilities in (0, 1) by construction."""
    from calibre import SplineCalibrator

    x, y = _dataset(9, n=800)
    f = SplineCalibrator(link="logit", clip_output=False, alpha=0.1).fit_transform(x, y)
    assert np.all(f > 0.0), f"minimum {f.min()} is not above 0"
    assert np.all(f < 1.0), f"maximum {f.max()} is not below 1"


@pytest.mark.parametrize("link", LINKS)
def test_clip_output_bounds_the_range(link):
    """With clipping on, output is inside [0, 1] for either link."""
    from calibre import SplineCalibrator

    x, y = _dataset(10, n=500)
    f = SplineCalibrator(link=link, clip_output=True, alpha=0.1).fit_transform(x, y)
    assert f.min() >= 0.0
    assert f.max() <= 1.0


@pytest.mark.parametrize(
    "cls_name", ["SplineCalibrator", "RegularizedIsotonicCalibrator"]
)
def test_scales_to_100k_without_degrading(cls_name):
    """Large n must fit quickly and still honour alpha.

    The previous regularized implementation put one parameter per unique score;
    above a few thousand its solve stopped converging and it silently fell back to
    unpenalised isotonic regression, so alpha became a no-op on exactly the sizes
    where smoothing matters. A fixed basis has no such regime.
    """
    import time

    import calibre

    cls = getattr(calibre, cls_name)
    rng = np.random.default_rng(11)
    n = 100_000
    x = np.sort(rng.random(n))
    y = (rng.random(n) < x).astype(float)

    start = time.perf_counter()
    cal = cls(alpha=1.0).fit(x, y)
    elapsed = time.perf_counter() - start

    assert elapsed < 20.0, f"{cls_name} took {elapsed:.1f}s at n={n}"

    fitted = cal.transform(x)
    assert np.all(np.diff(fitted) >= -1e-10), "not monotone at scale"
    # A staircase would collapse to a handful of levels; a smooth fit should not.
    assert len(np.unique(np.round(fitted, 9))) > 100, (
        "fit collapsed to a near-constant curve, which is what the silent "
        "fallback used to look like"
    )


def test_regularized_alpha_preserves_mean_calibration():
    """Raising alpha must not deflate the predictions.

    A ridge-toward-zero penalty drives E[beta] to 0 as alpha grows. A roughness
    penalty leaves straight lines alone, so the mean must track the base rate.
    """
    from calibre import RegularizedIsotonicCalibrator

    x, y = _dataset(12, n=2000)
    base = float(y.mean())
    for alpha in (0.0, 0.1, 1.0, 10.0, 1000.0):
        f = RegularizedIsotonicCalibrator(alpha=alpha).fit_transform(x, y)
        assert abs(float(f.mean()) - base) < 0.05, (
            f"alpha={alpha}: E[beta]={f.mean():.4f} drifted from base rate {base:.4f}"
        )


# --------------------------------------------------------------------------- #
# scikit-learn contract
# --------------------------------------------------------------------------- #


def _same_params(a: dict, b: dict) -> bool:
    """Compare two get_params() mappings elementwise.

    ``==`` is not enough: some calibrators take array-valued parameters (CDI's
    ``thresholds``), and comparing those with ``==`` yields an array, which
    ``assert`` cannot evaluate.

    Parameters
    ----------
    a
        Parameters before fitting.
    b
        Parameters after fitting.

    Returns
    -------
    bool
        True if every parameter is unchanged.
    """
    if a.keys() != b.keys():
        return False
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


@pytest.mark.parametrize("cls_name", ALL_CALIBRATORS)
def test_fit_does_not_mutate_constructor_params(cls_name):
    """fit() must leave get_params() untouched, or clone/GridSearchCV break.

    ``SmoothedIsotonicCalibrator`` used to coerce ``poly_order`` and
    ``min_window`` onto the instance inside fit, so a cloned estimator did not
    match the one it was cloned from.
    """
    import calibre

    cls = getattr(calibre, cls_name)
    x, y = _dataset(13, n=300)

    cal = cls()
    before = dict(cal.get_params())
    cal.fit(x, y)
    assert _same_params(before, cal.get_params())


@pytest.mark.parametrize("cls_name", ALL_CALIBRATORS)
def test_fit_does_not_mutate_out_of_range_params(cls_name):
    """Out-of-range arguments are corrected for the fit, not written back.

    The coercion path is the one that mutated state, so it needs its own case:
    passing valid values would never have caught the original bug.
    """
    from sklearn.base import clone

    import calibre

    cls = getattr(calibre, cls_name)
    x, y = _dataset(15, n=300)

    # Only some calibrators take these; the rest exercise the default path.
    kwargs = {}
    params = cls().get_params()
    if "poly_order" in params:
        kwargs["poly_order"] = 0
    if "min_window" in params:
        kwargs["min_window"] = 1

    cal = cls(**kwargs)
    before = dict(cal.get_params())
    cal.fit(x, y)
    assert _same_params(before, cal.get_params())
    assert _same_params(before, clone(cal).get_params())


@pytest.mark.parametrize(
    "cls_name", ["SplineCalibrator", "RegularizedIsotonicCalibrator"]
)
def test_transform_before_fit_raises(cls_name):
    """Predicting before fitting must fail clearly."""
    import calibre

    cls = getattr(calibre, cls_name)
    with pytest.raises(AttributeError, match="not fitted"):
        cls().transform(np.array([0.1, 0.5]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_knots": 2},
        {"degree": 0},
        {"alpha": -1.0},
        {"link": "probit"},
        {"knots": "spaced"},
        {"cv": 1},
    ],
)
def test_invalid_params_rejected_by_fit(kwargs):
    """Bad configuration must raise at fit time rather than produce a fit."""
    from calibre import SplineCalibrator

    x, y = _dataset(14, n=200)
    with pytest.raises(ValueError, match=r"(?i)must be"):
        SplineCalibrator(**kwargs).fit(x, y)


# --------------------------------------------------------------------------- #
# Monotonicity under tied scores
# --------------------------------------------------------------------------- #


def _tied_dataset(seed: int, n: int = 600, decimals: int = 2):
    """Generate scores with heavy ties, as a rounded or binned model produces.

    Parameters
    ----------
    seed
        Random seed.
    n
        Number of observations.
    decimals
        Rounding applied to the scores; fewer decimals means more ties.

    Returns
    -------
    tuple of ndarray
        Scores with repeated values, and binary labels.
    """
    rng = np.random.default_rng(seed)
    x = np.round(rng.uniform(0.0, 1.0, n), decimals)
    y = rng.binomial(1, x).astype(float)
    return x, y


@pytest.mark.parametrize("cls_name", MONOTONE_CALIBRATORS)
@pytest.mark.parametrize("decimals", [1, 2])
def test_monotone_under_tied_scores(cls_name, decimals):
    """Tied input scores must not break monotonicity.

    Tied scores are the ordinary case in calibration -- tree ensembles and any
    rounded or binned score produce them. ``SmoothedIsotonicCalibrator`` built
    its interpolant directly on the training scores with duplicates present,
    which silently discarded all but one observation per tie group and produced
    34 violations on this data in its default configuration.

    Zero violations, not a tolerance: every calibrator here is monotone by
    construction.
    """
    import calibre

    cls = getattr(calibre, cls_name)
    x, y = _tied_dataset(seed=7, decimals=decimals)

    cal = cls().fit(x, y)
    grid = np.linspace(0.0, 1.0, 4000)
    diffs = np.diff(cal.transform(grid))

    n_violations = int((diffs < -1e-9).sum())
    assert n_violations == 0, (
        f"{cls_name} produced {n_violations} violations on tied scores "
        f"(decimals={decimals}), worst {diffs.min():+.6f}"
    )


@pytest.mark.parametrize(
    "cls_name",
    # SplineCalibrator is excluded because it selects (n_knots, alpha) by
    # cross-validation, and KFold assigns folds by row position, so shuffling
    # legitimately changes which hyperparameters win. That is CV behaviour, not
    # tie handling.
    [c for c in MONOTONE_CALIBRATORS if c != "SplineCalibrator"],
)
def test_ties_do_not_depend_on_input_order(cls_name):
    """Shuffling the training rows must not change the fit.

    An interpolant built on duplicated abscissae keeps whichever tied point
    survived the sort, which makes the result depend on row order. Pooling ties
    first removes that dependence.
    """
    import calibre

    cls = getattr(calibre, cls_name)
    x, y = _tied_dataset(seed=11)

    grid = np.linspace(0.0, 1.0, 500)
    first = cls().fit(x, y).transform(grid)

    rng = np.random.default_rng(3)
    perm = rng.permutation(len(x))
    second = cls().fit(x[perm], y[perm]).transform(grid)

    np.testing.assert_allclose(first, second, atol=1e-9)
