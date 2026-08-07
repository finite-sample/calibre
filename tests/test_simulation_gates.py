"""Negative tests for the Monte Carlo gates in ``tests/simulation.py``.

The battery in ``test_monte_carlo.py`` rests entirely on four assertions --
``assert_unbiased``, ``assert_biased_upward``, ``assert_coverage`` and the two
Monte Carlo standard errors behind them. Every one of them is trivially
satisfiable by an implementation that checks nothing, and none of them had a test
that watched it fail. A suite built on a gate that cannot fail reports success
while testing nothing, which is worse than having no gate at all: it converts an
unverified codebase into one that certifies itself.

So each gate is exercised twice here. Once on input that satisfies its property,
where it must stay silent, and once on input that violates it, where it must
raise. The design is the one in `simcheck
<https://github.com/finite-sample/simcheck>`_, which was extracted from this
package's sibling repositories after the same gap turned up in three of them.
"""

from __future__ import annotations

import numpy as np
import pytest
from simcheck import binomial_band

from tests.simulation import (
    assert_biased_upward,
    assert_coverage,
    assert_unbiased,
    mc_se_mean,
    mc_se_proportion,
)

# --------------------------------------------------------------------------
# assert_unbiased
# --------------------------------------------------------------------------


def test_assert_unbiased_passes_on_a_centred_sample():
    """The gate must not fire on an estimator that is doing its job."""
    rng = np.random.default_rng(0)
    assert_unbiased(rng.normal(2.0, 0.5, 400), 2.0, label="centred")


def test_assert_unbiased_fails_on_a_shifted_sample():
    """A shift of one standard deviation over 400 draws cannot be missed."""
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError, match="standard errors"):
        assert_unbiased(rng.normal(2.5, 0.5, 400), 2.0, label="shifted")


def test_assert_unbiased_resolves_a_smaller_bias_as_the_study_grows():
    """The property that makes a replicate-derived tolerance worth having.

    The same call must be lenient in a small study and strict in a large one,
    with no threshold edited in between. A fixed tolerance could not do this.
    """
    rng = np.random.default_rng(1)
    small = rng.normal(2.02, 0.5, 30)
    large = rng.normal(2.02, 0.5, 40000)

    assert_unbiased(small, 2.0, label="too small a study to resolve 0.02")
    with pytest.raises(AssertionError):
        assert_unbiased(large, 2.0, label="large enough to resolve 0.02")


# --------------------------------------------------------------------------
# assert_biased_upward
# --------------------------------------------------------------------------


def test_assert_biased_upward_passes_when_the_bias_is_really_upward():
    """The gate must accept the case it exists to certify."""
    rng = np.random.default_rng(2)
    assert_biased_upward(rng.normal(2.5, 0.5, 400), 2.0, label="upward")


def test_assert_biased_upward_fails_on_an_unbiased_sample():
    """No bias is not an upward bias.

    This is the direction that matters: the plugin-versus-debiased argument in
    test_monte_carlo.py rests on this gate distinguishing a real upward bias from
    none, and a gate that accepted zero would make that argument vacuous.
    """
    rng = np.random.default_rng(3)
    with pytest.raises(AssertionError):
        assert_biased_upward(rng.normal(2.0, 0.5, 400), 2.0, label="unbiased")


def test_assert_biased_upward_fails_on_a_downward_bias():
    """And a bias in the wrong direction must not be reported as the right one."""
    rng = np.random.default_rng(4)
    with pytest.raises(AssertionError):
        assert_biased_upward(rng.normal(1.5, 0.5, 400), 2.0, label="downward")


# --------------------------------------------------------------------------
# assert_coverage
# --------------------------------------------------------------------------


def test_assert_coverage_passes_at_the_nominal_rate():
    """The gate must not fire on correctly calibrated intervals."""
    assert_coverage(950, 1000, 0.95, label="calibrated")


def test_assert_coverage_fails_on_under_coverage():
    """Intervals covering 80% of the time must fail a 95% claim."""
    with pytest.raises(AssertionError, match="outside"):
        assert_coverage(800, 1000, 0.95, label="under-covering")


def test_assert_coverage_fails_on_a_vacuous_interval():
    """Covering every single time is a defect too, and used to be invisible.

    An interval so wide it always covers is useless, and the old band could not
    say so. It took the standard error of the *observed* proportion, which is
    zero when every replication covers, so the band collapsed to a point and only
    a ``1/n`` floor kept the arithmetic alive at all. The band now comes from the
    null, where the standard error does not depend on what was observed.
    """
    with pytest.raises(AssertionError):
        assert_coverage(1000, 1000, 0.95, label="vacuous")


def test_assert_coverage_tightens_as_the_study_grows():
    """A 2.5-point shortfall is noise at R=100 and a failure at R=10000."""
    assert_coverage(93, 100, 0.95, label="too small a study to resolve 0.925")
    with pytest.raises(AssertionError):
        assert_coverage(9250, 10000, 0.95, label="large enough to resolve 0.925")


def test_the_coverage_band_matches_the_textbook_binomial_interval():
    """The band is nominal +- k sqrt(p(1-p)/n), not something invented here."""
    nominal, reps, sigmas = 0.95, 400, 3.0
    expected = sigmas * np.sqrt(nominal * (1 - nominal) / reps)
    low, high = binomial_band(nominal, reps, sigmas)
    assert low == pytest.approx(nominal - expected)
    assert high == pytest.approx(nominal + expected)


# --------------------------------------------------------------------------
# The standard errors underneath
# --------------------------------------------------------------------------


def test_the_monte_carlo_standard_error_of_a_mean_shrinks_as_one_over_root_r():
    """Four times the replications must halve it, not merely reduce it."""
    rng = np.random.default_rng(5)
    pool = rng.normal(0.0, 1.0, 64000)
    small = mc_se_mean(pool[:4000])
    large = mc_se_mean(pool[:16000])
    assert small / large == pytest.approx(2.0, rel=0.15)


def test_a_degenerate_study_gets_an_infinite_standard_error():
    """One draw admits no tolerance, and must not silently admit a tight one."""
    assert mc_se_mean(np.array([1.0])) == float("inf")
    assert mc_se_proportion(1, 1) == float("inf")


def test_the_proportion_standard_error_stays_usable_at_the_extremes():
    """Zero or total success must still yield a positive tolerance.

    ``sqrt(c(1-c)/n)`` is exactly zero when ``c`` is 0 or 1, which would make any
    deviation infinitely many standard errors and turn the gate into a coin
    flip on floating point. The floor is what stops that.
    """
    for hits in (0, 1000):
        assert mc_se_proportion(hits, 1000) > 0.0
