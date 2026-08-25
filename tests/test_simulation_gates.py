"""Negative tests for the Monte Carlo gates in ``tests/simulation.py``.

The battery in ``test_monte_carlo.py`` rests entirely on three assertions --
``assert_unbiased``, ``assert_biased_upward``, ``assert_coverage`` -- and the
Monte Carlo standard error behind them. Every one of them is trivially
satisfiable by an implementation that checks nothing, and none of them had a test
that watched it fail. A suite built on a gate that cannot fail reports success
while testing nothing, which is worse than having no gate at all: it converts an
unverified codebase into one that certifies itself.

So each gate is exercised twice here. Once on input that satisfies its property,
where it must stay silent, and once on input that violates it, where it must
raise. The design is the one in `simcheck
<https://github.com/finite-sample/simcheck>`_, which was extracted from this
package's sibling repositories after the same gap turned up in three of them.

The impossible inputs are here for the same reason as the ordinary ones. A gate
fails open when the arithmetic it is handed is degenerate -- no replications, a
constant estimator, a hit count outside the study -- and every one of those
started as a wrong answer somewhere upstream, so a gate that reads it as a pass
converts one bug into a certificate. All three were live in this module at some
point; the tests are what stops them coming back.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from simcheck import binomial_band

from tests.simulation import (
    assert_biased_upward,
    assert_coverage,
    assert_unbiased,
    mc_se_mean,
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


def test_assert_unbiased_fails_on_a_constant_estimator_that_misses():
    """An estimator that never varies and is wrong is the worst case, not the best.

    The bias t statistic is ``bias / mc_se``, and ``mc_se`` is zero when the
    estimator returns the same number every replication. Reading that as a t of
    zero says "no detectable bias" about the one estimator whose bias no number
    of replications could ever resolve.
    """
    with pytest.raises(AssertionError, match="no sampling variation"):
        assert_unbiased(np.full(400, 5.0), 0.0, label="constant and wrong")


def test_assert_unbiased_passes_on_a_constant_estimator_that_hits():
    """A deterministic estimator sitting exactly on the target is not a failure."""
    assert_unbiased(np.full(400, 2.0), 2.0, label="constant and right")


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


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_assert_biased_upward_fails_on_an_empty_study():
    """Zero replications certify nothing, and must not certify a bias.

    The excess is ``(mean - target) / se``, which is NaN when there is no mean.
    NaN compares False against everything, so a gate written as ``if excess <
    required: raise`` stays silent here while one written as ``if not (excess >=
    required)`` fires. The difference is invisible until it isn't.

    NumPy's warnings about the empty mean are the input being degenerate, which
    is the point of the test, so they are filtered rather than fixed.
    """
    with pytest.raises(AssertionError):
        assert_biased_upward(np.array([]), 0.0, label="no replications")


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


@pytest.mark.parametrize("hits", [-5, 150])
def test_assert_coverage_rejects_a_hit_count_outside_the_study(hits):
    """A count below zero or above the replication count is an accounting error.

    It has to be rejected rather than coerced. Coercion is what an expression
    like ``covered[:hits] = True`` does silently: NumPy reads a negative count as
    a slice from the end, so ``-5`` of ``100`` becomes 95 covered replications --
    exactly nominal, and the gate certifies the arithmetic that produced it.
    """
    with pytest.raises(ValueError, match=r"in \[0, 100\]"):
        assert_coverage(hits, 100, 0.95, label="impossible count")


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


# --------------------------------------------------------------------------
# The gates survive -O
# --------------------------------------------------------------------------

# Each gate below is given input it must reject, and the child exits non-zero if
# any of them stays quiet. The three are the same violations the negative tests
# above use, so a difference between the two runs is the ``-O`` flag and nothing
# else.
_OPTIMISED_CHILD = """
import numpy as np

from tests.simulation import assert_biased_upward, assert_coverage, assert_unbiased

if __debug__:
    raise SystemExit("the subprocess is not running under -O")

rng = np.random.default_rng(0)
cases = [
    (assert_unbiased, (rng.normal(2.5, 0.5, 400), 2.0), {"label": "shifted"}),
    (assert_biased_upward, (rng.normal(2.0, 0.5, 400), 2.0), {"label": "unbiased"}),
    (assert_coverage, (800, 1000, 0.95), {"label": "under-covering"}),
]

for gate, args, kwargs in cases:
    try:
        gate(*args, **kwargs)
    except AssertionError:
        continue
    raise SystemExit(f"{gate.__name__} did not fire under -O")
"""


def test_the_gates_still_fire_under_optimisation():
    """``python -O`` deletes ``assert``, and must not be able to delete a gate.

    Every test above runs in this session, where ``assert`` is live, so all of
    them pass against a gate built on a bare ``assert`` statement -- and that
    gate checks nothing under ``-O``. This module was in exactly that state
    before the gates moved onto simcheck: seven of its negative tests failed
    under ``python -O`` because nothing was raised.

    A subprocess is the only way to see it. The flag is read at compile time, so
    it cannot be turned on for part of a running interpreter. The child checks
    ``__debug__`` with an ``if`` rather than an ``assert``, since an ``assert``
    that ``-O`` deleted would be a test of the flag that the flag disables.
    """
    result = subprocess.run(
        [sys.executable, "-O", "-c", _OPTIMISED_CHILD],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
