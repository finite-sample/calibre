r"""Known-truth designs and Monte Carlo assertions.

This module exists so the test suite can ask the question that neither the
reference fixtures nor the property tests reach: **under a data-generating
process whose truth we know, is the estimator unbiased, and does a nominal 95%
interval cover 95% of the time?**

Closed-form population values
-----------------------------
Each design draws a true probability ``p`` and reports a score ``x``, related by
an affine map in logit space, ``logit(x) = slope * logit(p) + shift``. That map
is a strictly increasing bijection, so ``x`` determines ``p`` and the conditional
event probability is exactly ``E[y | x] = p``. The whole CORP decomposition
follows in closed form, for the Brier score:

===============  =========================================
quantity         population value
===============  =========================================
``UNC``          ``p_bar (1 - p_bar)``
``DSC``          ``Var(p)``
``MCB``          ``E[(x - p)^2]``
mean Brier       ``E[(x - p)^2] + E[p (1 - p)]``
===============  =========================================

and ``MCB - DSC + UNC = mean score`` holds in population as well as in sample.
The true :math:`\ell_1` calibration error is ``E|p - x|``; the true :math:`\ell_2`
error is exactly ``sqrt(MCB)`` -- which ties :func:`~calibre.score_decomposition`
to :func:`~calibre.debiased_calibration_error`, two independently written parts of
the package.

Expectations are evaluated by quadrature, not simulation, so the targets carry no
Monte Carlo error of their own.

Three constraints that are load-bearing
---------------------------------------
**No clipping in any link.** Clipping maps an interval of ``p`` onto one ``x``, so
``x`` stops determining ``p``, ``E[y | x]`` becomes an average over the clipped
region, and every population value above is quietly wrong. Working in logit space
keeps ``x`` inside ``(0, 1)`` with no clipping at all.

**Links are computed from the latent normal, never round-tripped through
``logit``.** For a logit-normal ``p``, ``logit(p) = mu + sigma * z`` holds exactly,
so both ``p`` and ``x`` are formed from ``z``. Recovering ``z`` from ``p``
numerically would overflow at the quadrature nodes, where ``p`` rounds to exactly
one.

**No global random state.** Draws come from :func:`numpy.random.default_rng` with
an explicit seed, so results do not depend on execution order.
``tests/data_generators.py`` uses the legacy global RNG throughout and is
deliberately not reused here; it also adds noise to the *score*, which breaks the
bijection these closed forms depend on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

__all__ = [
    "DESIGNS",
    "Design",
    "assert_biased_upward",
    "assert_coverage",
    "assert_unbiased",
    "mc_se_mean",
    "mc_se_proportion",
]

# Gauss-Hermite is exponentially accurate for these smooth integrands, and
# numpy's node solver overflows well before 400 nodes, so keep this modest.
_HERMITE_NODES = 120
_LEGENDRE_NODES = 2000


def _logistic(z: np.ndarray) -> np.ndarray:
    """Standard logistic function.

    Parameters
    ----------
    z
        Log-odds.

    Returns
    -------
    ndarray
        Probabilities in ``(0, 1)``.
    """
    # Branch to avoid overflow in exp for large |z|: the quadrature nodes reach
    # far into the tails, where exp(-z) overflows and warns even though the
    # result is correct.
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    tail = np.exp(z[~positive])
    out[~positive] = tail / (1.0 + tail)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_logistic`, guarded against exact 0 and 1.

    Parameters
    ----------
    p
        Probabilities.

    Returns
    -------
    ndarray
        Log-odds.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    return np.log(p) - np.log1p(-p)


@dataclass(frozen=True)
class Design:
    """A data-generating process whose population quantities are known exactly.

    Attributes
    ----------
    name : str
        Identifier, used in assertion messages.
    slope : float
        Multiplier on the log-odds. Above one is overconfident, below one
        underconfident, exactly one leaves the shape alone.
    shift : float
        Added to the log-odds, a pure prior shift.
    family : str
        ``"logit_normal"``, ``"uniform"`` or ``"atoms"``.
    mu : float
        Mean of the latent normal.
    sigma : float
        Standard deviation of the latent normal.
    atoms : tuple of float
        Support of ``p`` for the discrete family.
    atom_weights : tuple of float
        Probabilities of those atoms.
    """

    name: str
    slope: float = 1.0
    shift: float = 0.0
    family: str = "logit_normal"
    mu: float = 0.0
    sigma: float = 2.0
    atoms: tuple[float, ...] = ()
    atom_weights: tuple[float, ...] = ()
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    # The link and its inverse
    # ------------------------------------------------------------------ #

    def link(self, p: np.ndarray) -> np.ndarray:
        """Map true probabilities to reported scores.

        Parameters
        ----------
        p
            True probabilities.

        Returns
        -------
        ndarray
            Reported scores.
        """
        return _logistic(self.slope * _logit(p) + self.shift)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Recover the true probability from a reported score.

        Because the link is a bijection this is exactly ``E[y | x]``.

        Parameters
        ----------
        x
            Reported scores.

        Returns
        -------
        ndarray
            True conditional event probabilities.
        """
        return _logistic((_logit(x) - self.shift) / self.slope)

    def true_cep(self, x: np.ndarray) -> np.ndarray:
        """The true conditional event probability curve at ``x``.

        Parameters
        ----------
        x
            Reported scores.

        Returns
        -------
        ndarray
            ``E[y | x]``.
        """
        return self.inverse(np.asarray(x, dtype=float))

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def draw_p(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw true probabilities.

        Parameters
        ----------
        n
            Sample size.
        rng
            Generator.

        Returns
        -------
        ndarray
            True event probabilities.
        """
        if self.family == "atoms":
            return rng.choice(
                np.asarray(self.atoms), size=n, p=np.asarray(self.atom_weights)
            )
        if self.family == "uniform":
            return rng.uniform(0.0, 1.0, n)
        return _logistic(self.mu + self.sigma * rng.standard_normal(n))

    def sample(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw one dataset.

        Parameters
        ----------
        n
            Sample size.
        rng
            Generator.

        Returns
        -------
        tuple of ndarray
            ``(y, x, p_true)``: outcomes, reported scores, true probabilities.
        """
        p = self.draw_p(n, rng)
        x = self.link(p)
        y = rng.binomial(1, p).astype(float)
        return y, x, p

    # ------------------------------------------------------------------ #
    # Population quantities, by quadrature
    # ------------------------------------------------------------------ #

    def _nodes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quadrature nodes in ``p`` and ``x``, with probability weights.

        Returns
        -------
        tuple of ndarray
            ``(p_nodes, x_nodes, weights)``, weights summing to one.
        """
        if "nodes" in self._cache:
            return self._cache["nodes"]

        if self.family == "atoms":
            p = np.asarray(self.atoms, dtype=float)
            w = np.asarray(self.atom_weights, dtype=float)
            x = self.link(p)
        elif self.family == "uniform":
            raw, weights = np.polynomial.legendre.leggauss(_LEGENDRE_NODES)
            p = 0.5 * (raw + 1.0)
            w = 0.5 * weights
            x = self.link(p)
        else:
            # Gauss-Hermite against the standard normal. With z = sqrt(2) t,
            # int f(z) phi(z) dz = sum(w_i f(sqrt(2) t_i)) / sqrt(pi).
            t, weights = np.polynomial.hermite.hermgauss(_HERMITE_NODES)
            z = np.sqrt(2.0) * t
            latent = self.mu + self.sigma * z
            # Both p and x come from the latent value directly. Recovering the
            # latent from p would overflow where p rounds to exactly one.
            p = _logistic(latent)
            x = _logistic(self.slope * latent + self.shift)
            w = weights / np.sqrt(np.pi)

        w = w / w.sum()
        self._cache["nodes"] = (p, x, w)
        return p, x, w

    def _expect(self, values: np.ndarray) -> float:
        """Weighted mean of node values.

        Parameters
        ----------
        values
            Values evaluated at the quadrature nodes.

        Returns
        -------
        float
            The expectation.
        """
        _, _, w = self._nodes()
        return float(np.sum(w * np.asarray(values, dtype=float)))

    @property
    def p_bar(self) -> float:
        """Population base rate ``E[p]``.

        Returns
        -------
        float
            The base rate.
        """
        p, _, _ = self._nodes()
        return self._expect(p)

    @property
    def unc(self) -> float:
        """Population uncertainty ``p_bar (1 - p_bar)``.

        Returns
        -------
        float
            The irreducible term.
        """
        return self.p_bar * (1.0 - self.p_bar)

    @property
    def dsc(self) -> float:
        """Population discrimination ``Var(p)``.

        Returns
        -------
        float
            The discrimination term.
        """
        p, _, _ = self._nodes()
        return self._expect(p**2) - self.p_bar**2

    @property
    def mcb(self) -> float:
        """Population miscalibration ``E[(x - p)^2]``.

        Returns
        -------
        float
            The miscalibration term, for the Brier score.
        """
        p, x, _ = self._nodes()
        return self._expect((x - p) ** 2)

    @property
    def brier(self) -> float:
        """Population mean Brier score.

        Returns
        -------
        float
            ``E[(x - p)^2] + E[p (1 - p)]``.
        """
        p, _, _ = self._nodes()
        return self.mcb + self._expect(p * (1.0 - p))

    def _crossing(self) -> float | None:
        """Latent value where the link crosses the diagonal, if it does.

        ``x = p`` when ``slope * L + shift = L``, i.e. ``L = -shift / (slope - 1)``.

        Returns
        -------
        float or None
            The crossing point in logit space, or None when the link never
            crosses the diagonal.
        """
        if self.slope == 1.0:
            return None
        return -self.shift / (self.slope - 1.0)

    @property
    def ce_l1(self) -> float:
        r"""True :math:`\ell_1` calibration error ``E|p - x|``.

        Notes
        -----
        Computed by adaptive quadrature split at the diagonal crossing, not by the
        Gauss rule used for the other quantities. ``|x - p|`` has a kink where the
        link crosses the diagonal, and a polynomial quadrature converges slowly
        across a kink -- it was wrong in the third decimal for the designs whose
        crossing carries real mass, while remaining accurate for those where it
        does not.

        Returns
        -------
        float
            The true error.
        """
        if self.family == "atoms":
            p, x, _ = self._nodes()
            return self._expect(np.abs(x - p))

        from scipy.integrate import quad
        from scipy.stats import norm

        cross = self._crossing()
        integrand: Callable[[float], float]

        if self.family == "uniform":

            def uniform_integrand(p: float) -> float:
                return float(abs(self.link(np.asarray(p)) - p))

            integrand = uniform_integrand
            breaks = [0.0, 1.0]
            if cross is not None:
                p_star = float(_logistic(np.asarray(cross)))
                if 0.0 < p_star < 1.0:
                    breaks = [0.0, p_star, 1.0]
        else:

            def latent_integrand(z: float) -> float:
                p = _logistic(np.asarray(self.mu + self.sigma * z))
                return float(abs(self.link(p) - p) * norm.pdf(z))

            integrand = latent_integrand
            breaks = [-np.inf, np.inf]
            if cross is not None:
                z_star = (cross - self.mu) / self.sigma
                breaks = [-np.inf, float(z_star), np.inf]

        total = 0.0
        for lo, hi in pairwise(breaks):
            total += quad(integrand, lo, hi, limit=200)[0]
        return float(total)

    @property
    def ce_l2(self) -> float:
        r"""True :math:`\ell_2` calibration error, exactly ``sqrt(MCB)``.

        Returns
        -------
        float
            The true error.
        """
        return float(np.sqrt(self.mcb))

    def expected_unc_at(self, n: int) -> float:
        """Exact finite-sample expectation of the estimated ``UNC``.

        ``UNC`` is estimated as ``y_bar (1 - y_bar)``, and

        .. math:: E[\\bar{y}(1 - \\bar{y})] = p(1-p)\\left(1 - \\frac{1}{n}\\right)

        exactly, because ``Var(y_bar) = p_bar (1 - p_bar) / n`` however ``p`` is
        distributed. So this estimator's bias is known in closed form rather than
        merely asymptotically, which makes it the sharpest available check on the
        decomposition.

        Parameters
        ----------
        n
            Sample size.

        Returns
        -------
        float
            ``E[UNC_hat]`` at that sample size.
        """
        return self.unc * (1.0 - 1.0 / n)


def _build_designs() -> dict[str, Design]:
    """Construct the design set.

    Returns
    -------
    dict
        Name to design.
    """
    return {
        # True calibration error exactly zero. Also the case where binning adds
        # no bias, because E[y | x] = x holds within every bin.
        "calibrated": Design(name="calibrated", family="uniform"),
        # The canonical failure: log-odds inflated by 1.8.
        "overconfident": Design(name="overconfident", slope=1.8),
        # The opposite sign, so a one-sided correction cannot pass both.
        "underconfident": Design(name="underconfident", slope=0.6),
        # A pure prior shift, leaving the shape alone.
        "prior_shift": Design(name="prior_shift", shift=0.8),
        # A 2% base rate, where the interesting region is a sliver near zero.
        "rare_event": Design(name="rare_event", slope=1.6, mu=-4.0, sigma=1.0),
        # A discrete support, so equal-mass binning can be exact and binning bias
        # is removed from the comparison rather than assumed small.
        "discrete": Design(
            name="discrete",
            shift=0.5,
            family="atoms",
            atoms=(0.15, 0.35, 0.6, 0.85),
            atom_weights=(0.25, 0.25, 0.25, 0.25),
        ),
    }


DESIGNS = _build_designs()


# --------------------------------------------------------------------------- #
# Monte Carlo assertions
# --------------------------------------------------------------------------- #


def mc_se_mean(values: np.ndarray) -> float:
    """Standard error of a Monte Carlo mean.

    Parameters
    ----------
    values
        Replication estimates.

    Returns
    -------
    float
        ``sd / sqrt(R)``.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("inf")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def mc_se_proportion(hits: int, n: int) -> float:
    """Standard error of a Monte Carlo proportion.

    Parameters
    ----------
    hits
        Number of successes.
    n
        Number of replications.

    Returns
    -------
    float
        ``sqrt(c (1 - c) / n)``, floored so a proportion of exactly zero or one
        still admits a usable tolerance.
    """
    if n < 2:
        return float("inf")
    c = hits / n
    return float(np.sqrt(max(c * (1.0 - c), 1.0 / n) / n))


def assert_unbiased(
    estimates: np.ndarray,
    target: float,
    *,
    label: str,
    n_se: float = 3.0,
) -> None:
    """Assert a Monte Carlo mean sits within ``n_se`` standard errors of a target.

    The tolerance is derived from the design rather than chosen, so tightening it
    is a matter of raising the replication count, not of editing a number.

    Parameters
    ----------
    estimates
        One estimate per replication.
    target
        The population value.
    label
        Included in the failure message.
    n_se
        How many standard errors to allow.

    Raises
    ------
    AssertionError
        If the mean is further than ``n_se`` standard errors from ``target``.
    """
    estimates = np.asarray(estimates, dtype=float)
    mean = float(estimates.mean())
    se = mc_se_mean(estimates)
    deviation = abs(mean - target) / se if se > 0 else float("inf")
    assert deviation <= n_se, (
        f"{label}: mean {mean:.6f} vs target {target:.6f} is {deviation:.1f} "
        f"Monte Carlo standard errors away "
        f"(se {se:.6f}, R {estimates.size}, allowed {n_se})"
    )


def assert_biased_upward(
    estimates: np.ndarray,
    target: float,
    *,
    label: str,
    n_se: float = 3.0,
) -> None:
    """Assert a Monte Carlo mean sits detectably *above* a target.

    The complement of :func:`assert_unbiased`, for an estimator whose bias is the
    reason a corrected estimator exists.

    Parameters
    ----------
    estimates
        One estimate per replication.
    target
        The population value.
    label
        Included in the failure message.
    n_se
        How many standard errors above the target the mean must sit.

    Raises
    ------
    AssertionError
        If the mean is not clearly above ``target``.
    """
    estimates = np.asarray(estimates, dtype=float)
    mean = float(estimates.mean())
    se = mc_se_mean(estimates)
    excess = (mean - target) / se if se > 0 else 0.0
    assert excess >= n_se, (
        f"{label}: mean {mean:.6f} is only {excess:.1f} Monte Carlo standard "
        f"errors above target {target:.6f}; expected a detectable upward bias "
        f"(se {se:.6f}, R {estimates.size}, required {n_se})"
    )


def assert_coverage(
    hits: int,
    n: int,
    nominal: float,
    *,
    label: str,
    n_se: float = 3.0,
) -> None:
    """Assert empirical coverage matches a nominal level within Monte Carlo error.

    Parameters
    ----------
    hits
        Replications in which the interval contained the truth.
    n
        Total replications.
    nominal
        The claimed coverage.
    label
        Included in the failure message.
    n_se
        How many standard errors to allow.

    Raises
    ------
    AssertionError
        If empirical coverage is further than ``n_se`` standard errors from
        ``nominal``.
    """
    covered = hits / n
    se = mc_se_proportion(hits, n)
    deviation = abs(covered - nominal) / se if se > 0 else float("inf")
    assert deviation <= n_se, (
        f"{label}: coverage {covered:.1%} vs nominal {nominal:.1%} is "
        f"{deviation:.1f} Monte Carlo standard errors away "
        f"(se {se:.4f}, R {n}, allowed {n_se})"
    )
