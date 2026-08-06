"""Monotone spline calibration with a user-specified roughness penalty.

Same estimator family as :class:`calibre.SplineCalibrator`, with the emphasis
reversed: here the smoothing penalty is something you set and control, rather than
something cross-validation picks for you. Useful when you want a specific amount of
smoothing, or when you want to sweep ``alpha`` yourself.
"""

from __future__ import annotations

import logging

import numpy as np

from .._core import (
    VALID_LINKS,
    MonotoneSplineBasis,
    fit_monotone_spline,
    monotone_spline_basis,
)
from ..base import BaseCalibrator
from ..utils import check_arrays

logger = logging.getLogger(__name__)

__all__ = ["RegularizedIsotonicCalibrator"]


class RegularizedIsotonicCalibrator(BaseCalibrator):
    r"""Monotone calibration with an explicit roughness penalty.

    Solves

    .. math::
        \min_{\theta,\ \delta \ge 0}\
        \mathcal{L}\big(\theta + M\delta;\ y, w\big)
        + \alpha \lVert \Delta\delta \rVert^2

    where ``M`` is an I-spline design, so ``delta >= 0`` makes the fit monotone by
    construction, and :math:`\Delta\delta` is the second difference of the
    underlying B-spline coefficients.

    Parameters
    ----------
    alpha
        Roughness penalty. ``0`` gives an unpenalised monotone spline; larger
        values drive the fit toward the best monotone straight line.
    n_knots
        Number of knots in the basis.
    degree
        B-spline degree.
    knots
        ``"quantile"`` or ``"uniform"`` knot placement.
    link
        ``"logit"`` or ``"identity"``. See :class:`calibre.SplineCalibrator`.
    clip_output
        Clip calibrated values into ``[0, 1]``.
    enable_diagnostics
        Whether to enable plateau diagnostics analysis.

    Attributes
    ----------
    basis_ : MonotoneSplineBasis
        The fitted basis.
    intercept_ : float
        Fitted intercept, on the link scale.
    coef_ : ndarray of shape (n_basis,)
        Fitted non-negative increment coefficients.
    n_features_in_ : int
        Always 1. Present for scikit-learn compatibility.

    Notes
    -----
    **The penalty is on curvature, not on magnitude.** A ridge penalty
    :math:`\alpha\sum_i \beta_i^2` buys no smoothness at all: unconstrained its
    solution is :math:`\beta = y/(1+\alpha)`, a uniform deflation of every
    probability that breaks mean calibration by construction and drives all
    predictions to zero as :math:`\alpha` grows. A second-difference penalty leaves
    any straight line unpenalised, so the identity map and the empirical base rate
    both survive it.

    **Why a fixed basis rather than one parameter per score.** Putting a parameter at
    every unique score makes this a smoothing-spline problem whose penalty operator
    scales like :math:`h^{-2} \sim n^{2}`, so the normal equations scale like
    :math:`n^{4}`. That is ill-conditioned in a way no solver choice repairs -- a
    constrained QP stops converging above a few thousand distinct scores, ADMM
    diverges, and a matrix-free least-squares solve fails to converge while the
    fitted mean collapses away from the base rate. A modest fixed basis with a
    coefficient penalty -- the P-spline construction of Eilers & Marx (1996), as used
    by the SCOP-splines of Pya & Wood (2015) -- has none of those regimes: it fits
    100,000 points in milliseconds with monotonicity guaranteed structurally.

    .. note::

       ``alpha=0`` no longer reduces to isotonic regression. It gives an
       *unpenalised monotone regression spline*, which is smooth rather than
       piecewise constant. For the exact isotonic fit use
       :class:`calibre.IsotonicCalibrator`; to remove isotonic's plateaus without
       leaving the non-parametric family, use
       :class:`calibre.CenteredIsotonicCalibrator`.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre import RegularizedIsotonicCalibrator
    >>>
    >>> rng = np.random.default_rng(0)
    >>> x = rng.random(500)
    >>> y = (rng.random(500) < x).astype(float)
    >>>
    >>> cal = RegularizedIsotonicCalibrator(alpha=1.0).fit(x, y)
    >>> fitted = cal.transform(np.linspace(0, 1, 200))
    >>> bool(np.all(np.diff(fitted) >= -1e-10))
    True

    See Also
    --------
    SplineCalibrator : Same estimator with the penalty chosen by cross-validation.
    CenteredIsotonicCalibrator : Non-parametric and plateau-free.
    IsotonicCalibrator : The exact isotonic fit.
    """

    #: Candidate roughness penalties searched when ``alpha="auto"``. Matches the
    #: grid SplineCalibrator has always used for the same parameter.
    ALPHA_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

    def __init__(
        self,
        alpha: float | str = "auto",
        n_knots: int = 10,
        degree: int = 3,
        knots: str = "quantile",
        link: str = "logit",
        cv: int = 5,
        scoring: str = "log_loss",
        random_state: int | None = 0,
        clip_output: bool = True,
        enable_diagnostics: bool = False,
    ):
        # Call base class for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.alpha = alpha
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.link = link
        self.clip_output = clip_output

    def _fit_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit the penalised monotone spline.

        Parameters
        ----------
        X
            Uncalibrated scores.
        y
            Targets: binary labels, or probabilities in ``[0, 1]``.
        sample_weight
            Non-negative per-observation weights.

        Raises
        ------
        ValueError
            If the configuration or the targets are invalid.
        """
        from ..selection import resolve_auto

        X, y = check_arrays(X, y)
        if self.link not in VALID_LINKS:
            raise ValueError(f"link must be one of {VALID_LINKS}, got {self.link!r}")
        # The Bernoulli likelihood requires y in [0, 1]; least squares on the
        # identity scale does not, so only the logit link enforces it.
        if self.link == "logit" and np.any((y < 0) | (y > 1)):
            raise ValueError(
                'y must lie in [0, 1] for link="logit" (it parameterises a '
                'Bernoulli likelihood); use link="identity" for unbounded targets'
            )

        # The penalty controls curvature and has no principled fixed value, so
        # it is selected unless the caller pins it. Stored on alpha_, never
        # written back onto self.alpha.
        self.alpha_ = resolve_auto(
            self.alpha,
            "alpha",
            self.ALPHA_GRID,
            lambda **kw: type(self)(
                n_knots=self.n_knots,
                degree=self.degree,
                knots=self.knots,
                link=self.link,
                clip_output=self.clip_output,
                **kw,
            ),
            X,
            y,
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            sample_weight=sample_weight,
        )

        basis = monotone_spline_basis(
            n_knots=self.n_knots, degree=self.degree, knots=self.knots
        ).fit(X)
        intercept, coef = fit_monotone_spline(
            basis.design(X),
            y,
            sample_weight=sample_weight,
            alpha=self.alpha_,
            link=self.link,
        )

        self.basis_: MonotoneSplineBasis = basis
        self.intercept_ = intercept
        self.coef_ = coef
        self.n_features_in_ = 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map scores through the fitted calibration curve.

        Parameters
        ----------
        X
            Scores to calibrate.

        Returns
        -------
        ndarray of shape (n_samples,)
            Calibrated probabilities.

        Raises
        ------
        AttributeError
            If called before :meth:`fit`.
        """
        from scipy.special import expit

        if not hasattr(self, "basis_"):
            raise AttributeError(
                f"{type(self).__name__} is not fitted yet. Call fit() first."
            )
        x = np.asarray(X, dtype=float).ravel()
        eta = self.intercept_ + self.basis_.design(x) @ self.coef_
        out = expit(eta) if self.link == "logit" else eta
        return np.clip(out, 0.0, 1.0) if self.clip_output else np.asarray(out, float)
