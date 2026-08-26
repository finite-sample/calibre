"""Monotone spline calibration.

A smooth, strictly monotone calibration map is the shape post-hoc calibration
benchmarks consistently favor: it corrects miscalibration without collapsing the
base model's score ordering into a staircase the way isotonic regression does.

Monotonicity here is structural rather than enforced afterwards. The design matrix
is an I-spline basis, on which non-negative coefficients give a non-decreasing
function, so the constraint is a box constraint on the coefficients and the fitted
curve cannot violate monotonicity at all.
"""

from __future__ import annotations

import logging
from numbers import Integral

import numpy as np

from .._core import (
    VALID_KNOTS,
    VALID_LINKS,
    MonotoneSplineBasis,
    PiecewiseLinear,
    aggregate_ties,
    fit_monotone_spline,
    monotone_spline_basis,
)
from ..base import BaseCalibrator
from ..utils import check_array_1d, check_arrays, check_fitted

logger = logging.getLogger(__name__)

__all__ = ["SplineCalibrator"]


class SplineCalibrator(BaseCalibrator):
    r"""Monotone spline calibration with cross-validated smoothing.

    Fits

    .. math::
        g\big(f(x)\big) = \theta + \sum_k \delta_k I_k(x),
        \qquad \delta_k \ge 0

    where the :math:`I_k` are I-splines (each non-decreasing) and :math:`g` is the
    link. Because every basis function is non-decreasing and every coefficient is
    non-negative, :math:`f` is non-decreasing by construction; the link is
    increasing, so the calibrated probability is too.

    Args:
        n_knots: Number of knots, or ``"auto"`` to select it by cross-validation.
            The basis has ``n_knots + degree - 1`` functions.
        degree: B-spline degree. 3 gives the usual cubic behavior.
        knots: ``"quantile"`` (default) places knots at score quantiles;
            ``"uniform"`` spaces them evenly. Quantile is normally right for
            calibration, where scores pile up wherever the base model is
            confident and uniform knots spend resolution on empty regions.
        alpha: Roughness penalty on the coefficient increments, or ``"auto"``
            to select it by cross-validation. A number pins it exactly. The loss
            is weight-normalized, so alpha does not change meaning with sample
            size or a constant rescaling of observation weights.
        link: ``"logit"`` (default) fits a penalized Bernoulli likelihood:
            log-loss is the proper score for binary labels, and predictions
            land in ``(0, 1)`` with no clipping. ``"identity"`` fits penalized
            least squares on the probability scale -- a single bounded linear
            solve.
        cv: Number of cross-validation folds. Stratified when ``y`` is binary.
        scoring: Proper scoring rule minimized during automatic selection.
            ``"auto"`` uses log loss for the logit link and squared error for
            the identity link.
        max_cv_samples: Cap on the number of observations used for
            *hyperparameter selection*. The final model is always refit on the
            full sample; this only bounds the cost of the search, which would
            otherwise fit the grid once per fold over every row (at n=100k
            that is ~50s against 0.3s for a single fit). Selecting two scalars
            from a large random subsample costs essentially nothing
            statistically. Set to ``None`` to search on all of the data.
        random_state: Seed for the cross-validation split. Defaults to ``0``
            so that ``fit`` is reproducible: cross-validation here only
            selects a hyperparameter, and a fit that silently returns a
            different curve on each identical call is a trap. Pass ``None`` to
            draw the split from the global RNG instead.
        clip_output: Clip calibrated values into ``[0, 1]``. A no-op for
            ``link="logit"``.
        enable_diagnostics: Whether to enable plateau diagnostics analysis.

    Attributes:
        basis_: The fitted basis. Its knots come from the same fit that
            produced ``coef_``.
        intercept_: Fitted intercept, on the link scale.
        coef_: Fitted non-negative increment coefficients.
        alpha_: The penalty actually used -- selected by cross-validation, or
            echoed back from ``alpha``.
        n_knots_: The knot count actually used.
        n_features_in_: Always 1. Present for scikit-learn compatibility.

    Notes:
        **Non-negative coefficients on a plain B-spline basis do not give
        monotonicity.** B-spline basis functions are bumps, so a non-negative
        combination of them is a non-negative *function* and nothing more -- a single
        non-negative coefficient already traces a curve that rises and then falls.
        Monotonicity requires non-negativity on the coefficient *differences*, which is
        exactly what the I-spline (cumulative) basis encodes; see
        :class:`calibre._core.MonotoneSplineBasis`. The basis follows Ramsay (1988),
        while the coefficient-difference penalty follows Eilers & Marx's (1996)
        P-splines. Pya & Wood's (2015) SCOP-splines in R's ``scam`` are a related
        reference model, not the same fitting algorithm.

        **Cross-validation selects a hyperparameter and then refits on all the data.**
        It is not a search for whichever fold's model scored best on its own validation
        split: that selects on noise and ships a model trained on only ``(cv-1)/cv`` of
        the sample. Logit-link fits are scored by log-loss; identity-link fits are
        scored by squared error, the loss they actually minimise.

    Examples:
        >>> import numpy as np
        >>> from calibre import SplineCalibrator
        >>>
        >>> rng = np.random.default_rng(0)
        >>> x = rng.random(500)
        >>> y = (rng.random(500) < x).astype(float)
        >>>
        >>> cal = SplineCalibrator(alpha=0.1).fit(x, y)
        >>> fitted = cal.transform(np.linspace(0, 1, 200))
        >>> bool(np.all(np.diff(fitted) >= -1e-10))     # monotone by construction
        True
        >>> bool(fitted.min() >= 0.0 and fitted.max() <= 1.0)
        True

    See Also:
        CenteredIsotonicCalibrator : Non-parametric interpolation, without tuning.
        IsotonicCalibrator : Piecewise-constant non-parametric calibration.
    """

    ALPHA_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    N_KNOTS_GRID = (5, 10, 20)

    def __init__(
        self,
        n_knots: int | str = "auto",
        degree: int = 3,
        knots: str = "quantile",
        alpha: float | str = "auto",
        link: str = "logit",
        cv: int = 5,
        scoring: str = "auto",
        max_cv_samples: int | None = 20_000,
        random_state: int | None = 0,
        clip_output: bool = True,
        enable_diagnostics: bool = False,
    ):
        # Call base class for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.alpha = alpha
        self.link = link
        self.cv = cv
        self.scoring = scoring
        self.max_cv_samples = max_cv_samples
        self.random_state = random_state
        self.clip_output = clip_output

    def _validate(self) -> None:
        """Check the configuration.

        Raises:
            ValueError: If any parameter is out of range.

        Notes:
            Validation lives here rather than in ``__init__`` so ``get_params`` and
            ``clone`` round-trip, and it raises rather than silently coercing -- a
            coerced value would persist and make ``clone`` return a differently
            configured estimator than the one it copied.
        """
        if isinstance(self.n_knots, str):
            if self.n_knots != "auto":
                raise ValueError(
                    f'n_knots must be an integer >= 3 or "auto", got {self.n_knots!r}'
                )
        elif (
            isinstance(self.n_knots, bool)
            or not isinstance(self.n_knots, Integral)
            or self.n_knots < 3
        ):
            raise ValueError(
                f'n_knots must be an integer >= 3 or "auto", got {self.n_knots!r}'
            )
        if (
            isinstance(self.degree, bool)
            or not isinstance(self.degree, Integral)
            or self.degree < 1
        ):
            raise ValueError(f"degree must be an integer >= 1, got {self.degree!r}")
        if self.knots not in VALID_KNOTS:
            raise ValueError(f"knots must be one of {VALID_KNOTS}, got {self.knots!r}")
        if self.link not in VALID_LINKS:
            raise ValueError(f"link must be one of {VALID_LINKS}, got {self.link!r}")
        if isinstance(self.alpha, str):
            if self.alpha != "auto":
                raise ValueError(
                    "alpha must be finite and non-negative or "
                    f'"auto", got {self.alpha!r}'
                )
        else:
            try:
                alpha = float(self.alpha)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "alpha must be finite and non-negative or "
                    f'"auto", got {self.alpha!r}'
                ) from exc
            if not np.isfinite(alpha) or alpha < 0:
                raise ValueError(
                    f"alpha must be finite and non-negative, got {self.alpha}"
                )
        if (
            isinstance(self.cv, bool)
            or not isinstance(self.cv, Integral)
            or self.cv < 2
        ):
            raise ValueError(f"cv must be an integer >= 2, got {self.cv!r}")
        if self.max_cv_samples is not None and (
            isinstance(self.max_cv_samples, bool)
            or not isinstance(self.max_cv_samples, Integral)
            or self.max_cv_samples < 2 * self.cv
        ):
            raise ValueError(
                f"max_cv_samples must be an integer >= 2*cv={2 * self.cv} "
                f"or None, got {self.max_cv_samples!r}"
            )

    def _fit_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Select the smoothing parameters, then fit on all the data.

        Args:
            X: Uncalibrated scores.
            y: Targets: binary labels, or probabilities in ``[0, 1]``.
            sample_weight: Non-negative per-observation weights.

        Raises:
            ValueError: If the configuration or the targets are invalid.
        """
        X, y = check_arrays(X, y)
        self._validate()
        w = (
            np.ones_like(y)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).ravel()
        )
        if w.shape != y.shape:
            raise ValueError("sample_weight must have the same shape as y")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("sample_weight must contain finite non-negative values")
        if np.sum(w) <= 0.0:
            raise ValueError("sample_weight must contain at least one positive weight")

        positive = w > 0.0
        X_fit, y_fit, w_fit = X[positive], y[positive], w[positive]
        fit_weight = None if sample_weight is None else w_fit
        # The Bernoulli likelihood requires positive-mass targets in [0, 1].
        if self.link == "logit" and np.any((y_fit < 0) | (y_fit > 1)):
            raise ValueError(
                'y must lie in [0, 1] for link="logit" (it parameterises a '
                'Bernoulli likelihood); use link="identity" for unbounded targets'
            )

        if self.alpha == "auto" or self.n_knots == "auto":
            n_knots, alpha = self._select_hyperparameters(X_fit, y_fit, fit_weight)
        else:
            n_knots, alpha = int(self.n_knots), float(self.alpha)

        basis = monotone_spline_basis(
            n_knots=n_knots, degree=self.degree, knots=self.knots
        ).fit(X_fit, sample_weight=fit_weight)
        intercept, coef = fit_monotone_spline(
            basis.design(X_fit),
            y_fit,
            sample_weight=fit_weight,
            alpha=alpha,
            link=self.link,
        )

        self.basis_ = basis
        self.intercept_ = intercept
        self.coef_ = coef
        self.alpha_ = alpha
        self.n_knots_ = n_knots
        self.n_features_in_ = 1

    def _select_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> tuple[int, float]:
        """Choose ``(n_knots, alpha)`` by cross-validated prediction loss.

        Args:
            X: Uncalibrated scores.
            y: Targets.
            sample_weight: Optional sample weights.

        Returns:
            n_knots: Selected knot count.
            alpha: Selected roughness penalty.

        """
        from ..selection import select_by_cv

        # A quantile-knot basis needs enough distinct scores to place its knots.
        n_unique = int(np.unique(X).size)
        if self.n_knots == "auto":
            knot_grid = [k for k in self.N_KNOTS_GRID if k <= max(3, n_unique - 1)] or [
                3
            ]
        else:
            knot_grid = [int(self.n_knots)]
        alpha_grid = (
            list(self.ALPHA_GRID) if self.alpha == "auto" else [float(self.alpha)]
        )
        scoring = (
            ("log_loss" if self.link == "logit" else "brier")
            if self.scoring == "auto"
            else self.scoring
        )
        best = select_by_cv(
            lambda **params: type(self)(
                degree=self.degree,
                knots=self.knots,
                link=self.link,
                scoring=self.scoring,
                clip_output=self.clip_output,
                **params,
            ),
            {"n_knots": knot_grid, "alpha": alpha_grid},
            X,
            y,
            sample_weight=sample_weight,
            cv=self.cv,
            scoring=scoring,
            max_cv_samples=self.max_cv_samples,
            random_state=self.random_state,
        )
        return int(best["n_knots"]), float(best["alpha"])

    def _predict_from(
        self,
        basis: MonotoneSplineBasis,
        intercept: float,
        coef: np.ndarray,
        X: np.ndarray,
        clip: bool,
    ) -> np.ndarray:
        """Evaluate a fitted basis/coefficient pair at ``X``.

        Args:
            basis: A fitted :class:`calibre._core.MonotoneSplineBasis`.
            intercept: Fitted intercept on the link scale.
            coef: Fitted non-negative increment coefficients.
            X: Points to evaluate at.
            clip: Whether to clip into ``[0, 1]``.

        Returns:
            ndarray of shape (n_samples,): Calibrated probabilities.
        """
        from scipy.special import expit

        eta = intercept + basis.design(X) @ coef
        out = expit(eta) if self.link == "logit" else eta
        return np.clip(out, 0.0, 1.0) if clip else np.asarray(out, dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map scores through the fitted calibration curve.

        Args:
            X: Scores to calibrate.

        Returns:
            ndarray of shape (n_samples,): Calibrated probabilities.

        """
        check_fitted(self, ["basis_", "intercept_", "coef_"])
        return self._predict_from(
            self.basis_,
            self.intercept_,
            self.coef_,
            check_array_1d(X),
            clip=self.clip_output,
        )

    def calibration_curve(self, n_points: int = 200) -> PiecewiseLinear:
        """Sample the fitted map onto a grid, for plotting or inspection.

        Args:
            n_points: Number of grid points across the fitted score range.

        Returns:
            PiecewiseLinear: The sampled curve.

        Raises:
            AttributeError: If called before :meth:`fit`.
        """
        if not hasattr(self, "basis_"):
            raise AttributeError(
                f"{type(self).__name__} is not fitted yet. Call fit() first."
            )
        if self._fit_data_X is None or self._fit_data_y is None:
            raise AttributeError(
                f"{type(self).__name__} has no retained training data."
            )
        x_unique, _, _ = aggregate_ties(
            self._fit_data_X, self._fit_data_y, self._fit_data_weight
        )
        grid = np.linspace(float(x_unique[0]), float(x_unique[-1]), n_points)
        return PiecewiseLinear(grid, self.transform(grid))
