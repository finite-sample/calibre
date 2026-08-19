"""Monotone spline calibration.

A smooth, strictly monotone calibration map is the shape post-hoc calibration
benchmarks consistently favour: it corrects miscalibration without collapsing the
base model's score ordering into a staircase the way isotonic regression does.

Monotonicity here is structural rather than enforced afterwards. The design matrix
is an I-spline basis, on which non-negative coefficients give a non-decreasing
function, so the constraint is a box constraint on the coefficients and the fitted
curve cannot violate monotonicity at all.
"""

from __future__ import annotations

import logging

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
from ..utils import check_arrays

logger = logging.getLogger(__name__)

__all__ = ["SplineCalibrator"]


def _log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-observation Bernoulli log-loss, safe at the boundaries.

    Args:
        y: Targets in ``[0, 1]``.
        p: Predicted probabilities.
        eps: Clipping bound, so a confident-and-wrong prediction contributes a
            large finite penalty rather than an infinite one that would leave
            every candidate incomparable.

    Returns:
        ndarray: Elementwise loss.
    """
    p = np.clip(p, eps, 1.0 - eps)
    return np.asarray(-(y * np.log(p) + (1.0 - y) * np.log1p(-p)), dtype=float)


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
        n_knots: Number of knots. The basis has ``n_knots + degree - 1``
            functions. Used only when ``alpha`` is given; otherwise
            cross-validation selects it.
        degree: B-spline degree. 3 gives the usual cubic behaviour.
        knots: ``"quantile"`` (default) places knots at score quantiles;
            ``"uniform"`` spaces them evenly. Quantile is normally right for
            calibration, where scores pile up wherever the base model is
            confident and uniform knots spend resolution on empty regions.
        alpha: Roughness penalty on the coefficient increments. ``None``
            (default) selects it, along with ``n_knots``, by cross-validation.
            A number fixes it and skips cross-validation.
        link: ``"logit"`` (default) fits a penalised Bernoulli likelihood:
            log-loss is the proper score for binary labels, and predictions
            land in ``(0, 1)`` with no clipping. ``"identity"`` fits penalised
            least squares on the probability scale -- a single bounded linear
            solve.
        cv: Number of cross-validation folds. Stratified when ``y`` is binary.
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
        :class:`calibre._core.MonotoneSplineBasis`. This is the construction behind the
        SCOP-splines of Pya & Wood (2015) in R's ``scam`` and the penalised B-splines
        of Eilers & Marx (1996).

        **Cross-validation selects a hyperparameter and then refits on all the data.**
        It is not a search for whichever fold's model scored best on its own validation
        split: that selects on noise and ships a model trained on only ``(cv-1)/cv`` of
        the sample. Folds are scored by log-loss -- a proper score -- rather than by
        :math:`R^2`.

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
        CenteredIsotonicCalibrator : Non-parametric, needs no tuning, also plateau-free.
        RegularizedIsotonicCalibrator : Same basis, penalty specified rather than tuned.
    """

    def __init__(
        self,
        n_knots: int = 10,
        degree: int = 3,
        knots: str = "quantile",
        alpha: float | None = None,
        link: str = "logit",
        cv: int = 5,
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
        if self.n_knots < 3:
            raise ValueError(f"n_knots must be at least 3, got {self.n_knots}")
        if self.degree < 1:
            raise ValueError(f"degree must be at least 1, got {self.degree}")
        if self.knots not in VALID_KNOTS:
            raise ValueError(f"knots must be one of {VALID_KNOTS}, got {self.knots!r}")
        if self.link not in VALID_LINKS:
            raise ValueError(f"link must be one of {VALID_LINKS}, got {self.link!r}")
        if self.alpha is not None and self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.cv < 2:
            raise ValueError(f"cv must be at least 2, got {self.cv}")
        if self.max_cv_samples is not None and self.max_cv_samples < 2 * self.cv:
            raise ValueError(
                f"max_cv_samples must be at least 2*cv={2 * self.cv}, "
                f"got {self.max_cv_samples}"
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
        # The Bernoulli likelihood requires y in [0, 1]; least squares on the
        # identity scale does not, so only the logit link enforces it.
        if self.link == "logit" and np.any((y < 0) | (y > 1)):
            raise ValueError(
                'y must lie in [0, 1] for link="logit" (it parameterises a '
                'Bernoulli likelihood); use link="identity" for unbounded targets'
            )

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

        if self.alpha is None:
            n_knots, alpha = self._select_hyperparameters(X, y, w)
        else:
            n_knots, alpha = self.n_knots, float(self.alpha)

        basis = monotone_spline_basis(
            n_knots=n_knots, degree=self.degree, knots=self.knots
        ).fit(X)
        intercept, coef = fit_monotone_spline(
            basis.design(X), y, sample_weight=w, alpha=alpha, link=self.link
        )

        self.basis_ = basis
        self.intercept_ = intercept
        self.coef_ = coef
        self.alpha_ = alpha
        self.n_knots_ = n_knots
        self.n_features_in_ = 1

    def _select_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> tuple[int, float]:
        """Choose ``(n_knots, alpha)`` by cross-validated log-loss.

        Args:
            X: Uncalibrated scores.
            y: Targets.
            w: Sample weights.

        Returns:
            n_knots: Selected knot count.
            alpha: Selected roughness penalty.
        """
        from sklearn.model_selection import KFold, StratifiedKFold

        # Selection only needs to rank candidates, so bound its cost. The winning
        # configuration is refit on the full sample by the caller.
        if self.max_cv_samples is not None and y.size > self.max_cv_samples:
            rng = np.random.default_rng(self.random_state)
            keep = rng.choice(y.size, size=self.max_cv_samples, replace=False)
            X, y, w = X[keep], y[keep], w[keep]

        # A quantile-knot basis needs enough distinct scores to place its knots.
        n_unique = int(np.unique(X).size)
        knot_grid = [k for k in (5, 10, 20) if k <= max(3, n_unique - 1)] or [3]
        alpha_grid = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

        is_binary = bool(np.all((y == 0.0) | (y == 1.0)))
        n_splits = self.cv
        if is_binary:
            counts = np.bincount(y.astype(int), minlength=2)
            n_splits = min(n_splits, int(counts[counts > 0].min()))
        n_splits = int(max(2, min(n_splits, y.size)))

        if is_binary:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
            splits = list(splitter.split(X.reshape(-1, 1), y))
        else:
            splits = list(
                KFold(
                    n_splits=n_splits, shuffle=True, random_state=self.random_state
                ).split(X.reshape(-1, 1))
            )

        best_score, best_knots, best_alpha = np.inf, knot_grid[0], alpha_grid[0]
        for n_knots in knot_grid:
            for alpha in alpha_grid:
                total = 0.0
                weight_total = 0.0
                failed = False
                for train_idx, val_idx in splits:
                    try:
                        basis = monotone_spline_basis(
                            n_knots=n_knots, degree=self.degree, knots=self.knots
                        ).fit(X[train_idx])
                        intercept, coef = fit_monotone_spline(
                            basis.design(X[train_idx]),
                            y[train_idx],
                            sample_weight=w[train_idx],
                            alpha=alpha,
                            link=self.link,
                        )
                        pred = self._predict_from(
                            basis, intercept, coef, X[val_idx], clip=True
                        )
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        logger.debug(
                            "fold failed at n_knots=%s alpha=%s: %s",
                            n_knots,
                            alpha,
                            exc,
                        )
                        failed = True
                        break
                    total += float(np.sum(w[val_idx] * _log_loss(y[val_idx], pred)))
                    weight_total += float(np.sum(w[val_idx]))

                if failed or weight_total <= 0:
                    continue
                score = total / weight_total
                if score < best_score:
                    best_score, best_knots, best_alpha = score, n_knots, alpha

        return best_knots, best_alpha

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

        Raises:
            AttributeError: If called before :meth:`fit`.
        """
        if not hasattr(self, "basis_"):
            raise AttributeError(
                f"{type(self).__name__} is not fitted yet. Call fit() first."
            )
        return self._predict_from(
            self.basis_,
            self.intercept_,
            self.coef_,
            np.asarray(X, dtype=float).ravel(),
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
        x_unique, _, _ = aggregate_ties(self._fit_data_X, self._fit_data_y)
        grid = np.linspace(float(x_unique[0]), float(x_unique[-1]), n_points)
        return PiecewiseLinear(grid, self.transform(grid))
