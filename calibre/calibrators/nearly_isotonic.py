"""Nearly-isotonic regression for flexible monotonic calibration.

This module provides nearly-isotonic regression, which relaxes the strict
monotonicity constraint by penalizing rather than prohibiting violations.
"""

from __future__ import annotations

import numpy as np

from .._core import (
    PiecewiseLinear,
    aggregate_ties,
    nearly_isotonic_path,
    weighted_pava,
)
from ..base import BaseCalibrator
from ..utils import check_array_1d, check_arrays, check_fitted


class _MassScaledNearlyIsotonic:
    """Fit a CV candidate at the full sample's penalty per unit weight."""

    def __init__(self, lam: float, full_mass: float, clip_output: bool):
        """Store the full-sample candidate and mass.

        Args:
            lam: Absolute lambda candidate for the final full-data fit.
            full_mass: Total observation weight in the full calibration sample.
            clip_output: Whether the candidate clips outputs to ``[0, 1]``.
        """
        self.lam = lam
        self.full_mass = full_mass
        self.clip_output = clip_output

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> _MassScaledNearlyIsotonic:
        """Fit the candidate after converting lambda to the fold's mass.

        Args:
            X: Training-fold scores.
            y: Training-fold targets.
            sample_weight: Non-negative per-observation weights.

        Returns:
            self: The fitted adapter.
        """
        fold_mass = float(len(y) if sample_weight is None else np.sum(sample_weight))
        fold_lam = self.lam * fold_mass / self.full_mass
        self.model_ = NearlyIsotonicCalibrator(
            lam=fold_lam,
            clip_output=self.clip_output,
        ).fit(X, y, sample_weight=sample_weight)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform scores with the fitted fold model.

        Args:
            X: Scores to transform.

        Returns:
            ndarray: Calibrated scores.
        """
        return self.model_.transform(X)


class NearlyIsotonicCalibrator(BaseCalibrator):
    r"""Nearly-isotonic regression for flexible monotonic calibration.

    This calibrator implements nearly-isotonic regression, which relaxes the
    strict monotonicity constraint of standard isotonic regression by penalizing
    rather than prohibiting violations. This allows for a more flexible fit
    while still maintaining a generally monotonic trend.

    Args:
        lam: Regularization parameter controlling the strength of monotonicity
            constraint. Higher values enforce stricter monotonicity.
        cv: Number of cross-validation folds used when a hyperparameter is
            left at ``"auto"``. Ignored when every hyperparameter is pinned.
        scoring: Proper scoring rule the ``"auto"`` search minimizes.
            ``"auto"`` (the default) uses log loss for probability targets and
            squared error otherwise.
            Deliberately not a calibration error: ECE and its relatives are
            minimised by a constant forecast, so selecting on one would reward
            throwing resolution away.
        random_state: Seed for the cross-validation split, so an ``"auto"``
            selection is reproducible.
        clip_output: Clip calibrated values into ``[0, 1]``. Appropriate for
            probability calibration; turn it off to recover the unconstrained
            optimum of the objective above, which is what the estimator is
            actually defined as.
        enable_diagnostics: Whether to enable plateau diagnostics analysis.


    Notes:
        Nearly-isotonic regression solves the following optimization problem:

        .. math::
            \min_{\beta} \tfrac{1}{2}\sum_{i=1}^{n} w_i(y_i - \beta_i)^2
            + \lambda \sum_{i=1}^{n-1} \max(0, \beta_i - \beta_{i+1})

        where :math:`\beta` is the calibrated output, :math:`y` are the true labels,
        and :math:`\lambda > 0` controls the strength of the monotonicity penalty.

        This formulation penalizes violations of monotonicity proportionally to their
        magnitude, allowing small violations when they significantly improve the fit.

        **Interpreting lam.** Read it as a bias-variance knob on pooling rather than
        as permission for non-monotone structure: ``lam = 0`` returns the data
        untouched, ``lam -> inf`` returns the isotonic fit, and intermediate values
        give shorter plateaus than isotonic regression -- finer granularity -- in
        exchange for bounded violations.

        **This is not the calibrator to reach for if you only want granularity.**
        Because the objective fits one value per observation to the labels, a small
        ``lam`` approaches the raw outcomes and overfits. Increasing ``lam`` buys
        proper-score performance back by pooling more values. In the committed
        ``overconfident`` benchmark, the automatically selected fit retains 54
        distinct values with a held-out Brier score of 0.1532; centered isotonic
        regression retains 1,514 at 0.1527. Use nearly-isotonic regression when
        bounded monotonicity violations are the feature you need. Use centered
        isotonic regression or the spline calibrator when you need resolution.

        The objective, lambda scale, and modified PAVA path algorithm follow
        Tibshirani, Hoefling & Tibshirani (2011, *Technometrics* 53(1), 54-61).
        Cross-language fixtures in ``tests/test_r_reference.py`` compare the
        implementation with the authors' R package, ``neariso``.

        Automatic selection keeps :math:`\lambda / \sum_i w_i` fixed between
        each training fold and the final full-data fit. Numeric ``lam`` values
        remain absolute penalties on the paper's scale.

    Examples:
        >>> import numpy as np
        >>> from calibre import NearlyIsotonicCalibrator
        >>>
        >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> y = np.array([0.12, 0.18, 0.35, 0.25, 0.55])
        >>>
        >>> cal = NearlyIsotonicCalibrator(lam=0.5)
        >>> _ = cal.fit(X, y)
        >>> X_calibrated = cal.transform(np.array([0.15, 0.35, 0.55]))

    See Also:
        IsotonicCalibrator : Strict monotonicity constraint
        SplineCalibrator : Penalized monotone spline calibration.
    """

    #: Number of data-scaled penalties searched when ``lam="auto"``. The upper
    #: endpoint is the smallest penalty that reaches the isotonic solution.
    N_AUTO_LAMBDAS = 9

    def __init__(
        self,
        lam: float | str = "auto",
        cv: int = 5,
        scoring: str = "auto",
        random_state: int | None = 0,
        clip_output: bool = True,
        enable_diagnostics: bool = False,
    ):
        # Call base class for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.lam = lam
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.clip_output = clip_output

    def _fit_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Implement the nearly-isotonic regression fitting logic.

        Args:
            X: The training input samples.
            y: The target values.
            sample_weight: Non-negative per-observation weights.

        Notes:
            This method implements the actual fitting logic. Data storage,
            diagnostics, and return value are handled by the base class fit() method.
        """
        X, y = check_arrays(X, y)

        # Pool tied scores first. Without this the objective double-counts tied
        # observations as independent, and the interpolant would be built on
        # repeated abscissae, where the surviving point depends on the sort's
        # tie-breaking.
        x_unique, y_mean, weight = aggregate_ties(X, y, sample_weight)
        self.lam_ = self._resolve_lam(X, y, y_mean, weight, sample_weight)
        beta = np.asarray(
            nearly_isotonic_path(y_mean, lam=self.lam_, sample_weight=weight)
        )

        if self.clip_output:
            beta = np.clip(beta, 0.0, 1.0)

        self.calibration_curve_ = PiecewiseLinear(x_unique, beta)
        self.n_features_in_ = 1

    def _resolve_lam(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_mean: np.ndarray,
        weight: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> float:
        """Return the lambda to fit with, selecting it if asked.

        Args:
            X: Uncalibrated scores.
            y: Targets.
            y_mean: Weighted target means on the unique-score grid.
            weight: Total weight at each unique score.
            sample_weight: Per-observation weights passed to :meth:`fit`.

        Returns:
            float: The penalty to use. Written to ``lam_``; the ``lam``
                constructor argument is never modified, so ``get_params``
                round trips.

        """
        from ..selection import resolve_auto

        isotonic = weighted_pava(y_mean, weight)
        cumulative_residual = np.cumsum(weight * (y_mean - isotonic))[:-1]
        lam_max = max(0.0, float(np.max(cumulative_residual, initial=0.0)))
        grid = np.linspace(0.0, lam_max, self.N_AUTO_LAMBDAS).tolist()

        return resolve_auto(
            self.lam,
            "lam",
            grid,
            lambda **kw: _MassScaledNearlyIsotonic(
                lam=float(kw["lam"]),
                full_mass=float(np.sum(weight)),
                clip_output=self.clip_output,
            ),
            X,
            y,
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            sample_weight=sample_weight,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map scores through the fitted calibration curve.

        Args:
            X: The values to be calibrated.

        Returns:
            X_calibrated: Calibrated values.

        """
        check_fitted(self, ["calibration_curve_"])
        return self.calibration_curve_(check_array_1d(X))
