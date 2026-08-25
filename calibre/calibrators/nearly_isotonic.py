"""Nearly-isotonic regression for flexible monotonic calibration.

This module provides nearly-isotonic regression, which relaxes the strict
monotonicity constraint by penalizing rather than prohibiting violations.
"""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np

from .._core import PiecewiseLinear, aggregate_ties, nearly_isotonic_path
from ..base import BaseCalibrator
from ..utils import check_arrays

logger = logging.getLogger(__name__)

# cvxpy ships no type information, so `cp.error.SolverError` cannot be resolved
# statically. Bind it once here.
_SolverError: type[Exception] = getattr(
    getattr(cp, "error", None), "SolverError", Exception
)


class NearlyIsotonicCalibrator(BaseCalibrator):
    r"""Nearly-isotonic regression for flexible monotonic calibration.

    This calibrator implements nearly-isotonic regression, which relaxes the
    strict monotonicity constraint of standard isotonic regression by penalizing
    rather than prohibiting violations. This allows for a more flexible fit
    while still maintaining a generally monotonic trend.

    Args:
        lam: Regularization parameter controlling the strength of monotonicity
            constraint. Higher values enforce stricter monotonicity.
        method: Solver for the optimization problem. Both are exact and agree
            to solver tolerance; ``path`` is the faster and needs no CVXPY.

            - ``'path'``: the exact solution path (O(n log n)).
            - ``'cvx'``: convex optimization via CVXPY.
        cv: Number of cross-validation folds used when a hyperparameter is
            left at ``"auto"``. Ignored when every hyperparameter is pinned.
        scoring: Proper scoring rule the ``"auto"`` search minimises.
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
            \min_{\beta} \sum_{i=1}^{n} (y_i - \beta_i)^2
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

        **This is not the calibrator to reach for if you want granularity.** The
        granularity above is real but it is not free, and here it is not even
        cheap. Because the objective fits one value per observation to the *labels*,
        a small ``lam`` returns something close to the raw 0/1 labels: lots of
        distinct values, all of them overfitted. Measured out of sample on a
        logit-inflated design at n=3000, ``lam=0.001`` keeps 1074 distinct values at
        a held-out Brier of 0.191 against isotonic's 0.116, and every step up the
        ``lam`` grid buys score back by giving granularity away until, by
        ``lam=100``, the fit *is* isotonic regression.

        That frontier is dominated. On the same data
        :class:`~calibre.CenteredIsotonicCalibrator` keeps 2647 distinct values at a
        held-out Brier of 0.1159 -- more granularity than any ``lam`` reaches, at a
        *better* score than isotonic. So there is no default ``lam`` worth moving to,
        and none is claimed: unlike
        :class:`~calibre.RelaxedPAVACalibrator`, whose default now breaks plateaus
        apart at a cost in the fifth decimal, this estimator's defaults leave it
        close to isotonic on purpose. Use it when you want *bounded monotonicity
        violations* -- the thing it uniquely provides -- and use CIR or the spline
        calibrators when you want resolution.

        **Scaling differs from the source paper.** Tibshirani, Hoefling & Tibshirani
        (2011, *Technometrics* 53(1), 54-61) put a factor of 1/2 on the squared-error
        term:

        .. math::
            \min_{\beta} \tfrac{1}{2} \sum_i (y_i - \beta_i)^2
            + \lambda_{\text{paper}} \sum_i \max(0, \beta_i - \beta_{i+1})

        The objective above omits it, so ``lam`` here is *twice* the paper's
        :math:`\lambda`:

        .. math:: \lambda_{\text{here}} = 2\,\lambda_{\text{paper}}

        Double any penalty value taken from the paper before passing it in. Both
        solvers are pinned against the authors' R implementation (``neariso``) in
        ``tests/test_r_reference.py``.

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
        RegularizedIsotonicCalibrator : L2 regularization with strict monotonicity
    """

    #: Candidate lambdas searched when ``lam="auto"``. Spans "essentially the raw
    #: data" to "essentially isotonic", logarithmically.
    LAM_GRID = (0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0)

    def __init__(
        self,
        lam: float | str = "auto",
        method: str = "path",
        cv: int = 5,
        scoring: str = "auto",
        random_state: int | None = 0,
        clip_output: bool = True,
        enable_diagnostics: bool = False,
    ):
        # Call base class for diagnostic support
        super().__init__(enable_diagnostics=enable_diagnostics)

        self.lam = lam
        self.method = method
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
            sample_weight: Must be None. This calibrator is not weight-aware, and
                silently discarding weights would quietly return a different
                estimator than the caller asked for, so they are rejected.

        Notes:
            This method implements the actual fitting logic. Data storage,
            diagnostics, and return value are handled by the base class fit() method.
        """
        self._reject_sample_weight(sample_weight)
        X, y = check_arrays(X, y)
        if self.method not in ("path", "cvx"):
            raise ValueError(f"method must be 'path' or 'cvx', got {self.method!r}")
        self.lam_ = self._resolve_lam(X, y)
        self.X_ = X
        self.y_ = y

        # Pool tied scores first. Without this the objective double-counts tied
        # observations as independent, and the interpolant would be built on
        # repeated abscissae, where the surviving point depends on the sort's
        # tie-breaking.
        x_unique, y_mean, weight = aggregate_ties(X, y)

        if self.method == "path":
            beta = np.asarray(
                nearly_isotonic_path(y_mean, lam=self.lam_, sample_weight=weight)
            )
        else:
            beta = self._solve_cvx(y_mean, weight)

        if self.clip_output:
            beta = np.clip(beta, 0.0, 1.0)

        self.calibration_curve_ = PiecewiseLinear(x_unique, beta)
        self.n_features_in_ = 1

    def _resolve_lam(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the lambda to fit with, selecting it if asked.

        Args:
            X: Uncalibrated scores.
            y: Targets.

        Returns:
            float: The penalty to use. Written to ``lam_``; the ``lam``
                constructor argument is never modified, so ``get_params``
                round trips.

        """
        from ..selection import resolve_auto

        return resolve_auto(
            self.lam,
            "lam",
            self.LAM_GRID,
            lambda **kw: type(self)(
                method=self.method, clip_output=self.clip_output, **kw
            ),
            X,
            y,
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map scores through the fitted calibration curve.

        Args:
            X: The values to be calibrated.

        Returns:
            X_calibrated: Calibrated values.

        Raises:
            AttributeError: If called before :meth:`fit`.
        """
        if not hasattr(self, "calibration_curve_"):
            raise AttributeError(
                f"{type(self).__name__} is not fitted yet. Call fit() first."
            )
        return self.calibration_curve_(np.asarray(X, dtype=float).ravel())

    def _solve_cvx(self, y_mean: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Solve the nearly-isotonic problem with CVXPY on the pooled grid.

        Args:
            y_mean: Weighted mean target at each unique score.
            weight: Total weight at each unique score.

        Returns:
            ndarray of shape (n_unique,): The optimal fitted values.
        """
        beta = cp.Variable(len(y_mean))

        # Penalty for non-monotonicity: sum of positive parts of decreases.
        monotonicity_penalty = cp.sum(cp.maximum(0, beta[:-1] - beta[1:]))

        # Weighted squared error, so pooled ties carry their original mass.
        obj = cp.Minimize(
            cp.sum(cp.multiply(weight, cp.square(beta - y_mean)))
            + self.lam_ * monotonicity_penalty
        )
        prob = cp.Problem(obj)

        try:
            # OSQP with polishing is what matches R's neariso to ~1e-16 on this
            # hinge objective; CLARABEL returns optimal_inaccurate here.
            prob.solve(solver=cp.OSQP, polishing=True)
            if (
                prob.status in ("optimal", "optimal_inaccurate")
                and beta.value is not None
            ):
                return np.asarray(beta.value, dtype=float)
            logger.warning(
                "Nearly-isotonic solve did not converge (status=%s); falling back "
                "to the exact path algorithm",
                prob.status,
            )
        except _SolverError as exc:
            logger.warning(
                "Nearly-isotonic solve failed (%s); falling back to the exact "
                "path algorithm",
                exc,
            )

        # The path solver computes the same estimator exactly, so it is a strictly
        # better fallback than switching to a different estimator entirely.
        return np.asarray(
            nearly_isotonic_path(y_mean, lam=self.lam_, sample_weight=weight)
        )
