"""Centered isotonic regression (CIR).

Isotonic regression's fitted curve is piecewise constant, so a flat block spreads
one pooled rate across a whole interval of scores. Every score inside the block
is mapped to the same probability, which discards the ranking information the
base model provided there.

CIR keeps the pooled estimate but stops pretending it applies uniformly across
the block: it places the estimate at the block's weighted-centroid score and
interpolates linearly between consecutive centroids. The result is strictly
increasing except possibly at the boundaries.

Reference
---------
Oron & Flournoy (2017), "Centered Isotonic Regression: Point and Interval
Estimation for Dose-Response Studies", *Statistics in Biopharmaceutical
Research* 9(3), 258-267. Pinned against that paper's R implementation
(``cir::cirPAVA``) in ``tests/test_r_reference.py``.
"""

from __future__ import annotations

import numpy as np

from .._core import (
    PiecewiseLinear,
    aggregate_ties,
    collapse_blocks,
    weighted_pava,
)
from ..base import BaseCalibrator
from ..utils import check_array_1d, check_arrays, check_fitted

__all__ = ["CenteredIsotonicCalibrator"]


class CenteredIsotonicCalibrator(BaseCalibrator):
    """Centered isotonic regression for granularity-preserving calibration.

    Runs weighted PAVA, collapses each flat block to its weighted-centroid
    predictor value, and interpolates linearly between those points. Compared
    with standard isotonic regression this preserves the score ordering inside
    what would otherwise be a plateau, at no cost in monotonicity.

    Args:
        clip_output: Clip calibrated values into ``[0, 1]``. Appropriate for
            probability calibration; turn it off to use the estimator on an
            unbounded target.
        enable_diagnostics: Run plateau diagnostics after fitting.

    Attributes:
        calibration_curve_: The fitted calibration map. ``.x`` holds the block
            centroids and ``.y`` their pooled values.
        n_features_in_: Always 1. Present for scikit-learn compatibility.

    Notes:
        Standard isotonic regression is the L2 projection onto the monotone cone and
        is optimal for that objective; CIR is not a minimizer of the same criterion.
        The justification is inferential rather than variational: within a flat block
        the data support a single pooled rate, and linear interpolation between
        consecutive pooled estimates is the minimal assumption that neither invents
        structure nor throws away the ordering. Oron & Flournoy report substantially
        lower estimation error than isotonic regression when monotonicity violations
        are present at sample sizes typical of dose-response studies.

        Extrapolation holds the end values constant, matching the convention of
        ``sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")``.

    Examples:
        >>> import numpy as np
        >>> from calibre import CenteredIsotonicCalibrator
        >>>
        >>> x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> y = np.array([0, 0, 1, 0, 1])
        >>>
        >>> cal = CenteredIsotonicCalibrator()
        >>> cal.fit(x, y)
        CenteredIsotonicCalibrator()

        PAVA gives ``[0, 0, 0.5, 0.5, 1]``, i.e. blocks ``{0.1,0.2} -> 0``,
        ``{0.3,0.4} -> 0.5``, ``{0.5} -> 1``. The interior block collapses to its
        centroid 0.35; the leading block anchors at its inner edge 0.2, so the curve
        stays flat to the left of it:

        >>> cal.calibration_curve_.x
        array([0.2 , 0.35, 0.5 ])
        >>> cal.transform(np.array([0.15, 0.35]))
        array([0. , 0.5])

    See Also:
        IsotonicCalibrator : The piecewise-constant fit CIR is derived from.
    """

    def __init__(
        self,
        *,
        clip_output: bool = True,
        enable_diagnostics: bool = False,
    ) -> None:
        super().__init__(enable_diagnostics=enable_diagnostics)
        self.clip_output = clip_output

    def _fit_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit the centered isotonic calibration map.

        Args:
            X: Uncalibrated scores.
            y: Targets: binary labels, or probabilities in ``[0, 1]``.
            sample_weight: Non-negative per-observation weights.

        Notes:
            All of the work happens here so that ``transform`` is a pure lookup.
        """
        X, y = check_arrays(X, y)

        # Pool tied scores first: the estimator is defined on an ordered
        # sequence, and an interpolant cannot be built on repeated abscissae.
        x_unique, y_mean, weight = aggregate_ties(X, y, sample_weight)

        fitted = weighted_pava(y_mean, weight)
        x_centroid, y_centroid = collapse_blocks(x_unique, fitted, weight)

        if self.clip_output:
            y_centroid = np.clip(y_centroid, 0.0, 1.0)

        self.calibration_curve_ = PiecewiseLinear(x_centroid, y_centroid)
        self.n_features_in_ = 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map scores through the fitted calibration curve.

        Args:
            X: Scores to calibrate.

        Returns:
            ndarray of shape (n_samples,): Calibrated probabilities.

        """
        check_fitted(self, ["calibration_curve_"])
        return self.calibration_curve_(check_array_1d(X))
