"""Shared numerical primitives for calibration.

Every calibrator in the package is built from the routines here rather than
reimplementing isotonic machinery, interpolation, or tie handling locally.

The primitives are pinned against R reference implementations by
``tests/test_r_reference.py``:

======================  ==================================
``weighted_pava``       ``isotone::gpava``, ``Iso::pava``
``aggregate_ties``      tie pooling used by all of the above
``shift_to_pava``       reduces to ``gpava`` when ``L == 0``
======================  ==================================
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "aggregate_ties",
    "collapse_blocks",
    "cumulative_max",
    "fit_monotone_spline",
    "monotone_projection",
    "monotone_spline_basis",
    "nearly_isotonic_path",
    "shift_to_pava",
    "weighted_pava",
    "MonotoneSplineBasis",
    "PiecewiseLinear",
    "StepFunction",
]


# --------------------------------------------------------------------------- #
# Tie handling
# --------------------------------------------------------------------------- #


def aggregate_ties(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool observations sharing an ``x`` value into one weighted point.

    Isotonic-family estimators are defined on an ordered sequence, so tied
    predictor values must be collapsed to a single point carrying the summed
    weight. Skipping this step is not a cosmetic issue: an interpolant built on
    duplicated abscissae silently discards all but one of the tied points, and
    which one survives depends on the sort's tie-breaking, making the result
    nondeterministic.

    Parameters
    ----------
    x
        Predictor values. Need not be sorted.
    y
        Target values.
    sample_weight
        Non-negative per-observation weights. Defaults to 1.

    Returns
    -------
    x_unique : ndarray of shape (n_unique,)
        Sorted unique values of ``x``.
    y_mean : ndarray of shape (n_unique,)
        Weighted mean of ``y`` within each tie group. Groups with zero total
        weight are reported as 0.
    weight : ndarray of shape (n_unique,)
        Total weight of each tie group.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([2.0, 1.0, 1.0, 2.0])
    >>> y = np.array([1.0, 0.0, 1.0, 0.0])
    >>> aggregate_ties(x, y)
    (array([1., 2.]), array([0.5, 0.5]), array([2., 2.]))
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    w = (
        np.ones_like(y)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).ravel()
    )

    x_unique, inverse = np.unique(x, return_inverse=True)
    n_groups = x_unique.size

    weight = np.bincount(inverse, weights=w, minlength=n_groups)
    weighted_y = np.bincount(inverse, weights=y * w, minlength=n_groups)
    y_mean = np.divide(
        weighted_y, weight, out=np.zeros_like(weighted_y), where=weight > 0.0
    )
    return x_unique, y_mean, weight


# --------------------------------------------------------------------------- #
# Isotonic regression
# --------------------------------------------------------------------------- #


def weighted_pava(y: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    """Weighted pool-adjacent-violators: the L2 projection onto the monotone cone.

    Solves, in O(n) time,

    .. math::
        \\min_{z_1 \\le \\cdots \\le z_n} \\sum_i w_i (y_i - z_i)^2

    by the classical stack algorithm: push each point as its own block, then
    merge backwards into weighted means while the last two blocks violate the
    ordering.

    Parameters
    ----------
    y
        Target values, in the order the monotonicity constraint applies to.
    sample_weight
        Non-negative weights. Defaults to 1.

    Returns
    -------
    ndarray of shape (n_samples,)
        Fitted values, non-decreasing, one per input element.

    Raises
    ------
    ValueError
        If ``y`` is empty, shapes disagree, or any weight is negative.

    Examples
    --------
    >>> import numpy as np
    >>> weighted_pava(np.array([1.0, 0.0]))
    array([0.5, 0.5])
    >>> weighted_pava(np.array([1.0, 0.0]), np.array([3.0, 1.0]))
    array([0.75, 0.75])
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size == 0:
        raise ValueError("y must not be empty")
    w = (
        np.ones_like(y)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).ravel()
    )
    if w.shape != y.shape:
        raise ValueError(f"sample_weight shape {w.shape} does not match y {y.shape}")
    if np.any(w < 0.0):
        raise ValueError("sample_weight must be non-negative")

    # Block stacks. means[k] is non-decreasing in k by construction.
    means = np.empty(y.size, dtype=float)
    weights = np.empty(y.size, dtype=float)
    counts = np.empty(y.size, dtype=np.intp)
    top = 0

    for i in range(y.size):
        means[top] = y[i]
        weights[top] = w[i]
        counts[top] = 1
        top += 1

        while top > 1 and means[top - 2] > means[top - 1]:
            w_merged = weights[top - 2] + weights[top - 1]
            if w_merged > 0.0:
                means[top - 2] = (
                    weights[top - 2] * means[top - 2]
                    + weights[top - 1] * means[top - 1]
                ) / w_merged
            else:
                # Both blocks carry zero weight: the objective is indifferent, so
                # take the unweighted mean to keep the result well defined.
                means[top - 2] = 0.5 * (means[top - 2] + means[top - 1])
            weights[top - 2] = w_merged
            counts[top - 2] += counts[top - 1]
            top -= 1

    return np.repeat(means[:top], counts[:top])


def monotone_projection(
    y: np.ndarray, sample_weight: np.ndarray | None = None
) -> np.ndarray:
    """The L2 projection of ``y`` onto the non-decreasing cone.

    An alias for :func:`weighted_pava`, named for the role it plays when a
    smoother's output needs to be made monotone. Prefer this over
    :func:`cumulative_max`, which is not a projection.

    Parameters
    ----------
    y
        Values to project.
    sample_weight
        Non-negative weights. Defaults to 1.

    Returns
    -------
    ndarray of shape (n_samples,)
        The closest non-decreasing sequence in weighted L2.
    """
    return weighted_pava(y, sample_weight)


def cumulative_max(y: np.ndarray) -> np.ndarray:
    """Enforce monotonicity by a running maximum.

    This is *not* the L2 projection. It only ever raises values, so it biases
    the result upward relative to :func:`monotone_projection`. It is retained
    because it preserves the original value at every point that is already in
    order, which is occasionally what a caller wants; when in doubt use
    :func:`monotone_projection`.

    Parameters
    ----------
    y
        Values to make non-decreasing.

    Returns
    -------
    ndarray of shape (n_samples,)
        ``out[i] = max(y[0], ..., y[i])``.

    Examples
    --------
    >>> import numpy as np
    >>> cumulative_max(np.array([0.1, 0.3, 0.2, 0.5, 0.4]))
    array([0.1, 0.3, 0.3, 0.5, 0.5])
    """
    return np.asarray(np.maximum.accumulate(np.asarray(y, dtype=float).ravel()))


def shift_to_pava(
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    L: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Isotonic regression with per-adjacency lower bounds on the increments.

    Solves

    .. math::
        \\min_{z} \\sum_i w_i (y_i - z_i)^2
        \\quad\\text{s.t.}\\quad z_{i+1} - z_i \\ge L_i

    in O(n) by a change of variables. With :math:`R_i = \\sum_{j<i} L_j` and
    :math:`u_i = z_i - R_i`, the constraint becomes

    .. math::
        u_{i+1} - u_i = (z_{i+1} - R_{i+1}) - (z_i - R_i) = z_{i+1} - z_i - L_i \\ge 0

    so the problem is a plain weighted PAVA on the shifted targets
    :math:`y_i - R_i`, after which the shift is added back.

    One parameter spans three estimators:

    - ``L = 0``   standard isotonic regression
    - ``L < 0``   epsilon-monotone: bounded decreases permitted
    - ``L > 0``   minimum slope: strictly increasing, so no plateaus at all

    Parameters
    ----------
    y
        Target values in constraint order.
    sample_weight
        Non-negative weights. Defaults to 1.
    L
        Lower bound on each of the ``n - 1`` increments. A scalar is broadcast.

    Returns
    -------
    ndarray of shape (n_samples,)
        Fitted values satisfying the increment constraints.

    Raises
    ------
    ValueError
        If ``L`` has the wrong length.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0.0, 1.0, 1.0, 0.0])
    >>> shift_to_pava(y)                                # standard isotonic
    array([0.        , 0.66666667, 0.66666667, 0.66666667])
    >>> bool(np.all(np.diff(shift_to_pava(y, L=0.05)) >= 0.05 - 1e-12))
    True
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n == 0:
        raise ValueError("y must not be empty")

    L_arr = np.broadcast_to(np.asarray(L, dtype=float), (max(n - 1, 0),)).astype(
        float, copy=False
    )
    if L_arr.size != n - 1:
        raise ValueError(f"L must have length {n - 1}, got {L_arr.size}")

    shift = np.zeros(n, dtype=float)
    if n > 1:
        np.cumsum(L_arr, out=shift[1:])

    return np.asarray(weighted_pava(y - shift, sample_weight) + shift, dtype=float)


def nearly_isotonic_path(
    y: np.ndarray,
    lam: float,
    sample_weight: np.ndarray | None = None,
    return_path: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[tuple[float, int]]]:
    """Nearly-isotonic regression by its exact solution path.

    Solves

    .. math::
        \\min_{\\beta} \\sum_i w_i (y_i - \\beta_i)^2
        + \\lambda \\sum_{i=1}^{n-1} \\max(0,\\ \\beta_i - \\beta_{i+1})

    exactly, in O(n log n), by following the solution path from
    :math:`\\lambda = 0` to the requested value.

    Monotonicity is penalised rather than imposed, so ``lam`` acts as a
    bias-variance knob on pooling: ``lam = 0`` returns the data untouched,
    ``lam -> inf`` returns the isotonic fit, and intermediate values give shorter
    plateaus than isotonic regression at the cost of bounded violations.

    Parameters
    ----------
    y
        Target values in constraint order.
    lam
        Penalty on monotonicity violations. See Notes on scaling.
    sample_weight
        Non-negative weights. Defaults to 1.
    return_path
        Also return the merge events as ``(lambda, n_blocks)`` pairs. The block
        count is an unbiased estimate of the fit's degrees of freedom, which is
        what makes ``lam`` selectable rather than guessed.

    Returns
    -------
    beta : ndarray of shape (n_samples,)
        The exact minimiser at ``lam``.
    path : list of (float, int), optional
        Returned when ``return_path`` is set.

    Raises
    ------
    ValueError
        If ``y`` is empty or ``lam`` is negative.

    Notes
    -----
    **Scaling.** The reference formulation (Tibshirani, Hoefling & Tibshirani
    2011, *Technometrics* 53(1), 54-61) carries a factor of one half on the
    squared-error term. The objective above does not, so

    .. math:: \\lambda_{\\text{here}} = 2\\,\\lambda_{\\text{paper}}

    A penalty value taken from the paper must be doubled before being passed
    here. This is verified against the authors' own R implementation
    (``neariso``) in ``tests/test_r_reference.py``.

    **How the path is computed.** Subgradient stationarity gives
    :math:`2 w_i(\\beta_i - y_i) + \\lambda(g_i - g_{i-1}) = 0`, where
    :math:`g_i \\in [0, 1]` is the subgradient of the hinge on adjacency
    :math:`i` and :math:`g_0 = g_n = 0`. Summing over a block :math:`A_k` of
    tied coordinates, the interior subgradients cancel telescopically and leave

    .. math::
        \\beta_k(\\lambda) = \\bar{y}_k
        - \\frac{\\lambda}{2}\\cdot\\frac{G_k - G_{k-1}}{W_k}

    with :math:`\\bar y_k` the block's weighted mean, :math:`W_k` its total
    weight, and :math:`G_k \\in \\{0, 1\\}` indicating whether the boundary
    between blocks :math:`k` and :math:`k+1` is violated. So each block's value
    is *linear in* :math:`\\lambda` with slope
    :math:`s_k = -(G_k - G_{k-1}) / (2 W_k)`, and two adjacent blocks meet after

    .. math::
        \\Delta_k = \\frac{\\beta_{k+1} - \\beta_k}{s_k - s_{k+1}}

    The two ingredients that make this correct are exactly the ones a naive
    implementation drops: the collision time is a gap divided by a *difference of
    slopes* (so block sizes govern the merge order), and block values *drift with*
    :math:`\\lambda` between merges instead of sitting still. Using the raw gap
    :math:`\\beta_k - \\beta_{k+1}` as the collision time, and freezing values
    between merges, yields a different -- and suboptimal -- estimator.

    Blocks never split as :math:`\\lambda` grows, so merges are permanent.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0.1, 0.4, 0.2, 0.5])
    >>> nearly_isotonic_path(y, lam=0.0)
    array([0.1, 0.4, 0.2, 0.5])
    >>> beta = nearly_isotonic_path(y, lam=1e6)          # -> isotonic
    >>> np.allclose(beta, weighted_pava(y))
    True
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n == 0:
        raise ValueError("y must not be empty")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")

    w = (
        np.ones_like(y)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).ravel()
    )
    if w.shape != y.shape:
        raise ValueError(f"sample_weight shape {w.shape} does not match y {y.shape}")
    if np.any(w < 0.0):
        raise ValueError("sample_weight must be non-negative")

    # Block state. Blocks are consecutive runs; `size` counts original indices so
    # the result can be expanded at the end.
    beta = y.copy()  # current value of each block
    block_w = w.copy()  # total weight of each block
    size = np.ones(n, dtype=np.intp)
    n_blocks = n

    path: list[tuple[float, int]] = [(0.0, n_blocks)]
    lam_curr = 0.0

    while n_blocks > 1 and lam_curr < lam:
        b = beta[:n_blocks]
        bw = block_w[:n_blocks]

        # G_k = 1 where the boundary between blocks k and k+1 is violated.
        violated = (b[:-1] > b[1:]).astype(float)
        G = np.zeros(n_blocks + 1, dtype=float)  # G[0] = G[n_blocks] = 0
        G[1:n_blocks] = violated

        # s_k = -(G_k - G_{k-1}) / (2 W_k), indexing G so that G[k] is the
        # boundary to the right of block k-1.
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = -(G[1:] - G[:-1]) / (2.0 * bw)
        slope[~np.isfinite(slope)] = 0.0  # zero-weight blocks do not move

        # Collision time for each adjacent pair.
        # A pair meets when beta_k + s_k*d == beta_{k+1} + s_{k+1}*d. The only
        # admissible condition is d > 0; the sign of `rate` on its own says
        # nothing, since a violated pair has gap < 0 and rate < 0 and so a
        # perfectly valid positive collision time.
        gap = b[1:] - b[:-1]
        rate = slope[:-1] - slope[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = gap / rate
        delta[~np.isfinite(delta)] = np.inf
        delta[delta <= 0.0] = np.inf

        if not np.any(np.isfinite(delta)):
            # Nothing will ever collide: advance to lam and stop.
            beta[:n_blocks] = b + slope * (lam - lam_curr)
            lam_curr = lam
            break

        step = float(np.min(delta))
        if lam_curr + step >= lam:
            beta[:n_blocks] = b + slope * (lam - lam_curr)
            lam_curr = lam
            break

        # Advance to the merge point, then merge every pair colliding there.
        beta[:n_blocks] = b + slope * step
        lam_curr += step

        merge_at = np.flatnonzero(delta <= step * (1.0 + 1e-12) + 1e-15)
        keep = np.ones(n_blocks, dtype=bool)
        for k in merge_at[::-1]:
            # Fold block k+1 into block k.
            block_w[k] += block_w[k + 1]
            size[k] += size[k + 1]
            keep[k + 1] = False

        idx = np.flatnonzero(keep)
        n_new = idx.size
        block_w[:n_new] = block_w[idx]
        size[:n_new] = size[idx]
        # Keep the value the merging blocks met at. The solution path is
        # continuous in lambda, so a merged block does NOT jump to the weighted
        # mean of its members: that mean is only its intercept at lambda = 0, and
        # the block has already drifted away from it by -(lambda/2)*dG/W.
        beta[:n_new] = beta[idx]
        n_blocks = n_new
        path.append((lam_curr, n_blocks))

    result = np.repeat(beta[:n_blocks], size[:n_blocks])
    if return_path:
        return result, path
    return result


def collapse_blocks(
    x: np.ndarray,
    fitted: np.ndarray,
    sample_weight: np.ndarray | None = None,
    anchor_terminal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink each flat run of an isotonic fit to a single representative point.

    This is the geometric step behind centered isotonic regression: a plateau
    carries exactly one piece of information -- a single pooled rate -- so it is
    represented at one ``x`` location rather than smeared across the whole
    interval as a step.

    Interior blocks collapse to their weighted centroid. Terminal blocks are
    treated differently when ``anchor_terminal`` is set: the first block is
    placed at its *last* ``x`` and the last block at its *first* ``x``, i.e. at
    the edge facing the rest of the data. Collapsing a terminal plateau to its
    centroid would invent a slope between the edge of the data and that centroid
    even though there is no observation beyond it to interpolate toward. This is
    why centered isotonic regression is strictly monotone in the interior but
    may stay flat near the boundaries.

    Parameters
    ----------
    x
        Strictly increasing predictor grid.
    fitted
        Non-decreasing fitted values on that grid.
    sample_weight
        Non-negative weights. Defaults to 1.
    anchor_terminal
        Anchor the first and last blocks at their inner edge instead of their
        centroid. Matches ``cir::cirPAVA``.

    Returns
    -------
    x_collapsed : ndarray of shape (n_blocks,)
        Representative predictor value for each flat block.
    y_collapsed : ndarray of shape (n_blocks,)
        The block's fitted value.

    Examples
    --------
    A single interior plateau collapses to its centroid:

    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> collapse_blocks(x, np.array([0.0, 0.5, 0.5, 1.0]))
    (array([1. , 2.5, 4. ]), array([0. , 0.5, 1. ]))

    A leading plateau anchors at its inner edge, so the curve stays flat out to
    the left boundary rather than sloping up from it:

    >>> collapse_blocks(x, np.array([0.0, 0.0, 0.5, 1.0]))
    (array([2., 3., 4.]), array([0. , 0.5, 1. ]))
    """
    x = np.asarray(x, dtype=float).ravel()
    fitted = np.asarray(fitted, dtype=float).ravel()
    w = (
        np.ones_like(fitted)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).ravel()
    )
    if x.size == 0:
        return x.copy(), fitted.copy()

    # Block boundaries wherever the fitted value changes.
    starts = np.concatenate(([0], np.flatnonzero(np.diff(fitted) != 0.0) + 1))
    ends = np.concatenate((starts[1:], [fitted.size]))
    n_blocks = starts.size

    x_collapsed = np.empty(n_blocks, dtype=float)
    y_collapsed = fitted[starts]

    for k, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
        if anchor_terminal and n_blocks > 1 and k == 0:
            x_collapsed[k] = x[hi - 1]
        elif anchor_terminal and n_blocks > 1 and k == n_blocks - 1:
            x_collapsed[k] = x[lo]
        else:
            block_w = w[lo:hi]
            total = block_w.sum()
            x_collapsed[k] = (
                np.dot(block_w, x[lo:hi]) / total if total > 0 else x[lo:hi].mean()
            )
    return x_collapsed, y_collapsed


# --------------------------------------------------------------------------- #
# Fitted-function objects
# --------------------------------------------------------------------------- #


class PiecewiseLinear:
    """A piecewise-linear function on a fixed knot grid, constant outside it.

    Built on :func:`numpy.interp` rather than ``scipy.interpolate.interp1d``:
    ``interp1d`` accepts duplicated abscissae and silently returns a value from
    whichever tied point happened to survive its internal sort.

    Parameters
    ----------
    x
        Knot locations. Must be strictly increasing.
    y
        Function value at each knot.

    Raises
    ------
    ValueError
        If the arrays are empty, disagree in length, or ``x`` is not strictly
        increasing.

    Examples
    --------
    >>> import numpy as np
    >>> f = PiecewiseLinear(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    >>> f(np.array([-1.0, 0.25, 2.0]))
    array([0.  , 0.25, 1.  ])
    """

    __slots__ = ("x", "y")

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.size == 0:
            raise ValueError("x must not be empty")
        if x.shape != y.shape:
            raise ValueError(f"x shape {x.shape} does not match y {y.shape}")
        if x.size > 1 and np.any(np.diff(x) <= 0):
            raise ValueError(
                "x must be strictly increasing; pool tied values with "
                "aggregate_ties() before constructing an interpolant"
            )
        self.x = x
        self.y = y

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        """Evaluate the function, holding the end values outside the knot range.

        Parameters
        ----------
        x_new
            Points to evaluate at.

        Returns
        -------
        ndarray
            Interpolated values.
        """
        x_new = np.asarray(x_new, dtype=float).ravel()
        if self.x.size == 1:
            return np.full(x_new.shape, self.y[0], dtype=float)
        out = np.interp(x_new, self.x, self.y, left=self.y[0], right=self.y[-1])
        return np.asarray(out, dtype=float)


class StepFunction:
    """A right-continuous step function on a fixed breakpoint grid.

    Parameters
    ----------
    x
        Breakpoints. Must be strictly increasing.
    y
        Value taken on ``[x[i], x[i+1])``.

    Raises
    ------
    ValueError
        If the arrays are empty, disagree in length, or ``x`` is not strictly
        increasing.

    Examples
    --------
    >>> import numpy as np
    >>> f = StepFunction(np.array([0.0, 1.0]), np.array([0.2, 0.8]))
    >>> f(np.array([-1.0, 0.5, 1.0, 5.0]))
    array([0.2, 0.2, 0.8, 0.8])
    """

    __slots__ = ("x", "y")

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.size == 0:
            raise ValueError("x must not be empty")
        if x.shape != y.shape:
            raise ValueError(f"x shape {x.shape} does not match y {y.shape}")
        if x.size > 1 and np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing")
        self.x = x
        self.y = y

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        """Evaluate the step function.

        Parameters
        ----------
        x_new
            Points to evaluate at.

        Returns
        -------
        ndarray
            Step values.
        """
        x_new = np.asarray(x_new, dtype=float).ravel()
        idx = np.searchsorted(self.x, x_new, side="right") - 1
        np.clip(idx, 0, self.x.size - 1, out=idx)
        return self.y[idx]


# --------------------------------------------------------------------------- #
# Monotone splines
# --------------------------------------------------------------------------- #

VALID_LINKS = ("logit", "identity")
VALID_KNOTS = ("uniform", "quantile")


class MonotoneSplineBasis:
    """An I-spline design matrix, on which non-negative coefficients are monotone.

    ``sklearn.preprocessing.SplineTransformer`` produces a **B-spline** basis. Its
    columns are bumps, so a non-negative combination of them is merely
    non-negative -- not monotone. Constraining B-spline coefficients to be
    non-decreasing *is* sufficient for a monotone spline, and that constraint
    becomes a plain non-negativity constraint after a change of variables:

    .. math::
        c_1 = \\theta,\\qquad c_j = \\theta + \\sum_{k \\le j} \\delta_k,
        \\qquad \\delta_k \\ge 0

    Substituting into :math:`f = \\sum_j c_j B_j` and collecting terms gives

    .. math::
        f = \\theta \\underbrace{\\sum_j B_j}_{=\\,1}
            + \\sum_{k\\ge 1} \\delta_k \\underbrace{\\sum_{j \\ge k} B_j}_{I_k}

    The :math:`I_k` are the **I-splines**: right-cumulative sums of the B-spline
    basis, each non-decreasing. The :math:`k = 0` term is constant by partition of
    unity, so it is dropped in favour of an explicit intercept, leaving a design
    whose every column is monotone.

    This is the same construction as the SCOP-splines of Pya & Wood (2015), used
    by R's ``scam``, and the penalised B-splines of Eilers & Marx (1996).

    Parameters
    ----------
    n_knots
        Number of knots. The basis has ``n_knots + degree - 1`` columns before the
        constant one is dropped.
    degree
        Polynomial degree of the B-splines.
    knots
        ``"quantile"`` places knots at data quantiles, ``"uniform"`` at equal
        spacing. Quantile is usually right for calibration, where scores cluster
        wherever the base model is confident.
    extrapolation
        Passed through to ``SplineTransformer``. ``"constant"`` holds the basis
        flat outside the knot range, so the fitted curve plateaus rather than
        diverging.

    Attributes
    ----------
    transformer_ : SplineTransformer
        The fitted B-spline transformer.
    n_basis_ : int
        Number of design columns, i.e. the required coefficient count.

    Notes
    -----
    Construct one of these per fit. Sharing a single instance across
    cross-validation folds is what produced the mismatch between retained knots
    and retained coefficients in earlier versions of this package.

    Examples
    --------
    >>> import numpy as np
    >>> basis = MonotoneSplineBasis(n_knots=6, degree=3)
    >>> x = np.linspace(0.0, 1.0, 200)
    >>> M = basis.fit(x).design(x)
    >>> bool(np.all(np.diff(M, axis=0) >= -1e-10))   # every column monotone
    True
    """

    __slots__ = (
        "degree",
        "extrapolation",
        "knots",
        "n_basis_",
        "n_knots",
        "transformer_",
    )

    def __init__(
        self,
        n_knots: int = 10,
        degree: int = 3,
        knots: str = "quantile",
        extrapolation: str = "constant",
    ) -> None:
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.extrapolation = extrapolation

    def fit(self, x: np.ndarray) -> MonotoneSplineBasis:
        """Place the knots from ``x``.

        Parameters
        ----------
        x
            Predictor values.

        Returns
        -------
        MonotoneSplineBasis
            self, for chaining.

        Raises
        ------
        ValueError
            If the configuration is invalid.
        """
        from sklearn.preprocessing import SplineTransformer

        if self.n_knots < 3:
            raise ValueError(f"n_knots must be at least 3, got {self.n_knots}")
        if self.degree < 1:
            raise ValueError(f"degree must be at least 1, got {self.degree}")
        if self.knots not in VALID_KNOTS:
            raise ValueError(f"knots must be one of {VALID_KNOTS}, got {self.knots!r}")

        x = np.asarray(x, dtype=float).ravel().reshape(-1, 1)
        # Quantile knots collapse if the scores are too concentrated; fall back
        # rather than emit a degenerate basis.
        knots = self.knots
        if knots == "quantile":
            qs = np.quantile(x.ravel(), np.linspace(0, 1, self.n_knots))
            if np.unique(qs).size < self.n_knots:
                knots = "uniform"

        self.transformer_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            knots=knots,
            extrapolation=self.extrapolation,
            include_bias=True,
        )
        self.transformer_.fit(x)
        self.n_basis_ = self.transformer_.transform(x[:1]).shape[1] - 1
        return self

    def design(self, x: np.ndarray) -> np.ndarray:
        """Build the monotone design matrix at ``x``.

        Parameters
        ----------
        x
            Points to evaluate the basis at.

        Returns
        -------
        ndarray of shape (n_samples, n_basis_)
            Every column non-decreasing in ``x``.

        Raises
        ------
        AttributeError
            If called before :meth:`fit`.
        """
        if not hasattr(self, "transformer_"):
            raise AttributeError("MonotoneSplineBasis is not fitted yet.")
        x = np.asarray(x, dtype=float).ravel().reshape(-1, 1)
        B = self.transformer_.transform(x)
        # I_k = sum_{j >= k} B_j, then drop the constant k = 0 column.
        ispline = np.cumsum(B[:, ::-1], axis=1)[:, ::-1]
        return np.ascontiguousarray(ispline[:, 1:])


def monotone_spline_basis(
    n_knots: int = 10,
    degree: int = 3,
    knots: str = "quantile",
    extrapolation: str = "constant",
) -> MonotoneSplineBasis:
    """Construct a fresh :class:`MonotoneSplineBasis`.

    Parameters
    ----------
    n_knots
        Number of knots.
    degree
        B-spline degree.
    knots
        ``"quantile"`` or ``"uniform"``.
    extrapolation
        ``SplineTransformer`` extrapolation mode.

    Returns
    -------
    MonotoneSplineBasis
        An unfitted basis. Always a new object, so callers cannot accidentally
        share mutable state across folds.
    """
    return MonotoneSplineBasis(
        n_knots=n_knots, degree=degree, knots=knots, extrapolation=extrapolation
    )


def _difference_matrix(p: int) -> Any:
    """First-difference operator on the increment vector.

    Because the increments are already the first differences of the B-spline
    coefficients, differencing them once more gives a **second**-order penalty on
    the coefficients. Second order is the right choice: it leaves any straight
    line unpenalised, so the identity map and the empirical base rate both survive
    shrinkage. A first-order penalty would instead pull the fit toward flat.

    Parameters
    ----------
    p
        Number of increments.

    Returns
    -------
    sparse matrix of shape (p - 1, p)
        The difference operator, or an empty operator when ``p < 2``.
    """
    from scipy import sparse

    if p < 2:
        return sparse.csr_matrix((0, p))
    return sparse.diags([-1.0, 1.0], [0, 1], shape=(p - 1, p), format="csr")


def fit_monotone_spline(
    design: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    alpha: float = 0.0,
    link: str = "logit",
) -> tuple[float, np.ndarray]:
    """Fit a monotone spline by penalised, non-negativity-constrained regression.

    Solves

    .. math::
        \\min_{\\theta,\\ \\delta \\ge 0}\\;
        \\mathcal{L}\\!\\left(\\theta + M\\delta;\\ y, w\\right)
        + \\alpha \\lVert \\Delta\\delta \\rVert^2

    where ``M`` is a monotone design (see :class:`MonotoneSplineBasis`), so
    ``delta >= 0`` makes the fit monotone by construction -- no post-hoc
    projection and no tolerance to tune.

    Parameters
    ----------
    design
        Monotone design matrix of shape ``(n_samples, n_basis)``.
    y
        Targets in ``[0, 1]``.
    sample_weight
        Non-negative weights. Defaults to 1.
    alpha
        Roughness penalty on the increments.
    link
        ``"logit"`` fits a penalised Bernoulli likelihood, so predictions live in
        ``(0, 1)`` with no clipping and the objective is the proper score for
        binary labels. ``"identity"`` fits penalised least squares on the
        probability scale, which is a single bounded linear solve and is therefore
        easy to check against an external QP solver.

    Returns
    -------
    intercept : float
        The unconstrained intercept.
    delta : ndarray of shape (n_basis,)
        Non-negative increment coefficients.

    Raises
    ------
    ValueError
        If ``link`` is unknown, ``alpha`` is negative, or shapes disagree.

    Notes
    -----
    Both objectives are convex on the feasible set ``delta >= 0``, so the returned
    solution is a global optimum: the identity case is a convex QP solved by
    bounded least squares, and the logit case is a convex penalised likelihood
    solved by L-BFGS-B.
    """
    from scipy import sparse
    from scipy.optimize import lsq_linear, minimize
    from scipy.special import expit

    M = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if M.ndim != 2:
        raise ValueError(f"design must be 2-D, got shape {M.shape}")
    if M.shape[0] != y.size:
        raise ValueError(f"design has {M.shape[0]} rows but y has {y.size}")
    if link not in VALID_LINKS:
        raise ValueError(f"link must be one of {VALID_LINKS}, got {link!r}")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    w = (
        np.ones_like(y)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).ravel()
    )
    if w.shape != y.shape:
        raise ValueError(f"sample_weight shape {w.shape} does not match y {y.shape}")
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative")

    n, p = M.shape
    D = _difference_matrix(p)
    # Intercept unbounded, increments non-negative -- that is the whole constraint.
    lower = np.concatenate(([-np.inf], np.zeros(p)))
    upper = np.full(p + 1, np.inf)

    if link == "identity":
        sw = np.sqrt(w)
        top = np.column_stack([np.ones(n), M]) * sw[:, None]
        blocks = [sparse.csr_matrix(top)]
        rhs = [sw * y]
        if alpha > 0 and D.shape[0] > 0:
            pen = sparse.hstack([sparse.csr_matrix((D.shape[0], 1)), D], format="csr")
            blocks.append(np.sqrt(alpha) * pen)
            rhs.append(np.zeros(D.shape[0]))
        A = sparse.vstack(blocks, format="csr")
        b = np.concatenate(rhs)
        result = lsq_linear(A, b, bounds=(lower, upper), lsmr_tol="auto", max_iter=500)
        z = np.asarray(result.x, dtype=float)
        return float(z[0]), z[1:]

    # Logit link: penalised Bernoulli negative log-likelihood.
    DtD = (D.T @ D) if D.shape[0] > 0 else sparse.csr_matrix((p, p))

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        theta, delta = z[0], z[1:]
        eta = theta + M @ delta
        # logaddexp(0, eta) is log(1 + exp(eta)) without overflow.
        loss = float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)))
        resid = w * (expit(eta) - y)
        grad = np.empty_like(z)
        grad[0] = float(resid.sum())
        grad[1:] = M.T @ resid
        if alpha > 0 and p > 1:
            pen_vec = DtD @ delta
            loss += alpha * float(delta @ pen_vec)
            grad[1:] += 2.0 * alpha * pen_vec
        return loss, grad

    # Start from a flat fit at the base rate; delta = 0 is feasible.
    base = float(
        np.clip(np.average(y, weights=w) if w.sum() > 0 else 0.5, 1e-6, 1 - 1e-6)
    )
    z0 = np.zeros(p + 1)
    z0[0] = float(np.log(base / (1.0 - base)))

    result = minimize(
        objective,
        z0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None)] + [(0.0, None)] * p,
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-10},
    )
    z = np.asarray(result.x, dtype=float)
    return float(z[0]), np.maximum(z[1:], 0.0)
