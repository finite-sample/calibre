"""Independent numerical checks for bounded-increment PAVA."""

from __future__ import annotations

import numpy as np
import pytest

from calibre._core import shift_to_pava


def _cvx_projection(
    y: np.ndarray, weight: np.ndarray, bound: float | np.ndarray
) -> tuple[np.ndarray, float]:
    """Solve the documented quadratic program without using PAVA."""
    import cvxpy as cp

    z = cp.Variable(y.size)
    lower = np.broadcast_to(np.asarray(bound, dtype=float), (y.size - 1,))
    objective = cp.sum(cp.multiply(weight, cp.square(y - z)))
    problem = cp.Problem(cp.Minimize(objective), [cp.diff(z) >= lower])
    problem.solve(
        solver=cp.OSQP,
        eps_abs=1e-10,
        eps_rel=1e-10,
        polishing=True,
    )
    assert problem.status == "optimal"
    return np.asarray(z.value, dtype=float), float(problem.value)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("bound", [-0.1, -0.02, 0.0, 0.01, 0.05])
def test_shift_to_pava_matches_independent_convex_program(seed, bound):
    """The reduction must attain the weighted constrained optimum."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=5 + 2 * seed)
    weight = rng.lognormal(size=y.size)

    expected, expected_objective = _cvx_projection(y, weight, bound)
    got = shift_to_pava(y, weight, L=bound)
    got_objective = float(np.sum(weight * (y - got) ** 2))

    assert np.all(np.diff(got) >= bound - 1e-12)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-8)
    assert got_objective == pytest.approx(expected_objective, abs=1e-9)


def test_shift_to_pava_matches_oracle_with_local_bounds():
    """The vector-bound path used by CDI must obey every local constraint."""
    rng = np.random.default_rng(91)
    y = rng.normal(size=17)
    weight = rng.uniform(0.1, 3.0, size=y.size)
    bound = rng.uniform(-0.08, 0.04, size=y.size - 1)

    expected, expected_objective = _cvx_projection(y, weight, bound)
    got = shift_to_pava(y, weight, L=bound)
    got_objective = float(np.sum(weight * (y - got) ** 2))

    assert np.all(np.diff(got) >= bound - 1e-12)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-8)
    assert got_objective == pytest.approx(expected_objective, abs=1e-9)
