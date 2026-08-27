"""Behavior shared by every public binary calibrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from calibre import (
    CDIIsotonicCalibrator,
    CenteredIsotonicCalibrator,
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SplineCalibrator,
)

if TYPE_CHECKING:
    from collections.abc import Callable

FACTORIES: dict[str, Callable[[], object]] = {
    "cdi": lambda: CDIIsotonicCalibrator(thresholds=[0.5]),
    "centered": CenteredIsotonicCalibrator,
    "isotonic": IsotonicCalibrator,
    "nearly": lambda: NearlyIsotonicCalibrator(lam=5.0),
    "relaxed": lambda: RelaxedPAVACalibrator(min_increment=-0.02),
    "spline": lambda: SplineCalibrator(alpha=0.1),
}


@pytest.fixture
def calibration_data() -> tuple[np.ndarray, np.ndarray]:
    """A nondegenerate binary calibration sample."""
    x = np.linspace(0.02, 0.98, 80)
    y = (np.sin(np.arange(x.size) * 1.7) + x > 0.6).astype(float)
    return x, y


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
def test_fit_rejects_multidimensional_scores(factory, calibration_data):
    """Multiple features cannot be flattened into one score sequence."""
    x, y = calibration_data
    with pytest.raises(ValueError, match="X must be 1-dimensional"):
        factory().fit(np.column_stack([x, x]), np.tile(y, 2))


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
@pytest.mark.parametrize("which", ["score", "target"])
def test_fit_rejects_nonfinite_data(factory, calibration_data, which):
    """Non-finite calibration observations must fail at the public boundary."""
    x, y = (values.copy() for values in calibration_data)
    if which == "score":
        x[3] = np.nan
    else:
        y[3] = np.inf
    with pytest.raises(ValueError, match="must contain only finite values"):
        factory().fit(x, y)


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
def test_fit_rejects_multidimensional_sample_weight(factory, calibration_data):
    """Observation weights follow the same one-dimensional array contract."""
    x, y = calibration_data
    with pytest.raises(ValueError, match="sample_weight must be 1-dimensional"):
        factory().fit(x, y, sample_weight=np.ones_like(y).reshape(-1, 1))


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
def test_fit_rejects_mismatched_sample_weight(factory, calibration_data):
    """Every calibrator reports the shared weight-shape contract."""
    x, y = calibration_data
    with pytest.raises(ValueError, match="sample_weight must have the same shape as y"):
        factory().fit(x, y, sample_weight=np.ones(y.size - 1))


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
@pytest.mark.parametrize(
    ("sample_weight", "match"),
    [
        (lambda n: np.r_[np.ones(n - 1), -1.0], "non-?negative"),
        (lambda n: np.r_[np.ones(n - 1), np.nan], "finite"),
        (lambda n: np.zeros(n), "positive"),
    ],
    ids=["negative", "nonfinite", "zero-mass"],
)
def test_fit_rejects_invalid_sample_weight(
    factory, calibration_data, sample_weight, match
):
    """Every calibrator enforces the documented observation-weight domain."""
    x, y = calibration_data
    with pytest.raises(ValueError, match=match):
        factory().fit(x, y, sample_weight=sample_weight(y.size))


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
def test_transform_uses_the_same_input_contract(factory, calibration_data):
    """Prediction must not flatten dimensions or pass non-finite scores onward."""
    x, y = calibration_data
    fitted = factory().fit(x, y)
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        fitted.transform(x.reshape(-1, 1))
    with pytest.raises(ValueError, match="must contain only finite values"):
        fitted.transform(np.array([0.2, np.nan, 0.8]))


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
def test_failed_refit_removes_the_previous_curve(factory, calibration_data):
    """A failed refit cannot leave an old model callable as if it were current."""
    x, y = calibration_data
    fitted = factory().fit(x, y)
    fitted.transform(x[:3])
    with pytest.raises(ValueError, match="same length"):
        fitted.fit(x, y[:-1])
    with pytest.raises(NotFittedError):
        fitted.transform(x[:3])


def test_failed_diagnostics_leave_no_fitted_estimator(monkeypatch, calibration_data):
    """A fit that raises after numerical training must remain transactional."""
    x, y = calibration_data
    calibrator = IsotonicCalibrator(enable_diagnostics=True)

    def fail_diagnostics():
        raise RuntimeError("diagnostics failed")

    monkeypatch.setattr(calibrator, "_run_diagnostics", fail_diagnostics)
    with pytest.raises(RuntimeError, match="diagnostics failed"):
        calibrator.fit(x, y)
    with pytest.raises(NotFittedError):
        calibrator.transform(x[:3])


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
def test_transform_before_fit_raises_not_fitted(factory):
    """Every calibrator reports the standard sklearn fitted-state exception."""
    with pytest.raises(NotFittedError):
        factory().transform(np.array([0.2, 0.8]))


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
def test_fit_transform_rejects_unknown_parameters(factory, calibration_data):
    """A misspelled fitting argument must not be silently discarded."""
    x, y = calibration_data
    with pytest.raises(TypeError):
        factory().fit_transform(x, y, sampl_weight=np.ones_like(y))
