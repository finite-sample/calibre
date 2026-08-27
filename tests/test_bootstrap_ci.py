"""Reference and contract tests for ``bootstrap_ci``."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.stats import bootstrap as scipy_bootstrap

from calibre import bootstrap_ci, calibration_report
from calibre.metrics import brier_score


@pytest.fixture
def evaluation_sample():
    """Held-out binary outcomes and probability forecasts."""
    rng = np.random.default_rng(41)
    y_pred = rng.uniform(0.02, 0.98, 80)
    y_true = rng.binomial(1, y_pred).astype(float)
    return y_true, y_pred


def test_api_uses_standard_default_and_keyword_only_options():
    """The public contract follows the package's evaluation conventions."""
    signature = inspect.signature(bootstrap_ci)
    assert list(signature.parameters) == [
        "metric",
        "y_true",
        "y_pred",
        "level",
        "n_resamples",
        "random_state",
        "method",
    ]
    for name in ("metric", "y_true", "y_pred"):
        assert (
            signature.parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
    for name in ("level", "n_resamples", "random_state", "method"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["method"].default == "bca"


@pytest.mark.parametrize("method", ["percentile", "basic", "bca"])
def test_matches_scipy_paired_bootstrap(evaluation_sample, method):
    """Match the mature reference on identical paired resamples."""
    y_true, y_pred = evaluation_sample
    seed = 17
    actual = bootstrap_ci(
        brier_score,
        y_true,
        y_pred,
        level=0.9,
        n_resamples=199,
        random_state=seed,
        method=method,
    )
    indices = np.arange(y_true.size)

    def paired_brier(row_indices):
        selected = np.asarray(row_indices, dtype=np.intp)
        return brier_score(y_true[selected], y_pred[selected])

    reference = scipy_bootstrap(
        (indices,),
        paired_brier,
        vectorized=False,
        confidence_level=0.9,
        n_resamples=199,
        batch=199,
        method=method,
        rng=np.random.default_rng(seed),
    )
    assert actual["lower"] == pytest.approx(reference.confidence_interval.low)
    assert actual["upper"] == pytest.approx(reference.confidence_interval.high)
    assert actual["bias"] == pytest.approx(
        np.mean(reference.bootstrap_distribution) - brier_score(y_true, y_pred)
    )


def test_bounds_scipy_batch_memory(monkeypatch):
    """A realistic large evaluation must not ask SciPy for an O(n-squared) batch."""
    observed = {}

    def fake_bootstrap(data, statistic, **kwargs):
        observed.update(kwargs)
        value = statistic(np.asarray([0, 1, 2]))
        return type(
            "Result",
            (),
            {
                "bootstrap_distribution": np.asarray([value, value + 0.01]),
                "confidence_interval": type(
                    "Interval", (), {"low": value - 0.01, "high": value + 0.01}
                )(),
            },
        )()

    monkeypatch.setattr("calibre.evaluation.scipy_bootstrap", fake_bootstrap)
    n = 10_000
    y_pred = np.linspace(0.01, 0.99, n)
    y_true = (y_pred >= 0.5).astype(float)
    bootstrap_ci(brier_score, y_true, y_pred, n_resamples=1000)

    assert observed["batch"] < n
    assert observed["batch"] * n * np.dtype(np.intp).itemsize <= 8 * 1024**2
    assert "rng" in observed


def test_result_contains_only_defined_quantities(evaluation_sample):
    """A degenerate flag is unnecessary when degenerate intervals raise."""
    y_true, y_pred = evaluation_sample
    result = bootstrap_ci(brier_score, y_true, y_pred, n_resamples=50)
    assert set(result) == {
        "estimate",
        "lower",
        "upper",
        "level",
        "n_resamples",
        "method",
        "bias",
    }


def test_custom_bc_method_is_not_supported(evaluation_sample):
    """Expose only interval methods implemented by the SciPy reference."""
    y_true, y_pred = evaluation_sample
    with pytest.raises(ValueError, match="method must be one of"):
        bootstrap_ci(brier_score, y_true, y_pred, n_resamples=20, method="bc")


@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        (np.array([]), np.array([]), "must not be empty"),
        (np.array([0.0]), np.array([0.2]), "at least two observations"),
        (np.array([0.0, 0.5]), np.array([0.2, 0.8]), "binary outcomes"),
        (np.array([0.0, 1.0]), np.array([-0.1, 0.8]), "probabilities"),
        (np.array([0.0]), np.array([0.2, 0.8]), "same shape"),
    ],
    ids=["empty", "one-row", "nonbinary", "probability", "shape"],
)
def test_rejects_invalid_evaluation_samples(y_true, y_pred, match):
    """The bootstrap cannot repair malformed evaluation data."""
    with pytest.raises(ValueError, match=match):
        bootstrap_ci(brier_score, y_true, y_pred, n_resamples=20)


@pytest.mark.parametrize(
    "metric",
    [
        pytest.param(lambda y, p: np.nan, id="nan"),
        pytest.param(lambda y, p: np.inf, id="infinite"),
        pytest.param(lambda y, p: np.array([0.1, 0.2]), id="vector"),
    ],
)
def test_rejects_nonfinite_or_nonscalar_metric_results(evaluation_sample, metric):
    """A failed statistic cannot silently become a malformed interval."""
    y_true, y_pred = evaluation_sample
    with pytest.raises(ValueError, match="metric must return a finite scalar"):
        bootstrap_ci(metric, y_true, y_pred, n_resamples=20)


def test_rejects_a_noncallable_metric(evaluation_sample):
    """Fail at the named public boundary instead of deep inside SciPy."""
    y_true, y_pred = evaluation_sample
    with pytest.raises(TypeError, match="metric must be callable"):
        bootstrap_ci("brier", y_true, y_pred, n_resamples=20)


def test_degenerate_bootstrap_distribution_is_not_an_interval(evaluation_sample):
    """Zero resampling variation means uncertainty could not be estimated."""
    y_true, y_pred = evaluation_sample
    with pytest.raises(RuntimeError, match="bootstrap distribution is degenerate"):
        bootstrap_ci(lambda y, p: 1.0, y_true, y_pred, n_resamples=20)


def test_calibration_report_only_intervals_the_regular_proper_score(
    evaluation_sample,
):
    """Do not report invalid ordinary-bootstrap intervals for errors at zero."""
    y_true, y_pred = evaluation_sample
    report = calibration_report(
        y_true,
        y_pred,
        include_brier_interval=True,
        interval_n_resamples=50,
    )
    assert set(report.intervals) == {"brier_score"}
