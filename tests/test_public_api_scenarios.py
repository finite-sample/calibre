"""Expectation-driven workflows for every public numerical API.

Each fixture has a known data-generating process. The tests assert the answer
implied by that process, then cross-check APIs that claim to report the same
quantity. This complements unit and property tests: returning a plausible number
is not enough here.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted

import calibre
import calibre.plots as calibre_plots
import calibre.utils as calibre_utils
from calibre import (
    CalibrationReport,
    CDIIsotonicCalibrator,
    CenteredIsotonicCalibrator,
    IsotonicCalibrator,
    MonotonicMixin,
    NearlyIsotonicCalibrator,
    RegularizedIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SmoothedIsotonicCalibrator,
    SplineCalibrator,
    TemperatureScaler,
    binned_calibration_error,
    bootstrap_ci,
    brier_score,
    calibration_curve,
    calibration_report,
    classwise_decomposition,
    classwise_ece,
    classwise_reliability,
    confidence_bands,
    consistency_bands,
    corp_reliability,
    correlation_metrics,
    cross_val_calibrate,
    debiased_calibration_error,
    detect_plateaus,
    expected_calibration_error,
    make_folds,
    maximum_calibration_error,
    mean_calibration_error,
    miscalibration_profile,
    plugin_calibration_error,
    run_plateau_diagnostics,
    score_decomposition,
    select_by_cv,
    smooth_calibration_error,
    sweep_calibration_error,
    tie_preservation_score,
    top_label_ece,
    unique_value_counts,
)
from calibre.diagnostics import analyze_plateau_simple
from calibre.utils import (
    check_array_1d,
    check_arrays,
    check_consistent_length,
    check_fitted,
    clip_to_range,
    ensure_1d,
    find_unique_sorted,
    group_by_value,
    interpolate_monotonic,
    restore_order,
    sort_by_x,
    validate_parameters,
)


@pytest.fixture(scope="module")
def exact_binary_calibration() -> tuple[np.ndarray, np.ndarray]:
    """Five forecast groups whose observed event rates equal their forecasts."""
    probabilities = np.repeat(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 100)
    outcomes = np.concatenate(
        [
            np.r_[
                np.ones(int(probability * 100)), np.zeros(100 - int(probability * 100))
            ]
            for probability in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
    )
    return probabilities, outcomes


def _overconfident(probabilities: np.ndarray) -> np.ndarray:
    odds = probabilities / (1.0 - probabilities)
    return odds**2 / (1.0 + odds**2)


def test_public_namespace_is_fully_accounted_for():
    """A new public entry must be assigned an expectation-driven workflow."""
    expected = {
        "BaseCalibrator",
        "CDIIsotonicCalibrator",
        "CalibrationReport",
        "CenteredIsotonicCalibrator",
        "IsotonicCalibrator",
        "MonotonicMixin",
        "NearlyIsotonicCalibrator",
        "RegularizedIsotonicCalibrator",
        "RelaxedPAVACalibrator",
        "SmoothedIsotonicCalibrator",
        "SplineCalibrator",
        "TemperatureScaler",
        "binned_calibration_error",
        "bootstrap_ci",
        "brier_score",
        "calibration_curve",
        "calibration_report",
        "classwise_decomposition",
        "classwise_ece",
        "classwise_reliability",
        "confidence_bands",
        "consistency_bands",
        "corp_reliability",
        "correlation_metrics",
        "cross_val_calibrate",
        "debiased_calibration_error",
        "detect_plateaus",
        "expected_calibration_error",
        "make_folds",
        "maximum_calibration_error",
        "mean_calibration_error",
        "metrics",
        "miscalibration_profile",
        "plots",
        "plugin_calibration_error",
        "run_plateau_diagnostics",
        "score_decomposition",
        "select_by_cv",
        "smooth_calibration_error",
        "sweep_calibration_error",
        "tie_preservation_score",
        "top_label_ece",
        "unique_value_counts",
    }
    assert set(calibre.__all__) == expected


def test_secondary_public_namespaces_are_fully_accounted_for():
    """Public utility and plotting additions must enter a realistic workflow."""
    assert set(calibre_utils.__all__) == {
        "check_array_1d",
        "check_arrays",
        "check_consistent_length",
        "check_fitted",
        "clip_to_range",
        "ensure_1d",
        "find_unique_sorted",
        "group_by_value",
        "interpolate_monotonic",
        "restore_order",
        "sort_by_x",
        "validate_parameters",
    }
    assert set(calibre_plots.__all__) == {
        "PALETTE",
        "SEMANTIC",
        "color_cycle",
        "plot_calibrator_comparison",
        "plot_classwise_reliability",
        "plot_ece_bin_sensitivity",
        "plot_mcb_dsc_plane",
        "plot_miscalibration_profile",
        "plot_reliability_diagram",
        "plot_resolution_frontier",
        "plot_resolution_loss",
        "plot_score_decomposition",
        "style_context",
    }


def test_public_array_utilities_preserve_a_score_label_workflow():
    """Every exported array helper must preserve values, groups, and ordering."""
    scores_column = np.array([[0.8], [0.2], [0.2], [1.1]])
    labels_column = np.array([[1], [0], [1], [1]])
    scores, labels = check_arrays(scores_column, labels_column)
    np.testing.assert_array_equal(scores, [0.8, 0.2, 0.2, 1.1])
    np.testing.assert_array_equal(labels, [1.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(check_array_1d(scores), scores)
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        check_array_1d(scores_column)
    check_consistent_length(scores, labels)

    clipped = clip_to_range(scores)
    np.testing.assert_array_equal(clipped, [0.8, 0.2, 0.2, 1.0])
    np.testing.assert_array_equal(ensure_1d(scores_column), scores)

    order, sorted_scores, sorted_labels = sort_by_x(clipped, labels)
    np.testing.assert_array_equal(sorted_scores, [0.2, 0.2, 0.8, 1.0])
    np.testing.assert_array_equal(sorted_labels, [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(restore_order(sorted_labels, order), labels)

    unique, starts = find_unique_sorted(sorted_scores)
    np.testing.assert_array_equal(unique, [0.2, 0.8, 1.0])
    np.testing.assert_array_equal(starts, [0, 2, 3])
    groups, indices = group_by_value(sorted_scores, sorted_labels)
    assert [group.tolist() for group in groups] == [[0.0, 1.0], [1.0], [1.0]]
    assert [index.tolist() for index in indices] == [[0, 1], [2], [3]]

    interpolated = interpolate_monotonic(
        np.array([0.0, 0.5, 1.0]),
        np.array([0.1, 0.4, 0.9]),
        np.array([-0.1, 0.25, 0.75, 1.1]),
    )
    np.testing.assert_allclose(interpolated, [0.1, 0.25, 0.65, 0.9])

    calibrator = IsotonicCalibrator()
    with pytest.raises(ValueError, match="must be fitted"):
        check_fitted(calibrator)
    calibrator.fit(clipped, labels)
    check_fitted(calibrator)
    validate_parameters(alpha=0.1, n_splits=5, percentile=95)


def test_public_plot_style_helpers_are_bounded_and_restore_global_state():
    """Plot styling must cycle deterministically and remain scoped to its context."""
    import matplotlib as mpl

    colours = calibre_plots.color_cycle(len(calibre_plots.PALETTE) + 1)
    assert colours[: len(calibre_plots.PALETTE)] == list(calibre_plots.PALETTE)
    assert colours[-1] == calibre_plots.PALETTE[0]

    before = mpl.rcParams["savefig.dpi"]
    with calibre_plots.style_context(**{"savefig.dpi": 144}):
        assert mpl.rcParams["savefig.dpi"] == 144
    assert mpl.rcParams["savefig.dpi"] == before


def test_binary_metrics_recover_exact_groupwise_calibration(exact_binary_calibration):
    """Every calibration-error estimator must vanish when every group is exact."""
    probabilities, outcomes = exact_binary_calibration
    expected_brier = float(
        np.mean(
            np.array([0.1, 0.3, 0.5, 0.7, 0.9]) * np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        )
    )

    assert mean_calibration_error(outcomes, probabilities) == pytest.approx(0.0)
    assert binned_calibration_error(outcomes, probabilities, n_bins=5) == pytest.approx(
        0.0, abs=1e-12
    )
    assert expected_calibration_error(
        outcomes, probabilities, n_bins=5
    ) == pytest.approx(0.0, abs=1e-12)
    assert maximum_calibration_error(
        outcomes, probabilities, n_bins=5
    ) == pytest.approx(0.0, abs=1e-12)
    assert plugin_calibration_error(outcomes, probabilities, n_bins=5) == pytest.approx(
        0.0, abs=1e-12
    )
    assert debiased_calibration_error(outcomes, probabilities, n_bins=5) == 0.0
    assert sweep_calibration_error(outcomes, probabilities) == pytest.approx(
        0.0, abs=1e-12
    )
    assert smooth_calibration_error(outcomes, probabilities) == pytest.approx(
        0.0, abs=1e-12
    )
    assert brier_score(outcomes, probabilities) == pytest.approx(expected_brier)

    prob_true, prob_pred, counts = calibration_curve(outcomes, probabilities, n_bins=5)
    np.testing.assert_allclose(prob_true, np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
    np.testing.assert_allclose(prob_pred, prob_true)
    np.testing.assert_array_equal(counts, np.full(5, 100))

    details = binned_calibration_error(
        outcomes, probabilities, n_bins=5, return_details=True
    )
    assert details["bce"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_array_equal(details["bin_counts"], counts)


def test_binary_metrics_match_manual_errors_on_overconfidence(exact_binary_calibration):
    """Binned summaries must equal their definitions, not merely be positive."""
    probabilities, outcomes = exact_binary_calibration
    reported = probabilities + 0.1 * (probabilities - 0.5)
    truth = np.repeat(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 100)
    group_errors = np.abs(np.unique(reported) - np.unique(truth))

    assert mean_calibration_error(outcomes, reported) == pytest.approx(0.0, abs=1e-12)
    assert expected_calibration_error(outcomes, reported, n_bins=5) == pytest.approx(
        np.mean(group_errors)
    )
    assert maximum_calibration_error(outcomes, reported, n_bins=5) == pytest.approx(
        np.max(group_errors)
    )
    assert binned_calibration_error(outcomes, reported, n_bins=5) == pytest.approx(
        np.sqrt(np.mean(group_errors**2))
    )


def test_resolution_and_plateau_diagnostics_report_known_structure():
    """Structural diagnostics must describe ties along the score axis exactly."""
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    calibrated = np.array([0.2, 0.2, 0.2, 0.8, 0.8, 0.8])

    assert detect_plateaus(calibrated) == [(0, 2, 0.2), (3, 5, 0.8)]
    diagnostics = run_plateau_diagnostics(scores[::-1], calibrated[::-1])
    assert diagnostics["n_plateaus"] == 2
    assert [plateau["x_range"] for plateau in diagnostics["plateaus"]] == [
        (0.1, 0.3),
        (0.7, 0.9),
    ]
    assert [plateau["n_samples"] for plateau in diagnostics["plateaus"]] == [3, 3]
    assert analyze_plateau_simple(scores, 0, 2, 0.2, 0) == diagnostics["plateaus"][0]

    counts = unique_value_counts(calibrated, scores)
    assert counts == {
        "n_unique_y_pred": 2,
        "n_unique_y_orig": 6,
        "unique_value_ratio": 1 / 3,
    }
    assert tie_preservation_score(scores, scores) == 1.0
    assert tie_preservation_score(scores, np.full_like(scores, 0.5)) == 0.0

    correlations = correlation_metrics(
        np.array([0, 0, 0, 1, 1, 1]),
        calibrated,
        x=calibrated,
        y_orig=calibrated,
    )
    assert correlations["spearman_corr_to_x"] == pytest.approx(1.0)
    assert correlations["spearman_corr_to_y_orig"] == pytest.approx(1.0)
    assert correlations["spearman_corr_orig_to_calib"] == pytest.approx(1.0)


def test_corp_decomposition_and_bands_agree_on_exact_calibration(
    exact_binary_calibration,
):
    """CORP's diagram, decomposition, and two band constructors must agree."""
    probabilities, outcomes = exact_binary_calibration
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    diagram = corp_reliability(probabilities, outcomes)

    np.testing.assert_allclose(diagram.x, levels)
    np.testing.assert_allclose(diagram.cep, levels)
    np.testing.assert_array_equal(diagram.weight, np.full(5, 100))
    np.testing.assert_allclose(diagram(levels), levels)

    decomposition = score_decomposition(probabilities, outcomes)
    assert decomposition["MCB"] == pytest.approx(0.0, abs=1e-12)
    assert decomposition["DSC"] > 0.0
    assert decomposition["mean_score"] == pytest.approx(
        decomposition["MCB"] - decomposition["DSC"] + decomposition["UNC"]
    )

    consistent = consistency_bands(
        probabilities, outcomes, n_resamples=100, random_state=7
    )
    confident = confidence_bands(
        probabilities, outcomes, n_resamples=100, random_state=7
    )
    for key in ("x", "lower", "upper"):
        np.testing.assert_allclose(consistent[key], confident[key])
    assert np.all(consistent["lower"] <= consistent["upper"])


def test_weighted_decomposition_equals_literal_frequency_replication():
    """Integer frequency weights must be equivalent to repeated observations."""
    probabilities = np.array([0.1, 0.4, 0.8, 0.9])
    outcomes = np.array([0.0, 1.0, 0.0, 1.0])
    weights = np.array([1, 3, 2, 4])
    weighted = score_decomposition(probabilities, outcomes, sample_weight=weights)
    repeated = score_decomposition(
        np.repeat(probabilities, weights), np.repeat(outcomes, weights)
    )
    assert weighted == pytest.approx(repeated)


def test_bootstrap_and_report_reproduce_their_component_metrics(
    exact_binary_calibration,
):
    """The one-call report and interval wrapper must not reinterpret metrics."""
    probabilities, outcomes = exact_binary_calibration
    interval = bootstrap_ci(
        brier_score, outcomes, probabilities, n_resamples=200, random_state=11
    )
    assert interval["estimate"] == pytest.approx(brier_score(outcomes, probabilities))
    assert interval["lower"] <= interval["estimate"] <= interval["upper"]

    report = calibration_report(outcomes, probabilities, n_bins=5)
    assert isinstance(report, CalibrationReport)
    assert report.brier == pytest.approx(brier_score(outcomes, probabilities))
    assert report.mcb == pytest.approx(0.0, abs=1e-12)
    assert report.debiased_ece == 0.0
    assert report.n_distinct == 5
    assert report.to_dict()["n"] == outcomes.size
    assert "Brier" in str(report)


@pytest.mark.parametrize(
    "calibrator",
    [
        IsotonicCalibrator(),
        CenteredIsotonicCalibrator(),
        NearlyIsotonicCalibrator(),
        RegularizedIsotonicCalibrator(),
        RelaxedPAVACalibrator(),
        SmoothedIsotonicCalibrator(),
        SplineCalibrator(),
        CDIIsotonicCalibrator(),
    ],
    ids=lambda calibrator: type(calibrator).__name__,
)
def test_default_calibrators_recover_a_known_monotone_distortion(
    calibrator,
    exact_binary_calibration,
):
    """Every default calibrator must improve a realistic global distortion."""
    probabilities, outcomes = exact_binary_calibration
    raw = _overconfident(probabilities)
    fitted = calibrator.fit(raw, outcomes)
    calibrated = fitted.transform(raw)

    check_is_fitted(fitted)
    assert np.all(np.isfinite(calibrated))
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    assert np.all(np.diff(calibrated[np.argsort(raw)]) >= -1e-10)
    assert brier_score(outcomes, calibrated) < brier_score(outcomes, raw)

    expected = np.repeat(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 100)
    np.testing.assert_allclose(calibrated, expected, atol=0.03)


@pytest.mark.parametrize(
    "calibrator",
    [
        IsotonicCalibrator(),
        CenteredIsotonicCalibrator(clip_output=False),
        NearlyIsotonicCalibrator(clip_output=False),
        RegularizedIsotonicCalibrator(link="identity", clip_output=False),
        RelaxedPAVACalibrator(clip_output=False),
        SplineCalibrator(link="identity", clip_output=False),
    ],
    ids=lambda calibrator: type(calibrator).__name__,
)
def test_continuous_target_calibrators_preserve_an_unbounded_monotone_signal(
    calibrator,
):
    """Identity-scale workflows must not silently clip continuous targets."""
    scores = np.linspace(-1.0, 1.0, 200)
    target = 2.0 + 3.0 * scores + 0.1 * np.sin(5.0 * scores)
    fitted = calibrator.fit(scores, target)
    prediction = fitted.transform(scores)

    assert prediction.min() < 0.0
    assert prediction.max() > 1.0
    assert np.all(np.diff(prediction) >= -1e-10)
    assert np.sqrt(np.mean((prediction - target) ** 2)) < 0.01


@pytest.mark.parametrize(
    "factory",
    [
        IsotonicCalibrator,
        CenteredIsotonicCalibrator,
        lambda: RegularizedIsotonicCalibrator(alpha=0.01, n_knots=5),
        lambda: RelaxedPAVACalibrator(epsilon=0.0),
        lambda: SplineCalibrator(alpha=0.001, n_knots=5),
        CDIIsotonicCalibrator,
        lambda: CDIIsotonicCalibrator(
            thresholds=[0.2, 0.5, 0.8],
            threshold_weights=[1.0, 3.0, 1.0],
            gamma=0.8,
            window=3,
        ),
    ],
)
def test_zero_weight_observation_has_no_effect(factory):
    """Weight-aware calibrators must ignore an observation carrying zero weight."""
    scores = np.linspace(0.05, 0.95, 12)
    outcomes = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1], dtype=float)
    weights = np.array([1, 2, 1, 3, 1, 2, 4, 1, 2, 1, 3, 2], dtype=float)
    grid = np.linspace(0.0, 1.0, 101)

    baseline = factory().fit(scores, outcomes, sample_weight=weights).transform(grid)
    augmented = (
        factory()
        .fit(
            np.r_[scores, 0.99],
            np.r_[outcomes, 2.0],
            sample_weight=np.r_[weights, 0.0],
        )
        .transform(grid)
    )
    np.testing.assert_allclose(augmented, baseline, atol=1e-10)


@pytest.mark.parametrize(
    "factory",
    [
        CenteredIsotonicCalibrator,
        lambda: RelaxedPAVACalibrator(epsilon=0.0, min_slope=0.0),
    ],
)
def test_zero_weight_observation_cannot_create_an_interpolation_knot(factory):
    """A zero-mass score between fitted points must not bend the fitted map."""
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    outcomes = np.array([0.0, 0.0, 0.0, 1.0])
    weights = np.array([1.0, 2.0, 3.0, 1.0])
    grid = np.linspace(0.0, 1.0, 101)

    baseline = factory().fit(scores, outcomes, sample_weight=weights).transform(grid)
    augmented = (
        factory()
        .fit(
            np.r_[scores, 0.7],
            np.r_[outcomes, 1.0],
            sample_weight=np.r_[weights, 0.0],
        )
        .transform(grid)
    )

    np.testing.assert_allclose(augmented, baseline, atol=1e-12)


def test_zero_weight_observation_does_not_change_diagnostics_or_curve_range():
    """Derived spline outputs must use the same positive-mass training support."""
    scores = np.linspace(0.1, 0.9, 20)
    outcomes = np.r_[np.zeros(10), np.ones(10)]
    weights = np.ones(scores.size)
    baseline = SplineCalibrator(alpha=0.01, n_knots=5, enable_diagnostics=True).fit(
        scores, outcomes, sample_weight=weights
    )
    augmented = SplineCalibrator(alpha=0.01, n_knots=5, enable_diagnostics=True).fit(
        np.r_[scores, 2.0],
        np.r_[outcomes, 1.0],
        sample_weight=np.r_[weights, 0.0],
    )

    np.testing.assert_allclose(
        augmented.calibration_curve().x, baseline.calibration_curve().x
    )
    assert augmented.get_diagnostics() == baseline.get_diagnostics()


def test_selection_and_out_of_fold_calibration_choose_the_known_direction(
    exact_binary_calibration,
):
    """Cross-validation must prefer the true increasing map.

    Its out-of-fold predictions must also improve the proper score.
    """
    probabilities, outcomes = exact_binary_calibration
    raw = _overconfident(probabilities)
    folds = make_folds(raw, outcomes, cv=5, random_state=3)
    validation = np.concatenate([valid for _, valid in folds])
    np.testing.assert_array_equal(np.sort(validation), np.arange(outcomes.size))
    for train, valid in folds:
        assert not np.intersect1d(train, valid).size

    selected = select_by_cv(
        lambda **kwargs: IsotonicCalibrator(**kwargs),
        {"increasing": [False, True]},
        raw,
        outcomes,
        cv=5,
        scoring="brier",
        random_state=3,
    )
    assert selected == {"increasing": True}

    out_of_fold = cross_val_calibrate(
        IsotonicCalibrator(), raw, outcomes, cv=5, random_state=3
    )
    assert out_of_fold.shape == outcomes.shape
    assert brier_score(outcomes, out_of_fold) < brier_score(outcomes, raw)


def test_monotonic_mixin_enforces_its_declared_contract():
    """The public mixin must detect and repair the same decreasing sequence."""
    values = np.array([0.1, 0.4, 0.3, 0.8])
    assert not MonotonicMixin.check_monotonicity(values)
    repaired = MonotonicMixin.enforce_monotonicity(values)
    np.testing.assert_allclose(repaired, np.array([0.1, 0.4, 0.4, 0.8]))
    assert MonotonicMixin.check_monotonicity(repaired)


@pytest.fixture(scope="module")
def multiclass_temperature_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four-class probabilities sharpened by one known global temperature."""
    rng = np.random.default_rng(123)
    truth = rng.dirichlet(np.ones(4) * 0.8, size=3000)
    outcomes = np.array([rng.choice(4, p=row) for row in truth])
    reported = truth**2
    reported /= reported.sum(axis=1, keepdims=True)
    return truth, reported, outcomes


def test_multiclass_apis_agree_and_temperature_scaling_recovers_global_distortion(
    multiclass_temperature_case,
):
    """Every multiclass summary must reduce to its documented binary calculation."""
    truth, reported, outcomes = multiclass_temperature_case
    rows = np.arange(outcomes.size)
    scaler = TemperatureScaler().fit(reported, outcomes)
    calibrated = scaler.transform(reported)

    assert scaler.temperature_ == pytest.approx(2.0, rel=0.1)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
    np.testing.assert_array_equal(calibrated.argmax(axis=1), reported.argmax(axis=1))
    assert -np.mean(np.log(calibrated[rows, outcomes])) < -np.mean(
        np.log(reported[rows, outcomes])
    )

    decompositions = classwise_decomposition(reported, outcomes)
    diagrams = classwise_reliability(reported, outcomes)
    profile = miscalibration_profile(reported, outcomes)
    assert len(decompositions) == len(diagrams) == reported.shape[1]
    np.testing.assert_allclose(profile["mcb"], [part["MCB"] for part in decompositions])
    for part, diagram in zip(decompositions, diagrams, strict=True):
        assert part["mean_score"] == pytest.approx(
            part["MCB"] - part["DSC"] + part["UNC"]
        )
        assert np.all(np.diff(diagram.cep) >= -1e-12)

    manual_classwise = np.mean(
        [
            debiased_calibration_error(
                (outcomes == column).astype(float), reported[:, column], n_bins=10
            )
            for column in range(reported.shape[1])
        ]
    )
    assert classwise_ece(reported, outcomes, n_bins=10) == pytest.approx(
        manual_classwise
    )

    confidence = reported.max(axis=1)
    correct = (reported.argmax(axis=1) == outcomes).astype(float)
    assert top_label_ece(reported, outcomes, n_bins=10) == pytest.approx(
        debiased_calibration_error(correct, confidence, n_bins=10)
    )
    assert classwise_ece(truth, outcomes, n_bins=10) < classwise_ece(
        reported, outcomes, n_bins=10
    )
