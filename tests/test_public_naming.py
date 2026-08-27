"""Public naming conventions are consistent across calibre's API."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from calibre import (
    BaseCalibrator,
    CalibrationReport,
    CDIIsotonicCalibrator,
    CenteredIsotonicCalibrator,
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SplineCalibrator,
    calibration_report,
    plugin_calibration_error,
    root_mean_squared_calibration_error,
    smooth_calibration_error,
    sweep_calibration_error,
    unique_value_counts,
)
from calibre.diagnostics import detect_plateaus, run_plateau_diagnostics
from calibre.evaluation import ReliabilityDiagram
from calibre.multiclass import (
    TemperatureScaler,
    classwise_decomposition,
    classwise_ece,
    classwise_reliability,
    miscalibration_profile,
    top_label_ece,
)
from calibre.plots import (
    plot_calibrator_comparison,
    plot_miscalibration_profile,
    plot_resolution_frontier,
    plot_resolution_loss,
)
from calibre.plots._style import color_cycle
from calibre.selection import cross_val_calibrate, resolve_auto, select_by_cv


def _parameter_names(callable_: object) -> list[str]:
    return list(inspect.signature(callable_).parameters)


@pytest.mark.parametrize(
    ("function", "names"),
    [
        (classwise_decomposition, ["y_true", "y_pred", "score"]),
        (miscalibration_profile, ["y_true", "y_pred"]),
        (classwise_ece, ["y_true", "y_pred", "n_bins", "estimator"]),
        (top_label_ece, ["y_true", "y_pred", "n_bins", "estimator"]),
        (classwise_reliability, ["y_true", "y_pred"]),
        (TemperatureScaler.fit, ["self", "X", "y"]),
        (TemperatureScaler.transform, ["self", "X"]),
        (TemperatureScaler.fit_transform, ["self", "X", "y"]),
    ],
)
def test_multiclass_uses_sklearn_and_metric_argument_names(function, names):
    assert _parameter_names(function) == names


def test_multiclass_optional_controls_are_keyword_only():
    assert (
        inspect.signature(classwise_decomposition).parameters["score"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(classwise_ece).parameters["n_bins"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(top_label_ece).parameters["n_bins"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_miscalibration_profile_uses_descriptive_result_keys():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ]
    )

    assert set(miscalibration_profile(y_true, y_pred)) == {
        "classwise_miscalibration",
        "relative_miscalibration_spread",
        "worst_classes",
        "interpretation",
    }


def test_prediction_granularity_uses_prediction_vocabulary():
    result = unique_value_counts(
        np.array([0.2, 0.2, 0.7]),
        original_predictions=np.array([0.1, 0.4, 0.9]),
    )

    assert _parameter_names(unique_value_counts) == [
        "predictions",
        "original_predictions",
    ]
    assert set(result) == {
        "n_unique_predictions",
        "n_unique_original_predictions",
        "unique_prediction_ratio",
    }


def test_plateau_diagnostics_use_descriptive_nonduplicated_fields():
    result = run_plateau_diagnostics(
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0.25, 0.25, 0.75, 0.75]),
    )

    assert _parameter_names(run_plateau_diagnostics) == [
        "input_scores",
        "calibrated_predictions",
    ]
    assert _parameter_names(detect_plateaus) == [
        "calibrated_predictions",
        "min_width",
    ]
    assert (
        inspect.signature(detect_plateaus).parameters["min_width"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert set(result["plateaus"][0]) == {
        "plateau_id",
        "input_score_range",
        "calibrated_value",
        "n_observations",
        "support",
    }


def test_calibration_error_norm_and_bandwidth_names_are_consistent():
    assert _parameter_names(plugin_calibration_error) == [
        "y_true",
        "y_pred",
        "n_bins",
        "norm",
        "sample_weight",
    ]
    assert _parameter_names(sweep_calibration_error) == [
        "y_true",
        "y_pred",
        "norm",
        "return_n_bins",
    ]
    assert _parameter_names(smooth_calibration_error) == [
        "y_true",
        "y_pred",
        "bandwidth",
        "return_bandwidth",
    ]


def test_rmsce_details_use_complete_names():
    details = root_mean_squared_calibration_error(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        n_bins=2,
        return_details=True,
    )

    assert set(details) == {
        "root_mean_squared_calibration_error",
        "bin_counts",
        "bin_weights",
        "bin_score_minimums",
        "bin_score_maximums",
        "bin_prediction_means",
        "bin_event_rates",
    }


def test_report_and_reliability_objects_use_descriptive_attribute_names():
    report = calibration_report(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), n_bins=2
    )
    diagram = ReliabilityDiagram(
        np.array([0.2, 0.8]),
        np.array([0.1, 0.9]),
        np.array([2.0, 2.0]),
    )

    assert "n_observations" in CalibrationReport.__dataclass_fields__
    assert "n" not in CalibrationReport.__dataclass_fields__
    assert report.n_observations == 4
    assert hasattr(diagram, "prediction_weights")
    assert not hasattr(diagram, "weights")
    np.testing.assert_array_equal(diagram.prediction_weights, [2.0, 2.0])
    np.testing.assert_allclose(diagram(np.array([0.5])), [0.5])
    assert _parameter_names(ReliabilityDiagram.__call__) == [
        "self",
        "new_predictions",
    ]


def test_selection_helpers_use_descriptive_names_and_keyword_only_controls():
    assert _parameter_names(select_by_cv) == [
        "calibrator_factory",
        "param_grid",
        "X",
        "y",
        "sample_weight",
        "cv",
        "scoring",
        "max_cv_samples",
        "random_state",
    ]
    assert _parameter_names(cross_val_calibrate) == [
        "calibrator",
        "X",
        "y",
        "sample_weight",
        "cv",
        "random_state",
    ]
    assert _parameter_names(resolve_auto) == [
        "parameter_value",
        "parameter_name",
        "parameter_grid",
        "calibrator_factory",
        "X",
        "y",
        "cv",
        "scoring",
        "random_state",
        "minimum_value",
        "sample_weight",
    ]

    for function, first_optional in (
        (select_by_cv, "sample_weight"),
        (cross_val_calibrate, "sample_weight"),
        (resolve_auto, "cv"),
    ):
        assert (
            inspect.signature(function).parameters[first_optional].kind
            is inspect.Parameter.KEYWORD_ONLY
        )


def test_plotting_helpers_use_descriptive_public_names():
    assert _parameter_names(plot_calibrator_comparison)[:2] == [
        "calibrators",
        "input_scores",
    ]
    assert _parameter_names(plot_resolution_loss)[:2] == [
        "calibrated_predictions",
        "input_scores",
    ]
    assert _parameter_names(plot_resolution_frontier)[2] == "error_bars"
    assert "show_interpretation" in _parameter_names(plot_miscalibration_profile)
    assert "interpretation_width" in _parameter_names(plot_miscalibration_profile)
    assert "show_reading" not in _parameter_names(plot_miscalibration_profile)
    assert _parameter_names(color_cycle) == ["n_colors"]


def test_estimator_constructor_options_are_keyword_only():
    classes = [
        BaseCalibrator,
        IsotonicCalibrator,
        CenteredIsotonicCalibrator,
        NearlyIsotonicCalibrator,
        SplineCalibrator,
        TemperatureScaler,
    ]
    for class_ in classes:
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in inspect.signature(class_).parameters.values()
        )

    cdi_parameters = inspect.signature(CDIIsotonicCalibrator).parameters
    assert cdi_parameters["thresholds"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in cdi_parameters.items()
        if name != "thresholds"
    )

    relaxed_parameters = inspect.signature(RelaxedPAVACalibrator).parameters
    assert (
        relaxed_parameters["min_increment"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in relaxed_parameters.items()
        if name != "min_increment"
    )


def test_old_positional_optional_estimator_arguments_are_rejected():
    with pytest.raises(TypeError):
        IsotonicCalibrator(0.0)
    with pytest.raises(TypeError):
        CDIIsotonicCalibrator([0.5], [1.0])
    with pytest.raises(TypeError):
        RelaxedPAVACalibrator(0.0, False)
