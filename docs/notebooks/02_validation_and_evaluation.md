---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Validation and evaluation

Software validation and statistical evaluation answer different questions.

- Software validation asks whether a calibrator returns finite probabilities with
  the documented shape, bounds, and monotonicity.
- Statistical evaluation asks whether calibration improves predictions on new data.

A calibrator can pass every software contract and still hurt held-out performance.
Conversely, fitting and scoring an isotonic-family method on the same observations
can report perfect in-sample calibration without establishing that it generalizes.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibre import (
    CenteredIsotonicCalibrator,
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SplineCalibrator,
    brier_score,
    calibration_curve,
    score_decomposition,
    smooth_calibration_error,
)

RANDOM_STATE = 42
random_generator = np.random.default_rng(RANDOM_STATE)
```

## Draw independent calibration and test samples

The first design applies a monotone distortion to known event probabilities. The
calibration and test samples are independent draws from the same population.

```{code-cell} ipython3
def probability(log_odds):
    """Map finite log odds to probabilities."""
    return 1.0 / (1.0 + np.exp(-log_odds))


def draw_sample(n_observations=2000, *, scale=1.0, shift=0.0):
    """Draw forecasts and binary outcomes under a log-odds distortion."""
    true_probability = random_generator.beta(2.0, 2.0, n_observations)
    outcomes = random_generator.binomial(1, true_probability).astype(float)
    true_log_odds = np.log(true_probability / (1.0 - true_probability))
    forecasts = probability(scale * true_log_odds + shift)
    return forecasts, outcomes


calibration_predictions, calibration_outcomes = draw_sample(scale=1.7, shift=0.2)
test_predictions, test_outcomes = draw_sample(scale=1.7, shift=0.2)

print(
    f"calibration sample: {calibration_outcomes.size}; "
    f"test sample: {test_outcomes.size}"
)
```

## Fit the candidates once

Relaxed PAVA has no defensible universal increment bound, so the example supplies one
explicitly. CDI isotonic is omitted because its thresholds must come from a real
decision problem on the forecast scale.

```{code-cell} ipython3
def candidate_calibrators():
    """Return fresh calibrators for one independent fit."""
    return {
        "isotonic": IsotonicCalibrator(),
        "centered isotonic": CenteredIsotonicCalibrator(),
        "nearly isotonic": NearlyIsotonicCalibrator(),
        "spline": SplineCalibrator(),
        "relaxed PAVA (-0.01)": RelaxedPAVACalibrator(min_increment=-0.01),
    }


fitted_calibrators = {
    name: calibrator.fit(calibration_predictions, calibration_outcomes)
    for name, calibrator in candidate_calibrators().items()
}
```

## Validate software contracts

All public calibrators return one finite probability per input observation. Isotonic,
centered isotonic, and the spline are monotone by definition. Nearly isotonic and a
negative Relaxed PAVA bound deliberately permit decreases, so monotonicity would be
the wrong contract for them.

```{code-cell} ipython3
monotone_methods = {"isotonic", "centered isotonic", "spline"}
dense_grid = np.linspace(0.0, 1.0, 1001)

for name, calibrator in fitted_calibrators.items():
    transformed_test = calibrator.transform(test_predictions)
    transformed_grid = calibrator.transform(dense_grid)

    assert transformed_test.shape == test_predictions.shape
    assert np.all(np.isfinite(transformed_test))
    assert np.all((0.0 <= transformed_test) & (transformed_test <= 1.0))
    if name in monotone_methods:
        assert np.all(np.diff(transformed_grid) >= -1e-12)

    print(
        f"{name:22s} finite, bounded, correct shape"
        + (", monotone" if name in monotone_methods else "")
    )
```

The assertions are executable documentation. A contract violation stops the Sphinx
build instead of being converted to a warning or an infinite score.

## Evaluate on untouched outcomes

Held-out Brier score is the method-selection quantity. The remaining columns explain
whether a change came from calibration, discrimination, or prediction granularity.

```{code-cell} ipython3
def evaluation_row(method, predictions):
    """Return held-out evaluation measures for one prediction vector."""
    decomposition = score_decomposition(test_outcomes, predictions)
    return {
        "method": method,
        "brier": brier_score(test_outcomes, predictions),
        "miscalibration": decomposition["miscalibration"],
        "discrimination": decomposition["discrimination"],
        "smece": smooth_calibration_error(test_outcomes, predictions),
        "distinct_predictions": np.unique(predictions).size,
    }


rows = [evaluation_row("uncalibrated", test_predictions)]
for name, calibrator in fitted_calibrators.items():
    rows.append(evaluation_row(name, calibrator.transform(test_predictions)))

evaluation = pd.DataFrame(rows).sort_values("brier")
evaluation.assign(
    brier=evaluation["brier"].round(4),
    miscalibration=evaluation["miscalibration"].round(4),
    discrimination=evaluation["discrimination"].round(4),
    smece=evaluation["smece"].round(4),
)
```

Lower Brier is better. Lower MCB and smECE indicate less measured miscalibration,
while higher DSC indicates more useful separation between observations. The number of
distinct predictions is structural: it measures retained granularity, not calibration
quality.

## Use calibrated data as a negative control

Calibration should help a persistently distorted forecaster. It should not be expected
to improve a forecaster that already reports the true event probability. Finite-sample
calibrator fits can make that negative control worse by learning noise.

```{code-cell} ipython3
control_calibration_predictions, control_calibration_outcomes = draw_sample()
control_test_predictions, control_test_outcomes = draw_sample()

control_rows = [
    {
        "method": "uncalibrated",
        "brier": brier_score(control_test_outcomes, control_test_predictions),
    }
]
for name, calibrator in candidate_calibrators().items():
    calibrator.fit(control_calibration_predictions, control_calibration_outcomes)
    calibrated = calibrator.transform(control_test_predictions)
    control_rows.append(
        {
            "method": name,
            "brier": brier_score(control_test_outcomes, calibrated),
        }
    )

control_evaluation = pd.DataFrame(control_rows)
baseline_brier = float(
    control_evaluation.loc[
        control_evaluation["method"] == "uncalibrated", "brier"
    ].iloc[0]
)
control_evaluation["brier_change"] = (
    control_evaluation["brier"] - baseline_brier
)
control_evaluation.assign(
    brier=control_evaluation["brier"].round(4),
    brier_change=control_evaluation["brier_change"].round(4),
)
```

This control prevents a weak test from defining success as "the code ran" or "the
calibration metric became smaller." A useful validation suite needs both a positive
case where correction is possible and a negative case where unnecessary correction
can do harm.

## Stress ties, rare events, and clipped boundaries

The following designs exercise forecast patterns common in production. Each design
draws separate calibration and test outcomes.

```{code-cell} ipython3
def draw_stress_sample(design, n_observations=2000):
    """Draw one stress-test sample with known conditional event probabilities."""
    if design == "rare events":
        true_probability = random_generator.beta(1.0, 15.0, n_observations)
    else:
        true_probability = random_generator.beta(2.0, 2.0, n_observations)

    outcomes = random_generator.binomial(1, true_probability).astype(float)
    true_log_odds = np.log(true_probability / (1.0 - true_probability))
    forecasts = probability(1.6 * true_log_odds + 0.2)

    if design == "heavy ties":
        forecasts = np.round(forecasts, 1)
    elif design == "clipped boundaries":
        forecasts = np.clip(forecasts, 0.1, 0.9)
        forecasts[forecasts == 0.1] = 0.0
        forecasts[forecasts == 0.9] = 1.0

    return forecasts, outcomes


stress_rows = []
for design in ("heavy ties", "rare events", "clipped boundaries"):
    stress_calibration_predictions, stress_calibration_outcomes = draw_stress_sample(
        design
    )
    stress_test_predictions, stress_test_outcomes = draw_stress_sample(design)
    baseline = brier_score(stress_test_outcomes, stress_test_predictions)

    for name, calibrator in candidate_calibrators().items():
        calibrator.fit(stress_calibration_predictions, stress_calibration_outcomes)
        calibrated = calibrator.transform(stress_test_predictions)
        stress_rows.append(
            {
                "design": design,
                "method": name,
                "brier_improvement": baseline
                - brier_score(stress_test_outcomes, calibrated),
                "distinct_predictions": np.unique(calibrated).size,
            }
        )

stress_evaluation = pd.DataFrame(stress_rows)
stress_evaluation.assign(
    brier_improvement=stress_evaluation["brier_improvement"].round(4)
)
```

Positive Brier improvement means calibration helped on that held-out sample. A method
can improve Brier while retaining few distinct values, or retain many values without
improving Brier. The two columns answer different questions.

## Plot one held-out reliability comparison

```{code-cell} ipython3
centered_predictions = fitted_calibrators["centered isotonic"].transform(
    test_predictions
)

figure, axis = plt.subplots(figsize=(6, 5))
for name, predictions in (
    ("uncalibrated", test_predictions),
    ("centered isotonic", centered_predictions),
):
    event_rates, prediction_means, _ = calibration_curve(
        test_outcomes,
        predictions,
        n_bins=10,
    )
    axis.plot(prediction_means, event_rates, "o-", label=name)

axis.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
axis.set_xlabel("Mean prediction")
axis.set_ylabel("Observed event rate")
axis.set_title("Held-out reliability curves")
axis.grid(alpha=0.3)
axis.legend()
figure.tight_layout()
plt.show()
```

The curve is a descriptive view whose appearance depends on the binning rule. Use the
proper score for selection, and use CORP components, smECE, and granularity measures to
interpret the selected model.

## Evaluation checklist

1. Fit the classifier, calibrator, and method-selection rule on distinct data roles.
2. Compare methods with a proper score on observations outside every calibrator fit.
3. Reserve a final test set if the comparison itself selects a method.
4. Include a distorted positive control and an already calibrated negative control.
5. Stress ties, rare events, probability boundaries, small samples, and weighting when
   they occur in the intended application.
6. Report calibration, discrimination, and resolution separately rather than treating
   any one metric as a complete verdict.
