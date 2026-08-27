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

# A worked calibrator comparison

No calibrator is best for every population. This tutorial compares several methods
on three known distortions and keeps calibration fitting separate from evaluation.
It is a worked example, not a portable speed benchmark or a universal ranking.

The comparison uses held-out Brier score as the selection criterion because it is a
proper score. CORP miscalibration (MCB), discrimination (DSC), smECE, and the number
of distinct predictions explain why the proper score changed.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from calibre import (
    CenteredIsotonicCalibrator,
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SplineCalibrator,
    brier_score,
    score_decomposition,
    smooth_calibration_error,
)

RANDOM_STATE = 42
random_generator = np.random.default_rng(RANDOM_STATE)
```

## Generate known distortions

The population event probability is sampled first. Outcomes come from that
probability, while forecasts apply a known transformation on the log-odds scale:

- overconfident forecasts multiply the true log odds by 1.8;
- underconfident forecasts multiply them by 0.55;
- shifted forecasts add 0.8 to the true log odds.

These are monotone distortions. A separate nonmonotone design would be needed to
evaluate methods intended to permit ranking violations.

```{code-cell} ipython3
def probability(log_odds):
    """Map finite log odds to probabilities."""
    return 1.0 / (1.0 + np.exp(-log_odds))


def make_scenario(scale=1.0, shift=0.0, n_observations=3000):
    """Draw outcomes and forecasts under a known log-odds distortion."""
    true_probability = random_generator.beta(2.0, 2.0, n_observations)
    outcomes = random_generator.binomial(1, true_probability).astype(float)
    true_log_odds = np.log(true_probability / (1.0 - true_probability))
    forecasts = probability(scale * true_log_odds + shift)
    return forecasts, outcomes


scenarios = {
    "overconfident": make_scenario(scale=1.8),
    "underconfident": make_scenario(scale=0.55),
    "shifted": make_scenario(shift=0.8),
}
```

## Define comparable candidates

The candidates below are general-purpose calibrators. Relaxed PAVA needs an explicit
scientific bound, so the example states one. CDI isotonic is not included because it
requires application-specific decision thresholds on the forecast scale; dropping
arbitrary thresholds into a generic benchmark would not evaluate its intended use.

```{code-cell} ipython3
def candidate_calibrators():
    """Return fresh calibrators for one scenario."""
    return {
        "isotonic": IsotonicCalibrator(),
        "centered isotonic": CenteredIsotonicCalibrator(),
        "nearly isotonic": NearlyIsotonicCalibrator(),
        "spline": SplineCalibrator(),
        "relaxed PAVA (-0.01)": RelaxedPAVACalibrator(min_increment=-0.01),
    }
```

## Fit on calibration data and evaluate elsewhere

Every candidate sees the same calibration rows and the same held-out rows. Errors are
allowed to stop the documentation build; silently assigning an infinite score would
hide a broken calibrator.

```{code-cell} ipython3
def evaluation_row(scenario, method, outcomes, predictions):
    """Summarize one held-out prediction vector."""
    decomposition = score_decomposition(outcomes, predictions)
    return {
        "scenario": scenario,
        "method": method,
        "brier": brier_score(outcomes, predictions),
        "miscalibration": decomposition["miscalibration"],
        "discrimination": decomposition["discrimination"],
        "smece": smooth_calibration_error(outcomes, predictions),
        "distinct_predictions": np.unique(predictions).size,
    }


rows = []
for scenario_name, (forecasts, outcomes) in scenarios.items():
    (
        calibration_forecasts,
        test_forecasts,
        calibration_outcomes,
        test_outcomes,
    ) = train_test_split(
        forecasts,
        outcomes,
        test_size=0.5,
        stratify=outcomes,
        random_state=RANDOM_STATE,
    )

    rows.append(
        evaluation_row(
            scenario_name,
            "uncalibrated",
            test_outcomes,
            test_forecasts,
        )
    )

    for method_name, calibrator in candidate_calibrators().items():
        calibrator.fit(calibration_forecasts, calibration_outcomes)
        calibrated_predictions = calibrator.transform(test_forecasts)
        rows.append(
            evaluation_row(
                scenario_name,
                method_name,
                test_outcomes,
                calibrated_predictions,
            )
        )

results = pd.DataFrame(rows)
results.assign(
    brier=results["brier"].round(4),
    miscalibration=results["miscalibration"].round(4),
    discrimination=results["discrimination"].round(4),
    smece=results["smece"].round(4),
)
```

## Read the comparison

Lower Brier is the decision rule. MCB and smECE diagnose calibration, DSC measures
useful separation, and distinct predictions describe resolution. Neither a low
calibration error nor a large number of distinct predictions is sufficient by itself.

```{code-cell} ipython3
for scenario_name in scenarios:
    scenario_results = results[results["scenario"] == scenario_name]
    ordered = scenario_results.sort_values("brier")
    print(f"\n{scenario_name}")
    for row in ordered.itertuples():
        print(
            f"  {row.method:22s} Brier {row.brier:.4f}  "
            f"MCB {row.miscalibration:.4f}  DSC {row.discrimination:.4f}  "
            f"distinct {row.distinct_predictions}"
        )
```

The lowest value in each block describes this simulated split only. Repeating the
simulation with new populations, or using repeated validation splits on real data,
is necessary before making a stable method choice.

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for axis, column, title, log_scale in (
    (axes[0], "brier", "Held-out Brier score", False),
    (axes[1], "miscalibration", "CORP miscalibration", False),
    (axes[2], "distinct_predictions", "Distinct held-out predictions", True),
):
    pivoted = results.pivot(index="scenario", columns="method", values=column)
    pivoted.plot.bar(ax=axis, legend=False)
    axis.set_title(title)
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=0)
    if log_scale:
        axis.set_yscale("log")

handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, loc="outside lower center", ncol=3)
figure.tight_layout(rect=(0, 0.15, 1, 1))
plt.show()
```

## Choosing a method in an application

Use the same sequence on real data:

1. Fit each candidate on calibration observations the classifier did not train on.
2. Compare candidates on separate validation observations with log loss or Brier
   score.
3. Inspect held-out calibration error, discrimination, and resolution to understand
   the choice.
4. After choosing, evaluate once on a final test set that played no role in fitting or
   selection.

Isotonic and centered isotonic fit once with no tuning search. The automatic nearly
isotonic and spline fits cross-validate their regularization choices, so they generally
cost more. Runtime depends on data size, candidate grids, software versions, and
hardware; measure it on the intended workload rather than copying a timing from this
tutorial.
