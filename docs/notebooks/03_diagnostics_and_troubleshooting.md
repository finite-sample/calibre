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

# Plateau Diagnostics Demo

This notebook demonstrates the structural diagnostics for flat regions in an isotonic calibration curve.

## Overview

A plateau says that a fitted calibrator pooled a range of adjacent scores. It does not, by itself, say whether those scores truly carry the same risk. `run_plateau_diagnostics` reports each plateau's extent and supporting sample count. Use held-out scoring to decide whether a less restrictive calibrator generalizes better.

+++

## 1. Generate Synthetic Data

Let's create two scenarios:
- **Scenario A**: Data with genuine flat regions (noise-based flattening)
- **Scenario B**: Data with smooth trends but small sample size (limited-data flattening)

```{code-cell} ipython3
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from calibre import (
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    SplineCalibrator,
    brier_score,
    tie_preservation_score,
)
from calibre.diagnostics import run_plateau_diagnostics

np.random.seed(42)
```

```{code-cell} ipython3
def create_genuine_plateau_data(n=200, noise_level=0.05):
    """Create data with genuine flat regions."""
    X = np.sort(np.random.uniform(0, 1, n))

    # Create true probabilities with intentional flat regions
    y_true = np.zeros(n)
    y_true[: n // 4] = 0.1  # Flat low region
    y_true[n // 4 : n // 2] = np.linspace(0.1, 0.4, n // 4)  # Rising
    y_true[n // 2 : 3 * n // 4] = 0.4  # Flat middle region
    y_true[3 * n // 4 :] = np.linspace(0.4, 0.8, n // 4)  # Rising

    # Add small amount of noise
    y_true += np.random.normal(0, noise_level, n)
    y_true = np.clip(y_true, 0, 1)

    # Generate binary outcomes
    y_binary = np.random.binomial(1, y_true)

    return X, y_binary, y_true


def create_smooth_small_data(n=50):
    """Create smooth data with small sample size."""
    X = np.sort(np.random.uniform(0, 1, n))

    # Smooth sigmoid-like curve
    y_true = 1 / (1 + np.exp(-8 * (X - 0.5)))

    # Generate binary outcomes
    y_binary = np.random.binomial(1, y_true)

    return X, y_binary, y_true


# Generate both scenarios
X_genuine, y_genuine, y_true_genuine = create_genuine_plateau_data()
X_small, y_small, y_true_small = create_smooth_small_data()

print(f"Genuine plateau data: {len(X_genuine)} samples")
print(f"Small sample data: {len(X_small)} samples")
```

## 2. Basic Isotonic Regression with Diagnostics

Let's start with the simple wrapper that automatically runs diagnostics:

```{code-cell} ipython3
# Scenario A: Genuine plateaus (updated for v0.4.1)
print("=== Scenario A: Genuine Plateau Data ===")
cal_genuine = IsotonicCalibrator(enable_diagnostics=True)
cal_genuine.fit(X_genuine, y_genuine)

print("\nDiagnostic Summary:")
if cal_genuine.has_diagnostics():
    print(cal_genuine.diagnostic_summary())
else:
    print("No diagnostics available")

# Get calibrated predictions
y_cal_genuine = cal_genuine.transform(X_genuine)
```

```{code-cell} ipython3
# Scenario B: Small sample data (updated for v0.4.1)
print("=== Scenario B: Small Sample Data ===")
cal_small = IsotonicCalibrator(enable_diagnostics=True)
cal_small.fit(X_small, y_small)

print("\nDiagnostic Summary:")
if cal_small.has_diagnostics():
    print(cal_small.diagnostic_summary())
else:
    print("No diagnostics available")

y_cal_small = cal_small.transform(X_small)
```

## 3. Advanced Diagnostic Analysis

For more detailed analysis, call `run_plateau_diagnostics` directly:

```{code-cell} ipython3
# Split data for more thorough analysis (train/test)
X_train, X_test, y_train, y_test = train_test_split(
    X_genuine, y_genuine, test_size=0.3, random_state=42
)

# First fit calibrator and get predictions
cal = IsotonicCalibrator()
cal.fit(X_train, y_train)
y_cal_train = cal.transform(X_train)

# Run plateau diagnostics on the calibrated results. The diagnosis is
# structural -- it reads the calibrated curve, not the labels -- so the
# true outcomes are not an argument.
results = run_plateau_diagnostics(X_train, y_cal_train)

print("Detailed diagnostic results:")
print(f"Detected {results['n_plateaus']} plateau regions")

if results["n_plateaus"] > 0:
    print("\nPlateau details:")
    for i, plateau in enumerate(results["plateaus"]):
        print(f"  Plateau {i + 1}:")
        if "input_score_range" in plateau:
            print(
                f"    Input score range: [{plateau['input_score_range'][0]:.3f}, "
                f"{plateau['input_score_range'][1]:.3f}]"
            )
        if "calibrated_value" in plateau:
            print(f"    Calibrated value: {plateau['calibrated_value']:.3f}")
        if "n_observations" in plateau:
            print(f"    Observations: {plateau['n_observations']}")
        if "support" in plateau:
            print(f"    Support: {plateau['support']}")

if results["warnings"]:
    print("\nWarnings:")
    for warning in results["warnings"]:
        print(f"  ⚠️  {warning}")
```

## 4. Resolution Diagnostics

Count the distinct fitted values directly, and measure how many new ties calibration introduced:

```{code-cell} ipython3
# Compare original vs calibrated predictions
iso_basic = IsotonicRegression()
iso_basic.fit(X_genuine, y_genuine)
y_cal_basic = iso_basic.transform(X_genuine)

# Direct, structural summaries
tie_score = tie_preservation_score(X_genuine, y_cal_basic)
print(f"Distinct values: {np.unique(y_cal_basic).size}/{y_cal_basic.size}")
print(f"Tie preservation score: {tie_score:.3f}")
```

## 5. Comparison with Alternative Methods

Fit on the training split and compare strict isotonic regression with softer alternatives on held-out outcomes:

```{code-cell} ipython3
# Fit different calibrators
iso_strict = IsotonicCalibrator()
iso_nearly = NearlyIsotonicCalibrator(lam=1.0)
iso_reg = SplineCalibrator(alpha=0.1)

calibrators = {
    "Strict Isotonic": iso_strict.fit(X_train, y_train),
    "Nearly Isotonic": iso_nearly.fit(X_train, y_train),
    "Spline (fixed alpha)": iso_reg.fit(X_train, y_train),
}

# Compare held-out score and retained resolution
print("Method comparison:")
for name, cal in calibrators.items():
    y_pred = cal.transform(X_test)
    error = brier_score(y_test, y_pred)
    n_unique = len(np.unique(y_pred))

    print(f"  {name}:")
    print(f"    Held-out Brier score: {error:.3f}")
    print(f"    Distinct values: {n_unique}/{len(y_pred)}")
```

## 6. Visualization (if matplotlib available)

Let's create some visualizations to better understand the diagnostics:

+++

## 7. Acting on the Diagnostics

A sparse plateau is a prompt to inspect the score range, not proof that the plateau is wrong. Compare candidate calibrators on held-out Brier or log loss, examine how much resolution each retains, and collect more calibration data when the decision-relevant range has little support.

```{code-cell} ipython3
sparse = [
    plateau
    for plateau in results["plateaus"]
    if plateau["support"] in {"sparse", "very_sparse"}
]
print(f"{len(sparse)} of {results['n_plateaus']} plateaus have limited support.")
print("Choose among calibrators using the held-out comparison above.")
```

## 8. Summary and Best Practices

### What the diagnostics actually tell you

`run_plateau_diagnostics` is a **structural** check. It reports each flat region's score range and sample count. It does not use outcomes and cannot determine whether a plateau is statistically justified.

### Best Practices

1. **Run diagnostics** whenever you use isotonic regression on scores you care about ranking.
2. **Treat sparse plateaus as a data question**, not a modeling conclusion.
3. **Compare against a granularity-preserving method** such as `CenteredIsotonicCalibrator` or `SplineCalibrator`, and choose on held-out log loss or Brier score.
4. **Document the choice**, including the held-out numbers that justified it.

```{code-cell} ipython3
print("🎉 Plateau diagnostics demo completed!")
print("\nKey takeaways:")
print("1. Plateau diagnostics describe fitted structure, not truth")
print("2. Sparse plateaus identify score ranges needing scrutiny")
print("3. Held-out proper scores decide whether an alternative helps")
```
