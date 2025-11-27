## Calibre: Advanced Calibration Models

[![CI](https://github.com/finite-sample/calibre/workflows/CI/badge.svg)](https://github.com/finite-sample/calibre/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/calibre/)
[![PyPI version](https://img.shields.io/pypi/v/calibre.svg)](https://pypi.org/project/calibre/)
[![PyPI Downloads](https://static.pepy.tech/badge/calibre)](https://pepy.tech/projects/calibre)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/finite-sample/calibre/main/pyproject.toml&query=$.project.requires-python&label=Python)](https://github.com/finite-sample/calibre)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Calibration is a critical step in deploying machine learning models. While techniques like isotonic regression have been standard for this task, they come with significant limitations:

1. **Loss of granularity**: Traditional isotonic regression often collapses many distinct probability values into a small number of unique values, which can be problematic  for decision-making.

2. **Rigid monotonicity**: Perfect monotonicity might not always be necessary or beneficial; small violations might be acceptable if they better preserve the information content of the original predictions.

Calibre addresses these limitations by implementing a suite of advanced calibration techniques that provide more nuanced control over model probability calibration. Its methods are designed to preserve granularity while still favoring a generally monotonic trend.

- **Nearly-isotonic regression**: Allows controlled violations of monotonicity to better preserve data granularity
- **I-spline calibration**: Uses monotonic splines for smooth calibration functions
- **Relaxed PAVA**: Ignores "small" violations based on percentile thresholds in the data
- **Regularized isotonic regression:** Adds L2 regularization to standard isotonic regression for smoother calibration curves while maintaining monotonicity.
- **Locally smoothed isotonic:** Applies Savitzky-Golay filtering to isotonic regression results to reduce the "staircase effect" while preserving monotonicity.

### Benchmark

The notebook has [benchmark results](examples/benchmark.ipynb).

## Installation

```bash
pip install calibre
```

## Usage Examples

### Nearly Isotonic Regression with CVXPY

```python
import numpy as np
from calibre import NearlyIsotonicCalibrator

# Example data: model predictions and true binary outcomes
np.random.seed(42)
y_pred = np.sort(np.random.uniform(0, 1, 1000))  # Model probability predictions
y_true = np.random.binomial(1, y_pred, 1000)     # True binary outcomes

# Calibrate with different lambda values
cal_strict = NearlyIsotonicCalibrator(lam=10.0)
cal_strict.fit(y_pred, y_true)
y_calibrated_strict = cal_strict.transform(y_pred)

cal_relaxed = NearlyIsotonicCalibrator(lam=0.1)
cal_relaxed.fit(y_pred, y_true)
y_calibrated_relaxed = cal_relaxed.transform(y_pred)

# Now y_calibrated_relaxed will preserve more unique values
# while y_calibrated_strict will be more strictly monotonic
```

### I-Spline Calibration

```python
from calibre import SplineCalibrator

# Smooth calibration using I-splines with cross-validation
cal_ispline = SplineCalibrator(n_splines=10, degree=3, cv=5)
cal_ispline.fit(y_pred, y_true)
y_calibrated_ispline = cal_ispline.transform(y_pred)
```

### Relaxed PAVA

```python
from calibre import RelaxedPAVACalibrator

# Calibrate allowing small violations (threshold at 10th percentile)
cal_relaxed_pava = RelaxedPAVACalibrator(percentile=10, adaptive=True)
cal_relaxed_pava.fit(y_pred, y_true)
y_calibrated_relaxed = cal_relaxed_pava.transform(y_pred)

# This preserves more structure than standard isotonic regression
# while still correcting larger violations of monotonicity
```

### Regularized Isotonic

```python
from calibre import RegularizedIsotonicCalibrator

# Calibrate with L2 regularization
cal_reg_iso = RegularizedIsotonicCalibrator(alpha=0.1)
cal_reg_iso.fit(y_pred, y_true)
y_calibrated_reg = cal_reg_iso.transform(y_pred)
```

### Locally Smoothed Isotonic

```python
from calibre import SmoothedIsotonicCalibrator

# Apply local smoothing to reduce the "staircase" effect
cal_smoothed = SmoothedIsotonicCalibrator(window_length=7, poly_order=3, interp_method='linear')
cal_smoothed.fit(y_pred, y_true)
y_calibrated_smooth = cal_smoothed.transform(y_pred)
```

### 🔬 Plateau Diagnostics (New in v0.4.1)

Distinguish between **noise-based flattening** (good) and **limited-data flattening** (bad) in isotonic regression:

```python
from calibre import IsotonicCalibrator, run_plateau_diagnostics

# Automatic diagnostics with isotonic regression
cal = IsotonicCalibrator(enable_diagnostics=True)
cal.fit(y_pred, y_true)
y_calibrated = cal.transform(y_pred)

# Get human-readable diagnostic summary
if cal.has_diagnostics():
    print(cal.diagnostic_summary())

# Advanced analysis with standalone function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(y_pred, y_true, test_size=0.3)

results = run_plateau_diagnostics(X_train, y_train, y_calibrated)
print(f"Found {len(results)} plateau(s)")
```

**Key Diagnostic Methods:**
- **Bootstrap Tie Stability**: Measures plateau consistency across resamples
- **Conditional AUC**: Tests discrimination ability among tied pairs
- **Minimum Detectable Difference**: Statistical power analysis at boundaries
- **Progressive Sampling**: How diversity changes with sample size
- **Local Slope Testing**: Uses smooth fits to test genuine flatness

**Plateau Classifications:**
- **Supported**: High stability + low conditional AUC + flat slope → genuine plateaus
- **Limited-data**: Low stability + high conditional AUC + positive slope → artifacts
- **Inconclusive**: Mixed evidence requiring further investigation

```python
# Advanced diagnostic metrics
from calibre.metrics import (
    tie_preservation_score,
    plateau_quality_score, 
    calibration_diversity_index,
    progressive_sampling_diversity
)

# Measure tie preservation quality
tie_score = tie_preservation_score(y_pred, y_calibrated)

# Overall plateau quality
quality = plateau_quality_score(y_pred, y_true, y_calibrated)

# Granularity preservation
diversity = calibration_diversity_index(y_calibrated)

# Sample size analysis
sizes, diversities = progressive_sampling_diversity(y_pred, y_true)
```

### Evaluating Calibration Quality

```python
from calibre.metrics import (
    mean_calibration_error, 
    binned_calibration_error, 
    correlation_metrics,
    unique_value_counts
)

# Calculate error metrics
mce = mean_calibration_error(y_true, y_calibrated_strict)
bce = binned_calibration_error(y_true, y_calibrated_strict, n_bins=10)

# Check correlations
corr = correlation_metrics(y_true, y_calibrated_strict, y_orig=y_pred)
print(f"Correlation with true values: {corr['spearman_corr_to_y_true']:.4f}")
print(f"Correlation with original predictions: {corr['spearman_corr_orig_to_calib']:.4f}")

# Check granularity preservation
counts = unique_value_counts(y_calibrated_strict, y_orig=y_pred)
print(f"Original unique values: {counts['n_unique_y_orig']}")
print(f"Calibrated unique values: {counts['n_unique_y_pred']}")
print(f"Preservation ratio: {counts['unique_value_ratio']:.2f}")
```

### Evaluation Metrics

#### `mean_calibration_error(y_true, y_pred)`
Calculates the mean calibration error.

#### `binned_calibration_error(y_true, y_pred, x=None, n_bins=10, strategy='uniform', return_details=False)`
Calculates binned calibration error using uniform or quantile binning strategies.

#### `expected_calibration_error(y_true, y_pred, n_bins=10)`
Calculates the Expected Calibration Error (ECE), a weighted average of calibration errors across bins.

#### `maximum_calibration_error(y_true, y_pred, n_bins=10)`
Calculates the Maximum Calibration Error (MCE), the worst-case calibration error across all bins.

#### `calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')`
Generates calibration curve data points for plotting reliability diagrams.

#### `correlation_metrics(y_true, y_pred, x=None, y_orig=None)`
Calculates Spearman's correlation metrics.

#### `unique_value_counts(y_pred, y_orig=None, precision=6)`
Counts unique values in predictions to assess granularity preservation.

## When to Use Which Method

### Calibration Methods

- **NearlyIsotonicCalibrator**: When you want precise control over the monotonicity/granularity trade-off using convex optimization.

- **SplineCalibrator**: When you want a smooth calibration function rather than a step function, particularly for visualization and interpretation.

- **RelaxedPAVACalibrator**: When you want a simple, efficient approach that ignores "small" violations while correcting larger ones.

- **RegularizedIsotonicCalibrator**: When you need smoother calibration curves with L2 regularization to prevent overfitting.

- **SmoothedIsotonicCalibrator**: When you want to reduce the "staircase effect" of standard isotonic regression while preserving monotonicity.

### Plateau Diagnostics

- **IsotonicCalibrator with diagnostics**: Always use when applying isotonic regression to automatically detect and classify plateaus.

- **run_plateau_diagnostics()**: Use for comprehensive plateau analysis of calibration results.

- **Diagnostic Metrics**: Use `tie_preservation_score()`, `plateau_quality_score()`, and `progressive_sampling_diversity()` to quantitatively assess calibration quality beyond traditional error metrics.

**Decision Framework:**
1. **Run diagnostics first** with `IsotonicCalibrator(enable_diagnostics=True)`
2. **If limited-data plateaus detected**: Consider `NearlyIsotonicCalibrator`, `RegularizedIsotonicCalibrator`, or collecting more calibration data
3. **If supported plateaus**: Standard isotonic regression is appropriate
4. **If inconclusive**: Cross-validate between strict and soft methods

## References

1. Nearly-Isotonic Regression
Tibshirani, R. J., Hoefling, H., & Tibshirani, R. (2011).
Technometrics, 53(1), 54–61.
DOI:10.1198/TECH.2010.09281

2. A path algorithm for the fused lasso signal approximator.
Hoefling, H. (2010).
Journal of Computational and Graphical Statistics, 19(4), 984–1006.
DOI:10.1198/jcgs.2010.09208


**📖 Live Documentation**: https://finite-sample.github.io/calibre/

## License

MIT

## 🔗 Adjacent Repositories

- [gojiplus/robust_pava](https://github.com/gojiplus/robust_pava) — Increase uniqueness in isotonic regression by ignoring small violations
- [gojiplus/pyppur](https://github.com/gojiplus/pyppur) — pyppur: Python Projection Pursuit Unsupervised (Dimension) Reduction To Min. Reconstruction Loss or DIstance DIstortion
- [gojiplus/rmcp](https://github.com/gojiplus/rmcp) — R MCP Server
- [gojiplus/bloomjoin](https://github.com/gojiplus/bloomjoin) — bloomjoin: An R package implementing Bloom filter-based joins for improved performance with large datasets.
- [gojiplus/incline](https://github.com/gojiplus/incline) — Estimate Trend at a Point in a Noisy Time Series
