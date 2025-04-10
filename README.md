## Calibre: Advanced Calibration Models

[![PyPI version](https://img.shields.io/pypi/v/calibre.svg)](https://pypi.org/project/calibre/)
[![PyPI Downloads](https://static.pepy.tech/badge/calibre)](https://pepy.tech/projects/calibre)
[![Python Versions](https://img.shields.io/pypi/pyversions/calibre.svg)](https://pypi.org/project/calibre/)

Calibration is a critical step in deploying machine learning models. While techniques like isotonic regression have been standard for this task, they come with significant limitations:

1. **Loss of granularity**: Traditional isotonic regression often collapses many distinct probability values into a small number of unique values, which can be problematic  for decision-making.

2. **Rigid monotonicity**: Perfect monotonicity might not always be necessary or beneficial; small violations might be acceptable if they better preserve the information content of the original predictions.

Calibre addresses these limitations by implementing a suite of advanced calibration techniques that provide more nuanced control over model probability calibration. Its methods are designed to preserve granularity while still favoring a generally monotonic trend.

- **Nearly-isotonic regression**: Allows controlled violations of monotonicity to better preserve data granularity
- **I-spline calibration**: Uses monotonic splines for smooth calibration functions
- **Relaxed PAVA**: Ignores "small" violations based on percentile thresholds in the data
- **Regularized isotonic regression:** Adds L2 regularization to standard isotonic regression for smoother calibration curves while maintaining monotonicity.
- **Locally smoothed isotonic:** Applies Savitzky-Golay filtering to isotonic regression results to reduce the "staircase effect" while preserving monotonicity.
- **Adaptive smoothed isotonic:** Uses variable-sized smoothing windows based on data density to provide better detail in dense regions and smoother curves in sparse regions.

### Benchmark

The notebook has [benchmark results](benchmark.ipynb).

## Installation

```bash
pip install calibre
```

## Usage Examples

### Nearly Isotonic Regression with CVXPY

```python
import numpy as np
from calibre import NearlyIsotonicRegression

# Example data
np.random.seed(42)
x = np.sort(np.random.uniform(0, 1, 1000))
y_true = np.sin(2 * np.pi * x)
y = y_true + np.random.normal(0, 0.1, size=1000)

# Calibrate with different lambda values
cal_strict = NearlyIsotonicRegression(lam=10.0, method='cvx')
cal_strict.fit(x, y)
y_calibrated_strict = cal_strict.transform(x)

cal_relaxed = NearlyIsotonicRegression(lam=0.1, method='cvx')
cal_relaxed.fit(x, y)
y_calibrated_relaxed = cal_relaxed.transform(x)

# Now y_calibrated_relaxed will preserve more unique values
# while y_calibrated_strict will be more strictly monotonic
```

### I-Spline Calibration

```python
from calibre import ISplineCalibrator

# Smooth calibration using I-splines with cross-validation
cal_ispline = ISplineCalibrator(n_splines=10, degree=3, cv=5)
cal_ispline.fit(x, y)
y_ispline = cal_ispline.transform(x)
```

### Relaxed PAVA

```python
from calibre import RelaxedPAVA

# Calibrate allowing small violations (threshold at 10th percentile)
cal_relaxed_pava = RelaxedPAVA(percentile=10, adaptive=True)
cal_relaxed_pava.fit(x, y)
y_relaxed = cal_relaxed_pava.transform(x)

# This preserves more structure than standard isotonic regression
# while still correcting larger violations of monotonicity
```

### Regularized Isotonic

```python
from calibre import RegularizedIsotonicRegression

# Calibrate with L2 regularization
cal_reg_iso = RegularizedIsotonicRegression(alpha=0.1)
cal_reg_iso.fit(x, y)
y_reg_iso = cal_reg_iso.transform(x)
```

### Locally Smoothed Isotonic

```python
from calibre import SmoothedIsotonicRegression

# Apply local smoothing to reduce the “staircase” effect
cal_smoothed = SmoothedIsotonicRegression(window_length=7, poly_order=3, interp_method='linear')
cal_smoothed.fit(x, y)
y_smoothed = cal_smoothed.transform(x)
```

### Evaluating Calibration Quality

```python
from calibre import (
    mean_calibration_error, 
    binned_calibration_error, 
    correlation_metrics,
    unique_value_counts
)

# Calculate error metrics
mce = mean_calibration_error(y_true, y_calibrated)
bce = binned_calibration_error(y_true, y_calibrated, n_bins=10)

# Check correlations
corr = correlation_metrics(y_true, y_calibrated, x=x, y_orig=y)
print(f"Correlation with true values: {corr['spearman_corr_to_y_true']:.4f}")
print(f"Correlation with original predictions: {corr['spearman_corr_to_y_orig']:.4f}")

# Check granularity preservation
counts = unique_value_counts(y_calibrated, y_orig=y)
print(f"Original unique values: {counts['n_unique_y_orig']}")
print(f"Calibrated unique values: {counts['n_unique_y_pred']}")
print(f"Preservation ratio: {counts['unique_value_ratio']:.2f}")
```

## API Reference

### Calibration Functions

#### `nearly_isotonic_opt(x, y, lam=1.0)`
Implements nearly isotonic regression using convex optimization with CVXPY.

- **Parameters**:
  - `x`: Input features (for sorting)
  - `y`: Target values to calibrate
  - `lam`: Regularization parameter controlling the strength of monotonicity constraint
- **Returns**:
  - Calibrated values with controlled monotonicity

#### `nearly_isotonic_path(x, y, lam=1.0)`
Implements nearly isotonic regression using a path algorithm.

- **Parameters**:
  - `x`: Input features (for sorting)
  - `y`: Target values to calibrate
  - `lam`: Regularization parameter controlling the strength of monotonicity constraint
- **Returns**:
  - Calibrated values with controlled monotonicity

#### `ispline_calib(x, y, n_splines=10, degree=3, cv=5)`
Implements I-Spline calibration with cross-validation.

- **Parameters**:
  - `x`: Input features (predictions to calibrate)
  - `y`: Target values
  - `n_splines`: Number of spline basis functions
  - `degree`: Polynomial degree of spline basis functions
  - `cv`: Number of cross-validation folds
- **Returns**:
  - Calibrated values using monotonic splines

#### `relax_pava(x, y, percentile=10)`
Implements relaxed PAVA that ignores violations below a threshold.

- **Parameters**:
  - `x`: Input features (for sorting)
  - `y`: Target values to calibrate
  - `percentile`: Percentile of absolute differences to use as threshold
- **Returns**:
  - Calibrated values with relaxed monotonicity

#### `regularized_isotonic(x, y, alpha=0.1)`
Implements regularized isotonic regression using convex optimization with CVXPY.
- **Parameters**:
  - `x`: Input features (for sorting)
  - `y`: Target values to calibrate
  - `alpha`: Regularization strength parameter. Higher values result in smoother curves.
- **Returns**:
  - Calibrated values with regularized isotonic property

#### `locally_smoothed_isotonic(x, y, window_length=None, polyorder=3, interp_method='linear')`
Implements locally smoothed isotonic regression using Savitzky-Golay filtering.
- **Parameters**:
  - `x`: Input features (predictions to calibrate)
  - `y`: Target values
  - `window_length`: Window length for Savitzky-Golay filter. Should be odd. If None, automatically determined.
  - `polyorder`: Polynomial order for the Savitzky-Golay filter.
  - `interp_method`: Interpolation method to use ('linear', 'cubic', etc.)
- **Returns**:
  - Calibrated values with smoothed isotonic property

#### `adaptive_smoothed_isotonic(x, y, min_window=5, max_window=None, polyorder=3)`
Implements adaptive locally smoothed isotonic regression with variable window sizes.
- **Parameters**:
  - `x`: Input features (predictions to calibrate)
  - `y`: Target values
  - `min_window`: Minimum window length for smoothing
  - `max_window`: Maximum window length for smoothing. If None, automatically determined.
  - `polyorder`: Polynomial order for the filter
- **Returns**:
  - Calibrated values with adaptively smoothed isotonic property

### Evaluation Metrics

#### `mean_calibration_error(y_true, y_pred)`
Calculates the mean calibration error.

#### `binned_calibration_error(y_true, y_pred, x=None, n_bins=10)`
Calculates binned calibration error.

#### `correlation_metrics(y_true, y_pred, x=None, y_orig=None)`
Calculates Spearman's correlation metrics.

#### `unique_value_counts(y_pred, y_orig=None, precision=6)`
Counts unique values in predictions to assess granularity preservation.

## When to Use Which Method

- **nearly_isotonic_opt**: When you want precise control over the monotonicity/granularity trade-off and can afford the computational cost of convex optimization.

- **nearly_isotonic_path**: When you need an efficient algorithm for larger datasets that still provides control over monotonicity.

- **ispline_calib**: When you want a smooth calibration function rather than a step function, particularly for visualization and interpretation.

- **relax_pava**: When you want a simple, efficient approach that ignores "small" violations while correcting larger ones.

## References

1. Nearly-Isotonic Regression
Tibshirani, R. J., Hoefling, H., & Tibshirani, R. (2011).
Technometrics, 53(1), 54–61.
DOI:10.1198/TECH.2010.09281

2. A path algorithm for the fused lasso signal approximator.
Hoefling, H. (2010).
Journal of Computational and Graphical Statistics, 19(4), 984–1006.
DOI:10.1198/jcgs.2010.09208

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT
