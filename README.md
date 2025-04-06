## Calibre: Advanced Calibration Models

[![PyPI version](https://badge.fury.io/py/calibre.svg)](https://badge.fury.io/py/calibre)
[![PyPI downloads](https://img.shields.io/pypi/dm/calibre.svg)](https://pypi.org/project/calibre/)
[![Python Versions](https://img.shields.io/pypi/pyversions/calibre.svg)](https://pypi.org/project/calibre/)

Calibration is a critical step in deploying machine learning models. While techniques like isotonic regression have been standard for this task, they come with significant limitations:

1. **Loss of granularity**: Traditional isotonic regression often collapses many distinct probability values into a small number of unique values, which can be problematic  for decision-making.

2. **Rigid monotonicity**: Perfect monotonicity might not always be necessary or beneficial; small violations might be acceptable if they better preserve the information content of the original predictions.

Calibre addresses these issues by providing a suite of advanced calibration techniques that allow for more nuanced control over the calibration process, including:

- **Nearly-isotonic regression**: Allows controlled violations of monotonicity to better preserve data granularity
- **I-spline calibration**: Uses monotonic splines for smooth calibration functions
- **Relaxed PAVA**: Ignores "small" violations based on percentile thresholds in the data

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
from calibre import nearly_isotonic_opt

# Example data
np.random.seed(42)
x = np.sort(np.random.uniform(0, 1, 1000))
y_true = np.sin(2 * np.pi * x)
y = y_true + np.random.normal(0, 0.1, size=1000)

# Apply calibration with control over monotonicity strength
# Higher lambda = stronger monotonicity enforcement
y_calibrated_strict = nearly_isotonic_opt(x, y, lam=10.0)
y_calibrated_relaxed = nearly_isotonic_opt(x, y, lam=0.1)

# Now y_calibrated_relaxed will preserve more unique values
# while y_calibrated_strict will be more strictly monotonic
```

### Relaxed PAVA with Dynamic Threshold

```python
from calibre import relax_pava

# Allow violations below the 10th percentile of absolute differences
y_calibrated = relax_pava(x, y, percentile=10)

# This preserves more structure than standard isotonic regression
# while still correcting larger violations of monotonicity
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
