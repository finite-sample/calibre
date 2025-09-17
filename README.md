## Calibre: Advanced Calibration Models

[![CI](https://github.com/gojiplus/calibre/workflows/CI/badge.svg)](https://github.com/gojiplus/calibre/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/calibre.svg)](https://pypi.org/project/calibre/)
[![PyPI Downloads](https://static.pepy.tech/badge/calibre)](https://pepy.tech/projects/calibre)
[![Python Versions](https://img.shields.io/pypi/pyversions/calibre.svg)](https://pypi.org/project/calibre/)
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

- **NearlyIsotonicRegression (method='cvx')**: When you want precise control over the monotonicity/granularity trade-off and can afford the computational cost of convex optimization.

- **NearlyIsotonicRegression (method='path')**: When you need an efficient algorithm for larger datasets that still provides control over monotonicity.

- **ISplineCalibrator**: When you want a smooth calibration function rather than a step function, particularly for visualization and interpretation.

- **RelaxedPAVA**: When you want a simple, efficient approach that ignores "small" violations while correcting larger ones.

- **RegularizedIsotonicRegression**: When you need smoother calibration curves with L2 regularization to prevent overfitting.

- **SmoothedIsotonicRegression**: When you want to reduce the "staircase effect" of standard isotonic regression while preserving monotonicity.

## References

1. Nearly-Isotonic Regression
Tibshirani, R. J., Hoefling, H., & Tibshirani, R. (2011).
Technometrics, 53(1), 54–61.
DOI:10.1198/TECH.2010.09281

2. A path algorithm for the fused lasso signal approximator.
Hoefling, H. (2010).
Journal of Computational and Graphical Statistics, 19(4), 984–1006.
DOI:10.1198/jcgs.2010.09208

## Development

### Prerequisites

- Python 3.10+
- pip

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/gojiplus/calibre.git
cd calibre

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=calibre --cov-report=html

# Run specific test file
pytest tests/test_calibration.py

# Run tests in verbose mode
pytest -xvs tests/
```

### Code Quality

```bash
# Format code with Black
black calibre/ tests/

# Sort imports
isort calibre/ tests/

# Lint code  
flake8 calibre/ tests/
```

### Building Documentation

```bash
# Run the benchmark notebook to generate results
jupyter nbconvert --to notebook --execute benchmark.ipynb
```

### Continuous Integration

This project uses GitHub Actions for CI/CD:

- **Tests**: Run on Python 3.10, 3.11, 3.12 across Ubuntu, macOS, and Windows
- **Code Quality**: Black, isort, and flake8 checks (informational)
- **Coverage**: Automated coverage reporting via Codecov
- **Package Building**: Validates package can be built and installed

The main requirement is that tests pass. Code quality checks are informational to help maintain consistency.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Guidelines

1. Fork the repository and create a feature branch
2. Add tests for any new functionality  
3. Ensure tests pass: `pytest tests/`
4. Run code formatting: `black calibre/ tests/` and `isort calibre/ tests/`
5. Update documentation as needed
6. Submit a pull request with a clear description of changes

## License

MIT

## 🔗 Adjacent Repositories

- [gojiplus/robust_pava](https://github.com/gojiplus/robust_pava) — Increase uniqueness in isotonic regression by ignoring small violations
- [gojiplus/pyppur](https://github.com/gojiplus/pyppur) — pyppur: Python Projection Pursuit Unsupervised (Dimension) Reduction To Min. Reconstruction Loss or DIstance DIstortion
- [gojiplus/rmcp](https://github.com/gojiplus/rmcp) — R MCP Server
- [gojiplus/bloomjoin](https://github.com/gojiplus/bloomjoin) — bloomjoin: An R package implementing Bloom filter-based joins for improved performance with large datasets.
- [gojiplus/incline](https://github.com/gojiplus/incline) — Estimate Trend at a Point in a Noisy Time Series
