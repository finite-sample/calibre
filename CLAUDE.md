# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Calibre is a Python package for advanced probability calibration techniques in machine learning. It provides alternative calibration methods to traditional isotonic regression that better preserve probability granularity while maintaining monotonicity constraints.

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=calibre

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

# Type checking
mypy calibre/

# Lint (if using flake8 or similar - check project for specific linter)
```

### Build and Distribution
```bash
# Build package
python -m build

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Code Architecture

### Core Modules

**calibre/calibration.py**: Contains all calibration classes:
- `BaseCalibrator`: Abstract base class for all calibrators
- `NearlyIsotonicRegression`: Allows controlled violations of monotonicity (CVXPY-based)
- `ISplineCalibrator`: Smooth calibration using I-splines with cross-validation
- `RelaxedPAVA`: Ignores small violations based on percentile thresholds
- `RegularizedIsotonicRegression`: L2 regularized isotonic regression
- `SmoothedIsotonicRegression`: Applies Savitzky-Golay filtering to reduce staircase effects (supports both fixed and adaptive window sizing)

**calibre/metrics.py**: Evaluation metrics for calibration quality:
- `mean_calibration_error()`: Basic calibration error
- `binned_calibration_error()`: Binned approach with uniform/quantile strategies
- `expected_calibration_error()`: Expected calibration error (ECE)
- `maximum_calibration_error()`: Maximum calibration error (MCE)
- `brier_score()`: Brier score computation
- `calibration_curve()`: Calibration curve generation
- `correlation_metrics()`: Spearman correlations
- `unique_value_counts()`: Granularity preservation metrics

**calibre/utils.py**: Utility functions:
- `check_arrays()`: Input validation
- `sort_by_x()`: Sorting utilities
- `create_bins()`, `bin_data()`: Binning operations

### Key Dependencies
- **numpy, scipy**: Core numerical computing
- **scikit-learn**: Base classes and isotonic regression
- **cvxpy**: Convex optimization for nearly-isotonic regression
- **pandas**: Data manipulation
- **matplotlib**: Visualization (examples)

### Design Patterns
- All calibrators inherit from `BaseCalibrator` which extends sklearn's `BaseEstimator` and `TransformerMixin`
- Consistent `.fit(X, y)` and `.transform(X)` API following sklearn conventions
- Input validation through `check_arrays()` utility
- Type hints throughout codebase (Python 3.10+)

## Testing Structure
- Tests are in `tests/` directory
- Main test file: `tests/test_calibration.py`
- Uses pytest fixtures for test data generation
- Coverage reporting via pytest-cov

## Configuration
- **pyproject.toml**: Modern Python packaging configuration
- Tool configurations for Black, isort, mypy included in pyproject.toml
- Python 3.10+ required
- Development dependencies defined in `[project.optional-dependencies.dev]`

## Benchmarking
- **benchmark.ipynb**: Jupyter notebook with performance benchmarks comparing different calibration methods
- Contains visual comparisons and quantitative metrics for each calibrator