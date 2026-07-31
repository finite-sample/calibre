# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Calibre is a Python package for advanced probability calibration techniques in machine learning. It provides alternative calibration methods to traditional isotonic regression that better preserve probability granularity while maintaining monotonicity constraints.

**Current Version**: 0.7.0 (correctness release — see CHANGELOG)

### Import Structure
```python
# Import base classes
from calibre import BaseCalibrator, MonotonicMixin

# Import calibrators
from calibre import (
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    SplineCalibrator,
    RelaxedPAVACalibrator,
    RegularizedIsotonicCalibrator,
    SmoothedIsotonicCalibrator,
)

# Import standalone diagnostic functions
from calibre.diagnostics import run_plateau_diagnostics, detect_plateaus
```

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=calibre

# Run specific test file
pytest tests/test_calibrators_unit.py

# Run tests in verbose mode
pytest -xvs tests/
```

### Code Quality
```bash
# Format and lint code with ruff (unified tool)
ruff format calibre/ tests/
ruff check calibre/ tests/

# Fix automatically fixable issues
ruff check --fix calibre/ tests/

# Type checking
mypy calibre/
```

### Dependency Management
```bash
# Install/sync all dependencies including dev dependencies
uv sync --all-extras --dev

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --group dev package-name

# Update dependencies (regenerate uv.lock)
uv lock

# Update all dependencies to latest versions
uv lock --upgrade
```

### Build and Distribution
```bash
# Build package
uv build

# Install in development mode (after uv sync)
uv sync --all-extras --dev
```

## Development Workflow

### Dependency Management Best Practices

**IMPORTANT**: Always commit `uv.lock` changes when modifying dependencies. The CI/CD pipeline validates that `uv.lock` is consistent with `pyproject.toml`.

**When adding/updating dependencies:**
1. Run `uv add package-name` or `uv add --group dev package-name` 
2. This automatically updates both `pyproject.toml` and `uv.lock`
3. Commit both files together
4. CI will validate the lock file is up-to-date

**When pulling changes with new dependencies:**
1. Run `uv sync --all-extras --dev` to install new dependencies
2. This uses the exact versions specified in `uv.lock`

**Periodic dependency updates:**
1. Run `uv lock --upgrade` to update to latest compatible versions
2. Test thoroughly as this may introduce breaking changes
3. Commit the updated `uv.lock`

### CI/CD Integration

The CI pipeline uses `uv sync --locked` to ensure:
- Consistent dependency versions across all environments
- Fast builds with dependency caching based on `uv.lock` hash
- Deterministic behavior between local development and CI

## Code Architecture

### Core Modules

**calibre/base.py**: Base classes and mixins for all calibrators:
- `BaseCalibrator`: Abstract base class following sklearn transformer interface.
  Subclasses implement `_fit_impl(X, y, sample_weight)`; `fit()` is a template method
  that also runs diagnostics. Diagnostics live here, not in a separate mixin.
- `MonotonicMixin`: Utility mixin for monotonicity checking and enforcement

**calibre/calibrators/**: Modular calibrator implementations:
- `IsotonicCalibrator`: Standard isotonic regression (wraps sklearn)
- `CenteredIsotonicCalibrator`: Centered isotonic regression — collapses PAVA's flat
  blocks to their centroid and interpolates. The recommended default.
- `NearlyIsotonicCalibrator`: Penalises rather than forbids monotonicity violations.
  Two exact solvers: `method="path"` (default, pure NumPy) and `method="cvx"` (CVXPY).
  Note `lam` is 2x the source paper's lambda.
- `SplineCalibrator`: Monotone I-spline fit; CV picks `(n_knots, alpha)` on log-loss
- `RelaxedPAVACalibrator`: Bounds each adjacent increment — `epsilon` permits small
  decreases, `min_slope` forbids plateaus. Solved by shift-to-PAVA in O(n).
- `RegularizedIsotonicCalibrator`: Monotone spline with a second-difference (curvature)
  penalty. NOT ridge, and `alpha=0` is not isotonic regression.
- `SmoothedIsotonicCalibrator`: Savitzky-Golay smoothing of an isotonic fit (legacy)
- `CDIIsotonicCalibrator`: Cost- and data-informed isotonic (research)

**calibre/_core.py**: Shared numerical primitives. Every calibrator is built from
these rather than reimplementing isotonic machinery locally:
- `weighted_pava`, `monotone_projection`, `cumulative_max`
- `aggregate_ties` (pool tied scores — required before any interpolation)
- `shift_to_pava` (increment lower bounds via cumulative shift)
- `nearly_isotonic_path` (exact solution path)
- `collapse_blocks` (the geometric step behind CIR)
- `monotone_spline_basis` / `MonotoneSplineBasis`, `fit_monotone_spline`
- `PiecewiseLinear`, `StepFunction` (fitted-function objects, built on `np.interp`)

All are pinned against R reference implementations by `tests/test_r_reference.py`.

**calibre/diagnostics.py**: Standalone plateau diagnostic functions:
- `run_plateau_diagnostics()`: Returns a dict with `n_plateaus`, `plateaus`, `warnings`
- `detect_plateaus()`: Detect flat regions in calibration curves
- `analyze_plateau_simple()`: Describe one plateau (`x_range`, `value`, `n_samples`,
  `sample_density`)
- `diversity_learning_curve()`: How granularity changes with sample size

Note: this module is a stub relative to what earlier CHANGELOGs promised. Bootstrap
tie stability, conditional AUC among tied pairs, minimum detectable difference, and the
supported/limited-data/inconclusive classifier do not exist. `n_bootstraps` and
`random_state` on `run_plateau_diagnostics` are accepted and ignored.

**calibre/metrics.py**: Evaluation metrics for calibration quality:
- `mean_calibration_error()`: Bias, |E[p] - E[y]|. Changed in 0.7.0; it used to
  return mean absolute error, which is not a calibration error.
- `binned_calibration_error()`: Binned approach with uniform/quantile strategies
- `expected_calibration_error()`: Expected calibration error (ECE)
- `maximum_calibration_error()`: Maximum calibration error (MCE)
- `brier_score()`: Brier score computation
- `calibration_curve()`: Calibration curve generation
- `correlation_metrics()`: Spearman correlations
- `unique_value_counts()`: Granularity preservation metrics
- `tie_preservation_score()`: Measures how well ties are preserved during calibration
- `plateau_quality_score()`: Overall quality assessment of plateau regions
- `calibration_diversity_index()`: Measures granularity preservation
- `progressive_sampling_diversity()`: Analyzes how diversity changes with sample size

**calibre/utils/**: A package, not a module. `validation.py` and `array_ops.py`:
- `check_arrays()`, `check_array_1d()`, `check_fitted()`, `check_consistent_length()`,
  `validate_parameters()`
- `sort_by_x()`, `clip_to_range()`, `ensure_1d()`, `restore_order()`,
  `find_unique_sorted()`, `group_by_value()`, `interpolate_monotonic()`

### Key Dependencies
- **numpy, scipy**: Core numerical computing
- **scikit-learn**: Base classes and isotonic regression
- **cvxpy**: Convex optimization (`NearlyIsotonicCalibrator(method="cvx")`). Still a hard
  dependency because `nearly_isotonic.py` imports it at module level.

pandas, matplotlib and seaborn were removed as dependencies in 0.7.0 — nothing in the
package imported them.

### Design Patterns
- **Modular architecture**: Each calibrator in separate module under `calibrators/`
- **Base class inheritance**: All calibrators inherit from `BaseCalibrator` (extends sklearn's `BaseEstimator` and `TransformerMixin`)
- **Built-in diagnostics**: Enable via `enable_diagnostics=True` on any calibrator
- **Consistent API**: `.fit(X, y)` and `.transform(X)` following sklearn conventions
- **Standalone diagnostic functions**: Optional plateau analysis via `calibre.diagnostics` module
- **Input validation**: Through `check_arrays()` utility
- **Type hints**: Throughout codebase (Python 3.12+)

### Diagnostic Workflow
```python
# Built-in diagnostics approach (recommended)
from calibre import IsotonicCalibrator
import numpy as np

X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
y = np.array([0, 0, 1, 1, 1])

# Enable diagnostics during calibrator initialization
cal = IsotonicCalibrator(enable_diagnostics=True)
cal.fit(X, y)

# Access diagnostic results
if cal.has_diagnostics():
    diagnostics = cal.get_diagnostics()
    print(cal.diagnostic_summary())

# Standalone diagnostic functions approach
from calibre.diagnostics import run_plateau_diagnostics

# Run diagnostics on any calibration result
y_calibrated = cal.transform(X)
diagnostics = run_plateau_diagnostics(X, y, y_calibrated)
```

## Testing Structure
- Tests are in `tests/` directory
- Main test files:
  - `tests/test_calibrators_unit.py`: Unit tests for individual calibrator classes
  - `tests/test_diagnostics.py`: Plateau diagnostic testing
  - `tests/test_comprehensive_matrix.py`: Systematic testing across calibrator/data combinations
  - `tests/test_integration.py`: Full workflow and edge case testing
  - `tests/test_properties.py`: Mathematical property validation
  - `tests/test_metrics.py`: Calibration metrics testing
  - `tests/test_utils.py`: Utility function testing
  - `tests/test_monotone_spline.py`: Monotonicity guarantees for the spline calibrators
  - `tests/test_r_reference.py`: Cross-language checks against committed R fixtures
  - `tests/test_readme.py`: Executes every README code block and checks claimed output
  - `tests/data_generators.py`: Realistic test data generators for various calibration scenarios
- Uses pytest fixtures for test data generation
- Coverage reporting via pytest-cov
- Tests must fail rather than skip. Do not add `except Exception: pytest.skip(...)`.
- Total tests: 484, all passing (includes 52 doctests, collected via --doctest-modules)

## Configuration
- **pyproject.toml**: Modern Python packaging configuration
- Tool configuration for ruff, mypy, pytest, coverage, deptry, pydoclint in pyproject.toml
- Python 3.12+ required
- Dev dependencies are a PEP 735 `[dependency-groups]` entry, so `pip install -e ".[dev]"`
  does NOT work. Use `uv sync --all-extras --dev`.

## Interactive Examples
- **docs/source/notebooks/**: Jupyter notebooks with comprehensive examples and benchmarks
- Four focused notebooks covering getting started, validation, diagnostics, and performance comparison
- Executable via nbsphinx integration in documentation
- Located in `docs/source/notebooks/` (migrated from root examples/)

## CI/CD Configuration
- GitHub Actions workflow in `.github/workflows/ci.yml`
- **Optimized for efficiency**: CI skips when only documentation files are changed
- Test matrix: Python 3.12, 3.13, 3.14 on Ubuntu (primary), Python 3.12 on macOS/Windows
- Includes ruff lint/format checks as informational
- Package building and installation validation

### Files that skip CI when changed alone:
- All markdown files (`**.md`)
- Documentation directories (`docs/**`, `examples/**/*.md`)
- Project metadata (`LICENSE`, `citation.cff`, `CHANGELOG.md`, `CLAUDE.md`)

## Documentation
- **Sphinx documentation**: Comprehensive documentation with API reference, examples, and tutorials
- **Location**: `docs/` directory with source in `docs/source/`
- **Live site**: https://finite-sample.github.io/calibre/
- **Build locally**: `cd docs && make html` (requires `pip install -e ".[docs]"`)
- **Auto-deployment**: GitHub Pages deployment via `.github/workflows/docs.yml`

### Documentation Structure:
- Installation guide and quick start
- Comprehensive API reference with auto-generated docstrings
- Usage examples (basic and advanced)
- Performance benchmarks and comparisons
- Contributing guidelines

## Known Issues and Expected Behavior
- Some calibration methods may produce bounds violations (fixed with `np.clip`)
- All monotone methods are monotone by construction — a violation is a bug, not a
  tolerance. `tests/test_monotone_spline.py` asserts exactly zero.
- Assert provable properties, not thresholds tuned to whatever the code happened to do

## Code Quality Standards (v0.4.1+)
- **Line length**: 88 characters (ruff; E501 is ignored, the formatter handles it)
- **Complexity**: Functions should have complexity ≤10 (measured by McCabe)
- **Type hints**: Required throughout codebase (Python 3.12+ typing)
- **Import management**: No unused imports or variables
- **Formatting**: Automatic via `ruff format`
- **Testing**: Comprehensive test coverage with realistic data generators

### Diagnostics: what exists
See the `calibre/diagnostics.py` section above for the authoritative list. Only two
things are implemented: plateau detection with a sample-count density label, and
`diversity_learning_curve` / `progressive_sampling_diversity`. Earlier CHANGELOGs
advertised bootstrap tie stability, conditional AUC among tied pairs, minimum
detectable difference, and a supported/limited-data/inconclusive classifier; none of
those were ever written. Do not restate them here or anywhere else.