# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Calibre is a Python package for advanced probability calibration techniques in machine learning. It provides alternative calibration methods to traditional isotonic regression that better preserve probability granularity while maintaining monotonicity constraints.

**Current Version**: 0.4.0 (major architectural overhaul)

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
    SmoothedIsotonicCalibrator
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
- `BaseCalibrator`: Abstract base class following sklearn transformer interface
- `DiagnosticMixin`: Optional mixin for diagnostic capabilities
- `MonotonicMixin`: Utility mixin for monotonicity checking and enforcement

**calibre/calibrators/**: Modular calibrator implementations:
- `IsotonicCalibrator`: Standard isotonic regression calibration
- `NearlyIsotonicCalibrator`: Allows controlled violations of monotonicity (CVXPY-based)
- `SplineCalibrator`: Smooth calibration using I-splines with cross-validation
- `RelaxedPAVACalibrator`: Ignores small violations based on percentile thresholds
- `RegularizedIsotonicCalibrator`: L2 regularized isotonic regression
- `SmoothedIsotonicCalibrator`: Applies Savitzky-Golay filtering to reduce staircase effects

**calibre/diagnostics.py**: Standalone plateau diagnostic functions:
- `run_plateau_diagnostics()`: Comprehensive plateau analysis
- `detect_plateaus()`: Detect flat regions in calibration curves
- `analyze_plateau()`: Classify individual plateaus
- `classify_plateau()`: Classify plateaus as supported/limited-data/inconclusive

**calibre/metrics.py**: Evaluation metrics for calibration quality:
- `mean_calibration_error()`: Basic calibration error
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

**calibre/diagnostics.py** (NEW in v0.4.0, simplified in v0.4.1): Plateau diagnostic tools:
- `PlateauInfo`: Data structure for plateau information
- `PlateauAnalyzer`: Basic plateau identification and analysis
- `IsotonicDiagnostics`: Comprehensive diagnostic engine with 6 methods:
  - Bootstrap tie stability analysis
  - Cross-fit stability testing
  - Conditional AUC among tied pairs
  - Minimum detectable difference calculations
  - Progressive sampling analysis
  - Local slope testing with smooth monotone fits
- `analyze_plateaus()`: Convenience function for full diagnostic analysis

**calibre/utils.py**: Utility functions:
- `check_arrays()`: Input validation
- `sort_by_x()`: Sorting utilities
- `create_bins()`, `bin_data()`: Binning operations
- `extract_plateaus()`: Identify plateau regions in calibrated output
- `bootstrap_resample()`: Generate bootstrap resamples
- `compute_delong_ci()`: DeLong confidence intervals for AUC
- `minimum_detectable_difference()`: Statistical power analysis

**calibre/visualization.py** (NEW in v0.4.0): Plotting tools for diagnostics:
- `plot_plateau_diagnostics()`: Comprehensive diagnostic visualization
- `plot_stability_heatmap()`: Bootstrap stability visualization
- `plot_progressive_sampling()`: Sample size vs diversity plots
- `plot_calibration_comparison()`: Compare different calibration methods
- `plot_mdd_analysis()`: Minimum detectable difference visualization

### Key Dependencies
- **numpy, scipy**: Core numerical computing
- **scikit-learn**: Base classes and isotonic regression
- **cvxpy**: Convex optimization for nearly-isotonic regression
- **pandas**: Data manipulation
- **matplotlib**: Visualization (examples)

### Design Patterns
- **Modular architecture**: Each calibrator in separate module under `calibrators/`
- **Base class inheritance**: All calibrators inherit from `BaseCalibrator` (extends sklearn's `BaseEstimator` and `TransformerMixin`)
- **Built-in diagnostics**: Enable via `enable_diagnostics=True` on any calibrator
- **Consistent API**: `.fit(X, y)` and `.transform(X)` following sklearn conventions
- **Standalone diagnostic functions**: Optional plateau analysis via `calibre.diagnostics` module
- **Input validation**: Through `check_arrays()` utility
- **Type hints**: Throughout codebase (Python 3.10+)

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
  - `tests/data_generators.py`: Realistic test data generators for various calibration scenarios
  - `tests/conftests.py`: Shared pytest fixtures and configuration
- Uses pytest fixtures for test data generation
- Coverage reporting via pytest-cov
- **Expected behavior**: ~6-8 tests may be skipped when calibrators reach mathematical limits (this is normal)
- Total tests: ~170+, with 160+ typically passing

## Configuration
- **pyproject.toml**: Modern Python packaging configuration
- Tool configurations for Black, isort, mypy included in pyproject.toml
- Python 3.10+ required
- Development dependencies defined in `[project.optional-dependencies.dev]`

## Interactive Examples
- **docs/source/notebooks/**: Jupyter notebooks with comprehensive examples and benchmarks
- Four focused notebooks covering getting started, validation, diagnostics, and performance comparison
- Executable via nbsphinx integration in documentation
- Located in `docs/source/notebooks/` (migrated from root examples/)

## CI/CD Configuration
- GitHub Actions workflow in `.github/workflows/ci.yml`
- **Optimized for efficiency**: CI skips when only documentation files are changed
- Test matrix: Python 3.10, 3.11, 3.12 on Ubuntu (primary), Python 3.11 on macOS/Windows
- Includes code quality checks (Black, isort, flake8) as informational
- Coverage reporting via Codecov
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
- Regularized isotonic regression may have 15-20% monotonicity violations (expected)
- Mathematical property tests skip when algorithms reach inherent limitations
- Test thresholds have been relaxed to reflect realistic algorithm performance

## Code Quality Standards (v0.4.1+)
- **Line length**: 88 characters maximum (configured in Black and flake8)
- **Complexity**: Functions should have complexity ≤10 (measured by McCabe)
- **Type hints**: Required throughout codebase (Python 3.10+ typing)
- **Import management**: No unused imports or variables
- **Formatting**: Automatic via Black with 88-character line length
- **Testing**: Comprehensive test coverage with realistic data generators

### Key Diagnostic Features (v0.4.1)
- **Plateau classification**: Automatic classification as supported/limited-data/inconclusive
- **Bootstrap analysis**: Tie stability across resamples (P̂_tie ∈ [0,1])
- **Conditional AUC**: Discrimination among tied pairs (AUC_tie = P(S⁺ > S⁻ | (i,j) ∈ T))
- **Statistical power**: Minimum detectable difference calculations
- **Progressive sampling**: How diversity changes with sample size
- **Local slope testing**: Uses smooth monotone fits to validate genuine flatness
- **Decision framework**: Guides when to trust isotonic regression vs. consider alternatives