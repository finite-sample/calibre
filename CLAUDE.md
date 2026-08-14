# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Calibre is a Python package for advanced probability calibration techniques in machine learning. It provides alternative calibration methods to traditional isotonic regression that better preserve probability granularity while maintaining monotonicity constraints.

**Current Version**: 0.9.0 (see CHANGELOG)

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

# Plotting (optional: pip install 'calibre[plots]')
from calibre.plots import plot_reliability_diagram, plot_resolution_loss
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
- `plugin_calibration_error()`: The uncorrected ℓp estimator on equal-mass bins.
  Exists so plugin, debiased and sweep can be compared at one norm and one binning
  rule; the other three public estimators differ in both, which makes any plot of
  them together show different quantities rather than different bias.
- `expected_calibration_error()`: Expected calibration error (ECE). ℓ1 on
  **uniform-width** bins, unlike the bias-aware estimators.
- `maximum_calibration_error()`: Maximum calibration error (MCE)
- `brier_score()`: Brier score computation
- `calibration_curve()`: Calibration curve generation
- `correlation_metrics()`: Spearman correlations
- `unique_value_counts()`: Granularity preservation metrics
- `tie_preservation_score()`: Measures how well ties are preserved during calibration
- `plateau_quality_score()`: Overall quality assessment of plateau regions
- `calibration_diversity_index()`: Measures granularity preservation
- `progressive_sampling_diversity()`: Analyzes how diversity changes with sample size

**calibre/plots/**: Plotting. matplotlib is an **optional** extra
(`pip install 'calibre[plots]'`) and must stay one:
- No module-level `import matplotlib` anywhere under `calibre/`. Every plot function
  starts `require_matplotlib()` (`plots/_deps.py`). `calibre/__init__.py` exposes
  `plots` through a PEP 562 `__getattr__`, so `import calibre` imports nothing new.
  `tests/test_plots_deps.py` enforces this in a subprocess — an in-process check
  would pass regardless, because the test session has matplotlib loaded.
- matplotlib is also in the `dev` and `test` groups, so plots tests always run and
  never skip. Its lower bound is stated in four places (`plots` extra, `dev`, `test`,
  `docs`); PEP 735 groups cannot reference a project's own extras, so bump together.
- **Plots draw; they do not compute.** Functions take an already-computed object.
  Bands are a parameter, never an implicit flag — `consistency_bands` is a thousand
  PAV refits. `plot_calibrator_comparison` refuses an unfitted calibrator rather than
  fitting it. Two deliberate exceptions where sweeping *is* the plot:
  `plot_ece_bin_sensitivity`, `plot_resolution_frontier`.
- Single panel → `ax=None`, returns that `Axes` (the same object when supplied).
  Multi-panel → `axes=None`, returns a `Figure`.
- `plt` is touched in exactly one place: `_style.get_axes()`, `ax is None` branch.
  `plt.show()`, `plt.gca()`, `plt.style.use()` and `rcParams` assignment are banned
  and tested against.
- Every artist carries a stable label; internal ones are prefixed `_calibre:`, which
  matplotlib hides from legends. This is what makes the tests tractable.
- **No baseline-image tests.** Three OSes × three Pythons, and a pixel diff tests
  antialiasing rather than meaning. Assert the *claim* instead: barcode tick count ==
  distinct-value count exactly; decomposition panels reproduce the identity to 1e-12;
  plugin ECE rises with bin count while debiased does not.

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
  - `tests/test_evaluation.py`: `calibre.evaluation` — CORP, decompositions, bands
  - `tests/test_selection.py`: Cross-validated model selection
  - `tests/test_multiclass.py`: Temperature scaling and per-class calibration
  - `tests/test_report.py`: The text report
  - `tests/test_utils.py`: Utility function testing
  - `tests/test_monotone_spline.py`: Monotonicity guarantees for the spline calibrators
  - `tests/test_bootstrap_bias.py`: Bias of the bootstrap intervals
  - `tests/test_r_reference.py`: Cross-language checks against committed R fixtures
  - `tests/test_relplot_reference.py`: Checks against relplot's published numbers
  - `tests/test_readme.py`: Executes every README code block and checks claimed output
  - `tests/test_benchmarks.py`: The benchmark harness's schema and its guards
  - `tests/test_plots_deps.py`: The optional-matplotlib contract, in a subprocess
  - `tests/test_plots_contracts.py`, `tests/test_plots_claims.py`,
    `tests/test_plots_degenerate.py`: The plotting API's rules, what each picture
    claims, and what it does on degenerate input
  - `tests/simulation.py`, `tests/test_monte_carlo.py`,
    `tests/test_simulation_gates.py`: The Monte Carlo layer — see below
  - `tests/data_generators.py`: Realistic test data generators for various calibration scenarios
- Uses pytest fixtures for test data generation
- Coverage reporting via pytest-cov
- Tests must fail rather than skip. Do not add `except Exception: pytest.skip(...)`.
- The suite is all-passing; the total is deliberately not written down here,
  because a hardcoded count is wrong by the next PR and nothing checks it. Run
  `pytest` for the number. Doctests are included, collected via `--doctest-modules`.
- **`-n auto` is deliberately not in `addopts`.** Locally it is worth it (~140s on
  eight workers against ~600s serial, identical results), so run `pytest -n auto`
  yourself. It must not be the default: xdist sizes `auto` from `os.cpu_count()`,
  which on a containerised runner reports the host's CPUs rather than the ones the
  container can use. Setting it in `addopts` took CI's Ubuntu test step from 7m37s
  to 24m26s and timed out the 20-minute wheel job.
- A repo-root `conftest.py` sets `MPLBACKEND=Agg` and closes figures after each test.
  It must import nothing but the standard library: setting the backend only works
  before matplotlib is first imported.

### The Monte Carlo layer, and simcheck

`tests/simulation.py` holds data-generating processes whose population values are
known in closed form, and the assertions that use them. It asks what neither the
R fixtures nor the property tests reach: under a process whose truth we know, is
the estimator unbiased, and does a nominal 95% interval cover 95% of the time?
`tests/test_monte_carlo.py` is the battery; `tests/test_simulation_gates.py`
tests the assertions themselves.

[simcheck](https://github.com/finite-sample/simcheck) supplies the gates. It is a
`dev` and `test` dependency-group entry pinned to a git URL — never a published
extra, since PyPI rejects direct URL references in project metadata. Three rules
hold there:

- **Never a bare `assert` in a gate.** `python -O` deletes `assert` statements, so
  a module whose whole product is assertions passes everything under
  optimisation. Raise `AssertionError` explicitly.
  `test_the_gates_still_fire_under_optimisation` runs the gates in a `python -O`
  subprocess and fails if any stays quiet.
- **Every gate has a test that watches it fail.** A gate that cannot fail is worse
  than no gate: it converts an unverified suite into one that certifies itself.
  Degenerate input counts — an empty study, a constant estimator, a hit count
  outside the study. Each of those has silently passed at some point.
- **Delegating is not passing through.** Two wrappers deliberately do more than
  call simcheck, and both have a negative test that fails without it. Do not
  "simplify" them back. `assert_unbiased` rejects a constant estimator that
  misses the target, which simcheck reads as a bias t of zero. `assert_coverage`
  takes a hit count to `simcheck.assert_count_rate`, not `assert_coverage`,
  because the count gate validates `0 <= hits <= n` — expanding the count into a
  boolean array turns a negative count into a slice from the end.

`assert_biased_upward` has no simcheck equivalent: every gate there tests a
property an estimator is supposed to have, and this one certifies a defect. It
stays local and is a candidate to upstream, as is `assert_unbiased`'s guard.

## Benchmarks

`benchmarks/` is an importable package, not shipped in the wheel. It produces the
numbers in README.md and `docs/source/examples/benchmarks.rst`; both read its
committed CSVs, so **the docs build never re-runs the benchmark and never hits the
network**.

```bash
python -m benchmarks.run --quick      # offline, ~1 min, what CI exercises
python -m benchmarks.run --n-jobs 8   # the committed grid, ~5 min
python -m benchmarks.aggregate        # raw.csv -> summary.csv, paired.csv
python -m benchmarks.figures          # -> docs/source/_static/bench/
```

Everything tunable lives in `benchmarks/config.py`. **Do not change it and the
committed results in the same commit without saying why in the message.**

Two entries there are pre-registrations, and each is enforced rather than merely
stated — a rule recorded only in prose is one nothing notices you breaking.

- `PRIMARY_METRICS` fixes the headline metric before the grid runs, so it cannot
  be chosen after seeing the numbers. `figures.headline_table` reads
  `PRIMARY_METRICS[0]` for the ranking rather than naming a column itself. Do
  not inline the name back.
- `CALIBRATOR_DEFAULTS_ONLY` says every calibre calibrator is built at library
  defaults, since tuning them against an untuned scikit-learn baseline would
  decide the comparison by construction.
  `test_calibre_methods_are_built_at_library_defaults` compares each one against
  a bare instance of its class, so a hyperparameter appearing in
  `methods._build` fails rather than quietly flattering the results.

If you change a calibrator's defaults, re-run the grid and re-aggregate — the
committed results go stale silently otherwise. `docs/source/_static/bench/headline.csv`
is generated by `figures.py` and included by the docs page, so the table cannot
drift from the results.

The harness carries guards that fail loudly rather than promises: `calibre_isotonic`
must reproduce `sklearn_isotonic` to 1e-12, `aggregate.py` refuses to summarise a
cell missing seeds, and figures are drawn through `calibre.plots` so a plotting
regression breaks the benchmark build. `tests/test_benchmarks.py` covers the schema
and those guards.

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