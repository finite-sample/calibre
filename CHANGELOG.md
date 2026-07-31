# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`calibre.evaluation`: CORP reliability diagrams and score decompositions.**
  Implements Dimitriadis, Gneiting & Jordan (*PNAS* 2021). A binned reliability diagram
  makes the analyst pick the bins, and the picture changes with the choice; CORP removes
  the choice by estimating conditional event probabilities with isotonic regression via
  PAV — machinery this package already owned and already pinned against R.
  - `corp_reliability(x, y)` — the diagram, with no bin count to tune.
  - `score_decomposition(x, y, score=...)` — `mean_score = MCB - DSC + UNC`
    (miscalibration, discrimination, uncertainty) for the Brier or log score.
    `MCB` and `DSC` are non-negative by construction, and the identity is exact.
  - `consistency_bands` / `confidence_bands` — resampling-based uncertainty
    quantification. Resampling only; the paper's asymptotic route is not implemented,
    and that limit is documented rather than left for a user to discover.

  Pinned against R's `reliabilitydiag` on five datasets (calibrated, overconfident,
  squashed, heavily tied, rare-event): every component agrees to **1e-16 or better**.
  The Python ecosystem has no equivalent — the scikit-learn request for this
  decomposition ([#23767](https://github.com/scikit-learn/scikit-learn/issues/23767))
  has been open since 2022.

- **`calibre.selection`: cross-validation shared by every calibrator.** Previously the
  only CV lived inside `SplineCalibrator` as a private method.
  - `cross_val_calibrate(calibrator, X, y)` — out-of-fold calibrated probabilities.
    **This is a precondition for honest evaluation, not a refinement of it:** for any
    isotonic-family calibrator, in-sample `MCB` is *exactly* zero regardless of how the
    model generalises, because the calibrator and the CORP diagnostic are the same PAV
    projection and PAV is idempotent. Measured on 1500 points, in-sample MCB is 0.0
    while the out-of-fold estimate is 0.0028.
  - `select_by_cv`, `make_folds`, `resolve_auto` — the shared primitives.
  - Selection scores on a strictly proper scoring rule (log-loss by default, Brier
    available). ECE is deliberately rejected as a criterion: it is biased and depends on
    its binning, so selecting on it optimises binning artifacts.

### Changed

- **`lam`, `alpha` and `epsilon` now default to `"auto"`** on
  `NearlyIsotonicCalibrator`, `RegularizedIsotonicCalibrator` and
  `RelaxedPAVACalibrator`, resolved by cross-validation at fit time and recorded on
  `lam_` / `alpha_` / `epsilon_`. Passing a number pins it as before. These are pure
  bias-variance knobs, so the old fixed defaults (`lam=1.0`, `alpha=0.1`) were not
  neutral choices but hidden wrong answers — `RegularizedIsotonicCalibrator` even had
  the same `(n_knots, alpha)` pair that `SplineCalibrator` has always tuned.
  The constructor arguments are never written back to, so `get_params` still round trips
  and `clone` still reproduces the estimator.
- `CDIIsotonicCalibrator` is deliberately excluded from auto-selection: its
  `thresholds`, `bandwidth` and `gamma` encode economic domain knowledge rather than a
  bias-variance tradeoff, and tuning them away would defeat the estimator.
- `RegularizedIsotonicCalibrator` fits now depend on row order, as `SplineCalibrator`
  already did, because `KFold` assigns folds by position. That is cross-validation
  behaviour, and the monotonicity guarantees are unaffected.

## [0.7.1] - 2026-07-30

A follow-up to 0.7.0's correctness work. An audit of what 0.7.0 left behind found one
calibrator still breaking the package's central guarantee, three public parameters that
were accepted and ignored, and a test suite that converted its own failures into skips.

### 💥 BREAKING CHANGES

- **`SmoothedIsotonicCalibrator` was not monotone on tied scores** — in its *default*
  configuration. It was the last module still building its interpolant with
  `scipy.interpolate.interp1d` directly on the training scores, duplicates and all,
  which keeps whichever tied point survived the sort. Measured on 600 scores rounded to
  two decimals: 34 monotonicity violations, worst −0.0268. Tied scores are the ordinary
  case in calibration — tree ensembles and any rounded or binned score produce them. It
  now pools ties with `aggregate_ties` and interpolates with `PiecewiseLinear`, like
  every other calibrator. Zero violations.
  - `interp_method` is **removed**. Its documented `"cubic"` value produced 1424
    violations out of 4999 (worst −0.1127): monotonicity was enforced on the knots and
    cubic interpolation put the overshoot back between them. `"linear"` was the only
    safe value, so the parameter was a footgun with no valid alternative. No test ever
    passed anything but `"linear"`.
  - The fit now happens in `fit`. `transform` previously re-ran isotonic regression and
    Savitzky-Golay smoothing on every call.
  - Window lengths now count *distinct* scores rather than observations.
- **`run_plateau_diagnostics` no longer takes `y`, `n_bootstraps` or `random_state`.**
  All three were accepted and ignored — `n_bootstraps` and `random_state` were even
  commented as such in the source while the docstring documented them as live. The
  diagnosis is structural: it reads the calibrated curve, not the outcomes. Old
  three-argument calls now raise `TypeError` rather than silently rebinding.
- **`analyze_plateau_simple` no longer takes `y_calibrated`**, which it never read.

### Fixed

- **`fit()` no longer mutates constructor parameters.** `SmoothedIsotonicCalibrator`
  wrote coerced `poly_order` and `min_window` values back onto the instance, so
  `get_params()` did not round trip and `sklearn.base.clone` produced a different
  estimator. Validated values now live on `poly_order_` and `min_window_`.
- **`CDIIsotonicCalibrator` exposed fitted state as hyperparameters.** As a
  `@dataclass`, its "fitted attributes" (`_fitted`, `_L`, `_R`, `_z_fit`, …) landed in
  the generated `__init__`, which is what scikit-learn inspects — so `get_params()`
  returned fitted arrays, `clone()` copied a fit into a supposedly fresh estimator, and
  `repr()` printed the arrays. They are now `field(init=False, repr=False)`.

### Changed

- **The test suite no longer converts failures into skips.** Nine
  `except Exception: pytest.skip(...)` handlers were hiding three failing assertions,
  and because `skip` aborts the whole test, every calibrator after the first failure in
  each loop went unchecked. Expectations that were simply wrong have been corrected
  rather than suppressed: `NearlyIsotonicCalibrator` is asserted to reduce violations as
  lambda rises (measured 82 → 49 → 0) instead of being held to a fixed tolerance it
  cannot meet by design, and the granularity floor is applied only to the calibrators
  that claim granularity preservation.
- `SmoothedIsotonicCalibrator` is now documented as *not* preserving granularity: the
  running maximum that restores monotonicity re-flattens the curve wherever the filter
  dipped, retaining roughly 13–16% of distinct input values. This is long-standing
  behaviour, identical in 0.7.0; it was simply never measured.
- Docs no longer advertise diagnostics that do not exist. `CLAUDE.md` and the
  diagnostics notebook claimed bootstrap tie stability, conditional AUC among tied
  pairs, and minimum detectable difference; none were ever implemented.
- **Every example notebook ran end to end for the first time in several releases.** All
  four were failing at their import cell — 0.7.0 dropped `matplotlib` and `pandas` as
  runtime dependencies without adding them to the docs group, and
  `03_diagnostics_and_troubleshooting` had no import cell at all. 27 of 33 code cells
  raised, and the tracebacks were published to the docs site as cell output because
  `nbsphinx_allow_errors` was `True`. That flag is now `False`, so a failing cell fails
  the build; `matplotlib` and `pandas` are in the `docs` dependency group; and
  `boxplot(labels=...)` is updated to matplotlib 3.9's `tick_labels`. All 33 cells now
  execute cleanly and the plots render.

## [0.7.0] - 2026-07-30

Correctness release. Several estimators did not compute what they claimed; each of
those claims is now verified against a reference implementation or a numerical
optimum, and the test suite asserts the guarantees rather than restating them.

### 💥 BREAKING CHANGES

- **`SplineCalibrator` was not monotone.** It combined a B-spline basis with
  `Ridge(positive=True)`; non-negative coefficients on a B-spline basis give a
  non-negative function, not a monotone one. Measured before the fix, it produced a
  non-monotone calibration map on 12 of 12 random datasets, with up to 746 violations
  out of 1999 intervals. It now uses an I-spline basis on which non-negative
  coefficients *are* monotone, so monotonicity is structural.
  - `n_splines` renamed to `n_knots`. It was always passed straight through as
    `n_knots`, so the old name was simply wrong.
  - New: `knots`, `alpha`, `link`, `random_state`, `max_cv_samples`, `clip_output`.
  - Cross-validation now tunes `(n_knots, alpha)` on log-loss and refits on all the
    data. It previously kept whichever fold scored best on its own validation split —
    selection on noise, and the shipped model saw only `(cv-1)/cv` of the sample.
  - It also stored mismatched parameters: one mutable transformer was refit per fold,
    so the retained knots came from the last fold and the coefficients from the best.
- **`RelaxedPAVACalibrator`: `percentile` and `adaptive` replaced by `epsilon` and
  `min_slope`.** The old threshold was a percentile of `|diff(y)|`; with binary labels
  those differences are all 0 or 1, so it collapsed to either "never binds" or "never
  constrains" and the relaxation was a no-op for the package's main use case. `epsilon`
  is now an absolute tolerance, and `min_slope` runs the other way to forbid plateaus.
- **`RegularizedIsotonicCalibrator` is a monotone spline with a curvature penalty, not
  ridge-penalised isotonic regression.** `alpha * sum(beta^2)` buys no smoothness:
  unconstrained it is `beta = y/(1+alpha)`, a uniform deflation that breaks mean
  calibration and drives every prediction to zero as `alpha` grows. `alpha=0` no
  longer reduces to isotonic regression — use `IsotonicCalibrator` for that.
- **`NearlyIsotonicCalibrator.method` now defaults to `"path"`**, and the path solver
  actually solves the stated objective. The previous implementation used the raw level
  gap as its collision time and never let block values drift with lambda; at a matched
  lambda it returned objective 0.07625 against the true optimum 0.03750. Also
  documented: `lam` here is **twice** the lambda of Tibshirani, Höfling & Tibshirani
  (2011), because the squared-error term omits the factor of one half.
- **`mean_calibration_error` returns `|E[p] - E[y]|`.** It previously returned
  `mean(|p - y|)` — mean absolute error, which is minimised by hard 0/1 predictions and
  is nonzero for a perfectly calibrated model. Use
  `sklearn.metrics.mean_absolute_error` for the old quantity.
- **`calibre.visualization` removed.** It was never exported, had no tests, was absent
  from the API docs, indexed diagnostic keys the current `diagnostics.py` does not
  emit, and called `plt.cm.get_cmap`, removed in matplotlib 3.9.
- **`matplotlib`, `seaborn` and `pandas` are no longer dependencies.** `seaborn` and
  `pandas` were imported nowhere in the package; `matplotlib` went with the
  visualization module. Runtime dependencies drop from 7 to 4.

### Added

- **`CenteredIsotonicCalibrator`** — centered isotonic regression (Oron & Flournoy,
  2017). Collapses each of PAVA's flat blocks to its weighted-centroid score and
  interpolates, so the fit is strictly increasing in the interior. Non-parametric,
  nothing to tune, O(n). Over 30 held-out splits it beats plain isotonic on Brier in
  24, and returns ~1900 distinct values where isotonic returns 56.
- **`sample_weight`** on `fit` for `IsotonicCalibrator` and
  `CenteredIsotonicCalibrator`. Calibrators that cannot honour weights now raise
  rather than silently discarding them.
- **`calibre/_core.py`** — the shared numerical primitives every calibrator is built
  from: `weighted_pava`, `aggregate_ties`, `shift_to_pava`, `nearly_isotonic_path`,
  `collapse_blocks`, `monotone_spline_basis`, `fit_monotone_spline`, `PiecewiseLinear`,
  `StepFunction`.
- **Cross-language reference tests.** Committed fixtures in `tests/fixtures/r/` pin the
  estimators against `stats::isoreg`, `Iso::pava`, `isotone::gpava`, `cir::cirPAVA`,
  `neariso` and `scam(bs="mpi")`. `experiments/r_reference/gen_fixtures.R` regenerates
  them. The nearly-isotonic solver matches the authors' own R implementation to ~1e-16.
- `README.md` code blocks are executed by `tests/test_readme.py`, which also checks
  that any claimed output is the output actually produced.

### Fixed

- **Integer labels silently truncated.** `check_arrays` preserved `int64`, so pooling
  two labels averaged 0 and 1 to `0`. `RelaxedPAVACalibrator.fit(X, y)` with integer
  0/1 labels — the documented usage — returned only 0s and 1s.
- **Tied scores produced nondeterministic output.** Four calibrators built
  `scipy.interpolate.interp1d` on duplicated abscissae, which silently drops one of the
  tied points; combined with an unstable `argsort`, which one survived varied between
  runs. Tied scores are now pooled into one weighted point.
- **All work moved from `transform` into `fit`.** Four calibrators re-ran their whole
  solve on every `transform` call, so a solver failure surfaced at predict time as a
  silent fallback. `NearlyIsotonicCalibrator.transform` at n=100,000: 875 ms → 0.26 ms.
- Second-difference penalties are computed on the actual, unevenly spaced score grid
  rather than in index space.
- Doctests are now collected (`--doctest-modules`) and all 52 pass. Six docstrings
  stated numerically wrong results, including `brier_score` claiming 0.142 for a case
  that yields 0.098.
- Parameter validation happens in `fit` and raises instead of silently coercing, so
  `get_params`/`clone` round-trip.
- Removed `tests/conftests.py`, which pytest never loaded because of the trailing `s`.

### Changed

- Test suite grew from ~170 to 484 tests, all passing. Assertions that could not fail
  were replaced: a monotonicity test that permitted 35 violations out of 49, an
  "improvement" test satisfied by a 9% regression, and a granularity test that passed
  with 2 distinct values out of 400.
- `mypy` and `ruff` are clean; the test data generators reseed per request, so results
  no longer depend on test execution order.

## [0.6.0] - 2025-12-26

### Changed
- Code quality and type-safety pass across the package: type hints throughout,
  consolidated tooling, and a modular `calibrators/` package layout.
- Minimum Python raised to 3.12; CI matrix is 3.12, 3.13 and 3.14.

## [0.5.0] - 2025-11-27

### 💥 BREAKING CHANGES
- **Python Version Requirement**: Minimum Python version increased from 3.10 to 3.11
  (raised again to 3.12 in 0.6.0)
  - Updated CI test matrix to support Python 3.11, 3.12, and 3.13
  - Removed Python 3.10 from supported versions
  - Users must upgrade to Python 3.11+ to use this version

### Changed
- **🐍 Modern Python Features**: Leveraged Python 3.11+ capabilities
  - Added `from __future__ import annotations` to all modules for cleaner type hints
  - Updated development tooling configuration for Python 3.11 target version
  - Modernized type annotations throughout the codebase

### Improved
- **🛠️ Development Tooling**: Consolidated to ruff-only workflow
  - Removed black, isort, and flake8 dependencies in favor of unified ruff tooling
  - Updated CI/CD pipeline to use ruff for both linting and formatting
  - Simplified development workflow with single tool for code quality

## [0.4.2] - 2025-11-27

### Improved
- **📖 Documentation Quality & Consistency**: Comprehensive docstring improvements
  - Standardized import paths across all examples to use main package imports (`from calibre import`)
  - Enhanced mathematical notation with proper LaTeX formulation for optimization problems
  - Added detailed documentation for private methods (`_transform_cvx`, `_transform_path`)
  - Standardized parameter descriptions across all calibrator classes
  - Added missing `enable_diagnostics` parameter documentation to all calibrators
  - Fixed module docstring duplication in base classes

### Fixed
- **🔧 CI/CD Improvements**: Streamlined continuous integration
  - Fixed dependency installation in CI to use new uv dependency groups format (`--group dev`)
  - Removed unnecessary Codecov upload step from CI workflow
  - Removed redundant README validation job from CI
  - Updated documentation deployment to trigger on every commit to main branch

### Developer Experience
- Improved code maintainability with consistent documentation standards
- Better developer onboarding with standardized examples across all calibrators
- More reliable CI pipeline with proper dependency management

## [0.4.1] - 2025-01-23

### Changed
- **🏗️ Simplified Diagnostic Architecture**: Streamlined BaseCalibrator diagnostic system
  - Removed complex diagnostic parameters (`n_bootstraps`, `random_state`) from BaseCalibrator
  - Simplified to single `enable_diagnostics` boolean parameter
  - Diagnostic functions now called from standalone `diagnostics.py` module
  - Cleaner inheritance pattern for all calibrator classes
  - Maintained backward compatibility for diagnostic functionality

### Fixed
- Corrected diagnostic function signatures in tests
- Fixed imports and references to removed diagnostic parameters
- Improved code formatting and consistency across codebase

### Documentation
- Updated CLAUDE.md to reflect simplified diagnostic approach
- Removed references to deprecated diagnostic parameters in examples
- Updated usage patterns for cleaner API

## [0.4.0] - 2025-09-18

### Added
- **🔬 Plateau Diagnostics System**: Revolutionary diagnostic tools to distinguish between noise-based flattening (good) and limited-data flattening (bad) in isotonic regression
  - `IsotonicDiagnostics` class: Comprehensive plateau analysis with 6 diagnostic methods
  - `PlateauAnalyzer` class: Individual plateau identification and characterization
  - `IsotonicRegressionWithDiagnostics`: Drop-in replacement for sklearn's IsotonicRegression with integrated diagnostics
  - Bootstrap tie stability analysis across resamples
  - Cross-fit stability testing for plateau consistency
  - Conditional AUC computation among tied pairs with DeLong confidence intervals
  - Minimum detectable difference (MDD) calculations with statistical power analysis
  - Progressive sampling diversity curves for sample size effects
  - Local slope testing using smooth monotone fits

- **📊 Advanced Diagnostic Metrics**: New metrics for plateau quality assessment
  - `tie_preservation_score()`: Measures quality of tie preservation in calibration
  - `plateau_quality_score()`: Overall quality assessment for plateaus
  - `calibration_diversity_index()`: Granularity preservation metric
  - `progressive_sampling_diversity()`: Sample size vs diversity analysis

- **🔧 Enhanced Utility Functions**: Extended utility toolkit for plateau analysis
  - `extract_plateaus()`: Extract plateau regions from isotonic regression output
  - `bootstrap_resample()`: Bootstrap resampling utilities
  - `compute_delong_ci()`: AUC confidence intervals using DeLong method
  - `minimum_detectable_difference()`: Statistical power calculations for two proportions

- **📈 Visualization Module**: Comprehensive plotting tools for diagnostic analysis
  (removed in 0.7.0 — see that entry)
  - `plot_plateau_diagnostics()`: Multi-panel diagnostic visualization
  - `plot_stability_heatmap()`: Bootstrap stability visualization
  - `plot_progressive_sampling()`: Sample size analysis plots
  - `plot_calibration_comparison()`: Method comparison charts
  - `plot_mdd_analysis()`: Minimum detectable difference visualization

- **📚 Interactive Demo**: Complete tutorial and best practices guide
  - Interactive notebooks with comprehensive tutorials and practical examples
  - Decision framework for choosing between strict and soft calibration methods
  - Real-world scenarios and interpretation guidance
  - Performance comparison across different calibration approaches

- **🧪 Comprehensive Test Suite**: Full test coverage for diagnostic functionality
  - `tests/test_diagnostics.py`: Complete test suite for all diagnostic components
  - Edge case handling and integration tests
  - Performance and accuracy validation

### Technical Implementation
- **Mathematical Foundation**: Implementation based on rigorous statistical theory
  - Tie stability index: P̂_tie ∈ [0,1] computed across bootstrap samples
  - Conditional AUC: AUC_tie = P(S⁺ > S⁻ | (i,j) ∈ T) with confidence intervals
  - MDD calculation: MDD ≈ (z₁₋α/₂ + z₁₋β)√(p̂(1-p̂)(1/m + 1/n))
  - Progressive sampling curves with trend analysis
  - Local slope testing with bootstrap confidence intervals

- **Classification System**: Automatic plateau classification
  - **Supported**: High stability + low conditional AUC + flat slope → genuine plateaus
  - **Limited-data**: Low stability + high conditional AUC + positive slope → artifacts
  - **Inconclusive**: Mixed evidence requiring further investigation

- **Integration**: Seamless integration with existing calibre ecosystem
  - Maintains sklearn-style API consistency
  - Works with all existing calibration methods
  - Backward compatible design

### Impact
This release addresses a critical gap in calibration methodology by providing the first comprehensive diagnostic system for isotonic regression plateaus. Users can now make principled, evidence-based decisions about when to use strict isotonic regression versus softer alternatives, significantly improving calibration quality in practice.

## [0.3.0] - 2025-09-17

### Added
- **Comprehensive Testing Framework**: Added extensive test suite for validation and quality assurance
  - `tests/data_generators.py`: Realistic test data generators with 8 miscalibration patterns (overconfident neural networks, underconfident random forests, sigmoid distortion, imbalanced binary, multi-modal, weather forecasting, click-through rate, medical diagnosis)
  - `tests/test_properties.py`: Mathematical property validation tests for bounds, monotonicity, calibration improvement, and granularity preservation
  - `tests/test_comprehensive_matrix.py`: Comprehensive test matrix covering ~400 test combinations across all calibrators, patterns, sample sizes, and noise levels
  - `tests/validation/calibration_validation.ipynb`: Visual validation notebook with reliability diagrams and performance comparisons

### Fixed
- **ISpline Bounds Issue**: Fixed ISplineCalibrator producing values slightly above 1.0 by adding `np.clip(predictions, 0, 1)` to ensure strict [0,1] bounds
- **Import Issues**: Resolved relative import issues in test modules

### Changed
- **Enhanced CI/CD**: Simplified GitHub Actions workflow with informational linting checks
- **Documentation**: Updated CLAUDE.md with comprehensive development commands and testing instructions

### Technical Improvements
- **Mathematical Validation**: Comprehensive validation of all calibration methods across realistic scenarios
- **Edge Case Handling**: Robust testing for extreme scenarios (perfect calibration, constant predictions, extreme imbalance, small samples)
- **Performance Benchmarking**: Systematic evaluation across multiple data patterns and calibrator configurations

### Quality Assurance
- **Proof of Correctness**: Visual and quantitative validation that all calibration methods are mathematically sound
- **Real-World Testing**: Validation on scenarios mimicking medical diagnosis, click-through rates, weather forecasting, and fraud detection
- **Property Preservation**: Confirmed bounds preservation, monotonicity control, granularity preservation, and ranking correlation maintenance

## [0.2.1] - Previous Release

### Features
- Core calibration algorithms implementation
- Basic metrics and utilities
- Initial CI/CD setup

---

**Note**: This release represents a major advancement in validation and testing, ensuring the package is production-ready with comprehensive mathematical guarantees and real-world scenario validation.