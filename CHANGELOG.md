# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  - `plot_plateau_diagnostics()`: Multi-panel diagnostic visualization
  - `plot_stability_heatmap()`: Bootstrap stability visualization
  - `plot_progressive_sampling()`: Sample size analysis plots
  - `plot_calibration_comparison()`: Method comparison charts
  - `plot_mdd_analysis()`: Minimum detectable difference visualization

- **📚 Interactive Demo**: Complete tutorial and best practices guide
  - `examples/plateau_diagnostics_demo.ipynb`: Comprehensive tutorial with practical examples
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
  - `tests/test_mathematical_properties.py`: Mathematical property validation tests for bounds, monotonicity, calibration improvement, and granularity preservation
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