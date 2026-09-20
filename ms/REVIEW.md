# Manuscript revision and evidence audit

The previous manuscript described an economics-led method and experiments that
were not supported by the repository. The revision describes the implemented
software and uses the existing offline comparison. This follows the author's
approved software-and-methods positioning and existing-benchmark scope.

## Claims and evidence

| Claim or quantity | Evidence and unit | Interpretation retained |
|---|---|---|
| Common fitting and evaluation workflow | Public estimators, metrics, evaluation, selection, multiclass, report, and plotting modules; executable manuscript example | Implemented functionality, not a new calibration theorem |
| Centered isotonic and splines retain more values in the overconfidence design | `raw.csv`, one dataset/model/seed/method record; six-decimal distinct values divided by test-sample size where a fraction is used | Prediction granularity, not utility or CORP discrimination |
| Average score differences | Within-seed isotonic Brier minus method Brier; existing aggregator and generated tables/macros | Performance on the specified designs; uncertainty does not establish universal superiority |
| Intervals on mean paired Brier differences | Percentile bootstrap of 30 seed differences, 2,000 draws | Synthetic data-and-split variability; real-data split variability conditional on one fixed dataset; no multiplicity adjustment |
| Error against known probabilities | Mean absolute deviation from generating probabilities, synthetic observations only | In the tied-score design, latent individual probability error includes information lost by coarsening |
| CORP score components | Isotonic evaluation fit and proper-score identity on evaluation observations | Finite-sample decomposition, not unbiased population reliability or deployment refitting |
| Linear-time constrained projection | Cumulative-shift derivation, PAVA source, optimizer and reference tests | Linear after ordering; optional clipping can violate positive bounds |
| CDI-ISO | Local-bound construction and step-map implementation; dedicated tests | Experimental method with no demonstrated decision-utility benefit in this benchmark |

## Substantive replacements

- Removed the nonexistent plateau stability, concordance, power, progressive
  sampling, and classification suite. Actual plateau support labels summarize
  sample counts and do not establish flatness.
- Removed density-aware smoothing, the composite economic selection objective,
  the claimed 15-dataset study, and the unsupported clinical and credit cases.
- Replaced generic step-function descriptions with the implemented interpolation
  conventions. Distinguished penalized decreases from bounded increments and
  non-decreasing maps from strictly increasing maps.
- Described both spline and nearly-isotonic internal tuning. The defaults-only
  benchmark does not allocate equal optimization budgets.
- Identified the logistic and temperature comparators as the harness's own fits,
  rather than calls to scikit-learn's calibration wrapper.
- Separated Brier, CORP miscalibration, AUC, and granularity. The overconfidence
  comparison has lower mean Brier but larger MCB for centered isotonic and the
  spline than for isotonic; the text now states that qualification.
- Fixed the bibliography filename, corrected the Roelofs et al. authors and page
  range using the PMLR record, and added the original CIR, CORP, spline, smooth
  error, and constrained-projection sources. Removed unused references.

## Audit checks and scope

The raw-data audit found no duplicate keys or hard range failures. All expected
method/configuration/seed keys are required by the manuscript exporter, including
an entirely missing method or configuration. Missing truth error on real data is
intentional and shown as unavailable, never as zero. Required score and ranking
measurements must be finite. Score decomposition and distinct-count denominators
are checked before rendering.

The numerical rerun reproduced the previously committed Brier, MCB, distinct
counts, log loss, and synthetic truth error exactly. Runtime is remeasured.
Paired comparisons hold the inputs fixed within each replicate. The original
real-data sample is reused across seeds; the manuscript therefore narrows the
interpretation of its intervals rather than changing the bootstrap or pretending
these are independent datasets. Synthetic regimes are reported separately, not
pooled into a universal average. All methods and configurations remain visible.

The design is predictive, not causal. Treatment assignment, compliance, causal
identification, and instrumental-variable checks are inapplicable. First-stage
model fitting is separated from scoring by the existing out-of-fold and test
splits. The remaining training-size mismatch between out-of-fold calibration
scores and the refitted deployment model is stated. Grouped, temporal, subgroup,
full-vector multiclass, and economic utility guarantees are not inferred from
this binary experiment.

The provenance sweep flags design constants, figure dimensions, table widths,
and interval levels as unmatched numerals. These are not empirical estimates:
generator constants are identified in the source comment; measured results and
counts enter the prose through generated macros. LaTeX layout constants and
method-label numbers have no statistical interpretation. Numerical output tests
check the generated artifacts against their data rather than relying on prose
matching. The generic provenance script does not expand LaTeX macros, so its
orphan count is not used as an acceptance verdict.

A larger distinct-value count was considered and rejected as evidence of better
discrimination. A monotonicity restriction was considered and rejected as a
guarantee of strict rank preservation. Temperature's good performance is compatible
with the correctly specified synthetic distortion and does not establish its
superiority under arbitrary miscalibration. The headline claims retain these
limits.

## Validation

The full local suite passed before the final presentation edits (1,681 tests,
with four pre-existing R-reference skips for singleton or constant cases).
Subsequent targeted checks passed (435 tests, four existing skips), exercising the
manuscript exporter, benchmark provenance, README examples, calibrator
documentation, and numerical contracts. Both `make paper` and `make paper-check`
passed. Ruff linting and formatting, pydoclint, pyright, and the executable Sphinx
documentation build passed. The final `preen check --strict` also passed, including
its test suite and link checks; its only informational note recommends a source
layout instead of the existing flat package layout.

All 14 rendered pages were visually reviewed. Figures use shared comparison
axes, named interval methods, explicit denominators, and distinguishable map line
styles. Table groups remain together, labels are readable, and citations resolve.
The final figure-style edit was followed by exporter tests, linting, compilation,
and another inspection of the affected pages.
