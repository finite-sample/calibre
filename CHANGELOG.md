# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.0] - 2026-08-25

### Fixed

- Nearly-isotonic regression now uses the penalty scale defined in the source paper
  and matches the authors' R implementation, including weighted fits.
- Penalized I-spline loss is normalized by total observation weight, and failed
  numerical optimizers now raise instead of returning an invalid fitted model.
- Calibration fitting and transformation reject multidimensional and non-finite
  inputs consistently. A failed refit no longer leaves the previous curve fitted.
- Cross-validation rejects unsupported folds, ignores failed solver candidates, and
  forwards observation weights when producing out-of-fold predictions.

### Changed

- `SplineCalibrator` is the single penalized I-spline API. `alpha` and `n_knots`
  can each be fixed or selected by proper-score cross-validation.
- The public surface contains only maintained calibrators and one canonical key for
  each reported metric.
- Runtime calibration no longer depends on CVXPY. The independent CVXPY programs
  remain in the test suite as numerical oracles.
- Package documentation now uses the tested README as its landing page, and the
  committed benchmark was regenerated across 2,700 held-out method evaluations.

## [0.11.0] - 2026-08-24

### Fixed

- Plateau diagnostics now identify flat regions along the input-score axis, accept
  the same array-like inputs as calibrator fitting, and reject mismatched lengths.
- Fitted-state checks now use scikit-learn's estimator protocol. They no longer accept
  an unfitted isotonic model, reject a fitted non-isotonic model, or probe an arbitrary
  score that can violate a fitted calibrator's out-of-bounds policy.
- Identity-link spline cross-validation now uses squared error and the fitted model's
  clipping policy, matching the model it selects and ultimately fits. Calibrator-level
  automatic scoring uses log loss for probability targets and squared error for
  unbounded targets, and selection now raises when every candidate fit fails.
- `interpolate_monotonic(bounds_error=True)` now raises outside the training range
  instead of silently applying NumPy's boundary fill.
- CDI-ISO now rejects non-finite and all-zero sample weights instead of returning
  zero or NaN calibration curves.
- Binned metrics now reject zero bins instead of reporting zero calibration error.
- Accessing `calibre.plots` after a plain `import calibre` now loads the optional
  plotting namespace instead of recursing until Python raises `RecursionError`.
- Weighted spline fits now use sample weights when placing quantile knots.

### Changed

- Adopted the py-canon v1.1.0 fleet standard in full: `uv_build` backend with a
  static PEP 621 version, canon ruff/pyright/pydoclint configuration (E501 now
  enforced), the reusable docs workflow, and `preen check --strict` in CI.
- Documentation moved from `docs/source/` to a flat `docs/` Sphinx root; the
  tutorial notebooks are now executed by myst-nb instead of nbsphinx, removing
  the pandoc build dependency. README benchmark images are served from the
  documentation site instead of raw GitHub URLs.

## [0.10.0] - 2026-08-06

### Added

- **`calibre.plots`: the pictures.** matplotlib is an optional extra
  (`pip install 'calibre[plots]'`); importing calibre still imports nothing new, and
  a subprocess test enforces that.

  - `plot_reliability_diagram` — the CORP diagram, with consistency or confidence
    bands and a marginal density panel. Also reachable as `ReliabilityDiagram.plot()`.
  - `plot_score_decomposition` — `MCB`/`DSC`/`UNC` as three panels on independent
    scales. Drawn as one stacked bar the components are illegible: for any competent
    model `DSC` is around 0.10 while `MCB` is around 0.001, so the quantity the
    decomposition exists to expose is thinner than its own outline and the
    discrimination bar covers it outright. Every panel starts at zero.
  - `plot_mcb_dsc_plane` — forecasters at `(DSC, MCB)` with iso-score contours.
  - `plot_resolution_loss` — the collapse barcode: one tick per distinct output
    value, one strip per method. Isotonic's strip is countable by eye at 82 ticks
    while a resolution-preserving fit is solid ink at ~1900. Ticks are drawn
    semi-transparent so the strip is a density ramp rather than clipping to black.
  - `plot_resolution_frontier` — held-out score against distinct values retained,
    which answers the obvious objection that the extra values are noise.
  - `plot_calibrator_comparison` — overlaid calibration maps. **Refuses an unfitted
    calibrator** rather than fitting it, because fitting on the data you are about to
    display is the mistake the README warns about.
  - `plot_ece_bin_sensitivity` — plugin against debiased against sweep, over bin count.
  - `plot_miscalibration_profile`, `plot_classwise_reliability` — multiclass.

  Plots draw; they do not compute. Bands are a parameter and never an implicit flag,
  because `consistency_bands` is a thousand PAV refits. No function calls `plt.show()`,
  mutates `rcParams`, or reaches for the current figure, and each is tested for it.

  Colours are Okabe-Ito rather than matplotlib's `tab10`, whose red and green are
  indistinguishable under deuteranopia.

  There are no baseline-image tests. The CI matrix spans three operating systems and
  three Python versions, and a pixel diff reports that a pixel moved rather than
  whether the picture still says what it claims. Each plot's *claim* is asserted
  instead: the barcode's tick count must equal the distinct-value count exactly, the
  decomposition panels must reproduce `UNC + MCB - DSC = mean_score` to 1e-12, and the
  plugin ECE series must rise with bin count while the debiased one does not.

- **`plugin_calibration_error(y_true, y_pred, n_bins=15, p=2)`** in `calibre.metrics`.
  The uncorrected estimator that `debiased_calibration_error` corrects, on the same
  equal-mass tie-safe bins and at a caller-chosen norm. Comparing the existing
  estimators was a trap: `expected_calibration_error` is ℓ1 on uniform-width bins,
  `debiased_calibration_error` is ℓ2 on equal-mass bins, and `sweep_calibration_error`
  is ℓ1 on equal-mass bins, so plotting them together showed three different
  quantities disagreeing rather than one estimator being biased.

- **`sweep_calibration_error(..., return_n_bins=True)`** now reports the bin count the
  sweep settled on, which is half of what the estimator has to say.

- **`smooth_calibration_error`: smECE**, the smooth calibration error of Błasiok &
  Nakkiran (ICLR 2024). Replaces bins with a Gaussian kernel and chooses its own
  bandwidth by fixed point, so there is no parameter at all — not a bin count, not a
  bandwidth. It is a *consistent* calibration measure in the sense of Błasiok,
  Gopalan, Hu & Nakkiran (2023): bounded above and below by polynomial functions of
  the true distance to the nearest calibrated predictor. Binned ECE is not, which is
  why it can report a large error for a nearly calibrated predictor.

  Pinned against Apple's `relplot` across ten regimes — calibrated, over- and
  under-confident, shifted, heavily tied, rare-event, small-n, mass sitting exactly on
  0 and 1, and exactly-backwards forecasts — at the auto-selected bandwidth and four
  fixed ones. **Every value agrees to 1.1e-16.** relplot is not a dependency; the
  fixtures are committed and the generator is at
  `experiments/relplot_reference/gen_fixtures.py`, mirroring the R reference setup.

- **`bootstrap_ci(metric, y_true, y_pred, method="bc")`** — a bootstrap confidence
  interval for any callable of `(y_true, y_pred)`, with `"percentile"`, `"basic"`,
  `"bc"` and `"bca"` available and **bias correction as the default**.

  The default is not the percentile interval, and the reason is a property of what
  calibration errors *are*. The bootstrap resamples from the empirical measure, so
  `E[F*] = F`; a **linear** functional then satisfies `E[g(F*)] = g(F)` exactly, while
  a **convex** one satisfies `E[g(F*)] > g(F)` strictly, by Jensen. Proper scoring
  rules are plain means, hence linear. Every calibration error is a norm of a linear
  functional, hence convex. The size of the gap is curvature at `F`, which is unbounded
  at the kink `‖δ‖ = 0` and negligible far from it — so **the distortion is worst
  exactly when the model is well calibrated**, which is the case users most want an
  honest answer for.

  Measured (`experiments/bootstrap_bias/investigate.py`), bootstrap mean ÷ observed:

  | statistic | calibrated | miscalibrated |
  |---|---|---|
  | Brier score (linear) | 1.00x | 1.00x |
  | plugin ECE (convex) | 1.42x | 1.01x |
  | smECE | 1.33x | 1.04x |
  | `MCB` | 1.52x | 1.09x |

  The 1.42 is the predicted √2: the observed value is `‖δ‖` for sampling noise `δ`,
  while a resample gives `‖δ + ε‖` with `ε` of comparable variance, doubling the
  variance inside the norm. It **does not shrink with sample size** — 1.43, 1.44, 1.44,
  1.42 at n of 250, 1000, 4000 and 16000 — because both terms scale as `1/√n`. The
  decisive control is that the *signed* mean calibration error, which is linear, shows
  no shift at all (z = +0.009) while the *absolute* version on the same data through
  the same resampling shifts clearly (z = +0.411).

  Coverage of a true calibration error of exactly zero, at nominal 95%, using
  `debiased_calibration_error` (whose estimand really is the true error):

  | method | coverage | mean width |
  |---|---|---|
  | `percentile` | 77% | 0.063 |
  | `basic` | 98% | 0.063 |
  | **`bc`** | **95%** | **0.017** |

  So the default both hits the nominal level and is 3.6x tighter. `basic` over-covers
  and returns negative lower bounds for a non-negative quantity; `bc`, being a
  percentile method, cannot.

  Two caveats are documented rather than hidden. `bc` reads the bias off how many draws
  fall below the estimate, so it needs a tie correction to avoid collapsing to `[0, 0]`
  when a censored estimator sits on its boundary — `debiased_calibration_error` floors
  at zero on 59% of well-calibrated samples. A `degenerate` flag reports the collapse
  when it still happens. And the *plugin* estimator is biased, so its estimand is not
  zero and an interval for it correctly excludes zero; no interval method changes that.

  The result now also carries `bias`, the measured bootstrap shift, so the distortion is
  visible rather than inferred.

  `MCB` and `DSC` are excluded from `calibration_report`'s intervals altogether. They
  are functionals of an isotonic fit, for which the naive n-out-of-n bootstrap is
  inconsistent: a resample keeps only ~63% of rows distinct (measured 0.630 against a
  theoretical 0.632) and PAV overfits the duplicates, so the inflation tracks effective
  sample size — subsampling without replacement gives `MCB` of 0.0155, 0.0088 and 0.0056
  at m of 200, 500 and 1000 against 0.0036 observed at n = 2000. In practice it produced
  an interval both degenerate and sitting above its own estimate. `consistency_bands`
  and `confidence_bands` resample outcomes instead and remain correct there.

  `tests/test_bootstrap_bias.py` pins the mechanism rather than the constants: the
  linear control, the convex-versus-linear divergence on identical data, invariance to
  sample size, and the decay with distortion. If the explanation ever stops holding,
  those fail.

- **`calibration_report(y_true, y_pred, ci=False)`** — one call returning the CORP
  decomposition, three calibration-error estimators that disagree instructively, the
  bias, and the resolution retained, as an immutable dataclass that prints as an
  aligned block. The "just tell me if my model is calibrated" entry point. When
  intervals are requested it prints the caveat above alongside them, rather than
  relying on the reader having found the docstring.

- **A Monte Carlo battery: unbiasedness, coverage, size and power.**
  `tests/simulation.py` supplies data-generating processes whose population values
  are known in closed form, and `tests/test_monte_carlo.py` asserts the statistical
  properties the package implicitly claims. Every tolerance is a **Monte Carlo
  standard error** — `sd/√R` for a mean, `√(c(1−c)/R)` for a proportion — so
  tightening an assertion means raising the replication count, not editing a number,
  and a failure reports how many standard errors away it landed.

  Because each design's score is a strictly increasing, clipping-free function of a
  known `p_true`, `E[y | x] = p_true` exactly, and the whole CORP decomposition has
  a closed form: `UNC = p̄(1−p̄)`, `DSC = Var(p)`, `MCB = E[(x−p)²]`, with
  `MCB − DSC + UNC = mean score` holding in population. The true ℓ2 calibration
  error is exactly `√MCB`, which ties `score_decomposition` to
  `debiased_calibration_error` — two independently written estimators of one
  quantity that nothing previously required to agree.

  `UNC` gets the sharpest check available: `E[ȳ(1−ȳ)] = p̄(1−p̄)(1 − 1/n)` **exactly**,
  because `Var(ȳ) = p̄(1−p̄)/n` however `p` is distributed. Not an asymptotic target.

  Three findings, each now documented and asserted:

  1. **`debiased_calibration_error` is unbiased on the squared scale, not on the
     error scale.** Over 400 calibrated samples of 1500 observations, the debiased
     *sum* sits **1.3 standard errors** from zero — the correction works exactly —
     and is **negative on 53%** of samples, as an unbiased estimate of zero must be.
     The reported error is **15.7 standard errors** above zero, and the whole gap is
     introduced by `sqrt(max(·, 0))` discarding that negative half. No amount of data
     removes it. A new `squared=True` returns the unbiased quantity, for averaging
     across folds or comparing models.
  2. **The uncertainty bands are pointwise, not simultaneous, and the difference is
     not subtle.** Pointwise coverage of `consistency_bands` is exactly nominal —
     90.1%, 89.6%, 89.4% at n of 300, 1200, 4800 against a nominal 90% — while
     *simultaneous* coverage is ~0% at every n. So "my curve stayed inside the band,
     therefore it is calibrated" is a test with a false-positive rate near one. The
     word "pointwise" previously appeared only in a private helper's docstring.
  3. **`confidence_bands` under-cover the truth on small samples**: 78.2%, 86.9%,
     90.5% coverage of the *true* conditional event probability curve at n of 300,
     1200, 4800, nominal 90%. They are centered on an isotonic fit, which is biased at
     finite n; the shortfall vanishes as that bias does. Treat a 90% band on a few
     hundred observations as closer to an 80% one.

  Also asserted for the first time: that the plugin estimator is **detectably
  biased** where the debiased one is not (the justification for the latter existing,
  written down nowhere before); that fitted calibration maps **converge to the true
  inverse link** rather than merely being monotone; and that calibration helps a
  miscalibrated model by more than three standard errors, paired within replication.

- **A reproducible benchmark, with committed results.** `benchmarks/` is an
  importable package (not shipped in the wheel) run by `python -m benchmarks.run`:
  thirty seeds over ten methods and ten dataset/model cells, including three
  scikit-learn baselines. Within a cell the calibrator is the only thing that
  varies — the out-of-fold and test scores are computed once and shared — and the
  test split is touched exactly once.

  It replaces a README table and a page of star ratings that had no script behind
  them. The results are committed, so the docs build never re-runs the benchmark
  and never hits the network, and the docs table is generated from them rather
  than typed.

  Guards rather than promises: `calibre_isotonic` must reproduce
  `sklearn_isotonic` to 1e-12 on every row, `aggregate.py` refuses to summarize a
  cell missing any of its seeds, paired differences carry bootstrap intervals that
  are reported spanning zero when they do, and there is no composite score.
  Regimes where calibre loses are included at full weight and named: temperature
  scaling is six times more accurate against the known truth on `overconfident`,
  and on `breast_cancer/logreg` leaving the model uncalibrated beats isotonic.

### Changed

- **`RelaxedPAVACalibrator` now defaults to `min_slope="auto"`**, which resolves to
  `0.01 / n_unique`. PAVA's plateaus are an artifact of pooling adjacent violators,
  not a finding about the data, and at the old `min_slope=0.0` this estimator kept
  only 1-4% of the input's distinct values. Measured on logit-inflated designs at n
  from 300 to 3000, the new default retains 80-95% for a Brier cost in the fifth
  decimal, with monotonicity and the `[0, 1]` bounds intact at every n.

  The automatic slope applies **only on the untouched default path** — when
  `epsilon` was also left at `"auto"` and the cross-validated search settled on
  zero. Naming `epsilon` yourself, *including `epsilon=0`*, leaves the slope at zero,
  so `epsilon=0` keeps meaning plain isotonic regression and the documented
  epsilon sensitivity is unchanged. `min_slope=0.0` remains available explicitly.

- **Two tests that could not fail were replaced.**
  `test_consistency_bands_have_approximately_nominal_coverage` asserted
  `0.5 <= covered/trials <= 1.0` at a nominal level of 0.9 — a band covering half the
  time passed — and conflated pointwise with simultaneous coverage. The quantitative
  study moved to the Monte Carlo battery; what remains is a one-second guard on the
  qualitative fact, down from 31.7 seconds.

  `test_calibration_metrics_improvement` asserted
  `calibrated_brier <= original_brier * 2.0` and counted a 10% ECE *worsening* as an
  improvement, on a single draw of 400 observations. A calibrator that doubled the
  Brier score passed a test named for verifying improvement. Tightening the
  thresholds does not fix it: `NearlyIsotonicCalibrator(lam=1.0)` genuinely reports a
  worse ECE than the raw scores on that one draw (0.110 against 0.071) while
  improving in 40 of 40 independent draws at a larger sample size. The statistic is
  too noisy at that size to carry the claim, so the claim moved to the Monte Carlo
  battery and a non-deterioration bound stayed behind.

### Fixed

- **One eighth of the comprehensive test matrix had never run.**
  `_run_single_test` passed `noise_level` to every pattern outside a hard-coded
  exemption list, and that list was wrong in both directions: it exempted
  `click_through_rate`, which does accept one, and omitted `imbalanced_binary`,
  which does not. So every `imbalanced_binary` combination raised `TypeError` and
  was recorded as a *calibrator* failure — 216 of 1296 combinations, for every
  calibrator, at every sample size and noise level.

  It went unnoticed because the assertion was `success_rate >= 0.7`. One pattern
  in eight is 12.5%, comfortably inside a 30% allowance, so all seventeen
  calibrators scored exactly 87.5% and the suite reported success. The list is
  now derived from the generator's signature (`CalibrationDataGenerator.accepts`),
  so it cannot drift, and with the combinations actually running the real rate is
  100% — which is what is now asserted, per calibrator.

  The test is also parametrised by calibrator rather than looping over all 1296
  combinations in one 48-second test, so a failure names the calibrator instead
  of reporting an aggregate rate.

- `calibre.metrics.__all__` was declared partway down the module, above
  `debiased_calibration_error` and `sweep_calibration_error`, so
  `from calibre.metrics import *` silently omitted both. Moved to the end of the file,
  with a test that fails if any public metric is ever left out.

### Documentation

- **`NearlyIsotonicCalibrator` is documented as the wrong tool for granularity.**
  Its objective fits one value per observation to the labels, so a small `lam`
  returns something close to the raw 0/1 labels: measured out of sample,
  `lam=0.001` keeps 1074 distinct values at a held-out Brier of 0.191 against
  isotonic's 0.116, and every step up the grid buys score back by giving
  granularity away. That frontier is dominated outright —
  `CenteredIsotonicCalibrator` reaches 2647 distinct values at a *better* score
  than isotonic — so no default was invented to hide it.

- **The README documented none of this release.** It now covers `calibre.plots`,
  smECE, `calibration_report`, `bootstrap_ci` and `plugin_calibration_error`; its
  comparison table is regenerated from the committed benchmark grid; and it names
  the methods that beat calibre rather than only the ones that do not.


- **The API reference was two releases behind.** `calibre.evaluation` (all of 0.8.0),
  `calibre.multiclass` (all of 0.9.0), `calibre.selection` and `calibre.diagnostics`
  had no pages at all, and `CenteredIsotonicCalibrator` — the calibrator the README
  names as the recommended default — was absent from the calibrator page. The CORP
  decomposition, which no other maintained Python package ships, was undiscoverable in
  our own documentation. All now documented.
- Removed the star-rating table from the benchmarks page. It rated six calibrators
  across four dimensions with no script, no measurement and no provenance behind any
  of it, and listed `SmoothedIsotonicCalibrator`'s use case as "Visualization".
- `index.rst` described `RelaxedPAVACalibrator` as using "percentile thresholds in the
  data", which `relaxed_pava.py` says was removed as unworkable for binary labels.
- `installation.rst` listed pandas and matplotlib as runtime dependencies (dropped in
  0.7.0) and black, isort and flake8 as dev dependencies (replaced by ruff in 0.4.1).
- Fixed a malformed nested list in `run_plateau_diagnostics`'s docstring that made
  Sphinx emit an error. The docs now build clean under `-W`.

## [0.9.0] - 2026-07-31

### Added

- **`calibre.multiclass`: class-wise evaluation, and the diagnostic that tells you which
  multiclass method you need.**

  There is no single best multiclass calibration method — there are two regimes with
  different winners, and picking wrong costs about a factor of six. Measured against
  **known** true probabilities over 12 seeds on 5 classes:

  | miscalibration | uncalibrated | temperature | per-class (CIR) |
  |---|---|---|---|
  | global | 0.0821 | **0.0025** | 0.0165 |
  | class-dependent | 0.1043 | 0.0849 | **0.0173** |
  | class-dependent + shift | 0.0373 | 0.0276 | **0.0176** |

  The winner took 12/12 seeds in every row.

  - `miscalibration_profile(P, y)` — the diagnostic. Reports per-class miscalibration,
    its spread, and a plain-language reading. The spread is ~0.13 when the distortion is
    global and 0.38–0.92 when it is class-dependent, which is enough to choose a method.
    Built entirely on 0.8.0's `score_decomposition`.
  - `classwise_decomposition(P, y)` — the CORP `MCB`/`DSC`/`UNC` split per class. The
    identity is exact and the components non-negative in every class, inherited from the
    binary implementation rather than reimplemented; a 2-class problem agrees with
    `score_decomposition` to 1e-15.
  - `classwise_ece`, `top_label_ece` — built on the bias-aware, tie-safe estimators added
    in 0.8.0.
  - `classwise_reliability(P, y)` — one CORP reliability diagram per class.
  - `TemperatureScaler` — one parameter fitted by NLL. Ships because it *wins a whole
    regime*, not for completeness. **Never changes the predicted class**, so accuracy is
    exactly preserved — asserted on every row in the test suite. Its ceiling is asserted
    too: a test requires that it *fails* to fix a class-dependent distortion, because
    one parameter applied to every class cannot express that fix.

  Documented cost that no standard metric reveals: temperature scaling preserves each
  row's class ordering but reorders people *within* a class, at 49.6% of adjacent pairs.
  If you rank individuals by their probability of a given class, that is real.

  Scope is deliberate. Only **class-wise** calibration is targeted; canonical
  calibration is infeasible to verify beyond four or five classes. `OneVsRestCalibrator`
  is held for a later release — the evidence for it is strong (12/12 in the
  class-dependent regimes) but ICML 2025 ranks one-vs-rest isotonic *worst of seven* on
  NLL, so the "when not to use this" documentation is load-bearing and deserves its own
  release. Rank-preserving projection is out: exact projection does not converge at
  n=1500 (84s, 3.7e-02 row error), while the ε-relaxed version converges in 0.9s at
  ε=0.05 — a real result, but one that belongs in `rank-preserving-calibration`.

## [0.8.0] - 2026-07-31

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

- **Two bias-aware calibration error estimators** in `calibre.metrics`. The plugin binned
  ECE is biased upward — part of each bin's gap is sampling noise in the label mean —
  and the bias grows with the bin count, precisely when a finer picture is wanted. On
  4000 calibrated observations where the true error is zero, plugin ECE climbs from
  0.0134 at 5 bins to 0.0313 at 50.
  - `debiased_calibration_error` subtracts the per-bin Bernoulli variance (Bröcker 2012;
    Ferro & Fricker 2012; Kumar, Liang & Ma 2019). Checked against Kumar's reference
    implementation across 24 configurations: exact agreement on 18, worst difference
    4.3e-03, arising from a different bin-edge rule rather than a different estimator.
  - `sweep_calibration_error` chooses the bin count instead of fixing it, adding bins
    while the calibration curve stays monotone and stopping when it does not
    (Roelofs et al. 2022, Algorithm 1).
  - Both use equal-mass bins, which Roelofs et al. measure as less biased than the
    conventional equal-width. **Neither ever splits a group of tied predictions across a
    bin boundary** — a bin edge through the middle of a tie group compares a mean
    prediction against labels from an arbitrary subset of observations carrying that
    same prediction, measuring sort order rather than calibration. Clipping a forecast
    into `[0, 1]` routinely puts hundreds of observations on one value, so this is the
    common case rather than an exotic one.

- **`calibre.selection`: cross-validation shared by every calibrator.** Previously the
  only CV lived inside `SplineCalibrator` as a private method.
  - `cross_val_calibrate(calibrator, X, y)` — out-of-fold calibrated probabilities.
    **This is a precondition for honest evaluation, not a refinement of it:** for any
    isotonic-family calibrator, in-sample `MCB` is *exactly* zero regardless of how the
    model generalizes, because the calibrator and the CORP diagnostic are the same PAV
    projection and PAV is idempotent. Measured on 1500 points, in-sample MCB is 0.0
    while the out-of-fold estimate is 0.0028.
  - `select_by_cv`, `make_folds`, `resolve_auto` — the shared primitives.
  - Selection scores on a strictly proper scoring rule (log-loss by default, Brier
    available). ECE is deliberately rejected as a criterion: it is biased and depends on
    its binning, so selecting on it optimizes binning artifacts.

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
  behavior, and the monotonicity guarantees are unaffected.

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
  behavior, identical in 0.7.0; it was simply never measured.
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
  ridge-penalized isotonic regression.** `alpha * sum(beta^2)` buys no smoothness:
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
