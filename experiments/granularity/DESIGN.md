# Decision value of preserved granularity

This design implements the author's approved plan. Previous package benchmarks
were already inspected; this is not a blinded preregistration. This file and the
configuration are written before running the new simulation. No method must win.

The population is 1,000 equally weighted score types s=(j+0.5)/1000. The reported
score is logistic(1.8 logit(s)). Risks are s, 0.05+0.9s², 0.2/0.5/0.8 across equal
thirds, constant 0.3, or 0.5+0.35 sin(2 pi s). Each calibration sample draws types
with replacement and independent Bernoulli labels. Sizes are 200, 1000, and 5000;
seeds are 0–29. Methods share each sample. Evaluation integrates over the complete
finite population and known risks; there are no sampled evaluation labels.

For fitted Q=g(S), r=E[p(S)|Q]. Pooling loss is E[(p-r)²], miscalibration is
E[(r-Q)²], and total probability error is their sum. Expected Brier adds E[p(1-p)].
Conditional means use exact equal outputs, not approximate bins. Re-evaluate
rounded outputs at six decimals as a separate sensitivity. An injective map has
zero information loss even if its rankings or probability levels are wrong;
this oracle fact is not a claim that a finite-data user can decode it.

An action costs t and pays Y, so the implemented rule acts iff Q>t. Record
D(t)=E[(p-t)+]-E[(r-t)+] and
R(t)=E[(p-t)+]-E[(p-t)1{Q>t}] at 501 equally spaced thresholds including endpoints.
Their exact integrated values are pooling_loss/2 and total_error/2. Curves are
illustrations; never approximate the scalar integrals with a coarse display grid.
Uniform weighting is a reference convention, not an empirical cost distribution.

Select k percent for k=1,...,99. Compare expected successes with random boundary
tie-breaking and original-scoretie-breaking, original ranking, and oracle risk
ranking. Integrate random tie lotteries exactly. All outcomes are unaffected by
action. No treatment-effect or real-world welfare claim follows.

Report structural pair pooling and reversals among originally distinguishable
pairs, distinct counts, and Spearman as secondary quantities. Spearman is undefined
for a constant output; store this as missing, never zero. All population scores are
distinct. Output ties use exact equality; rounded sensitivity is explicitly labeled.

Means and paired differences against isotonic use 2,000 percentile bootstrap
resamples of 30 seeds (fixed bootstrap seed 20260919), separately for each design
and sample size. Intervals are 95%, pointwise and descriptive, with no multiple-test
adjustment. The same seed draws are used for all method contrasts. Intervals quantify Monte Carlo uncertainty in mean performance across calibration
samples, not a 95% range of performance for an individual fitted model. There is
no evaluation-sample uncertainty. There is no pooled winner.

Toy examples merge risks 0.3±d for d=0,0.05,0.15. Both coarse and fine forecasts
are calibrated and pooling loss is d². A separate constant-risk population with
artificially distinct forecasts demonstrates that granularity alone has no value.

Save every method, design, size, seed, and precision. Fail on missing cells or
nonfinite required results; never silently exclude failed fits. Save predictions
for exact re-evaluation. Keep metrics experiment-local and leave the main package
API and existing benchmark outputs unchanged. Report any deviations below.

## Deviations and implementation details

None at initialization. A pilot runs the existing methods on seed zero solely to
check execution and numerical contracts; it does not change the design or select
methods. All five designs remain in the final report.

## Subsequent theory and positioning review

After all existing simulation results had been inspected, the author chose score
replacement as the question and theory-first assessment as the next stage. The
note now derives boundary-crossing conditions and distinguishes information loss
from literal use of forecasts. This is a retrospective interpretation of the
existing experiment, not a new preregistration. No population, method, estimand,
seed, or reported simulation result changes. The assessment retains a methods
note: no novel finite-sample recoverability result has been established.
