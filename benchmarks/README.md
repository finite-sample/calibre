# The benchmark

Provenance for the claims calibre makes. Before this existed, the README's
comparison table and a page of star ratings had no script behind them.

```bash
python -m benchmarks.run --quick          # offline, ~1 min, what CI runs
python -m benchmarks.run --n-jobs 8       # the committed grid
python -m benchmarks.aggregate            # raw.csv -> summary.csv, paired.csv
python -m benchmarks.figures              # -> docs/_static/bench/*.svg
```

`benchmarks/` is not shipped in the wheel.

## The one rule

**Do not change `config.py` and the committed results in the same commit without
saying why in the message.** Everything that could be tuned to flatter calibre —
datasets, seeds, model and calibrator settings, the baseline, which metrics are
primary — lives in that one file so the choices are visible in one diff.

## What keeps it honest

**The calibrator is the only thing that varies within a cell.** For each
`(dataset, model, seed)` the out-of-fold scores and the test scores are computed
once and shared. Two calibrators being compared see byte-identical inputs, so a
difference between them cannot be resampling noise.

**The test split is touched exactly once.** Nothing is selected, tuned or
inspected on it. Calibrators fit on out-of-fold scores from `cross_val_predict`,
because a model's scores on its own training rows are already too good and a
calibrator fitted there learns the wrong correction.

**Library defaults only.** Tuning calibre's methods against an untuned isotonic
baseline would settle the comparison by construction. One asymmetry is worth
naming rather than hiding: `SplineCalibrator` and the `"auto"` default of the
relaxed calibrator choose their own hyperparameters by internal cross-validation.
That is a real advantage over a fixed competitor, and it is paid for in the fit
time the benchmark also records.

**Paired differences, not means of levels.** Seed variance dwarfs the effect being
measured, so `paired.csv` differences each method against the baseline *within*
seed and puts a bootstrap interval on the mean difference — resampling seeds,
because the seed is the unit of replication. An interval that spans zero is
reported spanning zero, and `beats_baseline` is false unless the lower bound
clears it.

**No composite score.** Score and resolution stay on separate axes. Folding them
into one number is where a thumb goes on the scale.

**A self-check that would catch us cheating.** `calibre_isotonic` must reproduce
`sklearn_isotonic` to 1e-12 on every row, asserted in `protocol.py`. calibre's
`IsotonicCalibrator` is a thin wrapper over scikit-learn's, so any divergence is a
bug — and a benchmark that hid it would be reporting calibre's advantage over its
own baseline.

**No silent dropping.** `aggregate.py` refuses to summarize a cell missing any of
its seeds, naming the offenders. A dataset that errored on half its seeds would
otherwise be averaged over whatever survived.

**Regimes where calibre loses are included at full weight** and named on the docs
page. `nonmonotone` exists because no monotone calibrator can express a
non-monotone truth. As it happens the measured loss showed up somewhere else
entirely — see below — which is the point of measuring.

## What the committed grid shows

Thirty seeds, held out, `overconfident` (logit inflated by 1.8):

| method | Brier | distinct values | error vs known truth |
|---|---|---|---|
| uncalibrated | 0.16037 | 1594 | 0.0808 |
| `sklearn_isotonic` | 0.15305 | **49** | 0.0255 |
| `calibre_relaxed_pava` | 0.15304 | **1356** | 0.0254 |
| `calibre_centered` | 0.15272 | 1514 | 0.0205 |
| `calibre_spline` | 0.15242 | 1595 | 0.0175 |
| `sklearn_temperature` | 0.15216 | 1599 | **0.0040** |

Two things to read off it. calibre's methods match or beat isotonic's score while
keeping around thirty times the distinct values, which is the claim — and
`calibre_relaxed_pava` is the cleanest demonstration, landing within 1e-5 of
isotonic's Brier while keeping 28 times its resolution. And **on this design
scikit-learn's temperature scaling is four times more accurate against the known
truth than calibre's best method** — because the distortion here *is* a pure
temperature change, so a one-parameter model is exactly specified. That is a
regime where calibre loses, and it is a real one.

`heavy_tie` isolates the same effect without the confound: `calibre_relaxed_pava`
scores 0.15319 against isotonic's 0.15317 — a difference in the fifth decimal —
while keeping 101 distinct values against 22.

Meanwhile `nonmonotone`, built expecting calibre to lose, has calibre winning
(0.21556 for the penalized spline against 0.22236 for Platt): the parametric
methods cannot follow the dip either, and they give up more.

Across the 80 non-baseline method-cells, 36 beat `sklearn_isotonic` with a
bootstrap interval clear of zero. The honest details: `sklearn_platt` and
`sklearn_temperature` are among the winners, and **`uncalibrated` beats the
baseline in one cell**: `breast_cancer/logreg`, by 0.00129 Brier with an interval
of [0.0003, 0.0025] and 22 of 30 seeds. Logistic regression is already close to
calibrated there and the test half is only ~228 rows, so isotonic's pooling costs
more than it buys. Calibrating is not free, and the benchmark says so.

Also visible: `NearlyIsotonicCalibrator` at its defaults is close to plain
isotonic on these designs — 54 distinct values against 49 — so its defaults are
not exercising what it exists for. That is documented on the class rather than
papered over with a new default: its resolution frontier is dominated by CIR,
which reaches more distinct values at a better score.

## Adding a dataset

Add a generator to `datasets.py` returning a `Dataset` with `p_true` set when the
truth is known. Synthetic designs construct the reported score directly and take
the identity "model", which isolates the calibrator from how well a classifier
happened to fit.

## netcal

Optional and off by default, enabled with `--include-netcal`. Not a hard
dependency: the moment it lags a Python release, a required import would make the
whole harness un-runnable — which silently stops the benchmark being re-run, the
failure this design exists to prevent. `aggregate.py` will not place a
partially-present method in the headline table.
