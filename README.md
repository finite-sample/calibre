# calibre

[![PyPI version](https://img.shields.io/pypi/v/calibre.svg)](https://pypi.org/project/calibre/)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://pepy.tech/badge/calibre)](https://pepy.tech/project/calibre)
[![CI](https://github.com/finite-sample/calibre/workflows/CI/badge.svg)](https://github.com/finite-sample/calibre/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-green.svg)](https://finite-sample.github.io/calibre/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Probability calibration that doesn't flatten your scores.**

Your classifier's probabilities are usually wrong — a model that says "80%" may be
right 60% of the time. Isotonic regression is the standard fix, and it works, but it
pays for accuracy with resolution: it is a step function, so it collapses many
distinct scores into a handful of values.

On the 2,000-point held-out set in the example below, isotonic regression turns
2,000 distinct scores into **82**. Everything inside a step becomes
indistinguishable — which matters as soon as you rank, threshold, or bucket the
output.

calibre gives you calibration methods that retain much more of that ordering while
correcting the probabilities.

## Install

```bash
pip install calibre           # core
pip install 'calibre[plots]'  # adds matplotlib for calibre.plots
```

Python 3.12+. Depends on numpy, scipy, scikit-learn and cvxpy. matplotlib is
optional and imported only when you use `calibre.plots`.

## The problem, in 20 lines

```python
import numpy as np
from sklearn.model_selection import train_test_split

from calibre import CenteredIsotonicCalibrator, IsotonicCalibrator

# An overconfident model: true log-odds z, but the model reports 1.8 * z.
rng = np.random.default_rng(0)
z = rng.normal(0, 2, 4000)
y = (rng.random(4000) < 1 / (1 + np.exp(-z))).astype(float)
scores = 1 / (1 + np.exp(-1.8 * z))

# Always fit the calibrator on data the model did not train on.
s_fit, s_test, y_fit, y_test = train_test_split(
    scores, y, test_size=0.5, random_state=0
)

isotonic = IsotonicCalibrator().fit(s_fit, y_fit)
centered = CenteredIsotonicCalibrator().fit(s_fit, y_fit)

print("distinct values, isotonic:", len(np.unique(isotonic.transform(s_test))))
print("distinct values, calibre: ", len(np.unique(centered.transform(s_test))))
# > distinct values, isotonic: 82
# > distinct values, calibre:  1863
```

Both are well calibrated. Only one of them still tells you which of two customers is
the riskier bet.

## Which calibrator should I use?

**If you don't want to think about it: `CenteredIsotonicCalibrator`.** It is
non-parametric, has nothing to tune, and preserves score ordering between pooled
isotonic blocks.

| You want | Use | Notes |
|---|---|---|
| A drop-in isotonic replacement, no tuning | `CenteredIsotonicCalibrator` | Collapses isotonic's flat steps to points and interpolates. O(n). |
| A smooth curve, and you can afford cross-validation | `SplineCalibrator` | Monotone spline; picks its own smoothing using the loss appropriate for its link. |
| A smooth curve with smoothing you control | `RegularizedIsotonicCalibrator` | Same model, you set `alpha` instead of tuning it. Fast. |
| Exactly scikit-learn's isotonic behavior | `IsotonicCalibrator` | Thin wrapper, plus optional plateau diagnostics. |
| Strict increase without output clipping | `RelaxedPAVACalibrator` | Forces a minimum step; clipping can flatten boundary values. |
| To allow small ranking violations if they fit better | `NearlyIsotonicCalibrator` | `lam` trades monotonicity against fit. Not the one to reach for if you want resolution — see its docstring. |
| Accuracy near specific decision thresholds | `CDIIsotonicCalibrator` | Research-grade; needs your operating thresholds. |

Every calibrator follows the scikit-learn transformer API: `.fit(scores, labels)` and
`.transform(scores)`, plus `sample_weight` where it is meaningful.

## What you actually get

Every number below comes from [`benchmarks/`](benchmarks/), whose results are
committed — `python -m benchmarks.run` reproduces them. This is the
`overconfident` design (a model reporting `1.8 * z` for true log-odds `z`), thirty
seeds, scored on a held-out half that nothing was tuned on. Lower Brier is better;
ΔBrier is the improvement over leaving the model uncalibrated.

| Method | Brier | ΔBrier | smECE | Distinct values | Beats isotonic |
|---|---|---|---|---|---|
| Uncalibrated | 0.1604 | — | 0.0835 | 1594 | — |
| `IsotonicCalibrator` | 0.1530 | +0.0073 | 0.0270 | **49** | baseline |
| `NearlyIsotonicCalibrator` | 0.1531 | +0.0072 | 0.0270 | 51 | 4/30 |
| `RelaxedPAVACalibrator` | 0.1530 | +0.0073 | 0.0270 | 1356 | 28/30 |
| `CenteredIsotonicCalibrator` | 0.1527 | +0.0076 | 0.0284 | 1514 | 25/30 |
| `RegularizedIsotonicCalibrator` | 0.1525 | +0.0079 | 0.0264 | 1596 | 24/30 |
| `SplineCalibrator` | 0.1524 | +0.0079 | 0.0259 | 1588 | 28/30 |
| Platt scaling (sklearn `method="sigmoid"`) | **0.1521** | **+0.0082** | 0.0251 | 1599 | 26/30 |
| Temperature scaling (sklearn `method="temperature"`) | 0.1522 | +0.0082 | 0.0251 | 1599 | 26/30 |

Read three things off it honestly.

**The Brier gains over isotonic are small.** The large win is the distinct-value
column: ~1400–1600 values instead of 49, at a Brier difference in the fourth
decimal. `RelaxedPAVACalibrator` is the cleanest case — it beats isotonic on 28 of
30 seeds by an average of 0.00001, which is to say it costs nothing, and keeps 29
times the resolution.

**scikit-learn's parametric methods win this design outright.** Both are
`CalibratedClassifierCV` options — `method="sigmoid"`, and `method="temperature"`
since 1.8. Both score better than anything in calibre, and against the *known*
truth they are four times more accurate (0.0064 and 0.0040, against 0.0169 for the
best calibre method). That is not an artefact: the distortion here is a pure
temperature change, so a one-parameter model is exactly specified and a
non-parametric one is paying for flexibility it does not need. If you know your
miscalibration has that shape, use them. calibre is for when you don't.

**smECE barely separates the methods**, because it is a calibration measure and
resolution is not miscalibration. That is a reason to look at more than one number,
which is what `calibration_report` below is for.

The cost is fit time: isotonic fits in 1.3 ms, `RelaxedPAVACalibrator` in 113 ms,
`RegularizedIsotonicCalibrator` in 0.6 s and `SplineCalibrator` in 2.3 s, the last
two because they cross-validate their own hyperparameters.

`nonmonotone` is in the grid because monotone methods should lose there. They
don't: `RegularizedIsotonicCalibrator` scores 0.2156 against Platt's 0.2224,
because the parametric methods cannot follow the dip either and give up more. And
on `breast_cancer/logreg`, *not calibrating at all* beats isotonic by 0.0013 with a
bootstrap interval clear of zero — the model was already close to calibrated and
the test half is small, so pooling costs more than it buys.

## Recipes

### Don't fit the calibrator on training predictions

This is the mistake that quietly ruins calibration. A model's scores on its own
training data are already too good, so a calibrator fitted there learns the wrong
correction. Use a held-out split, or out-of-fold predictions:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split

from calibre import CenteredIsotonicCalibrator

X, y = make_classification(n_samples=2000, n_features=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = LogisticRegression().fit(X_train, y_train)

# Out-of-fold predictions: every score comes from a model that did not see that row.
oof = cross_val_predict(
    LogisticRegression(), X_train, y_train, cv=5, method="predict_proba"
)[:, 1]

calibrator = CenteredIsotonicCalibrator().fit(oof, y_train)
calibrated = calibrator.transform(model.predict_proba(X_test)[:, 1])
print(
    f"{len(calibrated)} calibrated probabilities in "
    f"[{calibrated.min():.3f}, {calibrated.max():.3f}]"
)
# > 500 calibrated probabilities in [0.000, 1.000]
```

### Preserve more score resolution

Since 0.10.0 `RelaxedPAVACalibrator` defaults to `min_slope="auto"`. When its
automatic epsilon search selects strict monotonicity (`epsilon_ == 0`), it uses a
step of `0.01 / n_unique` to separate adjacent fitted values before output
clipping. On the benchmark grid that takes it from 11 distinct values to 124 on
`breast_cancer/logreg` while the Brier score moves in the fifth decimal.

Set `min_slope` yourself and disable clipping when you need a guaranteed gap:

```python
import numpy as np

from calibre import RelaxedPAVACalibrator

rng = np.random.default_rng(0)
scores = np.sort(rng.random(500))
labels = (rng.random(500) < scores).astype(float)

cal = RelaxedPAVACalibrator(min_slope=1e-4, clip_output=False)
fitted = cal.fit_transform(scores, labels)

steps = np.diff(fitted)
print("strictly increasing:", bool(np.all(steps > 0)))
print("smallest step:      ", round(float(steps.min()), 6))
print(
    "range:              ",
    (round(float(fitted.min()), 4), round(float(fitted.max()), 4)),
)
# > strictly increasing: True
# > smallest step:       0.0001
# > range:               (-0.0011, 1.002)
```

Note `clip_output=False`. Forcing 500 points apart by `1e-4` needs at least `0.05` of
range, so the fit runs slightly outside `[0, 1]` at the ends. Leaving the default
`clip_output=True` would clamp those tails and flatten them back together — 31 of the
499 steps become exactly zero — which defeats the point. Either turn clipping off, as
here, or pick a `min_slope` small enough that the fit stays inside the unit interval.

The same parameter runs the other way: `epsilon` permits decreases of up to that size,
which buys a closer fit at the cost of reordering some pairs.

### Weight your calibration set

```python
import numpy as np

from calibre import CenteredIsotonicCalibrator

rng = np.random.default_rng(0)
scores = rng.random(300)
labels = (rng.random(300) < scores).astype(float)
weights = rng.uniform(0.5, 2.0, 300)  # e.g. inverse sampling probabilities

calibrator = CenteredIsotonicCalibrator().fit(scores, labels, sample_weight=weights)
print("weighted fit:", calibrator.transform(np.array([0.25, 0.75])).round(3))
# > weighted fit: [0.149 0.671]
```

### Measure it

```python
import numpy as np

from calibre.metrics import (
    brier_score,
    expected_calibration_error,
    mean_calibration_error,
)

y_true = np.array([0, 0, 1, 1, 1, 0, 1, 1])
y_pred = np.array([0.1, 0.3, 0.6, 0.7, 0.9, 0.2, 0.8, 0.75])

print(f"Brier          {brier_score(y_true, y_pred):.4f}")  # lower is better
print(f"ECE            {expected_calibration_error(y_true, y_pred):.4f}")
print(f"bias           {mean_calibration_error(y_true, y_pred):.4f}")
# > Brier          0.0628
# > ECE            0.2313
# > bias           0.0813
```

`brier_score` is a proper scoring rule and the one to optimise.
`expected_calibration_error` is the familiar binned ECE — useful, but sensitive to the
bin count and blind to resolution. `mean_calibration_error` is calibration in the
large, `|mean(prediction) − base rate|`.

Binned ECE is also **biased upward**: part of each bin's gap is sampling noise in the
label mean rather than miscalibration, and the bias grows with the bin count — precisely
when you wanted a finer picture. Two estimators correct for it:

```python
import numpy as np

from calibre import debiased_calibration_error, sweep_calibration_error
from calibre.metrics import expected_calibration_error

rng = np.random.default_rng(0)
p = rng.uniform(0, 1, 4000)
y = rng.binomial(1, p).astype(float)  # calibrated by construction: true error is 0

print(f"plugin ECE  {expected_calibration_error(y, p, n_bins=15):.4f}")
print(f"debiased    {debiased_calibration_error(y, p, n_bins=15):.4f}")
print(f"sweep       {sweep_calibration_error(y, p):.4f}")
# > plugin ECE  0.0163
# > debiased    0.0000
# > sweep       0.0155
```

The true error here is zero, so the plugin's 0.0163 is entirely bias. Debiasing removes
it. The sweep estimator does not, on this sample — it targets the bin-count problem
rather than the within-bin bias, and the two are worth reaching for separately.

`debiased_calibration_error` subtracts the per-bin Bernoulli variance (Bröcker 2012;
Kumar et al. 2019) — verified against Kumar's reference implementation, exact on 18 of
24 cases. `sweep_calibration_error` chooses the bin count instead of fixing it, adding
bins while the calibration curve stays monotone and stopping when it doesn't (Roelofs
et al. 2022). Both use equal-mass bins, and neither ever splits a group of tied
predictions across a bin boundary.

Also available: `maximum_calibration_error`, `binned_calibration_error`,
`calibration_curve`, `correlation_metrics`, `unique_value_counts`,
and `tie_preservation_score`.

### Get every number at once

`calibration_report` runs the whole battery and prints it, so you do not pick the
one metric that flatters the model:

```python
import numpy as np

from calibre import calibration_report

rng = np.random.default_rng(0)
p = rng.uniform(0, 1, 2000)
y = rng.binomial(1, np.clip(p * 1.2, 0, 1)).astype(float)

print(calibration_report(y, p))
# > CalibrationReport  n=2,000  base rate 0.5760
# >
# >   Brier            0.1480
# >     = MCB          0.0110   (recalibration recovers this)
# >     - DSC          0.1072   (earned by the forecasts)
# >     + UNC          0.2442   (irreducible)
# >
# >   bias             0.0771   (mean forecast 0.4989)
# >   smECE            0.0769   (bandwidth 0.0771, chosen)
# >   debiased ECE     0.0871   (15 bins)
# >   plugin ECE       0.0929   (15 bins, uncorrected)
# >   sweep ECE        0.0771   (10 bins, chosen)
# >
# >   distinct values  2,000 of 2,000 (100.0%)
```

`smooth_calibration_error` is smECE, from Błasiok & Nakkiran (2024). It is the one
to reach for if you want a single number: unlike binned ECE it is *consistent* —
it goes to zero if and only if the forecaster is calibrated — and it has no bin
count for you to choose, which means no bin count for you to choose badly. calibre
pins it against the authors' own `relplot` implementation to 1.1e-16.

Every field is also available as an attribute (`report.brier`, `report.smece`, …)
rather than only as text.

### Put an interval on it

A calibration error computed on 2,000 rows is an estimate, and estimates deserve
intervals:

```python
import numpy as np

from calibre import bootstrap_ci
from calibre.metrics import brier_score, smooth_calibration_error

rng = np.random.default_rng(0)
p = rng.uniform(0, 1, 2000)
y = rng.binomial(1, p).astype(float)  # calibrated by construction

for name, metric in (("Brier", brier_score), ("smECE", smooth_calibration_error)):
    ci = bootstrap_ci(metric, y, p, n_resamples=400, random_state=0)
    print(f"{name:6s} {ci['estimate']:.4f}  [{ci['lower']:.4f}, {ci['upper']:.4f}]")
# > Brier  0.1604  [0.1516, 0.1684]
# > smECE  0.0223  [0.0199, 0.0226]
```

Look at the smECE row: the point estimate sits at the *top* of its interval. That
is not a bug, it is the correction working. **The naive bootstrap is biased upward
on calibration errors, and worst exactly when the model is well calibrated** —
which is when you most want to trust the number.

The reason is Jensen's inequality. Miscalibration is a convex functional of the
empirical distribution, so averaging it over resamples overshoots its value at the
centre. The truth here is zero by construction, and the percentile interval would
not contain it. `bootstrap_ci` therefore defaults to the bias-corrected interval
(`method="bc"`; `"percentile"`, `"basic"` and `"bca"` are also available). The
predicted inflation factor of √2 is measured at 1.40–1.42 and is invariant in `n`;
`experiments/bootstrap_bias/` reproduces the whole argument, including the linear
control — Brier, being linear, shows no bias at all.

### Measure it honestly

Scoring a calibrator on the data it was fit to does not merely flatter it. For any
isotonic-family calibrator it reports **perfect calibration by construction**, because
the calibrator and the diagnostic are the same PAV projection and PAV is idempotent. The
number is zero no matter how badly the model generalises:

```python
import numpy as np

from calibre import IsotonicCalibrator, cross_val_calibrate, score_decomposition

rng = np.random.default_rng(0)
scores = rng.uniform(0, 1, 1500)
labels = rng.binomial(1, scores).astype(float)

in_sample = IsotonicCalibrator().fit(scores, labels).transform(scores)
out_of_fold = cross_val_calibrate(IsotonicCalibrator(), scores, labels, cv=5)

print(f"MCB in-sample    {score_decomposition(in_sample, labels)['MCB']:.4f}")
print(f"MCB out-of-fold  {score_decomposition(out_of_fold, labels)['MCB']:.4f}")
# > MCB in-sample    0.0000
# > MCB out-of-fold  0.0030
```

`cross_val_calibrate` returns out-of-fold probabilities: each one comes from a model
that never saw that observation. Use those for any number you intend to believe.

### Decompose the score

`score_decomposition` splits a proper score into the three things you actually want to
know, following the CORP approach of Dimitriadis, Gneiting & Jordan (2021). It uses
isotonic regression to find the bins, so there is no bin count to choose and none to
tune in your favour:

```python
import numpy as np

from calibre import score_decomposition

rng = np.random.default_rng(0)
scores = rng.uniform(0, 1, 3000)
labels = rng.binomial(1, scores).astype(float)
overconfident = np.clip(1.6 * (scores - 0.5) + 0.5, 0, 1)

for name, x in (("honest", scores), ("overconfident", overconfident)):
    d = score_decomposition(x, labels)
    print(
        f"{name:14s} Brier {d['mean_score']:.4f} = "
        f"MCB {d['MCB']:.4f} - DSC {d['DSC']:.4f} + UNC {d['UNC']:.4f}"
    )
# > honest         Brier 0.1670 = MCB 0.0030 - DSC 0.0859 + UNC 0.2500
# > overconfident  Brier 0.1799 = MCB 0.0141 - DSC 0.0841 + UNC 0.2500
```

`MCB` is what recalibration would save you, `DSC` is what your scores buy over always
predicting the base rate, and `UNC` is the difficulty of the problem, which no
forecaster can change.

The split earns its keep here. Overconfidence cost 0.0129 of Brier score, and the
decomposition says where it went: `MCB` rose by 0.0111 — recoverable, just recalibrate —
while `DSC` fell by 0.0018, which is not recoverable. That small drop is the clipping at
0 and 1 collapsing 3000 distinct scores to 1841 and destroying ranking information with
them. A plain Brier score tells you the model got worse; this tells you which part you
can fix.

`mean_score = MCB - DSC + UNC` holds exactly, and both `MCB` and `DSC` are non-negative
by construction.

These numbers are pinned against R's `reliabilitydiag` to 1e-16 in the test suite.
`consistency_bands` and `confidence_bands` add resampling-based uncertainty.

### Inspect where a fit went flat

```python
import numpy as np

from calibre import IsotonicCalibrator, run_plateau_diagnostics

rng = np.random.default_rng(0)
scores = np.sort(rng.random(400))
labels = (rng.random(400) < scores).astype(float)

calibrator = IsotonicCalibrator().fit(scores, labels)
report = run_plateau_diagnostics(scores, calibrator.transform(scores))

print(f"{report['n_plateaus']} plateaus")
for plateau in report["plateaus"][:3]:
    low, high = plateau["x_range"]
    print(
        f"  [{low:.3f}, {high:.3f}] -> {plateau['value']:.3f} "
        f"({plateau['n_samples']} samples, {plateau['sample_density']})"
    )
# > 16 plateaus
# >   [0.000, 0.006] -> 0.000 (3 samples, very_sparse)
# >   [0.010, 0.163] -> 0.017 (58 samples, adequate)
# >   [0.163, 0.280] -> 0.103 (39 samples, adequate)
```

Plateaus flagged `very_sparse` rest on few observations. `report["warnings"]` collects
those as readable messages.

### Multiclass: find out which method you need

There is no single best multiclass calibration method. There are two regimes with
different winners, and picking wrong costs you roughly a factor of six. Measured against
**known** true probabilities, 12 seeds, 5 classes:

| miscalibration | uncalibrated | temperature | per-class (CIR) |
|---|---|---|---|
| global | 0.0821 | **0.0025** | 0.0165 |
| class-dependent | 0.1043 | 0.0849 | **0.0173** |

Temperature scaling applies one parameter to every class, so when the distortion really
is global it is exactly right — and when it differs by class it barely helps at all
(0.1043 → 0.0849). So measure before you choose:

```python
import numpy as np

from calibre import miscalibration_profile

rng = np.random.default_rng(0)
truth = rng.dirichlet(np.ones(5) * 0.7, size=4000)
labels = np.array([rng.choice(5, p=t) for t in truth])

# Each class distorted by a different exponent.
skewed = truth ** np.linspace(0.6, 2.4, 5)
scores = skewed / skewed.sum(axis=1, keepdims=True)

profile = miscalibration_profile(scores, labels)
print(f"spread {profile['spread']:.2f}")
print(profile["reading"])
# > spread 0.96
# > Miscalibration is concentrated in classes 0, 4, 3 (spread 0.96). A one-parameter method applies the same correction to every class and cannot express this; per-class calibration is likely to help.
```

A spread near 0.13 means the miscalibration is even across classes and
`TemperatureScaler` will likely capture it; 0.4 and above means it is concentrated and a
one-parameter method cannot express the fix.

Also available: `classwise_decomposition` (the MCB/DSC/UNC split per class),
`classwise_ece`, `top_label_ece`, and `classwise_reliability`.

One cost worth knowing, because no standard metric shows it: `TemperatureScaler` never
changes the predicted class — accuracy is exactly preserved — but it **does** reorder
people *within* a class, at 49.6% of adjacent pairs in our measurements. If you rank
individuals by their probability of a given class, that reordering is real.

## See it

```bash
pip install 'calibre[plots]'
```

matplotlib is an optional extra. Importing calibre pulls in nothing new without it,
and a subprocess test enforces that.

**The collapse barcode.** One thin tick per distinct output value, one strip per
method, drawn over the input range. The number of ticks *is* the number of distinct
values, so the loss is not asserted — it is visible.

![Resolution retained by each calibrator](https://finite-sample.github.io/calibre/_static/bench/resolution_loss.png)

scikit-learn's isotonic strip is sparse enough to count by eye. The calibre strips
are solid ink. Same data, same held-out Brier to the fourth decimal.

**The frontier.** The obvious objection to the barcode is that the extra values
might be noise. If they were, the methods keeping them would sit higher on the
score axis:

![Held-out score against distinct values retained](https://finite-sample.github.io/calibre/_static/bench/resolution_frontier.png)

They do not. The frontier is flat: two clusters four decades apart in resolution,
at the same height.

```python
import matplotlib

matplotlib.use("Agg")  # not needed interactively
import numpy as np

from calibre import CenteredIsotonicCalibrator, IsotonicCalibrator
from calibre.plots import plot_resolution_loss

rng = np.random.default_rng(0)
scores = rng.uniform(0, 1, 2000)
labels = rng.binomial(1, scores).astype(float)

ax = plot_resolution_loss(
    {
        "isotonic": IsotonicCalibrator().fit_transform(scores, labels),
        "centered": CenteredIsotonicCalibrator().fit_transform(scores, labels),
    },
    scores,
)
print("strips:", [t.get_text() for t in ax.get_yticklabels()])
# > strips: ['isotonic', 'centered']
```

Nine functions in all: `plot_reliability_diagram` (the CORP diagram, with
consistency or confidence bands), `plot_score_decomposition`, `plot_mcb_dsc_plane`,
`plot_resolution_loss`, `plot_resolution_frontier`, `plot_calibrator_comparison`,
`plot_ece_bin_sensitivity`, `plot_miscalibration_profile` and
`plot_classwise_reliability`.
Every one returns the `Axes` or `Figure`, so you keep full control of titles,
limits and saving. The palette is Okabe-Ito, which stays legible with any common
form of color blindness.

## Documentation

- [Full documentation](https://finite-sample.github.io/calibre/)
- [API reference](https://finite-sample.github.io/calibre/api/)
- [Worked examples](https://finite-sample.github.io/calibre/examples/)
- [Performance comparison](https://finite-sample.github.io/calibre/notebooks/04_performance_comparison.html)

## Contributing

```bash
git clone https://github.com/finite-sample/calibre.git
cd calibre
uv sync --all-groups
uv run pytest
```

The monotone and isotonic estimators are checked against reference implementations in
R — `isotone::gpava`, `Iso::pava`, `cir::cirPAVA`, `neariso` and `scam` — via committed
fixtures in `tests/fixtures/r/`. See `experiments/r_reference/gen_fixtures.R` for how
those were produced. Issues and pull requests welcome; please open an issue first for
anything large.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{calibre,
  title  = {calibre: Probability Calibration that Preserves Granularity},
  author = {Sood, Gaurav},
  url    = {https://github.com/finite-sample/calibre}
}
```

## References

- Oron & Flournoy (2017), "Centered Isotonic Regression: Point and Interval Estimation
  for Dose–Response Studies", *Statistics in Biopharmaceutical Research* 9(3).
- Tibshirani, Höfling & Tibshirani (2011), "Nearly-Isotonic Regression",
  *Technometrics* 53(1), 54–61.
- Pya & Wood (2015), "Shape constrained additive models", *Statistics and Computing*
  25(3), 543–559.
- Eilers & Marx (1996), "Flexible smoothing with B-splines and penalties",
  *Statistical Science* 11(2), 89–121.
- [Probability calibration in scikit-learn](https://scikit-learn.org/stable/modules/calibration.html)

## 🔗 Adjacent Repositories

- [finite-sample/streamcal](https://github.com/finite-sample/streamcal) — Always‑On Probability Calibration via Multiplicative‑Weights. Comparison to Batch Platt & Isotonic
- [finite-sample/rank-preserving-calibration](https://github.com/finite-sample/rank-preserving-calibration) — Rank preserving calibration of multiclass prob.
- [finite-sample/optimal-classification-cutoffs](https://github.com/finite-sample/optimal-classification-cutoffs) — Cutoffs for max. multiclass F1-score, etc.
- [finite-sample/winference](https://github.com/finite-sample/winference) — Calibrating pairwise rankings with accommodations for non-transitivity
- [finite-sample/tworeg](https://github.com/finite-sample/tworeg) — Two Regressions

✨ _Powered by [Adjacent](https://github.com/gojiplus/adjacent)_ 🚀
