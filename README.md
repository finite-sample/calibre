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

On a 2,000-point held-out set, isotonic regression turns 2,000 distinct scores into
**56**. Everything inside a step becomes indistinguishable — which matters as soon as
you rank, threshold, or bucket the output.

calibre gives you calibrators that fix the probabilities *and* keep the ordering.

## Install

```bash
pip install calibre
```

Python 3.12+. Depends on numpy, scipy, scikit-learn and cvxpy.

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
non-parametric, has nothing to tune, is monotone, and has no plateaus.

| You want | Use | Notes |
|---|---|---|
| A drop-in isotonic replacement, no tuning | `CenteredIsotonicCalibrator` | Collapses isotonic's flat steps to points and interpolates. O(n). |
| A smooth curve, and you can afford cross-validation | `SplineCalibrator` | Monotone spline; picks its own smoothing by CV on log-loss. |
| A smooth curve with smoothing you control | `RegularizedIsotonicCalibrator` | Same model, you set `alpha` instead of tuning it. Fast. |
| Exactly scikit-learn's isotonic behaviour | `IsotonicCalibrator` | Thin wrapper, plus optional plateau diagnostics. |
| Guaranteed strictly increasing output | `RelaxedPAVACalibrator(min_slope=...)` | Forces a minimum step between adjacent scores. |
| To allow small ranking violations if they fit better | `NearlyIsotonicCalibrator` | `lam` trades monotonicity against fit. |
| Accuracy near specific decision thresholds | `CDIIsotonicCalibrator` | Research-grade; needs your operating thresholds. |

Every calibrator follows the scikit-learn transformer API: `.fit(scores, labels)` and
`.transform(scores)`, plus `sample_weight` where it is meaningful.

## What you actually get

Held out over 30 random datasets (an overconfident logistic model; fit on one half,
scored on the other). Lower Brier is better; ΔBrier is the improvement over leaving
the model uncalibrated.

| Method | Brier | ΔBrier | ECE | Distinct values |
|---|---|---|---|---|
| Uncalibrated | 0.1581 | — | 0.0826 | 2000 |
| `IsotonicCalibrator` | 0.1515 | +0.0066 | 0.0265 | **56** |
| `CenteredIsotonicCalibrator` | 0.1511 | +0.0070 | 0.0272 | 1874 |
| `SplineCalibrator` | **0.1509** | **+0.0072** | 0.0258 | 1999 |
| `RelaxedPAVACalibrator(min_slope=1e-5)` | 0.1515 | +0.0066 | 0.0269 | 1941 |

Against plain isotonic on held-out Brier: `CenteredIsotonicCalibrator` wins 24/30
seeds, `SplineCalibrator` 26/30, `RelaxedPAVACalibrator` 28/30.

Two things worth reading honestly off that table. The Brier gains over isotonic are
**small** — the large win is the last column, ~1900 distinct values instead of 56. And
ECE barely moves, because ECE is computed on bins and is largely blind to the
resolution you just recovered; that is a reason to be careful with ECE, not a reason
to prefer isotonic.

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

### Guarantee no ties at all

`min_slope` forces a minimum gap between every adjacent pair of scores, so no two
inputs can come out equal:

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

Also available: `maximum_calibration_error`, `binned_calibration_error`,
`calibration_curve`, `correlation_metrics`, `unique_value_counts`,
`calibration_diversity_index`, `tie_preservation_score`, `plateau_quality_score`,
`progressive_sampling_diversity`.

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

## Documentation

- [Full documentation](https://finite-sample.github.io/calibre/)
- [API reference](https://finite-sample.github.io/calibre/api/)
- [Worked examples](https://finite-sample.github.io/calibre/examples/)
- [Performance comparison](https://finite-sample.github.io/calibre/notebooks/04_performance_comparison.html)

## Contributing

```bash
git clone https://github.com/finite-sample/calibre.git
cd calibre
uv sync --all-extras --dev
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

## Adjacent Repositories

- [gojiplus/pyppur](https://github.com/gojiplus/pyppur) — pyppur: Python Projection Pursuit Unsupervised (Dimension) Reduction To Min. Reconstruction Loss or DIstance DIstortion
- [gojiplus/rmcp](https://github.com/gojiplus/rmcp) — R MCP Server
- [gojiplus/bloomjoin](https://github.com/gojiplus/bloomjoin) — bloomjoin: An R package implementing Bloom filter-based joins for improved performance with large datasets.
- [gojiplus/incline](https://github.com/gojiplus/incline) — Estimate Trend at a Point in a Noisy Time Series
