---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Multiclass: Which Method Do You Actually Need?

There is no single best multiclass calibration method. There are **two regimes**, they
have different winners, and picking wrong costs roughly a factor of six.

This notebook shows the two regimes on data where the true probabilities are known — so
error is *measured*, not estimated — then shows how to tell which one you are in from
your own data.

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np

from calibre import (
    CenteredIsotonicCalibrator,
    TemperatureScaler,
    classwise_decomposition,
    classwise_ece,
    miscalibration_profile,
    top_label_ece,
)

rng = np.random.default_rng(20260731)
```

## 1. Two kinds of miscalibration

A model can be miscalibrated in two structurally different ways.

**Globally** — every class is distorted the same way, typically overconfidence. One
number describes the whole problem.

**Per class** — some classes are over-predicted, others under. No single number can
describe it.

Both look similar in aggregate metrics. They need completely different fixes.

```{code-cell}
def make(kind, n=6000, J=5, seed=0):
    """True probabilities, labels drawn from them, and a miscalibrated model."""
    r = np.random.default_rng(seed)
    truth = r.dirichlet(np.ones(J) * 0.7, size=n)
    y = np.array([r.choice(J, p=t) for t in truth])
    if kind == "global":
        skew = truth**2.2  # one exponent for every class
    else:
        skew = truth ** np.linspace(0.6, 2.4, J)  # a different exponent per class
    return skew / skew.sum(axis=1, keepdims=True), y, truth


P_global, y_g, truth_g = make("global")
P_perclass, y_p, truth_p = make("perclass")
print("global   : same exponent applied to all 5 classes")
print("perclass : exponents 0.6 ... 2.4, one per class")
```

A note on the first line, because it is easy to get wrong and it invalidates the
whole comparison if you do. Generating miscalibration as `truth ** k` renormalised **is**
a temperature distortion — temperature scaling inverts it exactly, by construction. Using
only that generator hands temperature scaling a rigged win. The per-class generator uses
a different exponent per column, which no single temperature can express.

+++

## 2. The measurement that settles it

With the truth in hand we can measure error directly rather than estimating calibration
error from bins.

```{code-cell}
def per_class_calibrate(P_cal, y_cal, P_test):
    """One CenteredIsotonicCalibrator per class, then renormalise rows."""
    out = np.zeros_like(P_test)
    for j in range(P_cal.shape[1]):
        cal = CenteredIsotonicCalibrator().fit(P_cal[:, j], (y_cal == j).astype(float))
        out[:, j] = cal.transform(P_test[:, j])
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def compare(P, y, truth, label):
    half = len(y) // 2
    cal, test = slice(0, half), slice(half, None)
    rows = [("uncalibrated", P[test])]
    rows.append(
        ("temperature", TemperatureScaler().fit(P[cal], y[cal]).transform(P[test]))
    )
    rows.append(("per-class (CIR)", per_class_calibrate(P[cal], y[cal], P[test])))

    print(f"\n{label}")
    print(f"  {'method':18s}{'TRUE error':>12s}{'accuracy':>11s}")
    for name, Q in rows:
        err = np.abs(Q - truth[test]).mean()
        acc = (Q.argmax(1) == y[test]).mean()
        print(f"  {name:18s}{err:12.5f}{acc:11.4f}")


compare(P_global, y_g, truth_g, "GLOBAL distortion")
compare(P_perclass, y_p, truth_p, "PER-CLASS distortion")
```

Two things to read off that.

**The winner flips.** Temperature scaling is ~6x better under a global distortion and
close to useless under a per-class one. Per-class calibration is the reverse.

**Temperature scaling never changes accuracy.** It is monotone in the logits, so the
predicted class is fixed by construction. Under the per-class distortion, per-class
calibration *gains accuracy* — reordering is exactly what fixes differently-distorted
classes, and a one-parameter method cannot do it.

+++

## 3. Telling which regime you are in

You do not have the true probabilities on real data. But you do not need them: the
**spread of miscalibration across classes** distinguishes the regimes.

`miscalibration_profile` computes per-class `MCB` — the CORP miscalibration component —
and reports its coefficient of variation.

```{code-cell}
for label, P, y in (("GLOBAL", P_global, y_g), ("PER-CLASS", P_perclass, y_p)):
    prof = miscalibration_profile(y, P)
    print(
        f"{label:10s} spread={prof['relative_miscalibration_spread']:.2f}   "
        f"per-class MCB x1000 = {np.round(prof['classwise_miscalibration'] * 1000, 2)}"
    )
    print(f"           {prof['interpretation']}\n")
```

```{code-cell}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (label, P, y) in zip(
    axes, (("global", P_global, y_g), ("per-class", P_perclass, y_p))
):
    prof = miscalibration_profile(y, P)
    values = prof["classwise_miscalibration"]
    ax.bar(np.arange(len(values)), values * 1000, color="steelblue")
    spread = prof["relative_miscalibration_spread"]
    ax.set_title(f"{label} distortion (spread {spread:.2f})")
    ax.set_xlabel("class")
axes[0].set_ylabel("miscalibration (MCB x1000)")
plt.tight_layout()
plt.show()
```

Flat bars mean one global correction will do. Uneven bars mean the fix differs by
class, and no single parameter can express it.

+++

## 4. Where the miscalibration lives

`classwise_decomposition` gives the full CORP split per class. `MCB` is what
recalibration would recover, `DSC` is the discrimination your scores already provide,
`UNC` is the difficulty of that class.

```{code-cell}
parts = classwise_decomposition(y_p, P_perclass)
print(f"{'class':>6s}{'Brier':>9s}{'MCB':>9s}{'DSC':>9s}{'UNC':>9s}")
for k, d in enumerate(parts):
    print(
        f"{k:6d}{d['mean_score']:9.4f}{d['miscalibration']:9.4f}"
        f"{d['discrimination']:9.4f}{d['uncertainty']:9.4f}"
    )

ok = all(
    abs(
        d["mean_score"] - (d["miscalibration"] - d["discrimination"] + d["uncertainty"])
    )
    < 1e-12
    for d in parts
)
print(f"\nidentity mean_score = MCB - DSC + UNC holds exactly in every class: {ok}")
```

Two scalar summaries, for when you need one number. `classwise_ece` averages the
one-vs-rest error over classes; `top_label_ece` asks only whether the predicted
class's confidence is right. Both use the bias-aware, tie-safe estimators, so
neither inherits the plugin bias that grows with the bin count.

```{code-cell}
print(f"{'model':22s}{'classwise ECE':>15s}{'top-label ECE':>15s}")
for name, P, y in (
    ("global distortion", P_global, y_g),
    ("per-class distortion", P_perclass, y_p),
):
    print(f"{name:22s}{classwise_ece(y, P):15.4f}{top_label_ece(y, P):15.4f}")
```

## 5. What temperature scaling costs you

It preserves each row's class ordering — the predicted class never moves. But it does
**not** preserve the ordering of *people within a class*, because the softmax denominator
makes every calibrated probability depend on the whole row.

If you rank individuals by their probability of a given class — triage, prioritisation,
any ranked list — that reordering is real, and no standard calibration metric shows it.

```{code-cell}
def within_class_inversions(P_before, Q_after):
    bad = tot = 0
    for j in range(Q_after.shape[1]):
        order = np.argsort(P_before[:, j], kind="mergesort")
        d = np.diff(Q_after[order, j])
        bad += int((d < -1e-9).sum())
        tot += d.size
    return 100.0 * bad / tot


half = len(y_g) // 2
Q_temp = TemperatureScaler().fit(P_global[:half], y_g[:half]).transform(P_global[half:])
Q_pc = per_class_calibrate(P_global[:half], y_g[:half], P_global[half:])

print(
    f"predicted class changed by temperature scaling: "
    f"{(Q_temp.argmax(1) != P_global[half:].argmax(1)).sum()} rows"
)
print(
    f"within-class pairs inverted, temperature : "
    f"{within_class_inversions(P_global[half:], Q_temp):.1f}%"
)
print(
    f"within-class pairs inverted, per-class   : "
    f"{within_class_inversions(P_global[half:], Q_pc):.1f}%"
)
```

Zero rows change their predicted class, and roughly half of all within-class pairs
get reordered. Those two facts are both true at once, and only the first one is usually
reported.

+++

## Summary

1. **Diagnose before choosing.** `miscalibration_profile` tells you whether your
   miscalibration is global or per-class. Spread near 0.13 means global; 0.4 and above
   means per-class.
2. **Global → `TemperatureScaler`.** One parameter, cannot overfit, accuracy exactly
   preserved, ~6x better than per-class methods in this regime.
3. **Per-class → per-class calibration.** Temperature scaling barely helps, and cannot
   change accuracy even when reordering is the fix.
4. **Measure on held-out or out-of-fold predictions.** In-sample miscalibration for an
   isotonic-family calibrator is identically zero — see the binary evaluation notebook.
5. **Know what you traded.** Temperature scaling preserves predicted classes and
   destroys within-class ranking. Both matter, depending on what you do with the output.

Only *class-wise* calibration is targeted here. Canonical calibration — requiring whole
probability vectors to be jointly correct — is infeasible to verify beyond four or five
classes.
