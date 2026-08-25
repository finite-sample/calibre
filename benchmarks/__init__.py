"""Reproducible benchmark for calibre's calibrators.

Run with::

    python -m benchmarks.run --quick     # offline, fast, what CI runs
    python -m benchmarks.run             # the committed grid
    python -m benchmarks.aggregate       # raw.csv -> summary.csv, paired.csv
    python -m benchmarks.figures         # summary.csv -> docs/_static/bench/

This package is not shipped in the wheel: ``[tool.hatch.build.targets.wheel]``
lists ``calibre`` only.

Why it exists
-------------
The claims this package makes -- that its calibrators match isotonic regression
on score while keeping two decades more resolution -- were previously asserted in
the README with no script behind them, and on a docs page carrying star ratings
that no measurement produced. This directory is the provenance.

How it stays honest
-------------------
The design choices that could be tuned to flatter calibre live in
:mod:`benchmarks.config`, in one diffable file, and the rule stated in
``benchmarks/README.md`` is that config and results do not change in the same
commit without the message saying why.

* Every calibrator in a cell sees the *identical* out-of-fold scores and the
  *identical* test scores, so the calibrator is the only thing that varies.
* The test split is touched exactly once, at the end.
* Calibrator hyperparameters are library defaults. Tuning calibre against an
  untuned isotonic baseline would settle the comparison by construction.
* Comparisons are **paired per seed** against the baseline, with a bootstrap
  interval, because seed variance dwarfs the effect being measured.
* There is no composite score. Resolution stays a separate axis from score;
  folding them into one number is where a thumb goes on the scale.
* ``calibre_isotonic`` must reproduce ``sklearn_isotonic`` to 1e-12 on every row.
  If calibre's wrapper ever diverges, the benchmark fails loudly rather than
  showing calibre spuriously ahead.
* Regimes where calibre is expected to lose -- ``nonmonotone`` above all -- are
  included at full weight and named on the docs page.
"""

from __future__ import annotations
