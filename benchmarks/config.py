"""The pre-registered benchmark configuration.

Everything that could be tuned to flatter calibre lives in this one file, so a
reviewer can see the whole set of choices in one diff. The rule, stated in
``benchmarks/README.md``: do not change this file and the committed results in
the same commit without saying why in the message.
"""

from __future__ import annotations

# Seeds are fixed and committed. `aggregate.py` refuses to summarize a cell that
# is missing any of them, so a dataset that errors cannot be quietly dropped.
SEEDS: tuple[int, ...] = tuple(range(30))

# Fraction of each dataset held out for the final scoring. The test half is
# touched exactly once, at the end.
TEST_SIZE = 0.4

# Folds used to produce out-of-fold model scores for the calibrator to fit on.
CV_FOLDS = 5

# Bin count for the fixed-bin error estimators. smECE and the sweep choose their
# own, which is the point of including them.
N_BINS = 15

# Bootstrap resamples for the paired comparison against the isotonic baseline.
N_BOOTSTRAP = 2000

# The method every other method is compared against. Beating scikit-learn's
# isotonic regression is the claim this package makes, so it is the baseline.
BASELINE = "sklearn_isotonic"

# Primary metrics, declared in advance so the headline cannot be chosen after
# seeing the numbers.
PRIMARY_METRICS = ("brier", "mcb")

# Datasets that need a network fetch, and are therefore opt-in.
REMOTE_DATASETS = frozenset(
    {"credit_g", "spambase", "adult", "bank_marketing", "covtype_bin"}
)

# Datasets large enough to be slow; behind --include-large.
LARGE_DATASETS = frozenset({"adult", "bank_marketing", "covtype_bin"})

# The subset used by --quick, for CI. Offline and small.
QUICK_DATASETS = ("breast_cancer", "overconfident", "heavy_tie", "nonmonotone")
QUICK_MODELS = ("logreg", "rf")
QUICK_SEEDS = (0, 1, 2)

# Calibrators run at library defaults. Tuning calibre's methods against an
# untuned isotonic baseline would be dishonest, so there is nothing to tune here.
CALIBRATOR_DEFAULTS_ONLY = True
