Choose calibration for a decision
=================================

A more granular prediction is valuable when the distinctions change useful
actions. It can also expose estimation noise. Specify the action problem before
comparing calibrators: a probability score, rank correlation, or count of unique
predictions alone cannot measure the economic value of those distinctions.

Two objectives
--------------

For a binary outcome Y, selecting a case earns ``benefit * Y - cost``; not
selecting earns zero. Reports average this payoff over **all eligible cases**.
At a known cost, a threshold policy acts when ``p > cost / benefit``. Equality
means no action. Include ``DecisionPolicy(rule="all")`` and
``DecisionPolicy(rule="none")`` as explicit comparators. Threshold columns must
be probabilities in [0, 1]; arbitrary original scores cannot be used as such.

For capacity, ``DecisionTask(capacity=20)`` selects exactly 20 cases, and
``DecisionTask(fraction=0.2)`` selects ``floor(0.2 * n)``. These are equal-sized,
equal-value slots, filled even if the expected net payoff is negative. An
at-most-capacity rule with abstention is a different problem and is not supported.
Every compared policy uses the same budget. Rank columns can be any finite
scores. There are three useful comparisons:

* Rank the calibrated probability alone, using a uniform lottery at boundary ties.
* Rank that probability, then use the retained original score to break ties.
* Rank the original score alone.

The evaluator integrates lotteries exactly. Its payoff is the expected payoff
over those lotteries for the observed outcomes, not the outcome of a random draw.
Remaining ties in the original score are also randomized. The original score is
a comparator, not an oracle: retaining it need not improve targeting.

Separate fitting, selection, and evaluation
-------------------------------------------

The following runnable example uses four disjoint samples. The base classifier
uses only training cases, calibrators use only calibration cases, and the policy
is chosen on validation cases. Final evaluation uses test cases once. Internal
proper-score tuning can still take place within the calibration sample.

.. literalinclude:: decision_example.py
   :language: python

``selection.selected`` remains the validation winner even if another candidate
has higher test payoff. Candidate order breaks exact validation ties. The test
report includes all frozen candidates, their action rates, and paired payoff
differences against the frozen reference. The selected candidate's test interval
is available as ``evaluation.intervals[selection.selected]``. No refitting occurs
between validation and test.

Repeat this workflow for a predeclared cost or capacity when several scenarios
matter. Do not average across scenarios without application-specific weights or
choose the scenario that looks best on the test set. To evaluate a fixed set of
policies without selecting, use :func:`~calibre.decision_report` directly.

Uncertainty and interpretation
------------------------------

By default, reports use 2,000 paired independent-case bootstrap draws and a fixed
seed. Each draw resamples the same cases for every policy and recomputes capacity
allocations within the resampled cohort. Intervals are percentile intervals for
payoff **differences**, conditional on fitted models and validation selection.
They exclude lottery realization noise, training variability, and selection
variability. Coverage is approximate; small samples and unstable ranking
boundaries can make it poor. Multiple candidate intervals have pointwise, not
simultaneous, coverage. Use ``n_resamples=0`` to omit intervals.

This sampling model requires independent, representative labeled cases.
Clusters, time dependence, and fixed deployment rosters require a separately
justified uncertainty analysis. Stable, unique case identifiers are mandatory
for selection and test evaluation; their overlap is rejected. Those checks
cannot detect upstream training leakage or identifiers changed between splits.

Payoffs concern selecting observed outcomes. They do not identify treatment
benefits when the action changes the outcome, and they assume labels are observed
for all eligible cases. No oracle grouping-loss estimate is inferred from these
labels. The known-risk decomposition remains a simulation tool. Use probability
diagnostics alongside payoff comparisons to understand why policies differ.
