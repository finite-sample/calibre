# Package changes implied by the decision argument

Implemented as an additive layer after author approval. Existing calibrators,
proper-score selection, and probability reports retain their interfaces.

## Implemented API

`calibre.decisions` adds `DecisionTask`, `DecisionPolicy`, `decision_report`,
`select_decision_policy`, and `evaluate_decision_policy`, with immutable reports
and selection records. All are also exported from `calibre`.

- Cost-based policies use payoff `a * (benefit * Y - cost)` and act strictly above
  the implied probability threshold; act-all/none are explicit comparators.
- Capacity policies fill integer slots or `floor(fraction * n)`, with exact
  expected boundary lotteries and optional original-score tie-breaking.
- Selection maximizes validation payoff for one declared objective, breaking ties
  by candidate order. Test evaluation keeps candidate rules and reference frozen.
- Unique validation/test case identifiers are required and overlap is rejected.
- Paired percentile intervals use independent-case bootstrap resampling,
  recomputing capacity allocations. They condition on fitting and selection.

The executable [four-sample example](../../docs/examples/decision_example.py)
separates classifier training, calibration, selection, and testing. Behavioral
checks live in `tests/test_decisions.py`; the full existing suite checks that the
extension preserves the previous package behavior. No dependencies were added.
The original proposal follows as a record of the rationale; references to missing
operations describe the package before this implementation.

## Original proposal

The paper motivates evaluating the calibrator, reported information, and action
rule together. Calibre already handles estimation and probability diagnostics;
the missing integration is from a declared decision problem to independently
evaluated policy payoff. This proposal does not claim population optimality or
commit to new public function names.

## Capability map

| Decision requirement | Current support | Missing operation | Smallest proposed change | Validation needed |
|---|---|---|---|---|
| Fit alternative probability maps on common data | Public calibrators with fit/transform interfaces | None for this argument | Reuse existing estimators | Existing estimator tests; document fitting versus evaluation data |
| Assess probability accuracy and reliability | Public `calibration_report`, proper scores, CORP decomposition, and bootstrap utilities | Economic payoff is not part of the standard report | Keep probability diagnostics as a companion to a decision report | Show that diagnostic rankings can differ from payoff rankings |
| Evaluate action worth taking at known benefit and cost | `regret_curves` exists only in the simulation and consumes true risks | Observed-outcome payoff for a fixed action rule | Add a small evaluator for binary actions with payoff a(bY-c), explicit benefits/costs, and act-all/none comparators | Hand-calculated payoffs, equality at the threshold, extreme costs, paired comparisons |
| Evaluate fixed-capacity allocation | `capacity_values` exists only in the simulation and assumes ordered population types | General observed-data evaluation with explicit scores and ties | Add an evaluator for equal-sized slots, supporting probability ranks, original ranks, and original-score tie-breaking | Exhaustive tie lotteries, row permutation invariance, boundary capacities, and monotone allocation equivalence |
| Distinguish reporting choices | Experiment compares random ties and original-score ties | Public representation of report plus action rule | Name the policy in each result and include probability-plus-score comparators | Same inputs and budget across candidates; no accidental conversion of arbitrary scores to probabilities |
| Select for a declared decision objective | Public `select_by_cv` selects estimator parameters on log loss or Brier | Selection among complete policies using payoff | Keep fitting separate; select a named candidate on supplied validation predictions for a supplied objective | Test labels cannot affect selection; deterministic handling of equal validation values |
| Evaluate a selected policy independently | Generic bootstrap is public; no selection-to-test workflow | Frozen policy evaluation with paired uncertainty and data-role checks | Evaluate on supplied test predictions, record selection provenance, check identifiers for overlap when available | No test reselection; paired bootstrap; explain that identifiers cannot certify upstream leakage absence |
| Explain why pooling matters | Experiment-local true-risk decomposition | An observed-data grouping-loss estimator with valid assumptions | Do not expose the oracle decomposition as an observed-label metric | Preserve known-risk identities as simulation checks; document non-identification from unique predictions |

Implementation locations examined: `calibre/report.py`, `calibre/evaluation.py`,
`calibre/selection.py`, and `experiments/granularity/study.py`. The current
experiment functions are not general-purpose production interfaces: the capacity
routine uses population order as the original ranking and permits fractional
boundary allocation. A public evaluator would need explicit score inputs and an
explicit capacity convention rather than silently inheriting those assumptions.

## Recommended sequence

First add evaluation of fixed policies for the two decision families. Accept
aligned held-out outcomes and predictions; do not refit calibrators or infer an
application's costs. Report payoff, action rate, baseline differences, and the
information/tie rule used. Users must explicitly choose any weights across cost
or capacity scenarios; do not combine the two families into one winner.

Then connect validation selection to frozen test evaluation. The selected policy
is the empirical validation winner among the declared candidates for the supplied
objective. It is not the population-optimal policy. A complete example must keep
underlying-model training, calibration, policy selection, and final evaluation
separate. Existing proper-score tuning remains available inside calibration.

Add paired uncertainty for observed-data evaluation with assumptions stated.
Independent-case resampling is suitable only for independent cases; deployment
batches, clustered outcomes, and time dependence require a separately justified
scheme. Capacity rules must be recomputed for resampled cohorts, and their
intervals must be distinguished from the current simulation's seed-bootstrap
intervals. None of this requires a new calibrator.

## Resolved implementation choices

- Fixed-policy evaluation, validation selection, and frozen test evaluation ship together.
- Capacity supports integer slots or a fraction rounded down to integer slots.
  Fractional action probabilities represent tie lotteries, not fractional slots.
- Uncertainty uses independent cases. Identifier checks detect declared overlap;
  users remain responsible for upstream data separation and sampling assumptions.

The author subsequently approved an expansion that preserves existing functionality.
The implementation follows those boundaries without adding a fitting algorithm.

## Contribution statement

Calibre's current contribution is a common set of calibration estimators and
probability diagnostics, accompanied by a reproducible decision-based comparison.
The proposed extension would turn that experiment-specific comparison into a
public workflow for specifying, comparing, and independently evaluating complete
calibration-and-decision procedures. Its software contribution must be established
by working capabilities and useful examples, not by claiming new decision theory.
