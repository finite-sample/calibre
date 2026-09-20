"""Evaluate binary decisions from already fitted predictions.

Payoffs describe selection of observed outcomes, not causal treatment effects.
Bootstrap intervals condition on model fitting and validation selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, kw_only=True)
class DecisionTask:
    """Specify one payoff objective, without averaging across scenarios.

    Attributes:
        benefit: Value of selecting a positive outcome; strictly positive.
        cost: Cost per selected case, including negative outcomes.
        capacity: Integer slots, or None for a probability threshold task.
        fraction: Fraction of cases to select, rounded down to integer slots.
            Cannot be combined with capacity. None means no fraction constraint.
    """

    benefit: float = 1.0
    cost: float = 0.0
    capacity: int | None = None
    fraction: float | None = None

    def __post_init__(self) -> None:
        """Validate payoff and capacity controls.

        Raises:
            ValueError: If the specification is invalid.
        """
        if not np.isfinite(self.benefit) or self.benefit <= 0:
            raise ValueError("benefit must be finite and positive")
        if not np.isfinite(self.cost) or self.cost < 0:
            raise ValueError("cost must be finite and nonnegative")
        if self.capacity is not None:
            if (
                isinstance(self.capacity, bool)
                or not isinstance(self.capacity, Integral)
                or self.capacity < 0
            ):
                raise ValueError("capacity must be a nonnegative integer")
            if self.fraction is not None:
                raise ValueError("choose capacity or fraction, not both")
        if self.fraction is not None and (
            not np.isfinite(self.fraction) or not 0 <= self.fraction <= 1
        ):
            raise ValueError("fraction must be finite and in [0, 1]")


@dataclass(frozen=True, kw_only=True)
class DecisionPolicy:
    """Specify a prediction column and an outcome-blind action rule.

    Attributes:
        prediction: Key in the supplied prediction mapping; None for baselines.
        rule: Probability threshold, capacity ranking, act all, or act none.
        tie_breaker: Optional score column for resolving capacity ranking ties.
            Remaining ties use the exact expectation of a uniform lottery.
    """

    prediction: str | None = None
    rule: Literal["threshold", "rank", "all", "none"] = "threshold"
    tie_breaker: str | None = None

    def __post_init__(self) -> None:
        """Validate prediction and tie-rule declarations.

        Raises:
            ValueError: If the specification is invalid.
        """
        if self.rule not in {"threshold", "rank", "all", "none"}:
            raise ValueError("unknown policy rule")
        if self.rule in {"threshold", "rank"}:
            if not isinstance(self.prediction, str) or not self.prediction:
                raise ValueError("threshold and rank policies need a prediction key")
        elif self.prediction is not None:
            raise ValueError("baseline policies do not use predictions")
        if self.tie_breaker is not None and (
            self.rule != "rank"
            or not isinstance(self.tie_breaker, str)
            or not self.tie_breaker
        ):
            raise ValueError("tie_breaker must be a nonempty key for a rank policy")


@dataclass(frozen=True)
class DecisionReport:
    """Store payoffs per eligible case and paired comparisons.

    Attributes:
        task: Objective used for every policy.
        policies: Frozen policy definitions.
        n_observations: Number of evaluation cases.
        values: Mean observed payoff, averaged over any boundary lottery.
        action_rates: Expected fraction selected.
        reference: Policy used as the comparison baseline.
        differences: Policy payoff minus reference payoff.
        intervals: Paired percentile bootstrap intervals for differences.
        interval_level: Requested confidence level, or None without intervals.
        n_resamples: Number of independent-case bootstrap draws.
        random_state: Bootstrap seed.
    """

    task: DecisionTask
    policies: Mapping[str, DecisionPolicy]
    n_observations: int
    values: Mapping[str, float]
    action_rates: Mapping[str, float]
    reference: str
    differences: Mapping[str, float]
    intervals: Mapping[str, tuple[float, float]]
    interval_level: float | None
    n_resamples: int
    random_state: int

    def __post_init__(self) -> None:
        """Copy mappings into read-only snapshots."""
        for name in ("policies", "values", "action_rates", "differences", "intervals"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True)
class DecisionSelection:
    """Freeze the empirical validation winner and its evaluation specification.

    Attributes:
        selected: Name of the highest validation payoff policy; first on ties.
        validation: Validation report, including all candidate policy definitions.
        case_ids: Unique validation identifiers for checking test separation.
    """

    selected: str
    validation: DecisionReport
    case_ids: frozenset[Hashable]

    def __post_init__(self) -> None:
        """Freeze identifiers and validate selection provenance.

        Raises:
            ValueError: If the specification is invalid.
        """
        object.__setattr__(
            self,
            "case_ids",
            _ids(tuple(self.case_ids), self.validation.n_observations),
        )
        if self.selected not in self.validation.policies:
            raise ValueError("selected must name a validation policy")


def _vector(value: ArrayLike, n: int | None = None) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError("inputs must be nonempty finite one-dimensional arrays")
    if n is not None and array.size != n:
        raise ValueError("all inputs must have the same length")
    return array


def _ids(case_ids: Sequence[Hashable], n: int) -> frozenset[Hashable]:
    if len(case_ids) != n:
        raise ValueError("case_ids must match the number of cases")
    for value in case_ids:
        if value is None or value != value:
            raise ValueError("case_ids cannot be missing")
    result = frozenset(case_ids)
    if len(result) != n:
        raise ValueError("case_ids must be unique")
    return result


def _actions(
    task: DecisionTask,
    policy: DecisionPolicy,
    predictions: Mapping[str, NDArray[np.float64]],
    n: int,
) -> NDArray[np.float64]:
    constrained = task.capacity is not None or task.fraction is not None
    if policy.rule in {"all", "none"}:
        if constrained:
            raise ValueError("capacity tasks require rank policies at the same budget")
        return np.full(n, float(policy.rule == "all"))
    if (policy.rule == "rank") != constrained:
        raise ValueError(
            "rank policies need capacity; threshold policies need no capacity"
        )
    if policy.prediction not in predictions:
        raise ValueError(f"missing prediction column: {policy.prediction}")
    scores = predictions[str(policy.prediction)]
    if policy.rule == "threshold":
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("threshold predictions must be probabilities in [0, 1]")
        return (scores > task.cost / task.benefit).astype(float)
    slots = (
        int(task.capacity)
        if task.capacity is not None
        else int(np.floor(n * (task.fraction or 0.0)))
    )
    if slots > n:
        raise ValueError("capacity exceeds the number of cases")
    secondary = np.zeros(n)
    if policy.tie_breaker is not None:
        if policy.tie_breaker not in predictions:
            raise ValueError(f"missing tie-breaker column: {policy.tie_breaker}")
        secondary = predictions[policy.tie_breaker]
    actions = np.zeros(n)
    if slots == 0:
        return actions
    order = np.lexsort((secondary, scores))[::-1]
    boundary = order[slots - 1]
    better = (scores > scores[boundary]) | (
        (scores == scores[boundary]) & (secondary > secondary[boundary])
    )
    tied = (scores == scores[boundary]) & (secondary == secondary[boundary])
    actions[better] = 1
    actions[tied] = (slots - np.count_nonzero(better)) / np.count_nonzero(tied)
    return actions


def decision_report(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    *,
    task: DecisionTask,
    policies: Mapping[str, DecisionPolicy],
    reference: str,
    n_resamples: int = 2000,
    interval_level: float = 0.95,
    random_state: int = 0,
) -> DecisionReport:
    """Compare fixed policies on common labeled cases.

    Capacity means exactly k slots (fraction rounds down), even when a selected
    case has negative expected payoff. Rank inputs may be arbitrary finite scores.
    Threshold inputs must be probabilities; equality means no action. Ties are
    integrated out, so intervals exclude realized lottery noise. Independent-case
    bootstrap resampling recomputes capacity allocations within each draw. These
    percentile intervals are approximate, conditional on fitting and selection,
    and may be unreliable for small samples or unstable ranking boundaries.

    Args:
        y_true: Binary observed outcomes for all eligible cases.
        predictions: Aligned prediction or score columns from fitted models.
        task: Explicit payoff objective and optional capacity constraint.
        policies: Nonempty mapping of names to fixed policy definitions.
        reference: Name of a supplied policy for paired differences.
        n_resamples: Bootstrap draws; zero disables intervals.
        interval_level: Confidence level strictly between zero and one.
        random_state: Nonnegative integer bootstrap seed.

    Returns:
        DecisionReport: Values and paired differences per eligible case.

    Raises:
        ValueError: If outcomes, predictions, policies, or controls are invalid.
    """
    y = _vector(y_true)
    if np.any((y != 0) & (y != 1)):
        raise ValueError("y_true must contain binary outcomes")
    if not policies or any(not isinstance(key, str) or not key for key in policies):
        raise ValueError("policies need nonempty string names")
    if reference not in policies:
        raise ValueError("reference must name a supplied policy")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, Integral)
        or n_resamples < 0
        or n_resamples == 1
    ):
        raise ValueError("n_resamples must be zero or an integer of at least two")
    if not np.isfinite(interval_level) or not 0 < interval_level < 1:
        raise ValueError("interval_level must be in (0, 1)")
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, Integral)
        or random_state < 0
    ):
        raise ValueError("random_state must be a nonnegative integer")
    columns = {key: _vector(value, y.size) for key, value in predictions.items()}
    actions = {
        name: _actions(task, policy, columns, y.size)
        for name, policy in policies.items()
    }
    payoff = task.benefit * y - task.cost
    values = {name: float(np.mean(action * payoff)) for name, action in actions.items()}
    differences = {name: value - values[reference] for name, value in values.items()}
    intervals = {}
    if n_resamples:
        rng = np.random.default_rng(random_state)
        draws = np.empty((n_resamples, len(policies)))
        names = list(policies)
        for draw in range(n_resamples):
            indices = rng.integers(0, y.size, size=y.size)
            sampled = {key: value[indices] for key, value in columns.items()}
            for column, policy in enumerate(policies.values()):
                action = _actions(task, policy, sampled, y.size)
                draws[draw, column] = np.mean(action * payoff[indices])
        draws -= draws[:, [names.index(reference)]]
        tail = (1 - interval_level) / 2
        bounds = np.quantile(draws, [tail, 1 - tail], axis=0)
        intervals = {
            name: (float(bounds[0, i]), float(bounds[1, i]))
            for i, name in enumerate(names)
        }
    return DecisionReport(
        task=task,
        policies=policies,
        n_observations=y.size,
        values=values,
        action_rates={name: float(np.mean(action)) for name, action in actions.items()},
        reference=reference,
        differences=differences,
        intervals=intervals,
        interval_level=interval_level if n_resamples else None,
        n_resamples=n_resamples,
        random_state=random_state,
    )


def select_decision_policy(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    *,
    task: DecisionTask,
    policies: Mapping[str, DecisionPolicy],
    reference: str,
    case_ids: Sequence[Hashable],
) -> DecisionSelection:
    """Freeze the first highest-payoff policy on validation cases.

    Supply predictions from models fitted without these cases. Selection includes
    every supplied policy, including baselines. No model is fitted or refitted.

    Args:
        y_true: Binary validation outcomes.
        predictions: Aligned validation predictions from already fitted models.
        task: Decision objective to freeze for final evaluation.
        policies: Named candidate policies, in tie-preference order.
        reference: Candidate used for paired comparisons on validation and test.
        case_ids: Unique nonmissing validation case identifiers.

    Returns:
        DecisionSelection: Frozen specification and validation provenance.
    """
    report = decision_report(
        y_true,
        predictions,
        task=task,
        policies=policies,
        reference=reference,
        n_resamples=0,
    )
    selected = max(report.values, key=lambda name: report.values[name])
    return DecisionSelection(selected, report, _ids(case_ids, report.n_observations))


def evaluate_decision_policy(
    selection: DecisionSelection,
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    *,
    case_ids: Sequence[Hashable],
    n_resamples: int = 2000,
    interval_level: float = 0.95,
    random_state: int = 0,
) -> DecisionReport:
    """Evaluate frozen candidates on disjoint test cases without reselection.

    The selected policy remains ``selection.selected`` regardless of test ranks.
    ID checks detect declared validation/test overlap, not leakage during fitting
    or misleading identifiers. All candidate comparisons are descriptive pointwise
    intervals, not simultaneous coverage or permission to select again on test.

    Args:
        selection: Frozen validation selection.
        y_true: Binary test outcomes.
        predictions: Test columns from the same fitted models used in selection.
        case_ids: Unique nonmissing test identifiers, disjoint from validation.
        n_resamples: Independent-case bootstrap draws; zero disables intervals.
        interval_level: Confidence level strictly between zero and one.
        random_state: Nonnegative integer bootstrap seed.

    Returns:
        DecisionReport: Test payoffs and paired intervals with the frozen reference.

    Raises:
        ValueError: If validation and test identifiers overlap.
    """
    identifiers = _ids(case_ids, _vector(y_true).size)
    if identifiers & selection.case_ids:
        raise ValueError("validation and test case_ids overlap")
    return decision_report(
        y_true,
        predictions,
        task=selection.validation.task,
        policies=selection.validation.policies,
        reference=selection.validation.reference,
        n_resamples=n_resamples,
        interval_level=interval_level,
        random_state=random_state,
    )
