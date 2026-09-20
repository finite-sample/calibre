"""Decision payoffs, tie lotteries, and separation of selection from evaluation."""

from dataclasses import FrozenInstanceError
from itertools import combinations

import numpy as np
import pytest

from calibre import (
    DecisionPolicy,
    DecisionTask,
    decision_report,
    evaluate_decision_policy,
    select_decision_policy,
)


def report(y, columns, *, task=None, policies=None, **kwargs):
    if policies is None:
        policies = {"model": DecisionPolicy(prediction="p")}
    return decision_report(
        y,
        columns,
        task=task or DecisionTask(cost=0.5),
        policies=policies,
        reference=next(iter(policies)),
        n_resamples=kwargs.pop("n_resamples", 0),
        **kwargs,
    )


def test_threshold_values_baselines_and_equality():
    result = report(
        [0, 1, 0, 1],
        {"p": [0, 0.5, 0.8, 1]},
        task=DecisionTask(benefit=2, cost=1),
        policies={
            "model": DecisionPolicy(prediction="p"),
            "all": DecisionPolicy(rule="all"),
            "none": DecisionPolicy(rule="none"),
        },
    )
    assert result.values == {"model": 0, "all": 0, "none": 0}
    assert result.action_rates == {"model": 0.5, "all": 1, "none": 0}
    assert result.intervals == {}
    assert result.interval_level is None


@pytest.mark.parametrize(("cost", "rate"), [(0, 0.75), (1, 0), (2, 0)])
def test_threshold_endpoints(cost, rate):
    result = report([0, 1, 0, 1], {"p": [0, 0.5, 0.8, 1]}, task=DecisionTask(cost=cost))
    assert result.action_rates["model"] == rate


@pytest.mark.parametrize("slots", range(6))
def test_capacity_matches_enumerated_uniform_lottery(slots):
    y = np.array([1, 0, 0, 1, 1])
    allocations = list(combinations(range(5), slots))
    expected = np.mean([sum(y[list(indices)]) / 5 for indices in allocations])
    result = report(
        y,
        {"p": np.ones(5)},
        task=DecisionTask(capacity=slots),
        policies={"model": DecisionPolicy(prediction="p", rule="rank")},
    )
    assert result.values["model"] == pytest.approx(expected)
    assert result.action_rates["model"] == pytest.approx(slots / 5)


def test_boundary_ties_original_score_and_remaining_ties():
    result = report(
        [0, 1, 0, 1],
        {"p": [0.2, 0.8, 0.8, 0.8], "s": [-3, 4, 2, 4]},
        task=DecisionTask(capacity=1),
        policies={
            "pooled": DecisionPolicy(prediction="p", rule="rank"),
            "retained": DecisionPolicy(prediction="p", rule="rank", tie_breaker="s"),
            "original": DecisionPolicy(prediction="s", rule="rank"),
        },
    )
    assert result.values["pooled"] == pytest.approx(1 / 6)
    assert result.values["retained"] == 0.25
    assert result.values["original"] == 0.25
    assert result.differences["retained"] == pytest.approx(1 / 12)


def test_rank_payoff_is_invariant_to_row_order_and_strict_monotone_transform():
    rng = np.random.default_rng(17)
    scores = rng.integers(-5, 5, 30)
    secondary = rng.integers(-5, 5, 30)
    y = rng.integers(0, 2, 30)
    policies = {"model": DecisionPolicy(prediction="p", rule="rank", tie_breaker="s")}
    first = report(
        y,
        {"p": scores, "s": secondary},
        task=DecisionTask(fraction=0.27),
        policies=policies,
    )
    order = rng.permutation(30)
    second = report(
        y[order],
        {"p": np.exp(scores[order]), "s": secondary[order]},
        task=DecisionTask(capacity=8),
        policies=policies,
    )
    assert first.values == pytest.approx(second.values)
    assert first.action_rates["model"] == pytest.approx(8 / 30)


def test_selection_is_frozen_and_test_does_not_reselect():
    policies = {
        "a": DecisionPolicy(prediction="a"),
        "b": DecisionPolicy(prediction="b"),
    }
    columns = {"a": [0.1, 0.9], "b": [0.9, 0.1]}
    selected = select_decision_policy(
        [0, 1],
        columns,
        task=DecisionTask(cost=0.5),
        policies=policies,
        reference="b",
        case_ids=["v0", "v1"],
    )
    policies.clear()
    result = evaluate_decision_policy(
        selected, [1, 0], columns, case_ids=["t0", "t1"], n_resamples=0
    )
    assert selected.selected == "a"
    assert result.values["b"] > result.values["a"]
    assert result.reference == "b"
    with pytest.raises(TypeError):
        selected.validation.policies["new"] = DecisionPolicy(rule="none")
    with pytest.raises(FrozenInstanceError):
        selected.selected = "b"
    with pytest.raises(ValueError, match="overlap"):
        evaluate_decision_policy(selected, [1, 0], columns, case_ids=["v0", "t1"])


def test_first_candidate_wins_validation_tie():
    selected = select_decision_policy(
        [0, 1],
        {},
        task=DecisionTask(cost=0.5),
        policies={
            "none": DecisionPolicy(rule="none"),
            "all": DecisionPolicy(rule="all"),
        },
        reference="none",
        case_ids=[1, 2],
    )
    assert selected.selected == "none"


def test_bootstrap_is_paired_seeded_and_recomputes_capacity():
    y = np.array([0, 1, 1, 0])
    scores = np.array([1, 4, 3, 2])
    policies = {
        "a": DecisionPolicy(prediction="a", rule="rank"),
        "b": DecisionPolicy(prediction="b", rule="rank"),
    }
    kwargs = {
        "task": DecisionTask(capacity=2),
        "policies": policies,
        "n_resamples": 100,
        "random_state": 9,
    }
    first = report(y, {"a": scores, "b": -scores}, **kwargs)
    second = report(y, {"a": scores, "b": -scores}, **kwargs)
    assert first.intervals == second.intervals
    assert first.intervals["a"] == (0, 0)
    rng = np.random.default_rng(9)
    differences = []
    for _ in range(100):
        indices = rng.integers(0, 4, 4)
        sample_y, sample_s = y[indices], scores[indices]
        # Enumerate equally likely allocations satisfying the top-two rule.
        values = []
        for direction in [1, -1]:
            ranks = direction * sample_s
            boundary = sorted(ranks, reverse=True)[1]
            allocations = [
                pair
                for pair in combinations(range(4), 2)
                if min(ranks[list(pair)]) >= boundary
                and all(i in pair for i in np.flatnonzero(ranks > boundary))
            ]
            values.append(
                np.mean([sum(sample_y[list(pair)]) / 4 for pair in allocations])
            )
        differences.append(values[1] - values[0])
    np.testing.assert_allclose(
        first.intervals["b"], np.quantile(differences, [0.025, 0.975])
    )


def test_constant_difference_has_zero_width_interval():
    result = report(
        [1] * 10,
        {},
        policies={
            "all": DecisionPolicy(rule="all"),
            "none": DecisionPolicy(rule="none"),
        },
        n_resamples=30,
    )
    assert result.intervals["none"] == (-0.5, -0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"benefit": 0},
        {"benefit": np.inf},
        {"cost": -1},
        {"cost": np.nan},
        {"capacity": -1},
        {"capacity": 1.5},
        {"capacity": True},
        {"capacity": 1, "fraction": 0.5},
        {"fraction": -0.1},
        {"fraction": np.nan},
        {"fraction": 1.1},
    ],
)
def test_invalid_tasks(kwargs):
    with pytest.raises(
        ValueError,
        match=r"benefit|cost|capacity|fraction|choose",
    ):
        DecisionTask(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"rule": "bogus"},
        {"prediction": ""},
        {"rule": "all", "prediction": "p"},
        {"prediction": "p", "tie_breaker": "s"},
        {"rule": "rank", "prediction": "p", "tie_breaker": ""},
    ],
)
def test_invalid_policies(kwargs):
    with pytest.raises(
        ValueError,
        match=r"prediction|policy|baseline|tie_breaker",
    ):
        DecisionPolicy(**kwargs)


@pytest.mark.parametrize(
    ("y", "p"),
    [
        ([], []),
        ([0, 2], [0, 1]),
        ([0, 1], [np.nan, 1]),
        ([0, 1], [0]),
        ([0, 1], [[0, 1]]),
        ([0, 1], [-1, 1]),
        ([0, 1], [0, 2]),
        ([np.inf], [0.5]),
    ],
)
def test_invalid_arrays(y, p):
    with pytest.raises(
        ValueError,
        match=r"inputs|length|binary|probabilities",
    ):
        report(y, {"p": p})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_resamples": -1},
        {"n_resamples": 1},
        {"n_resamples": True},
        {"n_resamples": 1.2},
        {"interval_level": 1},
        {"interval_level": np.nan},
        {"random_state": -1},
        {"random_state": None},
    ],
)
def test_invalid_interval_controls(kwargs):
    with pytest.raises(
        ValueError,
        match=r"n_resamples|interval_level|random_state",
    ):
        report([0, 1], {"p": [0.2, 0.8]}, **kwargs)


@pytest.mark.parametrize("ids", [[1], [1, 1], [None, 1], [float("nan"), 1]])
def test_invalid_case_ids(ids):
    with pytest.raises(
        ValueError,
        match="case_ids",
    ):
        select_decision_policy(
            [0, 1],
            {"p": [0.2, 0.8]},
            task=DecisionTask(),
            policies={"a": DecisionPolicy(prediction="p")},
            reference="a",
            case_ids=ids,
        )


@pytest.mark.parametrize(
    ("task", "policy", "columns"),
    [
        (DecisionTask(), DecisionPolicy(prediction="p", rule="rank"), {"p": [1, 2]}),
        (DecisionTask(capacity=1), DecisionPolicy(prediction="p"), {"p": [0.1, 0.9]}),
        (DecisionTask(capacity=1), DecisionPolicy(rule="all"), {}),
        (
            DecisionTask(capacity=3),
            DecisionPolicy(prediction="p", rule="rank"),
            {"p": [1, 2]},
        ),
        (DecisionTask(), DecisionPolicy(prediction="p"), {}),
        (
            DecisionTask(capacity=1),
            DecisionPolicy(prediction="p", rule="rank", tie_breaker="s"),
            {"p": [1, 2]},
        ),
    ],
)
def test_incompatible_task_or_missing_columns(task, policy, columns):
    with pytest.raises(
        ValueError,
        match=r"capacity|missing|rank policies",
    ):
        report([0, 1], columns, task=task, policies={"a": policy})


def test_documented_workflow_uses_disjoint_samples_and_expected_payoff():
    import runpy
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "docs/examples/decision_example.py"
    namespace = runpy.run_path(str(example))
    splits = [
        set(namespace[name]) for name in ("train", "calibration", "validation", "test")
    ]
    for left, right in combinations(splits, 2):
        assert left.isdisjoint(right)
    selection = namespace["selection"]
    evaluation = namespace["evaluation"]
    assert evaluation.n_observations == 600
    assert set(evaluation.policies) == set(selection.validation.policies)
    assert all(rate == pytest.approx(0.2) for rate in evaluation.action_rates.values())
    scores = namespace["scores"][namespace["test"]]
    labels = namespace["y"][namespace["test"]]
    expected = labels[np.argsort(scores)[-120:]].sum() / 600
    assert evaluation.values["original"] == pytest.approx(expected)
    assert evaluation.intervals["original"] == (0, 0)


def test_probability_score_and_payoff_can_rank_candidates_differently():
    from calibre import brier_score

    y = [0, 1]
    predictions = {"accurate": [0.1, 0.59], "useful": [0.59, 0.61]}
    assert brier_score(y, predictions["accurate"]) < brier_score(
        y, predictions["useful"]
    )
    result = report(
        y,
        predictions,
        task=DecisionTask(cost=0.6),
        policies={key: DecisionPolicy(prediction=key) for key in predictions},
    )
    assert result.values["useful"] > result.values["accurate"]


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_direct_selection_rejects_missing_validation_ids(missing):
    from calibre import DecisionSelection

    validation = report([0, 1], {"p": [0.2, 0.8]})
    with pytest.raises(ValueError, match="case_ids cannot be missing"):
        DecisionSelection("model", validation, frozenset([missing, "validation-1"]))
