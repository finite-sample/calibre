"""Fit, select, and evaluate calibration policies on separate samples."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from calibre import (
    CenteredIsotonicCalibrator,
    DecisionPolicy,
    DecisionTask,
    IsotonicCalibrator,
    evaluate_decision_policy,
    select_decision_policy,
)

X, y = make_classification(n_samples=2400, random_state=7)
train, calibration, validation, test = np.split(
    np.random.default_rng(8).permutation(2400), 4
)
model = LogisticRegression().fit(X[train], y[train])
scores = model.predict_proba(X)[:, 1]
calibrators = {
    "isotonic": IsotonicCalibrator().fit(scores[calibration], y[calibration]),
    "centered": CenteredIsotonicCalibrator().fit(scores[calibration], y[calibration]),
}
validation_predictions = {
    name: calibrator.transform(scores[validation])
    for name, calibrator in calibrators.items()
}
test_predictions = {
    name: calibrator.transform(scores[test]) for name, calibrator in calibrators.items()
}
validation_predictions["original"] = scores[validation]
test_predictions["original"] = scores[test]
policies = {
    "isotonic": DecisionPolicy(prediction="isotonic", rule="rank"),
    "centered": DecisionPolicy(prediction="centered", rule="rank"),
    "isotonic_with_score": DecisionPolicy(
        prediction="isotonic", rule="rank", tie_breaker="original"
    ),
    "original": DecisionPolicy(prediction="original", rule="rank"),
}
selection = select_decision_policy(
    y[validation],
    validation_predictions,
    task=DecisionTask(fraction=0.2),
    policies=policies,
    reference="original",
    case_ids=validation,
)
evaluation = evaluate_decision_policy(
    selection,
    y[test],
    test_predictions,
    case_ids=test,
)
selected_value = evaluation.values[selection.selected]
selected_difference = evaluation.differences[selection.selected]
selected_interval = evaluation.intervals[selection.selected]
