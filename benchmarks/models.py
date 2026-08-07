"""Base classifiers whose probabilities the calibrators are asked to fix.

Three, chosen because they fail in different ways and one of them barely fails at
all. Hyperparameters are fixed and unexplained by design: tuning the base model
would change what the calibrator is being asked to correct, and the benchmark is
about calibrators.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MODELS", "build"]

# Name -> a short note on what this model contributes to the comparison.
MODELS: dict[str, str] = {
    # Nearly calibrated already. Its job is to show that calibration does not
    # *hurt* when there is nothing to fix -- the cost every recommendation to
    # calibrate quietly carries.
    "logreg": "logistic regression, near-calibrated",
    # Miscalibrated toward the middle, and it emits heavily tied scores (k/300),
    # so it stresses tie handling and the granularity claim at the same time.
    "rf": "random forest, miscalibrated and heavily tied",
    # Overconfident at the extremes, which is the failure calibration is usually
    # reached for.
    "gbdt": "histogram gradient boosting, overconfident at the tails",
}


def build(name: str, seed: int) -> Any:
    """Construct one base model inside a preprocessing pipeline.

    Imputation and encoding live inside the pipeline so they are fitted on the
    training fold only. Doing them outside would leak test information into the
    scores the calibrator then learns from, which is the subtler cousin of the
    mistake the README warns about.

    Args:
        name: One of :data:`MODELS`.
        seed: Random state for the estimator.

    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline.

    Raises:
        ValueError: If the name is unknown.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(MODELS)}")

    if name == "logreg":
        estimator = LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=1
        )
    else:
        estimator = HistGradientBoostingClassifier(random_state=seed)

    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    prepare = ColumnTransformer(
        [
            ("numeric", numeric, _numeric_columns),
            ("categorical", categorical, _categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline([("prepare", prepare), ("model", estimator)])


def _numeric_columns(frame: Any) -> Any:
    """Select numeric columns, tolerating a plain array.

    Args:
        frame: Feature matrix, a DataFrame or an ndarray.

    Returns:
        list or slice: Column selector.
    """
    import numpy as np

    if hasattr(frame, "select_dtypes"):
        return list(frame.select_dtypes(include=["number", "bool"]).columns)
    return list(range(np.asarray(frame).shape[1]))


def _categorical_columns(frame: Any) -> Any:
    """Select non-numeric columns, tolerating a plain array.

    Args:
        frame: Feature matrix, a DataFrame or an ndarray.

    Returns:
        list: Column selector, empty for an ndarray.
    """
    if hasattr(frame, "select_dtypes"):
        return list(frame.select_dtypes(exclude=["number", "bool"]).columns)
    return []
