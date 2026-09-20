"""
Model definitions, training and hyperparameter tuning for the waiting-time regressor.

The challenge benchmark is a random forest regressor evaluated with MAE
(mean absolute error). This module builds a small registry of candidate
models (Random Forest, Extra Trees, Histogram gradient boosting,
LightGBM, and a Linear Model) behind a common pipeline interface, plus a
hyperparameter search helper that tunes a model against a fixed
chronological validation split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PathLike = str | Path

CATEGORICAL_FEATURES = ["gare"]

# Tree-based models get a cheap ordinal encoding of the station id (they
# split on thresholds, so an arbitrary integer code works fine and avoids
# blowing up dimensionality with 84 one-hot columns). The linear model needs
# a real one-hot encoding since it would otherwise treat station codes as an
# ordered quantity.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "random_forest": {
        "estimator": RandomForestRegressor,
        "encoder": "ordinal",
        "default_params": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": 42,
        },
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20],
            "model__min_samples_leaf": [1, 5],
        },
    },
    "extra_trees": {
        "estimator": ExtraTreesRegressor,
        "encoder": "ordinal",
        "default_params": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": 42,
        },
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20],
            "model__min_samples_leaf": [1, 5],
        },
    },
    "gradient_boosting": {
        "estimator": HistGradientBoostingRegressor,
        "encoder": "ordinal",
        "default_params": {
            "max_iter": 200,
            "learning_rate": 0.1,
            "max_depth": None,
            "random_state": 42,
        },
        "param_grid": {
            "model__max_iter": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [None, 10],
        },
    },
    "lightgbm": {
        "estimator": LGBMRegressor,
        "encoder": "ordinal",
        "default_params": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        },
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__num_leaves": [31, 63],
        },
    },
    "linear_regression": {
        "estimator": Ridge,
        "encoder": "onehot",
        "default_params": {"alpha": 1.0, "random_state": 42},
        "param_grid": {"model__alpha": [0.1, 1.0, 10.0]},
    },
}


def build_pipeline(model_name: str = "random_forest", **model_params: Any) -> Pipeline:
    """
    Build an untrained preprocessing + model pipeline for ``model_name``.

    :param model_name: key into :data:`MODEL_REGISTRY`
    :param model_params: overrides for that model's ``default_params`` entry
        (e.g. ``n_estimators=50`` for a faster, lower-accuracy model in tests)
    :return: untrained pipeline
    :raises ValueError: if ``model_name`` is not in :data:`MODEL_REGISTRY`
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]

    if spec["encoder"] == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[("gare", encoder, CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )

    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if spec["encoder"] == "onehot":
        steps.append(("scaler", StandardScaler()))

    params = {**spec["default_params"], **model_params}
    steps.append(("model", spec["estimator"](**params)))
    return Pipeline(steps)


def train_model(
    X: pd.DataFrame, y: pd.Series, model_name: str = "random_forest", **model_params: Any
) -> Pipeline:
    """
    Build (see :func:`build_pipeline`) and fit a pipeline on ``X``/``y``.

    :param X: training features
    :param y: training target
    :param model_name: key into :data:`MODEL_REGISTRY`
    :param model_params: overrides for that model's ``default_params`` entry
    :return: fitted pipeline
    """
    pipeline = build_pipeline(model_name, **model_params)
    pipeline.fit(X, y)
    return pipeline


def predict(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Predict the p0q0 waiting-time difference for each row of ``X``.

    :param pipeline: fitted pipeline
    :param X: features to predict on
    :return: predicted p0q0 values
    """
    return pipeline.predict(X)


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """
    Compute the mean absolute error, the challenge's official evaluation metric.

    :param y_true: ground-truth target values
    :param y_pred: predicted target values
    :return: mean absolute error
    """
    return float(mean_absolute_error(y_true, y_pred))


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Predict on ``X`` and compute the MAE against ``y``.

    :param pipeline: fitted pipeline
    :param X: features to predict on
    :param y: ground-truth target values
    :return: mean absolute error
    """
    return mae(y, predict(pipeline, X))


def hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "random_forest",
    param_grid: dict[str, list[Any]] | None = None,
    n_jobs: int = -1,
) -> tuple[Pipeline, dict[str, Any], float]:
    """
    Tune ``model_name`` against a fixed chronological validation split.

    Uses :class:`~sklearn.model_selection.GridSearchCV` with a
    :class:`~sklearn.model_selection.PredefinedSplit` so every candidate is
    scored only on ``(X_val, y_val)`` — never on a random subset of
    ``X_train`` — which matters here because the real test set is a
    disjoint, later time window (see :func:`src.utils.time_based_split`).

    The returned pipeline is refit on ``X_train`` *only* (``refit=False`` on
    the search itself) so its reported/plotted validation performance stays
    honest: refitting the winner on ``X_train`` + ``X_val`` combined — a
    reasonable thing to do once you commit to a final model — would let it
    see the very rows it's then scored against, and tree ensembles in
    particular can nearly memorize them, silently deflating the validation
    MAE. If you want a model trained on every locally available labelled row
    (e.g. right before predicting on the real, disjoint test set), refit
    separately with the returned ``best_params`` on the combined data.

    :param X_train: training features
    :param y_train: training target
    :param X_val: validation features
    :param y_val: validation target
    :param model_name: key into :data:`MODEL_REGISTRY`
    :param param_grid: grid to search; defaults to that model's ``param_grid`` entry
    :param n_jobs: number of parallel jobs for the grid search
    :return: tuple of the best pipeline (fit on ``X_train`` only), its
        hyperparameters, and its validation MAE
    :raises ValueError: if ``model_name`` is not in :data:`MODEL_REGISTRY`
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    grid = param_grid if param_grid is not None else MODEL_REGISTRY[model_name]["param_grid"]

    if not grid:
        pipeline = train_model(X_train, y_train, model_name=model_name)
        return pipeline, {}, evaluate(pipeline, X_val, y_val)

    X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_combined = pd.concat(
        [pd.Series(y_train).reset_index(drop=True), pd.Series(y_val).reset_index(drop=True)],
        ignore_index=True,
    )
    test_fold = np.concatenate([np.full(len(X_train), -1), np.zeros(len(X_val))])
    split = PredefinedSplit(test_fold)

    search = GridSearchCV(
        build_pipeline(model_name),
        param_grid=grid,
        scoring="neg_mean_absolute_error",
        cv=split,
        n_jobs=n_jobs,
        refit=False,
    )
    search.fit(X_combined, y_combined)

    best_params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    pipeline = train_model(X_train, y_train, model_name=model_name, **best_params)
    return pipeline, best_params, -search.best_score_
