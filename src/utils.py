"""Shared helpers: chronological validation split and model persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import joblib
import pandas as pd

PathLike = str | Path


def chronological_split_mask(date_column: pd.Series, validation_size: float = 0.2) -> np.ndarray:
    """
    Build a boolean mask (True = train) holding out the most recent dates.

    Exposed separately from :func:`time_based_split` so callers can apply the
    exact same train/validation boundary to other aligned arrays (e.g. the
    raw ``date`` column itself, for plotting predictions over time).

    :param date_column: date values, one per row
    :param validation_size: fraction of *unique dates* to hold out as validation
    :return: boolean array, True for rows kept as training data
    :raises ValueError: if ``validation_size`` is not between 0 and 1
    """
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    dates = pd.to_datetime(date_column).reset_index(drop=True)
    unique_dates = sorted(dates.unique())
    split_at = max(1, int(len(unique_dates) * (1 - validation_size)))
    cutoff = unique_dates[split_at]
    return (dates < cutoff).to_numpy()


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    date_column: pd.Series,
    validation_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split ``(X, y)`` chronologically on ``date_column``.

    The challenge's real test set covers dates strictly after the training
    dates (a disjoint temporal holdout), so a random split would leak
    information a plain accuracy check would not catch. This instead holds
    out the most recent ``validation_size`` fraction of *unique dates* as
    validation, keeping every stop from a given day on the same side.

    :param X: features
    :param y: target
    :param date_column: date values, one per row of ``X``
    :param validation_size: fraction of *unique dates* to hold out as validation
    :return: tuple of ``X_train, X_val, y_train, y_val``
    :raises ValueError: if ``X``, ``y`` and ``date_column`` have mismatched lengths
    """
    if len(X) != len(y) or len(X) != len(date_column):
        raise ValueError("X, y and date_column must have the same length")

    is_train = chronological_split_mask(date_column, validation_size)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X[is_train], X[~is_train], y[is_train], y[~is_train]


def save_model(model: Any, path: PathLike) -> None:
    """
    Persist a fitted model/pipeline to disk with joblib.

    :param model: fitted model or pipeline
    :param path: destination file path; parent directories are created as needed
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: PathLike) -> Any:
    """
    Load a model/pipeline previously saved with :func:`save_model`.

    :param path: path to the saved model file
    :return: the loaded model or pipeline
    """
    return joblib.load(Path(path))


def save_metadata(metadata: dict[str, Any], path: PathLike) -> None:
    """
    Persist a JSON-serializable metadata dict alongside a saved model.

    :param metadata: JSON-serializable metadata (e.g. chosen hyperparameters,
        validation score, training config)
    :param path: destination file path; parent directories are created as needed
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(path: PathLike) -> dict[str, Any]:
    """
    Load metadata previously saved with :func:`save_metadata`.

    :param path: path to the saved metadata file
    :return: the loaded metadata dict
    """
    with Path(path).open() as f:
        return json.load(f)
