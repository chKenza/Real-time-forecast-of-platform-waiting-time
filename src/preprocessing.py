"""
Data loading, cleaning and feature engineering for the SNCF waiting-time challenge.

Each row is a stop (train k, station s, date d). The raw ``x_*`` files carry
4 contextual columns (train, gare, date, arret) and 6 "past" delay columns
(p2q0, p3q0, p4q0, p0q2, p0q3, p0q4). The target ``p0q0`` (in ``y_train``) is
the difference between theoretical and observed waiting time, in minutes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PathLike = str | Path

CONTEXT_COLUMNS = ["train", "gare", "date", "arret"]
PAST_VALUE_COLUMNS = ["p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]
RAW_FEATURE_COLUMNS = CONTEXT_COLUMNS + PAST_VALUE_COLUMNS
TARGET_COLUMN = "p0q0"

# Model-ready features: station/stop/calendar context + the 6 past delays.
# "train" is intentionally excluded: train identifiers do not repeat between
# train and test splits (the challenge data is disjoint), so it cannot
# generalize as a model feature.
MODEL_FEATURE_COLUMNS = [
    "gare",
    "arret",
    "weekday",
    "month",
    "is_weekend",
    *PAST_VALUE_COLUMNS,
]

# A handful of rows (~0.05%) carry sentinel-like outliers (e.g. -1441, close
# to -24h in minutes) from upstream data glitches around day boundaries.
# Clipping keeps the model from being skewed by these without dropping rows.
DEFAULT_OUTLIER_CLIP = 30.0


def load_csv(path: PathLike) -> pd.DataFrame:
    """
    Load a raw challenge CSV, dropping the pandas-export index column(s).

    The provided files store an extra unnamed index column (sometimes two,
    e.g. ``Unnamed: 0`` and ``Unnamed: 0.1`` in ``x_train``) which carries no
    information and must not be treated as a feature.

    :param path: path to the raw CSV file
    :return: loaded dataframe, without any unnamed index column
    """
    path = Path(path)
    df = pd.read_csv(path)
    unnamed_columns = [c for c in df.columns if c.startswith("Unnamed")]
    return df.drop(columns=unnamed_columns)


def load_xy(x_path: PathLike, y_path: PathLike) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a feature file together with its target file.

    The two files are aligned by row order (there is no shared key column),
    so this also validates that both files have the same number of rows.

    :param x_path: path to the raw features CSV
    :param y_path: path to the raw target CSV, with a ``p0q0`` column
    :return: features dataframe and aligned target series
    :raises ValueError: if the target column is missing, or the row counts differ
    """
    X = load_csv(x_path)
    y_df = load_csv(y_path)
    if TARGET_COLUMN not in y_df.columns:
        raise ValueError(f"Expected column '{TARGET_COLUMN}' in {y_path}")
    if len(X) != len(y_df):
        raise ValueError(f"X and y have mismatched row counts: {len(X)} (X) vs {len(y_df)} (y)")
    y = y_df[TARGET_COLUMN]
    y.index = X.index
    return X, y


def validate_raw_columns(df: pd.DataFrame) -> None:
    """
    Raise if any expected raw feature column is missing from ``df``.

    :param df: raw features dataframe to check
    :raises ValueError: if one or more expected columns are missing
    """
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s): {missing}")


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive calendar features (weekday, month, weekend flag) from ``date``.

    :param df: raw features dataframe with a ``date`` column
    :return: copy of ``df`` with ``weekday``, ``month`` and ``is_weekend`` added
    """
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    df["weekday"] = dates.dt.weekday
    df["month"] = dates.dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df


def clip_outliers(
    df: pd.DataFrame,
    columns: list[str] = PAST_VALUE_COLUMNS,
    clip_value: float = DEFAULT_OUTLIER_CLIP,
) -> pd.DataFrame:
    """
    Clip extreme values in ``columns`` to +/- ``clip_value``.

    :param df: dataframe to clip
    :param columns: columns to clip
    :param clip_value: absolute value beyond which values are clipped
    :return: copy of ``df`` with ``columns`` clipped
    """
    df = df.copy()
    df[columns] = df[columns].clip(lower=-clip_value, upper=clip_value)
    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns the model is trained on, in a fixed order.

    :param df: dataframe with at least the :data:`MODEL_FEATURE_COLUMNS` columns
    :return: dataframe restricted to :data:`MODEL_FEATURE_COLUMNS`
    """
    return df[MODEL_FEATURE_COLUMNS]


def preprocess_features(df: pd.DataFrame, clip_value: float = DEFAULT_OUTLIER_CLIP) -> pd.DataFrame:
    """
    Run the full feature pipeline: validate, derive calendar features, clip, select.

    Takes a raw dataframe as returned by :func:`load_csv` and returns a
    dataframe ready to feed into a model (see :mod:`src.model`).

    :param df: raw features dataframe
    :param clip_value: absolute value beyond which past-delay values are clipped
    :return: model-ready dataframe restricted to :data:`MODEL_FEATURE_COLUMNS`
    """
    validate_raw_columns(df)
    df = add_date_features(df)
    df = clip_outliers(df, clip_value=clip_value)
    return select_model_features(df)


def get_default_data_paths(data_dir: PathLike) -> dict[str, Path]:
    """
    Resolve the canonical raw data file paths inside ``data_dir``.

    ``y_train`` is resolved with a glob because the challenge export ships it
    with a random suffix (e.g. ``y_train_final_j5KGWWK.csv``).

    :param data_dir: directory containing the raw challenge CSV files
    :return: mapping from ``x_train``/``y_train``/``x_test``/``y_sample`` to their paths
    :raises FileNotFoundError: if no ``y_train_final*.csv`` file is found
    """
    data_dir = Path(data_dir)
    y_train_matches = sorted(data_dir.glob("y_train_final*.csv"))
    if not y_train_matches:
        raise FileNotFoundError(f"No y_train_final*.csv file found in {data_dir}")
    return {
        "x_train": data_dir / "x_train_final.csv",
        "y_train": y_train_matches[0],
        "x_test": data_dir / "x_test_final.csv",
        "y_sample": data_dir / "y_sample_final.csv",
    }
