"""Shared pytest fixtures: small, deterministic synthetic datasets."""

from __future__ import annotations

import pandas as pd
import pytest

N_ROWS = 10


@pytest.fixture
def raw_features_df() -> pd.DataFrame:
    """
    Build a small raw features dataframe shaped like the challenge's x_* files.

    Spans 10 consecutive dates (2023-01-02, a Monday, through 2023-01-11) so
    tests can exercise weekday/weekend and chronological-split logic, and
    carries one clearly out-of-range value (p0q4 on the first row) to
    exercise outlier clipping.

    :return: raw features dataframe with :data:`~src.preprocessing.RAW_FEATURE_COLUMNS`
    """
    dates = pd.date_range("2023-01-02", periods=N_ROWS, freq="D")
    return pd.DataFrame(
        {
            "train": [f"T{i}" for i in range(N_ROWS)],
            "gare": ["AAA", "BBB"] * (N_ROWS // 2),
            "date": dates.strftime("%Y-%m-%d"),
            "arret": list(range(1, N_ROWS + 1)),
            "p2q0": [0.0, 1.0, -1.0, 2.0, -2.0, 0.0, 1.0, -1.0, 2.0, -2.0],
            "p3q0": [0.0] * N_ROWS,
            "p4q0": [0.0] * N_ROWS,
            "p0q2": [0.0] * N_ROWS,
            "p0q3": [0.0] * N_ROWS,
            "p0q4": [100.0] + [0.0] * (N_ROWS - 1),
        }
    )


@pytest.fixture
def target_series() -> pd.Series:
    """
    Build a target series aligned with :func:`raw_features_df`.

    :return: target series of length :data:`N_ROWS`
    """
    return pd.Series([float(i) - 5 for i in range(N_ROWS)], name="p0q0")
