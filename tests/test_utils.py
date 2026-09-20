"""Unit tests for :mod:`src.utils`."""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils import (
    chronological_split_mask,
    load_metadata,
    load_model,
    save_metadata,
    save_model,
    time_based_split,
)


class TestChronologicalSplitMask:
    """Tests for :func:`~src.utils.chronological_split_mask`."""

    def test_keeps_the_most_recent_dates_out_of_the_training_side(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=10, freq="D"))

        is_train = chronological_split_mask(dates, validation_size=0.2)

        assert is_train.sum() == 8
        assert (~is_train).sum() == 2
        assert list(is_train) == [True] * 8 + [False] * 2

    def test_keeps_every_row_of_a_repeated_date_on_the_same_side(self):
        dates = pd.Series(
            pd.to_datetime(["2023-01-01"] * 3 + ["2023-01-02"] * 3 + ["2023-01-03"] * 3)
        )

        is_train = chronological_split_mask(dates, validation_size=0.34)

        assert list(is_train) == [True] * 3 + [False] * 6

    def test_raises_when_validation_size_is_out_of_range(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D"))

        with pytest.raises(ValueError, match="validation_size"):
            chronological_split_mask(dates, validation_size=1.5)

    def test_raises_when_validation_size_is_zero(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D"))

        with pytest.raises(ValueError, match="validation_size"):
            chronological_split_mask(dates, validation_size=0.0)


class TestTimeBasedSplit:
    """Tests for :func:`~src.utils.time_based_split`."""

    def test_splits_x_and_y_consistently_with_the_date_column(self, raw_features_df, target_series):
        X_tr, X_val, y_tr, y_val = time_based_split(
            raw_features_df, target_series, raw_features_df["date"], validation_size=0.2
        )

        assert len(X_tr) == len(y_tr) == 8
        assert len(X_val) == len(y_val) == 2

    def test_validation_rows_are_strictly_later_than_training_rows(
        self, raw_features_df, target_series
    ):
        X_tr, X_val, _, _ = time_based_split(
            raw_features_df, target_series, raw_features_df["date"], validation_size=0.2
        )

        assert pd.to_datetime(X_tr["date"]).max() < pd.to_datetime(X_val["date"]).min()

    def test_raises_on_mismatched_lengths(self, raw_features_df, target_series):
        with pytest.raises(ValueError, match="same length"):
            time_based_split(raw_features_df, target_series.iloc[:-1], raw_features_df["date"])


class TestSaveModel:
    """Tests for :func:`~src.utils.save_model`."""

    def test_creates_parent_directories(self, tmp_path):
        destination = tmp_path / "nested" / "model.joblib"

        save_model({"a": 1}, destination)

        assert destination.exists()

    def test_saved_file_round_trips_through_load_model(self, tmp_path):
        destination = tmp_path / "model.joblib"
        original = {"coef": [1, 2, 3]}

        save_model(original, destination)

        assert load_model(destination) == original


class TestLoadModel:
    """Tests for :func:`~src.utils.load_model`."""

    def test_loads_a_previously_saved_object(self, tmp_path):
        destination = tmp_path / "model.joblib"
        save_model([1, 2, 3], destination)

        assert load_model(destination) == [1, 2, 3]

    def test_accepts_a_string_path(self, tmp_path):
        destination = tmp_path / "model.joblib"
        save_model("hello", destination)

        assert load_model(str(destination)) == "hello"


class TestSaveMetadata:
    """Tests for :func:`~src.utils.save_metadata`."""

    def test_writes_valid_json(self, tmp_path):
        destination = tmp_path / "meta.json"

        save_metadata({"validation_mae": 0.74, "tuned": True}, destination)

        assert destination.read_text().strip().startswith("{")

    def test_creates_parent_directories(self, tmp_path):
        destination = tmp_path / "nested" / "meta.json"

        save_metadata({"a": 1}, destination)

        assert destination.exists()

    def test_serializes_non_json_native_values_via_str(self, tmp_path):
        destination = tmp_path / "meta.json"
        non_native_value = tmp_path

        save_metadata({"path": non_native_value}, destination)

        assert load_metadata(destination)["path"] == str(non_native_value)


class TestLoadMetadata:
    """Tests for :func:`~src.utils.load_metadata`."""

    def test_round_trips_through_save_metadata(self, tmp_path):
        destination = tmp_path / "meta.json"
        original = {"model_name": "lightgbm", "validation_mae": 0.74}

        save_metadata(original, destination)

        assert load_metadata(destination) == original

    def test_accepts_a_string_path(self, tmp_path):
        destination = tmp_path / "meta.json"
        save_metadata({"a": 1}, destination)

        assert load_metadata(str(destination)) == {"a": 1}
