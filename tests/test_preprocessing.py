"""Unit tests for `src.preprocessing`"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    MODEL_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    add_date_features,
    clip_outliers,
    get_default_data_paths,
    load_csv,
    load_xy,
    preprocess_features,
    select_model_features,
    validate_raw_columns,
)


class TestLoadCsv:
    """Tests for :func:`~src.preprocessing.load_csv`."""

    def test_drops_a_single_unnamed_column(self, tmp_path):
        path = tmp_path / "x.csv"
        pd.DataFrame({"Unnamed: 0": [0, 1], "gare": ["AAA", "BBB"]}).to_csv(path, index=False)

        df = load_csv(path)

        assert list(df.columns) == ["gare"]

    def test_drops_multiple_unnamed_columns(self, tmp_path):
        path = tmp_path / "x.csv"
        pd.DataFrame({"Unnamed: 0": [0, 1], "Unnamed: 0.1": [0, 1], "gare": ["AAA", "BBB"]}).to_csv(
            path, index=False
        )

        df = load_csv(path)

        assert list(df.columns) == ["gare"]

    def test_preserves_regular_columns(self, tmp_path):
        path = tmp_path / "x.csv"
        pd.DataFrame({"gare": ["AAA"], "arret": [1]}).to_csv(path, index=False)

        df = load_csv(path)

        assert list(df.columns) == ["gare", "arret"]

    def test_accepts_a_string_path(self, tmp_path):
        path = tmp_path / "x.csv"
        pd.DataFrame({"gare": ["AAA"]}).to_csv(path, index=False)

        df = load_csv(str(path))

        assert list(df.columns) == ["gare"]


class TestLoadXy:
    """Tests for :func:`~src.preprocessing.load_xy`."""

    def _write(self, tmp_path, x_df: pd.DataFrame, y_df: pd.DataFrame) -> tuple:
        x_path = tmp_path / "x.csv"
        y_path = tmp_path / "y.csv"
        x_df.to_csv(x_path, index=False)
        y_df.to_csv(y_path, index=False)
        return x_path, y_path

    def test_aligns_features_and_target_by_row_order(self, tmp_path):
        x_df = pd.DataFrame({"gare": ["AAA", "BBB", "CCC"]})
        y_df = pd.DataFrame({"p0q0": [1.0, -2.0, 3.0]})
        x_path, y_path = self._write(tmp_path, x_df, y_df)

        X, y = load_xy(x_path, y_path)

        assert len(X) == len(y) == 3
        assert list(y) == [1.0, -2.0, 3.0]
        assert list(y.index) == list(X.index)

    def test_raises_when_target_column_missing(self, tmp_path):
        x_df = pd.DataFrame({"gare": ["AAA"]})
        y_df = pd.DataFrame({"not_p0q0": [1.0]})
        x_path, y_path = self._write(tmp_path, x_df, y_df)

        with pytest.raises(ValueError, match="p0q0"):
            load_xy(x_path, y_path)

    def test_raises_on_row_count_mismatch(self, tmp_path):
        x_df = pd.DataFrame({"gare": ["AAA", "BBB"]})
        y_df = pd.DataFrame({"p0q0": [1.0]})
        x_path, y_path = self._write(tmp_path, x_df, y_df)

        with pytest.raises(ValueError, match="mismatched row counts"):
            load_xy(x_path, y_path)


class TestValidateRawColumns:
    """Tests for :func:`~src.preprocessing.validate_raw_columns`."""

    def test_passes_when_all_columns_present(self, raw_features_df):
        validate_raw_columns(raw_features_df)  # should not raise

    def test_raises_when_a_column_is_missing(self, raw_features_df):
        df = raw_features_df.drop(columns=["arret"])

        with pytest.raises(ValueError, match="arret"):
            validate_raw_columns(df)

    def test_raises_when_several_columns_are_missing(self, raw_features_df):
        df = raw_features_df.drop(columns=["arret", "p2q0"])

        with pytest.raises(ValueError, match="arret"):
            validate_raw_columns(df)


class TestAddDateFeatures:
    """Tests for :func:`~src.preprocessing.add_date_features`."""

    def test_adds_the_three_calendar_columns(self, raw_features_df):
        df = add_date_features(raw_features_df)

        assert {"weekday", "month", "is_weekend"}.issubset(df.columns)

    def test_weekday_matches_the_calendar(self, raw_features_df):
        df = add_date_features(raw_features_df)

        # 2023-01-02 is a Monday (weekday 0).
        assert df.loc[0, "weekday"] == 0

    def test_flags_weekend_rows(self, raw_features_df):
        df = add_date_features(raw_features_df)

        # 2023-01-07 (row 5) is a Saturday, 2023-01-08 (row 6) a Sunday.
        assert df.loc[5, "is_weekend"] == 1
        assert df.loc[6, "is_weekend"] == 1
        assert df.loc[0, "is_weekend"] == 0

    def test_does_not_mutate_the_input(self, raw_features_df):
        original_columns = list(raw_features_df.columns)

        add_date_features(raw_features_df)

        assert list(raw_features_df.columns) == original_columns


class TestClipOutliers:
    """Tests for :func:`~src.preprocessing.clip_outliers`."""

    def test_clips_values_above_the_threshold(self, raw_features_df):
        df = clip_outliers(raw_features_df, clip_value=30.0)

        assert df.loc[0, "p0q4"] == 30.0

    def test_leaves_in_range_values_unchanged(self, raw_features_df):
        df = clip_outliers(raw_features_df, clip_value=30.0)

        assert df.loc[1, "p2q0"] == 1.0

    def test_clips_symmetrically(self, raw_features_df):
        df = raw_features_df.copy()
        df.loc[0, "p0q4"] = -100.0

        clipped = clip_outliers(df, clip_value=30.0)

        assert clipped.loc[0, "p0q4"] == -30.0

    def test_only_clips_the_requested_columns(self, raw_features_df):
        df = clip_outliers(raw_features_df, columns=["p0q4"], clip_value=30.0)

        assert df.loc[0, "p0q4"] == 30.0
        assert df.loc[0, "arret"] == raw_features_df.loc[0, "arret"]

    def test_does_not_mutate_the_input(self, raw_features_df):
        original_value = raw_features_df.loc[0, "p0q4"]

        clip_outliers(raw_features_df, clip_value=30.0)

        assert raw_features_df.loc[0, "p0q4"] == original_value


class TestSelectModelFeatures:
    """Tests for :func:`~src.preprocessing.select_model_features`."""

    def test_returns_exactly_the_model_feature_columns(self, raw_features_df):
        df = add_date_features(raw_features_df)

        selected = select_model_features(df)

        assert list(selected.columns) == MODEL_FEATURE_COLUMNS

    def test_excludes_the_train_identifier(self, raw_features_df):
        df = add_date_features(raw_features_df)

        selected = select_model_features(df)

        assert "train" not in selected.columns

    def test_raises_when_calendar_features_are_missing(self, raw_features_df):
        with pytest.raises(KeyError):
            select_model_features(raw_features_df)


class TestPreprocessFeatures:
    """Tests for :func:`~src.preprocessing.preprocess_features`."""

    def test_returns_model_ready_columns(self, raw_features_df):
        features = preprocess_features(raw_features_df)

        assert list(features.columns) == MODEL_FEATURE_COLUMNS

    def test_preserves_row_count(self, raw_features_df):
        features = preprocess_features(raw_features_df)

        assert len(features) == len(raw_features_df)

    def test_clips_outliers_using_the_given_clip_value(self, raw_features_df):
        features = preprocess_features(raw_features_df, clip_value=10.0)

        assert features.loc[0, "p0q4"] == 10.0

    def test_raises_when_a_required_raw_column_is_missing(self, raw_features_df):
        df = raw_features_df.drop(columns=["gare"])

        with pytest.raises(ValueError, match="gare"):
            preprocess_features(df)


class TestGetDefaultDataPaths:
    """Tests for :func:`~src.preprocessing.get_default_data_paths`."""

    def _touch_all(self, tmp_path, y_train_name: str = "y_train_final_abc123.csv"):
        for name in ("x_train_final.csv", "x_test_final.csv", "y_sample_final.csv", y_train_name):
            (tmp_path / name).write_text("")

    def test_resolves_the_four_expected_keys(self, tmp_path):
        self._touch_all(tmp_path)

        paths = get_default_data_paths(tmp_path)

        assert set(paths) == {"x_train", "y_train", "x_test", "y_sample"}
        assert paths["x_train"] == tmp_path / "x_train_final.csv"
        assert paths["x_test"] == tmp_path / "x_test_final.csv"
        assert paths["y_sample"] == tmp_path / "y_sample_final.csv"

    def test_resolves_y_train_via_glob(self, tmp_path):
        self._touch_all(tmp_path, y_train_name="y_train_final_j5KGWWK.csv")

        paths = get_default_data_paths(tmp_path)

        assert paths["y_train"] == tmp_path / "y_train_final_j5KGWWK.csv"

    def test_picks_the_lexicographically_first_match_when_several_exist(self, tmp_path):
        self._touch_all(tmp_path, y_train_name="y_train_final_bbb.csv")
        (tmp_path / "y_train_final_aaa.csv").write_text("")

        paths = get_default_data_paths(tmp_path)

        assert paths["y_train"] == tmp_path / "y_train_final_aaa.csv"

    def test_raises_when_no_y_train_file_exists(self, tmp_path):
        (tmp_path / "x_train_final.csv").write_text("")

        with pytest.raises(FileNotFoundError):
            get_default_data_paths(tmp_path)

    def test_raw_feature_columns_matches_context_plus_past(self):
        # Sanity check on the module-level constant the rest of the suite relies on.
        assert set(RAW_FEATURE_COLUMNS) == {
            "train",
            "gare",
            "date",
            "arret",
            "p2q0",
            "p3q0",
            "p4q0",
            "p0q2",
            "p0q3",
            "p0q4",
        }
