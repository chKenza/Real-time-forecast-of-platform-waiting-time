"""Unit tests for :mod:`src.evaluate`, one class per function."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluate import (
    main,
    parse_args,
    plot_actual_vs_predicted_over_time,
    plot_mae_by_weekday,
    plot_mae_comparison,
    plot_predicted_vs_actual,
    plot_residual_histogram,
    plot_residuals_vs_predicted,
)
from src.model import train_model
from src.preprocessing import preprocess_features
from src.utils import save_model


@pytest.fixture
def predictions(raw_features_df, target_series):
    """A small (y_true, y_pred, X_val) triple for the diagnostic-plot functions."""
    X = preprocess_features(raw_features_df)
    y_pred = target_series.to_numpy() + 0.5  # arbitrary, deterministic "predictions"
    return X, target_series, y_pred


class TestPlotPredictedVsActual:
    """Tests for :func:`~src.evaluate.plot_predicted_vs_actual`."""

    def test_writes_a_png_file(self, tmp_path, predictions):
        _, y_true, y_pred = predictions

        plot_predicted_vs_actual(y_true, y_pred, "mymodel", tmp_path)

        assert (tmp_path / "pred_vs_actual_mymodel.png").exists()


class TestPlotResidualHistogram:
    """Tests for :func:`~src.evaluate.plot_residual_histogram`."""

    def test_writes_a_png_file(self, tmp_path, predictions):
        _, y_true, y_pred = predictions

        plot_residual_histogram(y_true, y_pred, "mymodel", tmp_path)

        assert (tmp_path / "residual_hist_mymodel.png").exists()


class TestPlotResidualsVsPredicted:
    """Tests for :func:`~src.evaluate.plot_residuals_vs_predicted`."""

    def test_writes_a_png_file(self, tmp_path, predictions):
        _, y_true, y_pred = predictions

        plot_residuals_vs_predicted(y_true, y_pred, "mymodel", tmp_path)

        assert (tmp_path / "residuals_vs_predicted_mymodel.png").exists()


class TestPlotMaeByWeekday:
    """Tests for :func:`~src.evaluate.plot_mae_by_weekday`."""

    def test_writes_a_png_file(self, tmp_path, predictions):
        X_val, y_true, y_pred = predictions

        plot_mae_by_weekday(X_val, y_true, y_pred, "mymodel", tmp_path)

        assert (tmp_path / "mae_by_weekday_mymodel.png").exists()


class TestPlotActualVsPredictedOverTime:
    """Tests for :func:`~src.evaluate.plot_actual_vs_predicted_over_time`."""

    def test_writes_a_png_file(self, tmp_path, raw_features_df, predictions):
        _, y_true, y_pred = predictions

        plot_actual_vs_predicted_over_time(
            raw_features_df["date"], y_true, y_pred, "mymodel", tmp_path
        )

        assert (tmp_path / "daily_actual_vs_predicted_mymodel.png").exists()


class TestPlotMaeComparison:
    """Tests for :func:`~src.evaluate.plot_mae_comparison`."""

    def test_writes_a_png_file(self, tmp_path):
        plot_mae_comparison(["model_a", "model_b"], [0.7, 0.8], tmp_path)

        assert (tmp_path / "mae_comparison.png").exists()


class TestParseArgs:
    """Tests for :func:`~src.evaluate.parse_args`."""

    def test_defaults(self):
        args = parse_args([])

        assert args.data_dir == "data"
        assert args.model_path == ["models/model.joblib"]
        assert args.labels is None
        assert args.validation_size == 0.2
        assert args.clip_value == 30.0
        assert args.output_dir == "figures/models"

    def test_parses_multiple_model_paths_and_labels(self):
        args = parse_args(
            [
                "--model-path",
                "a.joblib",
                "b.joblib",
                "--labels",
                "model_a",
                "model_b",
            ]
        )

        assert args.model_path == ["a.joblib", "b.joblib"]
        assert args.labels == ["model_a", "model_b"]


class TestMain:
    """Tests for :func:`~src.evaluate.main` (integration: runs the real evaluation pipeline)."""

    def _write_dataset(self, data_dir, raw_features_df, target_series):
        data_dir.mkdir()
        raw_features_df.assign(**{"Unnamed: 0": range(len(raw_features_df))}).to_csv(
            data_dir / "x_train_final.csv", index=False
        )
        pd.DataFrame({"p0q0": target_series}).to_csv(
            data_dir / "y_train_final_testsuffix.csv", index=False
        )

    def _fit_and_save_model(self, model_path, raw_features_df, target_series):
        X = preprocess_features(raw_features_df)
        pipeline = train_model(X, target_series, model_name="random_forest", n_estimators=5)
        save_model(pipeline, model_path)

    def test_generates_the_per_model_figures(self, tmp_path, raw_features_df, target_series):
        data_dir = tmp_path / "data"
        out_dir = tmp_path / "figures"
        model_path = tmp_path / "model.joblib"
        self._write_dataset(data_dir, raw_features_df, target_series)
        self._fit_and_save_model(model_path, raw_features_df, target_series)

        main(
            [
                "--data-dir",
                str(data_dir),
                "--model-path",
                str(model_path),
                "--labels",
                "mymodel",
                "--validation-size",
                "0.3",
                "--output-dir",
                str(out_dir),
            ]
        )

        expected = {
            "pred_vs_actual_mymodel.png",
            "residual_hist_mymodel.png",
            "residuals_vs_predicted_mymodel.png",
            "mae_by_weekday_mymodel.png",
            "daily_actual_vs_predicted_mymodel.png",
        }
        assert expected.issubset({p.name for p in out_dir.iterdir()})
        assert not (out_dir / "mae_comparison.png").exists()  # only one model given

    def test_generates_the_comparison_chart_for_multiple_models(
        self, tmp_path, raw_features_df, target_series
    ):
        data_dir = tmp_path / "data"
        out_dir = tmp_path / "figures"
        model_path = tmp_path / "model.joblib"
        self._write_dataset(data_dir, raw_features_df, target_series)
        self._fit_and_save_model(model_path, raw_features_df, target_series)

        main(
            [
                "--data-dir",
                str(data_dir),
                "--model-path",
                str(model_path),
                str(model_path),
                "--labels",
                "model_a",
                "model_b",
                "--validation-size",
                "0.3",
                "--output-dir",
                str(out_dir),
            ]
        )

        assert (out_dir / "mae_comparison.png").exists()

    def test_raises_when_labels_length_does_not_match_model_paths(
        self, tmp_path, raw_features_df, target_series
    ):
        data_dir = tmp_path / "data"
        out_dir = tmp_path / "figures"
        model_path = tmp_path / "model.joblib"
        self._write_dataset(data_dir, raw_features_df, target_series)
        self._fit_and_save_model(model_path, raw_features_df, target_series)

        with pytest.raises(ValueError, match="--labels"):
            main(
                [
                    "--data-dir",
                    str(data_dir),
                    "--model-path",
                    str(model_path),
                    "--labels",
                    "a",
                    "b",
                    "--output-dir",
                    str(out_dir),
                ]
            )
