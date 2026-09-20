"""Unit tests for :mod:`src.train`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.train import main, parse_args
from src.utils import load_metadata, load_model

N_ROWS = 20


def _write_synthetic_dataset(data_dir: Path, with_test_set: bool = False) -> None:
    """Write a tiny x_train/y_train (and optionally x_test) dataset shaped like the real files."""
    dates = pd.date_range("2023-01-02", periods=N_ROWS // 2, freq="D").repeat(2)
    x_train = pd.DataFrame(
        {
            "Unnamed: 0": range(N_ROWS),
            "train": [f"T{i}" for i in range(N_ROWS)],
            "gare": ["AAA", "BBB"] * (N_ROWS // 2),
            "date": dates.strftime("%Y-%m-%d"),
            "arret": list(range(1, N_ROWS + 1)),
            "p2q0": [float(i % 5 - 2) for i in range(N_ROWS)],
            "p3q0": [0.0] * N_ROWS,
            "p4q0": [0.0] * N_ROWS,
            "p0q2": [float(i % 3 - 1) for i in range(N_ROWS)],
            "p0q3": [0.0] * N_ROWS,
            "p0q4": [0.0] * N_ROWS,
        }
    )
    y_train = pd.DataFrame({"p0q0": [float(i % 4 - 2) for i in range(N_ROWS)]})

    x_train.to_csv(data_dir / "x_train_final.csv", index=False)
    y_train.to_csv(data_dir / "y_train_final_testsuffix.csv", index=False)

    if with_test_set:
        x_test = x_train.drop(columns=["Unnamed: 0"]).head(5)
        x_test.to_csv(data_dir / "x_test_final.csv", index=False)


class TestParseArgs:
    """Tests for :func:`~src.train.parse_args`."""

    def test_defaults(self):
        args = parse_args([])

        assert args.data_dir == "data"
        assert args.model == "random_forest"
        assert args.tune is False
        assert args.predict_test is False
        assert args.validation_size == 0.2
        assert args.clip_value == 30.0

    def test_rejects_a_model_name_outside_the_registry(self):
        with pytest.raises(SystemExit):
            parse_args(["--model", "not_a_real_model"])

    def test_parses_the_tune_and_predict_test_flags(self):
        args = parse_args(["--tune", "--predict-test"])

        assert args.tune is True
        assert args.predict_test is True

    def test_parses_custom_paths_and_numeric_options(self):
        args = parse_args(
            [
                "--data-dir",
                "somewhere",
                "--output",
                "models/x.joblib",
                "--validation-size",
                "0.3",
                "--clip-value",
                "10",
                "--sample-frac",
                "0.5",
            ]
        )

        assert args.data_dir == "somewhere"
        assert args.output == "models/x.joblib"
        assert args.validation_size == 0.3
        assert args.clip_value == 10.0
        assert args.sample_frac == 0.5


class TestMain:
    """Tests for :func:`~src.train.main` (integration: runs the real training pipeline)."""

    def test_saves_a_model_and_metadata_with_default_params(self, tmp_path):
        _write_synthetic_dataset(tmp_path)
        output = tmp_path / "model.joblib"

        main(
            [
                "--data-dir",
                str(tmp_path),
                "--model",
                "random_forest",
                "--validation-size",
                "0.3",
                "--output",
                str(output),
            ]
        )

        assert output.exists()
        metadata = load_metadata(output.with_suffix(".json"))
        assert metadata["model_name"] == "random_forest"
        assert metadata["tuned"] is False
        assert isinstance(metadata["validation_mae"], float)

    def test_tune_records_tuned_true_and_searched_params(self, tmp_path):
        _write_synthetic_dataset(tmp_path)
        output = tmp_path / "model.joblib"

        main(
            [
                "--data-dir",
                str(tmp_path),
                "--model",
                "linear_regression",
                "--validation-size",
                "0.3",
                "--tune",
                "--output",
                str(output),
            ]
        )

        metadata = load_metadata(output.with_suffix(".json"))
        assert metadata["tuned"] is True

    def test_predict_test_writes_a_submission_and_a_production_model(self, tmp_path):
        _write_synthetic_dataset(tmp_path, with_test_set=True)
        output = tmp_path / "model.joblib"
        submission_output = tmp_path / "submission.csv"

        main(
            [
                "--data-dir",
                str(tmp_path),
                "--model",
                "random_forest",
                "--validation-size",
                "0.3",
                "--output",
                str(output),
                "--predict-test",
                "--submission-output",
                str(submission_output),
            ]
        )

        assert submission_output.exists()
        production_model_path = tmp_path / "model_production.joblib"
        assert production_model_path.exists()

        submission = pd.read_csv(submission_output)
        assert "p0q0" in submission.columns
        assert len(submission) == 5  # matches the synthetic x_test row count

    def test_sample_frac_reduces_the_training_set(self, tmp_path, capsys):
        _write_synthetic_dataset(tmp_path)
        output = tmp_path / "model.joblib"

        main(
            [
                "--data-dir",
                str(tmp_path),
                "--model",
                "random_forest",
                "--validation-size",
                "0.3",
                "--sample-frac",
                "0.5",
                "--output",
                str(output),
            ]
        )

        assert "Subsampled to" in capsys.readouterr().out

    def test_saved_model_can_be_reloaded_and_used_to_predict(self, tmp_path):
        _write_synthetic_dataset(tmp_path)
        output = tmp_path / "model.joblib"

        main(
            [
                "--data-dir",
                str(tmp_path),
                "--model",
                "random_forest",
                "--validation-size",
                "0.3",
                "--output",
                str(output),
            ]
        )

        pipeline = load_model(output)
        assert hasattr(pipeline, "predict")
