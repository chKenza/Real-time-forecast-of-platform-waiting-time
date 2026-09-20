"""Unit tests for :mod:`src.model`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from src.model import build_pipeline, evaluate, hyperparameter_search, mae, predict, train_model
from src.preprocessing import preprocess_features
from src.utils import time_based_split


@pytest.fixture
def model_features_df(raw_features_df) -> pd.DataFrame:
    """Model-ready features derived from the shared raw fixture."""
    return preprocess_features(raw_features_df)


class TestBuildPipeline:
    """Tests for :func:`~src.model.build_pipeline`."""

    def test_raises_for_an_unknown_model_name(self):
        with pytest.raises(ValueError, match="Unknown model_name"):
            build_pipeline("not_a_real_model")

    def test_ordinal_models_encode_the_station_ordinally(self):
        pipeline = build_pipeline("random_forest")

        transformers = dict(
            (name, t) for name, t, _ in pipeline.named_steps["preprocessor"].transformers
        )

        assert isinstance(transformers["gare"], OrdinalEncoder)
        assert list(pipeline.named_steps) == ["preprocessor", "model"]

    def test_onehot_models_encode_the_station_and_add_a_scaler(self):
        pipeline = build_pipeline("linear_regression")

        transformers = dict(
            (name, t) for name, t, _ in pipeline.named_steps["preprocessor"].transformers
        )

        assert isinstance(transformers["gare"], OneHotEncoder)
        assert list(pipeline.named_steps) == ["preprocessor", "scaler", "model"]

    def test_model_params_override_the_registry_defaults(self):
        pipeline = build_pipeline("random_forest", n_estimators=7)

        assert pipeline.named_steps["model"].n_estimators == 7

    def test_pipeline_is_not_yet_fitted(self):
        pipeline = build_pipeline("random_forest")

        with pytest.raises(Exception):
            check_is_fitted(pipeline.named_steps["model"])


class TestTrainModel:
    """Tests for :func:`~src.model.train_model`."""

    def test_returns_a_fitted_pipeline(self, model_features_df, target_series):
        pipeline = train_model(
            model_features_df, target_series, model_name="random_forest", n_estimators=5
        )

        check_is_fitted(pipeline.named_steps["model"])  # should not raise

    def test_fitted_pipeline_predicts_the_right_number_of_rows(
        self, model_features_df, target_series
    ):
        pipeline = train_model(
            model_features_df, target_series, model_name="random_forest", n_estimators=5
        )

        predictions = pipeline.predict(model_features_df)

        assert len(predictions) == len(model_features_df)

    def test_defaults_to_random_forest(self, model_features_df, target_series):
        pipeline = train_model(model_features_df, target_series, n_estimators=5)

        assert pipeline.named_steps["model"].__class__.__name__ == "RandomForestRegressor"


class TestPredict:
    """Tests for :func:`~src.model.predict`."""

    def test_returns_one_prediction_per_row(self, model_features_df, target_series):
        pipeline = train_model(
            model_features_df, target_series, model_name="random_forest", n_estimators=5
        )

        predictions = predict(pipeline, model_features_df)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(model_features_df)

    def test_matches_calling_pipeline_predict_directly(self, model_features_df, target_series):
        pipeline = train_model(
            model_features_df, target_series, model_name="random_forest", n_estimators=5
        )

        np.testing.assert_array_equal(
            predict(pipeline, model_features_df), pipeline.predict(model_features_df)
        )


class TestMae:
    """Tests for :func:`~src.model.mae`."""

    def test_zero_for_perfect_predictions(self):
        assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_matches_a_hand_computed_value(self):
        # |1-1| + |2-2| + |3-5| = 2, over 3 rows -> 2/3.
        assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(2 / 3)

    def test_accepts_numpy_arrays(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, -1.0])

        assert mae(y_true, y_pred) == 1.0

    def test_returns_a_python_float(self):
        assert isinstance(mae([1.0], [2.0]), float)


class TestEvaluate:
    """Tests for :func:`~src.model.evaluate`."""

    def test_matches_mae_of_manual_predictions(self, model_features_df, target_series):
        pipeline = train_model(
            model_features_df, target_series, model_name="random_forest", n_estimators=5
        )

        expected = mae(target_series, pipeline.predict(model_features_df))

        assert evaluate(pipeline, model_features_df, target_series) == expected


class TestHyperparameterSearch:
    """Tests for :func:`~src.model.hyperparameter_search`."""

    def test_raises_for_an_unknown_model_name(self, model_features_df, target_series):
        # model_name is validated before the data is touched, so an arbitrary
        # train/validation split is enough to exercise this.
        X_tr, y_tr = model_features_df.iloc[:8], target_series.iloc[:8]
        X_val, y_val = model_features_df.iloc[8:], target_series.iloc[8:]

        with pytest.raises(ValueError, match="Unknown model_name"):
            hyperparameter_search(X_tr, y_tr, X_val, y_val, model_name="not_a_real_model")

    def test_empty_param_grid_skips_the_search(self, raw_features_df, target_series):
        Xf = preprocess_features(raw_features_df)
        X_tr, X_val, y_tr, y_val = time_based_split(
            Xf, target_series, raw_features_df["date"], validation_size=0.2
        )

        pipeline, best_params, val_mae = hyperparameter_search(
            X_tr, y_tr, X_val, y_val, model_name="random_forest", param_grid={}
        )

        assert best_params == {}
        assert val_mae == evaluate(pipeline, X_val, y_val)

    def test_returns_the_best_params_and_a_matching_pipeline(self, raw_features_df, target_series):
        Xf = preprocess_features(raw_features_df)
        X_tr, X_val, y_tr, y_val = time_based_split(
            Xf, target_series, raw_features_df["date"], validation_size=0.2
        )

        pipeline, best_params, val_mae = hyperparameter_search(
            X_tr,
            y_tr,
            X_val,
            y_val,
            model_name="random_forest",
            param_grid={"model__n_estimators": [3, 5]},
        )

        assert best_params["n_estimators"] in (3, 5)
        assert pipeline.named_steps["model"].n_estimators == best_params["n_estimators"]
        assert isinstance(val_mae, float)

    def test_the_returned_pipeline_is_fit_on_the_training_split_only(
        self, raw_features_df, target_series
    ):
        # Regression test: hyperparameter_search used to refit the winning
        # pipeline on train+validation combined, which let it see the very
        # rows it was then scored against (see src/model.py's docstring).
        # Ridge has a deterministic, closed-form fit (no bootstrap/random_state
        # noise), so its coefficients exactly reveal which rows were used.
        Xf = preprocess_features(raw_features_df)
        X_tr, X_val, y_tr, y_val = time_based_split(
            Xf, target_series, raw_features_df["date"], validation_size=0.2
        )

        pipeline, _, _ = hyperparameter_search(
            X_tr,
            y_tr,
            X_val,
            y_val,
            model_name="linear_regression",
            param_grid={"model__alpha": [1.0]},
        )

        train_only = train_model(X_tr, y_tr, model_name="linear_regression", alpha=1.0)
        combined = train_model(
            pd.concat([X_tr, X_val], ignore_index=True),
            pd.concat([y_tr, y_val], ignore_index=True),
            model_name="linear_regression",
            alpha=1.0,
        )

        np.testing.assert_allclose(
            pipeline.named_steps["model"].coef_, train_only.named_steps["model"].coef_
        )
        assert not np.allclose(
            pipeline.named_steps["model"].coef_, combined.named_steps["model"].coef_
        )
