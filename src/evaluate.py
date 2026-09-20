"""
Generate validation-set diagnostic plots for one or more trained models.

Loads the raw training data, reproduces the same chronological train/validation
split used by ``src.train`` and for
each given model plots predicted-vs-actual, residuals, error by weekday, and
daily actual-vs-predicted trends on the held-out validation rows.

Usage
-----
    python -m src.evaluate --model-path models/model.joblib
    python -m src.evaluate \\
        --model-path models/random_forest.joblib models/lightgbm.joblib \\
        --labels random_forest lightgbm

Run ``python -m src.evaluate --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.model import mae  # noqa: E402
from src.preprocessing import get_default_data_paths, load_xy, preprocess_features  # noqa: E402
from src.utils import chronological_split_mask, load_model  # noqa: E402

DISPLAY_RANGE = (-15, 15)


def plot_predicted_vs_actual(
    y_true: pd.Series, y_pred: np.ndarray, label: str, out_dir: Path
) -> None:
    """
    Save a predicted-vs-actual density plot to ``out_dir``.

    :param y_true: ground-truth target values
    :param y_pred: predicted target values
    :param label: model name, used in the title and file name
    :param out_dir: directory to save the figure in
    """
    lo, hi = DISPLAY_RANGE
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    hb = ax.hexbin(
        y_true.clip(lo, hi), np.clip(y_pred, lo, hi), gridsize=40, cmap="Blues", mincnt=1
    )
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlabel("actual p0q0 (minutes)")
    ax.set_ylabel("predicted p0q0 (minutes)")
    ax.set_title(f"Predicted vs actual: {label}")
    ax.legend()
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out_dir / f"pred_vs_actual_{label}.png", dpi=150)
    plt.close(fig)


def plot_residual_histogram(
    y_true: pd.Series, y_pred: np.ndarray, label: str, out_dir: Path
) -> None:
    """
    Save a histogram of prediction residuals to ``out_dir``.

    :param y_true: ground-truth target values
    :param y_pred: predicted target values
    :param label: model name, used in the title and file name
    :param out_dir: directory to save the figure in
    """
    lo, hi = DISPLAY_RANGE
    residuals = (y_true.to_numpy() - y_pred).clip(lo, hi)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=41, color="#2b6cb0", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"Residuals (actual - predicted): {label}")
    ax.set_xlabel("residual (minutes)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / f"residual_hist_{label}.png", dpi=150)
    plt.close(fig)


def plot_residuals_vs_predicted(
    y_true: pd.Series, y_pred: np.ndarray, label: str, out_dir: Path
) -> None:
    """
    Save a residuals-vs-predicted density plot to ``out_dir``.

    :param y_true: ground-truth target values
    :param y_pred: predicted target values
    :param label: model name, used in the title and file name
    :param out_dir: directory to save the figure in
    """
    lo, hi = DISPLAY_RANGE
    residuals = np.clip(y_true.to_numpy() - y_pred, lo, hi)
    fig, ax = plt.subplots(figsize=(6, 4))
    hb = ax.hexbin(np.clip(y_pred, lo, hi), residuals, gridsize=40, cmap="Blues", mincnt=1)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("predicted p0q0 (minutes)")
    ax.set_ylabel("residual (actual - predicted)")
    ax.set_title(f"Residuals vs predicted: {label}")
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out_dir / f"residuals_vs_predicted_{label}.png", dpi=150)
    plt.close(fig)


def plot_mae_by_weekday(
    X_val: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, label: str, out_dir: Path
) -> None:
    """
    Save a bar chart of MAE by weekday to ``out_dir``.

    :param X_val: validation features, with a ``weekday`` column
    :param y_true: ground-truth target values
    :param y_pred: predicted target values
    :param label: model name, used in the title and file name
    :param out_dir: directory to save the figure in
    """
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df = pd.DataFrame(
        {"weekday": X_val["weekday"].to_numpy(), "abs_err": np.abs(y_true.to_numpy() - y_pred)}
    )
    by_weekday = df.groupby("weekday")["abs_err"].mean().sort_index()
    labels = [weekday_names[i] for i in by_weekday.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, by_weekday.values, color="#2b6cb0")
    ax.set_title(f"Validation MAE by weekday: {label}")
    ax.set_ylabel("MAE (minutes)")
    fig.tight_layout()
    fig.savefig(out_dir / f"mae_by_weekday_{label}.png", dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted_over_time(
    dates_val: pd.Series, y_true: pd.Series, y_pred: np.ndarray, label: str, out_dir: Path
) -> None:
    """
    Save a daily mean actual-vs-predicted line chart to ``out_dir``.

    :param dates_val: date values, one per validation row
    :param y_true: ground-truth target values
    :param y_pred: predicted target values
    :param label: model name, used in the title and file name
    :param out_dir: directory to save the figure in
    """
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_val).to_numpy(),
            "actual": y_true.to_numpy(),
            "predicted": y_pred,
        }
    )
    daily = df.groupby("date")[["actual", "predicted"]].mean().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(daily.index, daily["actual"], label="actual", marker="o", markersize=3)
    ax.plot(daily.index, daily["predicted"], label="predicted", marker="o", markersize=3)
    ax.set_title(f"Daily mean p0q0, actual vs predicted: {label}")
    ax.set_ylabel("mean p0q0 (minutes)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / f"daily_actual_vs_predicted_{label}.png", dpi=150)
    plt.close(fig)


def plot_mae_comparison(labels: list[str], maes: list[float], out_dir: Path) -> None:
    """
    Save a bar chart comparing validation MAE across models to ``out_dir``.

    :param labels: model names, in display order
    :param maes: validation MAE per model, aligned with ``labels``
    :param out_dir: directory to save the figure in
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, maes, color="#2b6cb0")
    ax.set_title("Validation MAE by model")
    ax.set_ylabel("MAE (minutes)")
    fig.tight_layout()
    fig.savefig(out_dir / "mae_comparison.png", dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation CLI.

    :param argv: argument list to parse; defaults to ``sys.argv[1:]``
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory with the raw CSV files.")
    parser.add_argument(
        "--model-path",
        nargs="+",
        default=["models/model.joblib"],
        help="One or more saved pipelines (joblib) to evaluate.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display name per --model-path (defaults to each file's stem).",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Must match the value used to train the model(s) being evaluated.",
    )
    parser.add_argument(
        "--clip-value",
        type=float,
        default=30.0,
        help="Must match the value used to train the model(s) being evaluated.",
    )
    parser.add_argument("--output-dir", default="figures/models", help="Where to save PNG figures.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Reproduce the validation split and generate diagnostic plots for each model.

    :param argv: argument list to parse; defaults to ``sys.argv[1:]``
    :raises ValueError: if ``--labels`` is given with a different length than ``--model-path``
    """
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels if args.labels is not None else [Path(p).stem for p in args.model_path]
    if len(labels) != len(args.model_path):
        raise ValueError("--labels must have the same length as --model-path")

    paths = get_default_data_paths(args.data_dir)
    print(f"Loading data from {args.data_dir} ...", flush=True)
    X_raw, y = load_xy(paths["x_train"], paths["y_train"])
    Xf = preprocess_features(X_raw, clip_value=args.clip_value)

    is_train = chronological_split_mask(X_raw["date"], args.validation_size)
    X_val = Xf.reset_index(drop=True)[~is_train]
    y_val = y.reset_index(drop=True)[~is_train]
    dates_val = X_raw["date"].reset_index(drop=True)[~is_train]
    print(f"Validation set: {X_val.shape}", flush=True)

    model_maes = []
    for model_path, label in zip(args.model_path, labels):
        print(f"Evaluating {label} ({model_path}) ...", flush=True)
        pipeline = load_model(model_path)
        y_pred = pipeline.predict(X_val)
        val_mae = mae(y_val, y_pred)
        print(f"  validation MAE: {val_mae:.4f}", flush=True)
        model_maes.append(val_mae)

        plot_predicted_vs_actual(y_val, y_pred, label, out_dir)
        plot_residual_histogram(y_val, y_pred, label, out_dir)
        plot_residuals_vs_predicted(y_val, y_pred, label, out_dir)
        plot_mae_by_weekday(X_val, y_val, y_pred, label, out_dir)
        plot_actual_vs_predicted_over_time(dates_val, y_val, y_pred, label, out_dir)

    if len(labels) > 1:
        plot_mae_comparison(labels, model_maes, out_dir)

    print(f"Saved figures to {out_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
