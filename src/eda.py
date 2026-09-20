"""
Exploratory data analysis: generate figures summarizing the raw data and the
effect of preprocessing.

Usage
-----
    python -m src.eda
    python -m src.eda --data-dir data --output-dir figures/EDA

Run ``python -m src.eda --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.preprocessing import (  # noqa: E402
    DEFAULT_OUTLIER_CLIP,
    PAST_VALUE_COLUMNS,
    TARGET_COLUMN,
    add_date_features,
    get_default_data_paths,
    load_xy,
)


def plot_target_distribution(y: pd.Series, out_dir: Path) -> None:
    """
    Save a histogram of the target distribution to ``out_dir``.

    :param y: target values
    :param out_dir: directory to save the figure in
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    clipped = y.clip(-20, 20)
    ax.hist(clipped, bins=41, color="#2b6cb0", edgecolor="white")
    ax.set_title(f"Distribution of {TARGET_COLUMN} (clipped to [-20, 20] for display)")
    ax.set_xlabel("p0q0 (minutes)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "target_distribution.png", dpi=150)
    plt.close(fig)


def plot_past_feature_boxplots(X: pd.DataFrame, out_dir: Path) -> None:
    """
    Save boxplots of the six past-delay features to ``out_dir``.

    :param X: raw features dataframe
    :param out_dir: directory to save the figure in
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [X[c].clip(-30, 30) for c in PAST_VALUE_COLUMNS]
    ax.boxplot(data, tick_labels=PAST_VALUE_COLUMNS, showfliers=True)
    ax.axhline(
        DEFAULT_OUTLIER_CLIP, color="red", linestyle="--", linewidth=1, label="clip threshold"
    )
    ax.axhline(-DEFAULT_OUTLIER_CLIP, color="red", linestyle="--", linewidth=1)
    ax.set_title("Past-delay feature distributions (clipped to [-30, 30] for display)")
    ax.set_ylabel("minutes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "past_feature_boxplots.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    """
    Save a heatmap of the past-delay features' correlation with the target.

    :param X: raw features dataframe
    :param y: target values
    :param out_dir: directory to save the figure in
    """
    df = X[PAST_VALUE_COLUMNS].copy()
    df[TARGET_COLUMN] = y
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation: past-delay features vs target")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_target_by_calendar(X_with_dates: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    """
    Save bar charts of the mean target by weekday and by month to ``out_dir``.

    :param X_with_dates: raw features dataframe with a ``date`` column
    :param y: target values
    :param out_dir: directory to save the figure in
    """
    df = add_date_features(X_with_dates)
    df[TARGET_COLUMN] = y.to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_weekday = df.groupby("weekday")[TARGET_COLUMN].mean().sort_index()
    weekday_labels = [weekday_names[i] for i in by_weekday.index]
    axes[0].bar(weekday_labels, by_weekday.values, color="#2b6cb0")
    axes[0].set_title("Mean p0q0 by weekday")
    axes[0].set_ylabel("mean p0q0 (minutes)")

    by_month = df.groupby("month")[TARGET_COLUMN].mean()
    axes[1].bar(by_month.index.astype(str), by_month.values, color="#2b6cb0")
    axes[1].set_title("Mean p0q0 by month")
    axes[1].set_xlabel("month")

    fig.tight_layout()
    fig.savefig(out_dir / "target_by_calendar.png", dpi=150)
    plt.close(fig)


def plot_station_volume(X: pd.DataFrame, out_dir: Path, top_n: int = 20) -> None:
    """
    Save a bar chart of the top stations by stop count to ``out_dir``.

    :param X: raw features dataframe
    :param out_dir: directory to save the figure in
    :param top_n: number of stations to show
    """
    counts = X["gare"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#2b6cb0")
    ax.set_title(f"Top {top_n} stations by number of stops")
    ax.set_xlabel("number of rows")
    fig.tight_layout()
    fig.savefig(out_dir / "station_volume.png", dpi=150)
    plt.close(fig)


def print_data_quality_summary(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Print row/column counts, missing values, and outlier counts to stdout.

    :param X: raw features dataframe
    :param y: target values
    """
    print(f"Rows: {len(X)}")
    print(f"Unique stations (gare): {X['gare'].nunique()}")
    print(f"Unique trains: {X['train'].nunique()}")
    print(f"Date range: {X['date'].min()} to {X['date'].max()}")
    print(f"Missing values per column:\n{X.isnull().sum()}")
    print(f"Target (p0q0) summary:\n{y.describe()}")
    for col in PAST_VALUE_COLUMNS:
        n_outliers = int((X[col].abs() > DEFAULT_OUTLIER_CLIP).sum())
        print(
            f"  {col}: {n_outliers} rows beyond +/-{DEFAULT_OUTLIER_CLIP:g} "
            f"({100 * n_outliers / len(X):.3f}%)"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the EDA CLI.

    :param argv: argument list to parse; defaults to ``sys.argv[1:]``
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory with the raw CSV files.")
    parser.add_argument("--output-dir", default="figures/EDA", help="Where to save PNG figures.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Load the training data, print a data-quality summary, and save all EDA figures.

    :param argv: argument list to parse; defaults to ``sys.argv[1:]``
    """
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = get_default_data_paths(args.data_dir)
    print(f"Loading data from {args.data_dir} ...", flush=True)
    X, y = load_xy(paths["x_train"], paths["y_train"])

    print_data_quality_summary(X, y)

    print(f"Generating figures in {out_dir} ...", flush=True)
    plot_target_distribution(y, out_dir)
    plot_past_feature_boxplots(X, out_dir)
    plot_correlation_heatmap(X, y, out_dir)
    plot_target_by_calendar(X, y, out_dir)
    plot_station_volume(X, out_dir)
    print("Done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
