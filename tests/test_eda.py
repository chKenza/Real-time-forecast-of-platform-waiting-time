"""Unit tests for `src.eda`"""

from __future__ import annotations

import pandas as pd

from src.eda import (
    main,
    parse_args,
    plot_correlation_heatmap,
    plot_past_feature_boxplots,
    plot_station_volume,
    plot_target_by_calendar,
    plot_target_distribution,
    print_data_quality_summary,
)


class TestPlotTargetDistribution:
    """Tests for :func:`~src.eda.plot_target_distribution`."""

    def test_writes_a_png_file(self, tmp_path, target_series):
        plot_target_distribution(target_series, tmp_path)

        assert (tmp_path / "target_distribution.png").exists()


class TestPlotPastFeatureBoxplots:
    """Tests for :func:`~src.eda.plot_past_feature_boxplots`."""

    def test_writes_a_png_file(self, tmp_path, raw_features_df):
        plot_past_feature_boxplots(raw_features_df, tmp_path)

        assert (tmp_path / "past_feature_boxplots.png").exists()


class TestPlotCorrelationHeatmap:
    """Tests for :func:`~src.eda.plot_correlation_heatmap`."""

    def test_writes_a_png_file(self, tmp_path, raw_features_df, target_series):
        plot_correlation_heatmap(raw_features_df, target_series, tmp_path)

        assert (tmp_path / "correlation_heatmap.png").exists()


class TestPlotTargetByCalendar:
    """Tests for :func:`~src.eda.plot_target_by_calendar`."""

    def test_writes_a_png_file(self, tmp_path, raw_features_df, target_series):
        plot_target_by_calendar(raw_features_df, target_series, tmp_path)

        assert (tmp_path / "target_by_calendar.png").exists()


class TestPlotStationVolume:
    """Tests for :func:`~src.eda.plot_station_volume`."""

    def test_writes_a_png_file(self, tmp_path, raw_features_df):
        plot_station_volume(raw_features_df, tmp_path)

        assert (tmp_path / "station_volume.png").exists()

    def test_respects_top_n(self, tmp_path, raw_features_df):
        plot_station_volume(raw_features_df, tmp_path, top_n=1)  # should not raise

        assert (tmp_path / "station_volume.png").exists()


class TestPrintDataQualitySummary:
    """Tests for :func:`~src.eda.print_data_quality_summary`."""

    def test_reports_the_row_count(self, capsys, raw_features_df, target_series):
        print_data_quality_summary(raw_features_df, target_series)

        assert f"Rows: {len(raw_features_df)}" in capsys.readouterr().out

    def test_reports_the_number_of_unique_stations(self, capsys, raw_features_df, target_series):
        print_data_quality_summary(raw_features_df, target_series)

        assert "Unique stations (gare): 2" in capsys.readouterr().out

    def test_reports_outlier_counts_for_every_past_column(
        self, capsys, raw_features_df, target_series
    ):
        print_data_quality_summary(raw_features_df, target_series)

        out = capsys.readouterr().out
        # p0q4 carries the one seeded outlier (100.0, beyond the default 30 clip).
        assert "p0q4: 1 rows beyond" in out
        assert "p2q0: 0 rows beyond" in out


class TestParseArgs:
    """Tests for :func:`~src.eda.parse_args`."""

    def test_defaults(self):
        args = parse_args([])

        assert args.data_dir == "data"
        assert args.output_dir == "figures/EDA"

    def test_parses_custom_paths(self):
        args = parse_args(["--data-dir", "somewhere", "--output-dir", "there"])

        assert args.data_dir == "somewhere"
        assert args.output_dir == "there"


class TestMain:
    """Tests for :func:`~src.eda.main` (integration: runs the real EDA pipeline)."""

    def test_generates_every_expected_figure(self, tmp_path, raw_features_df, target_series):
        data_dir = tmp_path / "data"
        out_dir = tmp_path / "figures"
        data_dir.mkdir()

        raw_features_df.assign(**{"Unnamed: 0": range(len(raw_features_df))}).to_csv(
            data_dir / "x_train_final.csv", index=False
        )
        pd.DataFrame({"p0q0": target_series}).to_csv(
            data_dir / "y_train_final_testsuffix.csv", index=False
        )

        main(["--data-dir", str(data_dir), "--output-dir", str(out_dir)])

        expected = {
            "target_distribution.png",
            "past_feature_boxplots.png",
            "correlation_heatmap.png",
            "target_by_calendar.png",
            "station_volume.png",
        }
        assert expected.issubset({p.name for p in out_dir.iterdir()})
