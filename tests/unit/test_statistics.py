"""Unit tests for utils.statistics."""

import json

import pandas as pd

from utils.statistics import ModelStatistics


class TestModelStatistics:
    """Tests for ModelStatistics."""

    def test_calculate_metrics_perfect_predictions(self):
        """With perfect predictions all metrics are 1.0."""
        # given
        y_test = pd.Series([1, 0, 1, 0])
        y_pred = pd.Series([1, 0, 1, 0])

        # when
        metrics = ModelStatistics.calculate_metrics(y_test, y_pred)

        # then
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_calculate_metrics_creates_parent_dir_if_needed(self, tmp_path):
        """Saving to a path with missing parent directory creates it."""
        # given
        output_file = tmp_path / "subdir" / "metrics.json"
        y_test = pd.Series([1, 0])
        y_pred = pd.Series([1, 0])

        # when
        ModelStatistics.calculate_metrics(y_test, y_pred, output_path=str(output_file))
        assert output_file.exists()

        # then
        with open(output_file) as f:
            data = json.load(f)

        assert "accuracy" in data
        assert "precision" in data
        assert "recall" in data
        assert "f1" in data
