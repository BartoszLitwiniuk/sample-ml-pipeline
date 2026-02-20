"""Unit tests for data.data_loader."""

import pandas as pd
import pytest

from data.data_loader import DataLoader


class TestDataLoaderLoadData:
    """Tests for DataLoader.load_data."""

    def test_load_data_returns_dataframe(self, tmp_path):
        """load_data returns a pandas DataFrame from a valid CSV."""
        # given
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n")

        # when
        result = DataLoader.load_data(str(csv_path))

        # then
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 3]
        assert result["b"].tolist() == [2, 4]

    def test_load_data_file_not_found_raises(self):
        """load_data raises when the path does not exist."""
        with pytest.raises(FileNotFoundError):
            DataLoader.load_data("/nonexistent/path/data.csv")
