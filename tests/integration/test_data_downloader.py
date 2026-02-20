"""Integration tests for data.data_downloader."""

import pandas as pd
import pytest

from data.data_downloader import DataDownloader

PG15_URL = (
    "https://github.com/dutangc/CASdatasets/raw/refs/heads/master/data/pg15training.rda"
)
DATASET_NAME = "pg15training"


@pytest.mark.integration
class TestDataDownloader:
    """Integration tests for DataDownloader.download_data (requires network)."""

    def test_download_data_fetches_and_saves_csv(self, tmp_path):
        """download_data fetches .rda from URL and saves CSV to output_path."""
        # given
        output_path = tmp_path / "pg15training.csv"

        # when
        DataDownloader.download_data(PG15_URL, str(output_path), DATASET_NAME)

        # then
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        assert len(df.columns) >= 1
