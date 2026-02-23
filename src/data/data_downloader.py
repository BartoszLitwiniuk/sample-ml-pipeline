import io

import rdata
import requests

from pathlib import Path

class DataDownloader:

    @staticmethod
    def download_data(url: str, output_path: str, dataset_name: str):
        """Download an R .rda dataset from `url` and save it as CSV to `output_path`.

        `output_path` may be a relative path including filename (e.g. "data/raw/pg15training.csv").
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data_response = requests.get(url)
        data_file = io.BytesIO(data_response.content)
        r_data = rdata.read_rda(data_file)[dataset_name]
        r_data.to_csv(out_path, index=False)
