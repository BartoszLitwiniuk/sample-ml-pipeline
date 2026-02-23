import pandas as pd
from pathlib import Path

from pathlib import Path
from utils.custom_logger import logger

class DataLoader:

    @staticmethod
    def load_data(dataset_path: str) -> pd.DataFrame:
        logger.debug(f"Load dataset from path: {dataset_path}")

        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

        return pd.read_csv(path)
