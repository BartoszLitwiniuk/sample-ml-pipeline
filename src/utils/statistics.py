import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils.custom_logger import logger


class ModelStatistics:
    @staticmethod
    def calculate_metrics(
        y_test: pd.Series, y_pred: pd.Series, output_path: str = None
    ) -> float:
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        logger.debug(f"Model metrics: {metrics}")

        if output_path:
            ModelStatistics._save_metrics(metrics, output_path)

        return metrics

    @staticmethod
    def _save_metrics(metrics: dict, output_path: str):
        logger.debug(f"Save model metrics to path: {output_path}")
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f)
