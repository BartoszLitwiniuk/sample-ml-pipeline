import argparse

import pandas as pd

from config import Config, load_config
from data.data_downloader import DataDownloader
from data.data_loader import DataLoader
from data.etl import DataPreparation
from model.model_trainer import ModelTrainer
from utils.custom_logger import logger
from utils.seed import set_seed
from utils.statistics import ModelStatistics


def ml_pipeline(config: Config):
    # Data preparation
    DataDownloader.download_data(
        config.data.url, config.data.dataset_file_path, config.data.dataset_name
    )
    dataset: pd.DataFrame = DataLoader.load_data(config.data.dataset_file_path)

    transformed_dataset: pd.DataFrame = DataPreparation.transform_dataset(dataset)
    X_train, X_test, y_train, y_test = DataPreparation.train_test_split(
        transformed_dataset, random_state=config.random_state
    )

    # Model training
    model_trainer = ModelTrainer(X_train, y_train, random_state=config.random_state)
    model_trainer.run_optimization(config.dataset.test_size, config.optuna)
    model_trainer.save(config.model.output_params_path, config.model.output_path)

    # Model evaluation
    y_pred = model_trainer.evaluate(X_test, y_test)
    ModelStatistics.calculate_metrics(y_test, y_pred, config.model.metrics_path)


def main(config_path: str):
    config: Config = load_config(config_path)
    logger.debug(config)

    set_seed(config.random_state)
    ml_pipeline(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()
    main(args.config_path)
