import json
import os
from typing import Any

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import OptunaConfig
from utils.custom_logger import logger
from utils.seed import DEFAULT_SEED


class ModelTrainer:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = DEFAULT_SEED,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.best_params: dict[str, Any] | None = None
        self.best_model: LGBMClassifier | None = None

    def _objective(self, trial, test_size: int, config: OptunaConfig) -> float:
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=test_size,
            random_state=self.random_state,
        )

        param: dict[str, Any] = {
            "objective": "binary",
            "n_jobs": -1,
            "random_state": self.random_state,
            "n_estimators": trial.suggest_int(
                "n_estimators", config.n_estimators.min, config.n_estimators.max
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                config.learning_rate.min,
                config.learning_rate.max,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth", config.max_depth.min, config.max_depth.max
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", config.num_leaves.min, config.num_leaves.max
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                config.min_child_samples.min,
                config.min_child_samples.max,
            ),
            "subsample": trial.suggest_float(
                "subsample", config.subsample.min, config.subsample.max
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                config.colsample_bytree.min,
                config.colsample_bytree.max,
            ),
        }

        model = LGBMClassifier(**param, verbosity=-1)
        model.fit(X_train_sub, y_train_sub)

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy

    def run_optimization(
        self, test_size: int, config: OptunaConfig
    ) -> tuple[LGBMClassifier, dict[str, Any]]:
        logger.debug(f"Starting optimization for n_trials={config.n_trials}")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            lambda trial: self._objective(trial, test_size, config),
            n_trials=config.n_trials,
        )

        self.best_params = study.best_params
        self.best_params["random_state"] = self.random_state
        self.best_model = LGBMClassifier(**self.best_params)
        self.best_model.fit(self.X_train, self.y_train)

        return self.best_model, self.best_params

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        logger.debug("Start evaluation..")
        return self.best_model.predict(X_test)

    def save(self, output_param_path: str, output_model_path: str):
        self._save_params(output_param_path)
        self._save_model(output_model_path)

    def _save_params(self, output_param_path: str):
        if not output_param_path:
            raise Exception(
                "Parameter output_param_path is empty or None: ${output_param_path}"
            )

        if not os.path.exists(output_param_path):
            os.makedirs(os.path.dirname(output_param_path), exist_ok=True)

        logger.debug("Save model params to path: ${output_param_path}")

        with open(output_param_path, "w") as f:
            json.dump(self.best_params, f)

    def _save_model(self, output_model_path: str):
        if not output_model_path:
            raise Exception(
                "Parameter output_model_path is empty or None: ${output_model_path}"
            )

        if not os.path.exists(output_model_path):
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

        logger.debug("Save model to path: ${output_model_path}")
        self.best_model.booster_.save_model(output_model_path)
