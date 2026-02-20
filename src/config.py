from dataclasses import dataclass

from utils.files import read_yaml


@dataclass
class IntRange:
    min: int
    max: int


@dataclass
class FloatRange:
    min: float
    max: float


@dataclass
class DataConfig:
    url: str
    dataset_file_path: str
    dataset_name: str


@dataclass
class DatasetConfig:
    test_size: float


@dataclass
class OptunaConfig:
    n_trials: int
    n_estimators: IntRange
    learning_rate: FloatRange
    max_depth: IntRange
    num_leaves: IntRange
    min_child_samples: IntRange
    subsample: FloatRange
    colsample_bytree: FloatRange


@dataclass
class ModelConfig:
    output_path: str
    output_params_path: str
    metrics_path: str


@dataclass
class Config:
    random_state: int
    data: DataConfig
    dataset: DatasetConfig
    optuna: OptunaConfig
    model: ModelConfig


def load_config(path: str) -> Config:
    config_file = read_yaml(path)

    return Config(
        random_state=config_file["random_state"],
        data=DataConfig(**config_file["data"]),
        dataset=DatasetConfig(**config_file["dataset"]),
        optuna=OptunaConfig(
            n_trials=config_file["optuna"]["n_trials"],
            n_estimators=IntRange(**config_file["optuna"]["n_estimators"]),
            learning_rate=FloatRange(**config_file["optuna"]["learning_rate"]),
            max_depth=IntRange(**config_file["optuna"]["max_depth"]),
            num_leaves=IntRange(**config_file["optuna"]["num_leaves"]),
            min_child_samples=IntRange(**config_file["optuna"]["min_child_samples"]),
            subsample=FloatRange(**config_file["optuna"]["subsample"]),
            colsample_bytree=FloatRange(**config_file["optuna"]["colsample_bytree"]),
        ),
        model=ModelConfig(**config_file["model"]),
    )
