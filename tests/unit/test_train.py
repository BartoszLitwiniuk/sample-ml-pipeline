import json

import numpy as np
import pandas as pd

from config import FloatRange, IntRange, OptunaConfig
from model.model_trainer import ModelTrainer


def _make_optuna_config(n_trials: int = 2) -> OptunaConfig:
    """Minimal OptunaConfig for fast integration tests."""
    return OptunaConfig(
        n_trials=n_trials,
        n_estimators=IntRange(min=5, max=20),
        learning_rate=FloatRange(min=0.05, max=0.2),
        max_depth=IntRange(min=2, max=4),
        num_leaves=IntRange(min=8, max=16),
        min_child_samples=IntRange(min=2, max=10),
        subsample=FloatRange(min=0.7, max=1.0),
        colsample_bytree=FloatRange(min=0.7, max=1.0),
    )


def _make_synthetic_data(
    n_samples: int = 200, n_features: int = 5, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic numeric X and binary y for training."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series((X.sum(axis=1) > 0).astype(int))
    return X, y


class TestModelTrainer:
    def test_run_optimization_returns_fitted_model_and_params(self):
        """run_optimization runs Optuna and returns a fitted model and best params."""
        # given
        X, y = _make_synthetic_data(n_samples=150, n_features=4)
        config = _make_optuna_config(n_trials=2)
        trainer = ModelTrainer(X, y, random_state=42)

        # when
        model, params = trainer.run_optimization(test_size=0.2, config=config)
        preds = model.predict(X.head(10))

        # then
        assert model is not None
        assert params is not None
        assert "n_estimators" in params
        assert "learning_rate" in params
        assert "random_state" in params

        assert preds.shape[0] == 10
        assert set(preds).issubset({0, 1})

    def test_save_writes_params_and_model_to_disk(self, tmp_path):
        """save() writes params JSON and model file to given paths."""

        # given
        X, y = _make_synthetic_data(n_samples=100, n_features=3)
        config = _make_optuna_config(n_trials=1)
        trainer = ModelTrainer(X, y, random_state=42)
        trainer.run_optimization(test_size=0.2, config=config)

        param_path = tmp_path / "params.json"
        model_path = tmp_path / "model.txt"

        # when
        trainer.save(str(param_path), str(model_path))

        # then
        assert param_path.exists()
        with open(param_path) as f:
            loaded = json.load(f)
        assert "n_estimators" in loaded and "random_state" in loaded

        assert model_path.exists()
        assert model_path.stat().st_size > 0
