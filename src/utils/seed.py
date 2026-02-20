"""Utility functions for setting random seeds to ensure reproducibility."""

import os
import random

import numpy as np

DEFAULT_SEED: int = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
