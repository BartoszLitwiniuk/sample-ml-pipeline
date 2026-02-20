"""Unit tests for utils.seed."""

import os
import random

import numpy as np

from utils.seed import set_seed


class TestSetSeed:
    """Tests for set_seed."""

    def test_set_seed_reproducibility_random(self):
        """Same seed produces same random sequence."""
        #  when
        set_seed(123)
        a = [random.random() for _ in range(5)]
        set_seed(123)
        b = [random.random() for _ in range(5)]
        # then
        assert a == b

    def test_set_seed_reproducibility_numpy(self):
        """Same seed produces same numpy random sequence."""
        # when
        set_seed(99)
        a = np.random.rand(5).tolist()
        set_seed(99)
        b = np.random.rand(5).tolist()
        # then
        assert a == b

    def test_set_seed_sets_pythonhashseed(self):
        """set_seed sets PYTHONHASHSEED environment variable."""
        # when
        set_seed(7)

        # then
        assert os.environ.get("PYTHONHASHSEED") == "7"
