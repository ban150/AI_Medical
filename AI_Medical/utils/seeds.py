"""Reproductibilité (entraînement, splits, numpy)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seeds(seed: int = 42) -> None:
    """Fixe les graines Python, NumPy et TensorFlow (meilleur effort sur GPU)."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
