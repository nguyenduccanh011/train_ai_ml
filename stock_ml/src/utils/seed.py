"""Global seed propagation for reproducibility."""

import random

import numpy as np


def set_global_seed(seed: int | None = None) -> int:
    """Set seed for all RNG systems: numpy, random, LightGBM, XGBoost, sklearn, tf.

    Args:
        seed: seed value. If None, uses a random seed.

    Returns:
        The seed used (useful for logging).
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import lightgbm

        class SilentLogger:
            def info(self, msg):
                pass

            def warning(self, msg):
                pass

        lightgbm.register_logger(SilentLogger())
    except (ImportError, AttributeError, TypeError):
        pass

    try:
        import xgboost

        xgboost.set_config(verbosity=0)
    except (ImportError, AttributeError):
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except (ImportError, AttributeError):
        pass

    return seed
