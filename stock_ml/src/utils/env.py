"""Path resolution for local research pipeline."""

import os


def _stock_ml_dir() -> str:
    """Absolute path to the stock_ml package root (this file is stock_ml/src/utils/env.py)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_data_dir(config_data_dir):
    """Resolve data directory from config.

    Args:
        config_data_dir: relative path from config (e.g., "../portable_data/...")

    Returns:
        Absolute path to the data directory.
    """
    env_override = os.environ.get("STOCK_DATA_DIR")
    if env_override:
        return env_override

    if config_data_dir is None:
        raise ValueError("data_dir is required")

    if os.path.isabs(config_data_dir):
        return os.path.normpath(config_data_dir)

    return os.path.normpath(os.path.join(_stock_ml_dir(), config_data_dir))


def get_results_dir():
    """Get results directory (stock_ml/results)."""
    env_override = os.environ.get("STOCK_RESULTS_DIR")
    if env_override:
        return env_override

    return os.path.join(_stock_ml_dir(), "results")


def get_experiment_dir(experiment_key: str) -> str:
    """Return path to a specific experiment subfolder inside results/.

    Args:
        experiment_key: e.g. "leading_v2__lightgbm"

    Returns:
        Absolute path: results/{experiment_key}/
    """
    return os.path.join(get_results_dir(), experiment_key)


def get_cache_dir() -> str:
    """Get cache directory (results/cache)."""
    cache_dir = os.path.join(get_results_dir(), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
