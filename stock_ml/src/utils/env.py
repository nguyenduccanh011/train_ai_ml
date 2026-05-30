"""Path resolution for local research pipeline."""

import os


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

    stock_ml_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(stock_ml_dir, config_data_dir))


def get_results_dir():
    """Get results directory (stock_ml/results)."""
    env_override = os.environ.get("STOCK_RESULTS_DIR")
    if env_override:
        return env_override

    stock_ml_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(stock_ml_dir, "results")


def get_experiment_dir(experiment_key: str) -> str:
    """Return path to a specific experiment subfolder inside results/.

    Args:
        experiment_key: e.g. "leading_v2__lightgbm"

    Returns:
        Absolute path: results/{experiment_key}/
    """
    return os.path.join(get_results_dir(), experiment_key)


