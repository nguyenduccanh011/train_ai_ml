"""
Environment detection and path resolution for local/Colab hybrid workflow.

All path constants and environment checks live here.
Other modules import from this file instead of hardcoding paths.
"""
import os

DRIVE_BASE = "/content/drive/MyDrive/stock_ml_hub"


def is_colab():
    """Detect if running inside Google Colab."""
    return os.path.exists("/content") and "COLAB_RELEASE_TAG" in os.environ


def resolve_data_dir(config_data_dir):
    """Resolve the data directory based on environment.

    Args:
        config_data_dir: relative path from models.yaml (e.g., "../portable_data/...")

    Returns:
        Absolute path to the data directory.
        - Local: resolves config_data_dir relative to stock_ml/
        - Colab: uses Drive mount path
    """
    env_override = os.environ.get("STOCK_DATA_DIR")
    if env_override:
        return env_override

    if is_colab():
        return os.path.join(DRIVE_BASE, "portable_data", "vn_stock_ai_dataset_cleaned")

    stock_ml_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(stock_ml_dir, config_data_dir))


def get_results_dir():
    """Get the results directory based on environment.

    Returns:
        - Local: stock_ml/results/
        - Colab: Drive results dir (also copies to local for export)
    """
    env_override = os.environ.get("STOCK_RESULTS_DIR")
    if env_override:
        return env_override

    if is_colab():
        return os.path.join(DRIVE_BASE, "results")

    stock_ml_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(stock_ml_dir, "results")


def get_env_info():
    """Print environment summary."""
    import shutil

    info = {
        "environment": "Google Colab" if is_colab() else "Local",
        "data_dir": resolve_data_dir("../portable_data/vn_stock_ai_dataset_cleaned"),
        "results_dir": get_results_dir(),
    }

    if is_colab():
        try:
            import torch
            info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
            info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"
        except Exception:
            info["gpu"] = "unknown"

        total, used, free = shutil.disk_usage("/content")
        info["disk_free"] = f"{free / 1e9:.1f} GB"
    else:
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

    data_exists = os.path.isdir(info["data_dir"])
    info["data_available"] = data_exists

    return info
