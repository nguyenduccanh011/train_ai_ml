"""MLflow experiment tracking for research reproducibility."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_git_dirty() -> bool:
    """Check if git working tree is dirty."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else False
    except Exception:
        return False


def data_fingerprint(symbols: list[str], start_date: str, end_date: str) -> str:
    """Compute stable hash of data specification.

    Args:
        symbols: sorted list of symbols
        start_date: data start date (YYYY-MM-DD)
        end_date: data end date (YYYY-MM-DD)

    Returns:
        SHA256 hex digest (16 chars)
    """
    spec = f"{','.join(sorted(symbols))}|{start_date}|{end_date}"
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


class MLFlowLogger:
    """Context manager for MLflow experiment tracking.

    Logs:
    - Config YAML (frozen)
    - Git commit + dirty flag
    - Data fingerprint
    - All metrics + CI bands
    - Output artifacts (CSV, JSON)
    """

    def __init__(
        self,
        tracking_uri: str | Path | None = None,
        experiment_name: str = "research",
    ):
        """Initialize MLFlow logger.

        Args:
            tracking_uri: MLflow tracking directory. Defaults to <project>/mlruns
            experiment_name: experiment name for grouping runs
        """
        self.tracking_uri = tracking_uri or Path("mlruns")
        self.experiment_name = experiment_name
        self.run = None
        self._in_context = False

    def __enter__(self):
        if not MLFLOW_AVAILABLE:
            return self

        tracking_uri = str(self.tracking_uri)
        if not tracking_uri.startswith(("http://", "https://", "file://")):
            tracking_uri = f"file:///{Path(tracking_uri).resolve()}"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run()
        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not MLFLOW_AVAILABLE or not self._in_context:
            return
        mlflow.end_run()
        self._in_context = False

    def log_config(self, config_dict: dict | Any):
        """Log config dict or dataclass as JSON artifact."""
        if not MLFLOW_AVAILABLE or not self._in_context:
            return

        if hasattr(config_dict, "__dataclass_fields__"):
            config_dict = asdict(config_dict)
        mlflow.log_dict(config_dict, "config.json")

    def log_reproducibility(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        seed: int,
    ):
        """Log code + data versioning info.

        Args:
            symbols: data symbols
            start_date: first date in data
            end_date: last date in data
            seed: random seed
        """
        if not MLFLOW_AVAILABLE or not self._in_context:
            return

        git_commit = get_git_commit()
        git_dirty = get_git_dirty()
        fingerprint = data_fingerprint(symbols, start_date, end_date)

        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("git_dirty", git_dirty)
        mlflow.log_param("data_fingerprint", fingerprint)
        mlflow.log_param("seed", seed)

    def log_metrics(self, metrics: dict[str, float]):
        """Log scalar metrics."""
        if not MLFLOW_AVAILABLE or not self._in_context:
            return
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(key, val)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None):
        """Log file as artifact.

        Args:
            local_path: file to upload
            artifact_path: destination folder in MLflow (optional)
        """
        if not MLFLOW_AVAILABLE or not self._in_context:
            return
        mlflow.log_artifact(str(local_path), artifact_path)

    def get_run_id(self) -> str:
        """Get current run ID."""
        if not MLFLOW_AVAILABLE or not self._in_context or not self.run:
            return "no_mlflow"
        return self.run.info.run_id
