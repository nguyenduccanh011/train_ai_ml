"""Experiment/backtest job submission routes."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException

from .jobs import attach_process, register_job

router = APIRouter(prefix="/api/v1", tags=["experiments"])


def _stock_ml_root() -> Path:
    """Get stock_ml root directory."""
    return Path(__file__).resolve().parents[2]


@router.post("/experiments")
def submit_experiment(payload: dict = Body(...)) -> dict:
    """Submit a backtest experiment config and queue it for execution.

    Expected payload:
    {
        "name": "my_exp",
        "config": {...}  # experiment config dict
    }
    Returns job_id for polling."""

    if not payload.get("name") or not payload.get("config"):
        raise HTTPException(400, detail="Missing 'name' or 'config' in payload")

    name = str(payload["name"]).strip()
    if not name or len(name) > 100:
        raise HTTPException(400, detail="Invalid experiment name (1-100 chars)")

    config = payload.get("config")
    if not isinstance(config, dict):
        raise HTTPException(400, detail="config must be a dict")

    # Write config to temp file
    stock_ml_root = _stock_ml_root()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        dir=stock_ml_root / "stock_ml" / "config" / "experiments",
        delete=False,
        encoding="utf-8",
    ) as f:
        config_path = Path(f.name)
        # Write config as YAML (simple dict dump)
        import yaml

        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Queue job via run_experiments.py
    job_id = f"exp_{name}_{config_path.stem}"
    log_path = stock_ml_root / "logs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    register_job(
        job_id,
        {
            "job_id": job_id,
            "type": "experiment",
            "name": name,
            "config_path": str(config_path),
            "status": "queued",
            "log": str(log_path),
        },
    )

    # Start subprocess
    try:
        log_fh = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "stock_ml.scripts.run_experiments",
                "--pending",
                str(stock_ml_root / "stock_ml" / "config" / "experiments"),
                "--done",
                str(stock_ml_root / "results" / "experiments_done"),
                "--failed",
                str(stock_ml_root / "results" / "experiments_failed"),
                "--parallel",
                "1",
            ],
            cwd=str(stock_ml_root.parent),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        attach_process(job_id, proc, log_path)
    except Exception as e:
        log_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)
        raise HTTPException(500, detail=f"Failed to start experiment: {e}")

    return {
        "job_id": job_id,
        "name": name,
        "config_path": str(config_path),
        "status": "queued",
    }


@router.get("/experiments/pending")
def list_pending_experiments() -> list[str]:
    """List pending experiment config files."""
    stock_ml_root = _stock_ml_root()
    pending_dir = stock_ml_root / "stock_ml" / "config" / "experiments"
    if not pending_dir.exists():
        return []
    return [
        f.stem for f in pending_dir.glob("*.yaml") if f.is_file() and not f.name.startswith(".")
    ]
