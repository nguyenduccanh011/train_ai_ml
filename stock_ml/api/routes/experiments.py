"""Experiment/backtest job submission routes (DB-First Phase 0.3).

DEPRECATED: Old YAML-based experiment submission is no longer supported.
Use template-based submission instead: POST /api/templates/{template_id}/submit
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1", tags=["experiments"])


def _stock_ml_root() -> Path:
    """Get stock_ml root directory."""
    return Path(__file__).resolve().parents[2]


@router.post("/experiments")
def submit_experiment_deprecated() -> dict:
    """DEPRECATED: Use template-based submission instead.

    New flow:
    1. Create ModelComponents via POST /api/model-library/components
    2. Create StrategyTemplate via POST /api/templates
    3. Submit via POST /api/templates/{template_id}/submit

    This endpoint is kept for backward compatibility but will error.
    """
    raise HTTPException(
        status_code=410,
        detail="YAML-based experiment submission is deprecated. "
        "Use template-based flow instead:\n"
        "1. Create rule components: POST /api/model-library/components\n"
        "2. Create template: POST /api/templates\n"
        "3. Submit: POST /api/templates/{template_id}/submit",
    )


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
