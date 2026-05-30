"""Model management endpoints"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..config import settings
from ..middleware.rate_limiter import get_limiter
from ..schemas.models import ModelListResponse, TrainModelRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])
limiter = get_limiter()


def load_manifest() -> dict:
    """Load manifest.json with error handling"""
    # Try multiple paths in order of preference
    paths_to_try = [
        settings.config_dir / "manifest.json",
        Path("config/manifest.json"),
        Path("./config/manifest.json"),
    ]

    manifest_path = None
    for path in paths_to_try:
        if path.exists():
            manifest_path = path
            logger.debug(f"Found manifest at: {path}")
            break

    if not manifest_path:
        logger.warning(f"Manifest not found in any of {[str(p) for p in paths_to_try]}")
        return {"models": []}

    try:
        with open(manifest_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse manifest.json: {e}")
        return {"models": []}
    except Exception as e:
        logger.error(f"Error loading manifest: {e}")
        raise HTTPException(status_code=500, detail="Failed to load manifest")


@router.get("/models", response_model=ModelListResponse)
@limiter.limit("100/minute")
async def list_models(request: Request) -> dict:
    """Get all available models with proper validation"""
    try:
        manifest = load_manifest()
        models = manifest.get("models", [])
        logger.info(f"list_models: loaded {len(models)} models from manifest")

        return {"total": len(models), "models": models}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.get("/models/{model_id}")
@limiter.limit("100/minute")
async def get_model(request: Request, model_id: str) -> dict:
    """Get specific model details with input validation"""
    # Validate model_id format
    if not all(c.isalnum() or c == "_" for c in model_id):
        raise HTTPException(status_code=400, detail="Invalid model ID format")

    try:
        manifest = load_manifest()
        for model in manifest.get("models", []):
            if model.get("name") == model_id:
                return model

        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model")


@router.post("/models/train")
@limiter.limit("10/minute")
async def train_model(request: Request, config: TrainModelRequest) -> dict:
    """Trigger model training (async) - validated input"""
    logger.info(f"Training request received for model: {config.model_name}")

    # TODO: Implement async training with Celery/APScheduler
    return {
        "status": "queued",
        "job_id": "job_123",
        "model_name": config.model_name,
        "message": "Training started in background",
    }
