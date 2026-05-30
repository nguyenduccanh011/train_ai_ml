"""Health check endpoints"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import psutil
import logging
from ..config import settings
from ..schemas.models import HealthCheckResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> dict:
    """Basic health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version
    }


@router.get("/health/detailed")
async def detailed_health() -> dict:
    """Detailed health check with system metrics"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app_version,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "data_dir_exists": settings.data_dir.exists(),
            "results_dir_exists": settings.results_dir.exists()
        }
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        raise HTTPException(status_code=500, detail="Failed to gather health metrics")
