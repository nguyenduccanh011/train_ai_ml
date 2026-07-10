"""Cache management routes: footprint + orphan stats for the dashboard panel."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["cache"])


def _results_dir() -> Path:
    from stock_ml.src.utils.env import get_results_dir

    return Path(get_results_dir())


def _dir_bytes(path: Path) -> int:
    """Total size of the files under a cache subdir (0 if it does not exist)."""
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@router.get("/cache/stats")
def cache_stats() -> dict:
    """Cache footprint (MB) + orphan summary the dashboard cache panel reads.

    Orphan counts come from a read-only GC dry-run; sizes are walked from
    results/cache/{features,predictions,_trash}.
    """
    from stock_ml.src.cache.garbage_collector import sweep

    cache_root = _results_dir() / "cache"
    features_bytes = _dir_bytes(cache_root / "features")
    predictions_bytes = _dir_bytes(cache_root / "predictions")
    trash_bytes = _dir_bytes(cache_root / "_trash")

    report = sweep(_results_dir(), dry_run=True)
    mb = 1024 * 1024
    return {
        "feature_cache_mb": round(features_bytes / mb, 1),
        "prediction_cache_mb": round(predictions_bytes / mb, 1),
        "orphan_count": report.orphan_count,
        "orphan_mb": round(report.orphan_bytes / mb, 1),
        "trash_mb": round(trash_bytes / mb, 1),
        "features_size": features_bytes,
        "total_size": features_bytes + predictions_bytes + trash_bytes,
    }
