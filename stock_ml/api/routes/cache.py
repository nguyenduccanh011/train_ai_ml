"""Cache management routes: footprint + orphan stats for the dashboard panel."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body

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


@router.post("/gc/sweep")
def gc_sweep(payload: dict = Body(default={})) -> dict:
    """Run a GC pass. Dry-run by default; quarantines only when ``{"apply": true}``.

    Attribution currently covers the legacy ``features/<set>/<key>`` and prediction
    caches only — the FeatureStore (``features/store/**``) is out of scope (§1.6), so
    an apply never touches it. The §1.6 empty-reference guard inside ``sweep()`` refuses
    to quarantine when nothing could be attributed to a run.
    """
    from stock_ml.src.cache.garbage_collector import sweep

    apply = bool(payload.get("apply", False))
    purge = payload.get("purge_older_than_days")
    report = sweep(
        _results_dir(),
        dry_run=not apply,
        purge_older_than_days=float(purge) if purge is not None else None,
    )
    mb = 1024 * 1024
    return {
        "dry_run": report.dry_run,
        "orphan_count": report.orphan_count,
        "orphan_mb": round(report.orphan_bytes / mb, 1),
        "quarantined": len(report.quarantined),
        "purged": len(report.purged),
    }


@router.post("/cache/purge-trash")
def purge_trash_route(payload: dict = Body(default={})) -> dict:
    """Permanently delete quarantine batches older than N days (default 7)."""
    from stock_ml.src.cache.garbage_collector import purge_trash

    days = float(payload.get("older_than_days", 7.0))
    trash_root = _results_dir() / "cache" / "_trash"
    before = _dir_bytes(trash_root)
    removed = purge_trash(_results_dir() / "cache", days)
    freed = before - _dir_bytes(trash_root)
    return {"purged_dirs": len(removed), "freed_mb": round(freed / (1024 * 1024), 1)}
