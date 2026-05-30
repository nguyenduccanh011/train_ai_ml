"""Model lifecycle management routes (pin/unpin/retire/delete)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

router = APIRouter(prefix="/api/v1", tags=["runs"])


def _results_dir() -> Path:
    """Get results directory from environment."""
    from stock_ml.src.utils.env import get_results_dir
    return Path(get_results_dir())


def _viz_dir() -> Path:
    """Get visualization directory."""
    import os
    stock_ml_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("STOCK_VIZ_DIR") or (stock_ml_root / "visualization"))


def _leaderboard_path() -> Path:
    """Get leaderboard.json path."""
    return _results_dir() / "leaderboard" / "leaderboard.json"


def _read_rows() -> list[dict[str, Any]]:
    """Read all leaderboard rows."""
    path = _leaderboard_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("rows", [])
    return data


def _find_row(run_id: str) -> dict[str, Any] | None:
    """Find a single row by run_id."""
    return next((r for r in _read_rows() if r.get("run_id") == run_id), None)


def _run_dir_for(row: dict[str, Any]) -> Path | None:
    """Resolve run directory from leaderboard row."""
    meta_rel = (row.get("artifacts") or {}).get("meta_json", "")
    if meta_rel:
        meta_path = (_results_dir() / meta_rel).resolve()
        if meta_path.exists():
            return meta_path.parent
    bundle = row.get("bundle", "")
    run_name = row.get("run_name", "")
    candidate = _results_dir() / "experiments" / bundle / run_name
    return candidate if candidate.exists() else None


_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _spawn_retrain(run_dir: Path, device: str = "cpu") -> str:
    """Spawn async retrain subprocess, return job_id."""
    job_id = uuid.uuid4().hex[:12]
    log_path = run_dir / "retrain.log"
    log_fh = log_path.open("w", encoding="utf-8")
    stock_ml_root = Path(__file__).resolve().parents[2]
    proc = subprocess.Popen(
        [sys.executable, "-m", "stock_ml.scripts.retrain_run", str(run_dir), "--device", device],
        cwd=str(stock_ml_root.parent),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "run_dir": str(run_dir),
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "log": str(log_path),
            "_proc": proc,
            "_log_fh": log_fh,
        }
    return job_id


def _job_status(job_id: str) -> dict[str, Any] | None:
    """Get job status, updating process exit code."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        proc = job["_proc"]
        code = proc.poll()
        if code is None:
            job["status"] = "running"
        else:
            job["status"] = "done" if code == 0 else "error"
            job["exit_code"] = code
            if not job["_log_fh"].closed:
                job["_log_fh"].close()
        return {k: v for k, v in job.items() if not k.startswith("_")}


def _set_state(run_dir: Path, state: str) -> None:
    """Write lifecycle.json with new state."""
    (run_dir / "lifecycle.json").write_text(
        json.dumps({"state": state, "updated_at": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )


def _rebuild_leaderboard() -> None:
    """Rebuild leaderboard.json from experiments dir."""
    from stock_ml.src.leaderboard.aggregator import rebuild_leaderboard
    out = _results_dir() / "leaderboard"
    rebuild_leaderboard(str(_results_dir() / "experiments"), str(out))


def _version_key_for(row: dict[str, Any]) -> str:
    """Get stable filesystem-safe dashboard key for a run."""
    ch = str(row.get("config_hash") or "").strip() or "nohash"
    return f"pin_{ch[:8]}"


def _export_run_to_dashboard(run_dir: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    """Export pinned run's trades.csv to visualization/data_<key>/ for dashboard."""
    from stock_ml.src.export.unified_export import export_version

    trades_csv = run_dir / "trades.csv"
    if not trades_csv.exists():
        return None
    version_key = _version_key_for(row)
    model_cfg = {
        "name": row.get("run_name", version_key),
        "color": "#2962ff",
        "marker_shape": "arrowUp",
        "market": row.get("market"),
        "market_family": row.get("market_family"),
        "timeframe": row.get("timeframe"),
        "schema": row.get("schema"),
    }
    return export_version(
        version_key,
        model_cfg,
        str(_results_dir()),
        str(_viz_dir()),
        trades_csv=str(trades_csv),
    )


def _rebuild_pinned_manifest() -> dict[str, Any]:
    """Rebuild visualization/manifest.json to contain exactly pinned runs."""
    from stock_ml.src.export.unified_export import generate_manifest

    pinned = [r for r in _read_rows() if r.get("state") == "pinned"]
    pinned_keys = {_version_key_for(r) for r in pinned}

    # Drop stale data dirs
    for d in _viz_dir().glob("data_pin_*"):
        if d.is_dir() and d.name[len("data_") :] not in pinned_keys:
            shutil.rmtree(d, ignore_errors=True)

    exported: list[dict[str, Any]] = []
    for row in pinned:
        run_dir = _run_dir_for(row)
        if run_dir is None:
            continue
        model = _export_run_to_dashboard(run_dir, row)
        if model is not None:
            exported.append(model)

    generate_manifest(exported, str(_viz_dir()), merge=False)
    return {"pinned": len(exported)}


def _quarantine_run_cache(row: dict[str, Any]) -> list[str]:
    """Move run's cache files to _trash, return moved paths."""
    from stock_ml.src.cache.garbage_collector import quarantine

    cache_root = _results_dir() / "cache"
    keys = row.get("cache_keys") or {}
    feat, pred = keys.get("features", ""), keys.get("predictions", "")
    targets: list[Path] = []
    if feat:
        targets += list((cache_root / "features").glob(f"*/{feat}.*"))
    if pred:
        p = cache_root / "predictions" / f"{pred}.pkl"
        if p.exists():
            targets.append(p)
    if not targets:
        return []
    moved = quarantine(targets, cache_root)
    return [str(p) for p in moved]


def _delete_run(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Delete run directory and its cache, rebuild leaderboard."""
    moved = _quarantine_run_cache(row)
    shutil.rmtree(run_dir, ignore_errors=True)
    _rebuild_leaderboard()
    _rebuild_pinned_manifest()
    return {"deleted_run_dir": str(run_dir), "quarantined_cache": moved}


def _resolve(run_id: str) -> tuple[dict[str, Any], Path]:
    """Resolve (row, run_dir) or raise 404."""
    row = _find_row(run_id)
    if row is None:
        raise HTTPException(404, detail=f"run not found: {run_id}")
    run_dir = _run_dir_for(row)
    if run_dir is None:
        raise HTTPException(404, detail=f"run dir not found for {run_id}")
    return row, run_dir


@router.get("/runs")
def list_runs(state: str = "", market: str = "") -> list[dict[str, Any]]:
    """List all runs, optionally filtered by state and market."""
    rows = _read_rows()
    if state:
        rows = [r for r in rows if r.get("state") == state]
    if market:
        rows = [r for r in rows if r.get("market") == market]
    return rows


@router.get("/runs/{run_id}/state")
def get_run_state(run_id: str) -> dict:
    """Get run's current lifecycle state."""
    row, run_dir = _resolve(run_id)
    return {"run_id": run_id, "state": row.get("state", "trained")}


@router.patch("/runs/{run_id}/state")
def patch_run_state(run_id: str, payload: dict = Body(...)) -> dict:
    """Change run lifecycle state (trained/pinned/retired)."""
    state = str(payload.get("state", "")).lower()
    if state not in {"trained", "pinned", "retired"}:
        raise HTTPException(400, detail=f"invalid state: {state!r}")
    _row, run_dir = _resolve(run_id)
    _set_state(run_dir, state)
    _rebuild_leaderboard()
    manifest_info = _rebuild_pinned_manifest()
    return {"run_id": run_id, "state": state, "dashboard": manifest_info}


@router.post("/runs/{run_id}/retrain")
def retrain(run_id: str, payload: dict = Body(default={})) -> dict:
    """Spawn async retrain job, return job_id."""
    _row, run_dir = _resolve(run_id)
    device = str(payload.get("device", "cpu"))
    job_id = _spawn_retrain(run_dir, device=device)
    return {"job_id": job_id, "run_id": run_id, "status": "running"}


@router.delete("/runs/{run_id}/cache")
def delete_cache(run_id: str) -> dict:
    """Quarantine run's cache files."""
    row, _run_dir = _resolve(run_id)
    moved = _quarantine_run_cache(row)
    return {"run_id": run_id, "quarantined_cache": moved}


@router.delete("/runs/{run_id}")
def delete_run(run_id: str) -> dict:
    """Delete run directory and its cache, rebuild leaderboard."""
    row, run_dir = _resolve(run_id)
    return _delete_run(run_dir, row)


@router.post("/runs/bulk-state")
def bulk_set_state(payload: dict = Body(...)) -> dict:
    """Bulk update run lifecycle state.

    Payload:
    {
        "state": "trained|pinned|retired",
        "filter": {
            "current_state": "",  # optional: only update rows with this state
            "market": "",         # optional: only update rows with this market
            "state": ""          # optional: another way to specify current_state
        }
    }
    """
    state = str(payload.get("state", "")).lower()
    if state not in {"trained", "pinned", "retired"}:
        raise HTTPException(400, detail=f"invalid state: {state!r}")

    filters = payload.get("filter", {})
    current_state = filters.get("current_state") or filters.get("state", "")
    market = filters.get("market", "")

    rows = _read_rows()
    matching = []
    updated_run_ids = []

    for row in rows:
        if current_state and row.get("state", "trained") != current_state:
            continue
        if market and row.get("market") != market:
            continue
        matching.append(row)

    # Update state for all matching rows
    for row in matching:
        run_dir = _run_dir_for(row)
        if run_dir is not None:
            _set_state(run_dir, state)
            updated_run_ids.append(row.get("run_id", ""))

    # Rebuild once
    if matching:
        _rebuild_leaderboard()
        _rebuild_pinned_manifest()

    return {
        "updated": len(updated_run_ids),
        "run_ids": updated_run_ids,
        "state": state,
    }


@router.delete("/runs/bulk")
def bulk_delete_runs(payload: dict = Body(...)) -> dict:
    """Bulk delete runs by state.

    Payload:
    {
        "state": "retired|trained|pinned",
        "confirm": true  # must be true to confirm deletion
    }
    """
    if not payload.get("confirm", False):
        raise HTTPException(400, detail="Must set confirm: true to delete")

    state = str(payload.get("state", "")).lower()
    if state not in {"trained", "pinned", "retired"}:
        raise HTTPException(400, detail=f"invalid state: {state!r}")

    rows = _read_rows()
    matching = [r for r in rows if r.get("state", "trained") == state]

    deleted_run_ids = []
    total_freed_mb = 0.0

    for row in matching:
        run_dir = _run_dir_for(row)
        if run_dir is None:
            continue

        cache_dir_before = _dir_size_mb(run_dir)
        _delete_run(run_dir, row)
        total_freed_mb += cache_dir_before
        deleted_run_ids.append(row.get("run_id", ""))

    # Rebuild once (delete_run rebuilds each time, but we'll rebuild again to be sure)
    if matching:
        _rebuild_leaderboard()
        _rebuild_pinned_manifest()

    return {
        "deleted": len(deleted_run_ids),
        "freed_mb": round(total_freed_mb, 1),
        "run_ids": deleted_run_ids,
    }


def _dir_size_mb(path: Path) -> float:
    """Calculate total size of directory in MB."""
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return round(total / (1024 * 1024), 1)


@router.get("/cache/stats")
def cache_stats() -> dict:
    """Get cache statistics: feature cache, predictions, orphans, trash."""
    from stock_ml.src.cache.garbage_collector import sweep

    cache_root = _results_dir() / "cache"
    features_dir = cache_root / "features"
    predictions_dir = cache_root / "predictions"
    trash_dir = cache_root / "_trash"

    # Feature cache size
    feature_cache_mb = _dir_size_mb(features_dir)

    # Prediction cache size
    prediction_cache_mb = _dir_size_mb(predictions_dir)

    # Trash size
    trash_mb = _dir_size_mb(trash_dir)

    # Orphan detection (dry-run)
    report = sweep(_results_dir(), dry_run=True)

    return {
        "feature_cache_mb": feature_cache_mb,
        "prediction_cache_mb": prediction_cache_mb,
        "orphan_count": report.orphan_count,
        "orphan_mb": round(report.orphan_bytes / (1024 * 1024), 1),
        "trash_mb": trash_mb,
        "referenced_features": len(report.referenced_features),
        "referenced_predictions": len(report.referenced_predictions),
    }


@router.post("/cache/purge-trash")
def purge_trash_endpoint(payload: dict = Body(default={})) -> dict:
    """Purge trash batches older than N days."""
    from stock_ml.src.cache.garbage_collector import purge_trash

    older_than_days = float(payload.get("older_than_days", 7.0))
    cache_root = _results_dir() / "cache"
    trash_before = _dir_size_mb(cache_root / "_trash")

    removed = purge_trash(cache_root, older_than_days=older_than_days)

    trash_after = _dir_size_mb(cache_root / "_trash")
    freed_mb = round(trash_before - trash_after, 1)

    return {
        "purged_dirs": len(removed),
        "freed_mb": freed_mb,
        "older_than_days": older_than_days,
    }


@router.post("/gc/sweep")
def gc_sweep(payload: dict = Body(default={})) -> dict:
    """Run garbage collector sweep on cache."""
    from stock_ml.src.cache.garbage_collector import sweep

    apply = bool(payload.get("apply", False))
    purge = payload.get("purge_older_than_days")
    report = sweep(
        _results_dir(),
        dry_run=not apply,
        purge_older_than_days=float(purge) if purge is not None else None,
    )
    return {
        "dry_run": report.dry_run,
        "orphan_count": report.orphan_count,
        "orphan_mb": round(report.orphan_bytes / (1024 * 1024), 1),
        "quarantined": len(report.quarantined),
        "purged": len(report.purged),
    }
