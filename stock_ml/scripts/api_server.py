"""Local API server for the Model Lifecycle UI.

Serves the leaderboard/dashboard static pages plus a small JSON API to manage
model lifecycle from the browser:

  GET    /api/runs                  list leaderboard rows (+ state/cache/artifacts)
  PATCH  /api/runs/{run_id}         change lifecycle state {state: pinned|trained|retired}
  POST   /api/runs/{run_id}/retrain spawn a retrain job, returns {job_id}
  GET    /api/jobs/{job_id}         poll job status
  DELETE /api/runs/{run_id}/cache   quarantine this run's cache, keep metrics
  DELETE /api/runs/{run_id}         delete run dir + its cache, drop from leaderboard
  POST   /api/gc/sweep              run GC sweep {apply: bool, purge_older_than_days?: float}

Run:
    python -m stock_ml.scripts.api_server --port 5176
Then open http://localhost:5176/visualization/leaderboard.html
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _results_dir() -> Path:
    from src.env import get_results_dir

    return Path(get_results_dir())


def _leaderboard_path() -> Path:
    return _results_dir() / "leaderboard" / "leaderboard.json"


def _read_rows() -> list[dict[str, Any]]:
    path = _leaderboard_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("rows", [])
    return data


def _find_row(run_id: str) -> dict[str, Any] | None:
    return next((r for r in _read_rows() if r.get("run_id") == run_id), None)


def _run_dir_for(row: dict[str, Any]) -> Path | None:
    """Resolve the run directory from a leaderboard row's artifacts/meta path."""
    meta_rel = (row.get("artifacts") or {}).get("meta_json", "")
    if meta_rel:
        meta_path = (_results_dir() / meta_rel).resolve()
        if meta_path.exists():
            return meta_path.parent
    # Fallback: experiments/<bundle>/<run_name>
    bundle = row.get("bundle", "")
    run_name = row.get("run_name", "")
    candidate = _results_dir() / "experiments" / bundle / run_name
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Job registry (in-memory) for async retrains
# ---------------------------------------------------------------------------

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _spawn_retrain(run_dir: Path, device: str = "cpu") -> str:
    job_id = uuid.uuid4().hex[:12]
    log_path = run_dir / "retrain.log"
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "stock_ml.scripts.retrain_run", str(run_dir), "--device", device],
        cwd=str(ROOT.parent),
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


# ---------------------------------------------------------------------------
# Lifecycle / cache / delete operations
# ---------------------------------------------------------------------------


def _set_state(run_dir: Path, state: str) -> None:
    (run_dir / "lifecycle.json").write_text(
        json.dumps({"state": state, "updated_at": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )


def _rebuild_leaderboard() -> None:
    from src.leaderboard.aggregator import rebuild_leaderboard

    out = _results_dir() / "leaderboard"
    rebuild_leaderboard(str(_results_dir() / "experiments"), str(out))


def _quarantine_run_cache(row: dict[str, Any]) -> list[str]:
    """Move this run's feature+prediction cache files to _trash. Returns moved paths."""
    from src.cache.garbage_collector import quarantine

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
    import shutil

    moved = _quarantine_run_cache(row)
    shutil.rmtree(run_dir, ignore_errors=True)
    _rebuild_leaderboard()
    return {"deleted_run_dir": str(run_dir), "quarantined_cache": moved}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app():
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Model Lifecycle API")

    def _resolve(run_id: str) -> tuple[dict[str, Any], Path]:
        """Return (row, run_dir) or raise 404. run_id passed as query param to
        tolerate '#' and '/' that break path routing."""
        row = _find_row(run_id)
        if row is None:
            raise HTTPException(404, f"run not found: {run_id}")
        run_dir = _run_dir_for(row)
        if run_dir is None:
            raise HTTPException(404, f"run dir not found for {run_id}")
        return row, run_dir

    @app.get("/api/runs")
    def list_runs(state: str = "", market: str = "") -> JSONResponse:
        rows = _read_rows()
        if state:
            rows = [r for r in rows if r.get("state") == state]
        if market:
            rows = [r for r in rows if r.get("market") == market]
        return JSONResponse(rows)

    @app.patch("/api/runs")
    def patch_run(run_id: str, payload: dict = Body(...)) -> dict:
        state = str(payload.get("state", "")).lower()
        if state not in {"trained", "pinned", "retired"}:
            raise HTTPException(400, f"invalid state: {state!r}")
        _row, run_dir = _resolve(run_id)
        _set_state(run_dir, state)
        _rebuild_leaderboard()
        return {"run_id": run_id, "state": state}

    @app.post("/api/runs/retrain")
    def retrain(run_id: str, payload: dict = Body(default={})) -> dict:
        _row, run_dir = _resolve(run_id)
        device = str(payload.get("device", "cpu"))
        job_id = _spawn_retrain(run_dir, device=device)
        return {"job_id": job_id, "run_id": run_id, "status": "running"}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        status = _job_status(job_id)
        if status is None:
            raise HTTPException(404, f"job not found: {job_id}")
        return status

    @app.delete("/api/runs/cache")
    def delete_cache(run_id: str) -> dict:
        row, _run_dir = _resolve(run_id)
        moved = _quarantine_run_cache(row)
        return {"run_id": run_id, "quarantined_cache": moved}

    @app.delete("/api/runs")
    def delete_run(run_id: str) -> dict:
        row, run_dir = _resolve(run_id)
        return _delete_run(run_dir, row)

    @app.post("/api/gc/sweep")
    def gc_sweep(payload: dict = Body(default={})) -> dict:
        from src.cache.garbage_collector import sweep

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

    # Static files served from stock_ml root (so /visualization and /results resolve)
    app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")
    return app


def main() -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Model Lifecycle API server")
    parser.add_argument("--port", type=int, default=5176)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    app = create_app()
    print(f"Model Lifecycle API on http://{args.host}:{args.port}")
    print(f"  Leaderboard: http://{args.host}:{args.port}/visualization/leaderboard.html")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
