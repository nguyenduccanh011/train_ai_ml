"""Job tracking routes (async experiment runs, backtest queue)."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1", tags=["jobs"])

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _job_status(job_id: str) -> dict[str, Any] | None:
    """Get job status, updating process exit code if still running."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        if "_proc" not in job:
            # No process attached (pre-computed or queued)
            return {k: v for k, v in job.items() if not k.startswith("_")}

        proc = job["_proc"]
        code = proc.poll()
        if code is None:
            job["status"] = "running"
        else:
            job["status"] = "done" if code == 0 else "error"
            job["exit_code"] = code
            if "_log_fh" in job and not job["_log_fh"].closed:
                job["_log_fh"].close()
        return {k: v for k, v in job.items() if not k.startswith("_")}


def register_job(job_id: str, metadata: dict[str, Any]) -> None:
    """Register a job without attaching a process."""
    with _JOBS_LOCK:
        _JOBS[job_id] = metadata


def attach_process(job_id: str, proc: subprocess.Popen, log_path: Path) -> None:
    """Attach a running process to an existing job."""
    with _JOBS_LOCK:
        if job_id in _JOBS:
            log_fh = log_path.open("w", encoding="utf-8")
            _JOBS[job_id]["_proc"] = proc
            _JOBS[job_id]["_log_fh"] = log_fh
            _JOBS[job_id]["status"] = "running"


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    """Get job status and metadata."""
    status = _job_status(job_id)
    if status is None:
        raise HTTPException(404, detail=f"job not found: {job_id}")
    return status


@router.get("/jobs")
def list_jobs(status: str = "") -> list[dict]:
    """List all jobs, optionally filtered by status."""
    with _JOBS_LOCK:
        jobs = [{k: v for k, v in job.items() if not k.startswith("_")} for job in _JOBS.values()]
    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    return jobs
