"""Test production FastAPI app (stock_ml.api.main)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture
def client(tmp_path):
    """Test client backed by a fresh file-based SQLite DB.

    Routes use the async ``get_db`` dependency; override it with an async session
    over a temp file so DB-backed endpoints work (empty data is fine here).
    """
    from stock_ml.api.main import app
    from stock_ml.db.base import Base
    from stock_ml.db.dependencies import get_db

    db_file = tmp_path / "test.db"
    sync_engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()

    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}")
    session_maker = async_sessionmaker(async_engine, expire_on_commit=False)

    async def override_get_db():
        async with session_maker() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_health_check(client):
    """Test health endpoint."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_leaderboard_endpoint(client):
    """Test leaderboard list endpoint."""
    resp = client.get("/api/v1/leaderboard")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data or "models" in data  # May be empty


def test_leaderboard_with_filters(client):
    """Test leaderboard with market filter."""
    resp = client.get("/api/v1/leaderboard?market=vn_stock&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("models", []), list)


def test_leaderboard_top_n(client):
    """Test top N models endpoint."""
    resp = client.get("/api/v1/leaderboard/top/5")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data


def test_list_runs(client):
    """Test list runs endpoint."""
    resp = client.get("/api/v1/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_list_runs_with_filters(client):
    """Test list runs with state filter."""
    resp = client.get("/api/v1/runs?state=trained")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_get_nonexistent_run_state(client):
    """Test get state for nonexistent run returns 404."""
    resp = client.get("/api/v1/runs/nonexistent-run/state")
    assert resp.status_code == 404


def test_get_nonexistent_run_trades(client):
    """Test get trades for nonexistent run returns 404."""
    resp = client.get("/api/v1/runs/nonexistent-run/trades")
    assert resp.status_code == 404


def test_cache_stats(client):
    """Test cache statistics endpoint."""
    resp = client.get("/api/v1/cache/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "features_size" in data or "total_size" in data


def test_list_jobs(client):
    """Test list jobs endpoint."""
    resp = client.get("/api/v1/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) or isinstance(data, dict)


def test_experiments_pending(client):
    """Test list pending experiments."""
    resp = client.get("/api/v1/experiments/pending")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) or isinstance(data, dict)


# --- Bulk lifecycle routes (/runs/bulk-state, /runs/bulk) ------------------------


def _seed_run(session_maker, run_id: str, state: str):
    """Insert one leaderboard row with the given lifecycle state (all NOT NULL cols filled)."""
    import asyncio
    from datetime import datetime

    from stock_ml.db.models.run import LeaderboardRunModel

    async def _insert():
        async with session_maker() as s:
            s.add(
                LeaderboardRunModel(
                    run_id=run_id,
                    bundle="test",
                    run_name=run_id,
                    config_hash="deadbeef",
                    generated_at=datetime(2026, 1, 1),
                    state=state,
                    strategy="s",
                    feature_set="f",
                    entry_model="lightgbm",
                    trades=0,
                    wr=0.0,
                    avg_pnl=0.0,
                    total_pnl=0.0,
                    pf=0.0,
                    avg_hold=0.0,
                    sharpe=0.0,
                    max_drawdown=0.0,
                    mdd_per_symbol=0.0,
                    yearly_consistency=0.0,
                    composite_score=0.0,
                    n_symbols=0,
                    first_test_year=2020,
                    last_test_year=2026,
                )
            )
            await s.commit()

    asyncio.run(_insert())


@pytest.fixture
def runs_client(tmp_path):
    """Client whose DB is pre-seeded with 2 trained + 1 retired run."""
    from stock_ml.api.main import app
    from stock_ml.db.base import Base
    from stock_ml.db.dependencies import get_db

    db_file = tmp_path / "runs.db"
    sync_engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()

    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}")
    session_maker = async_sessionmaker(async_engine, expire_on_commit=False)

    _seed_run(session_maker, "r-trained-1", "trained")
    _seed_run(session_maker, "r-trained-2", "trained")
    _seed_run(session_maker, "r-retired-1", "retired")

    async def override_get_db():
        async with session_maker() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_bulk_delete_requires_confirm(runs_client):
    """DELETE /runs/bulk without confirm must 400 — proves it hits the bulk route,
    not the /runs/{run_id} catch-all (which would 200 with deleted=false)."""
    resp = runs_client.request("DELETE", "/api/v1/runs/bulk", json={"state": "retired"})
    assert resp.status_code == 400


def test_bulk_delete_rejects_non_retired(runs_client):
    """Bulk delete is restricted to state=retired."""
    resp = runs_client.request(
        "DELETE", "/api/v1/runs/bulk", json={"state": "trained", "confirm": True}
    )
    assert resp.status_code == 400


def test_bulk_delete_removes_retired(runs_client):
    """Happy path: deletes exactly the retired rows, leaves the trained ones."""
    resp = runs_client.request(
        "DELETE", "/api/v1/runs/bulk", json={"state": "retired", "confirm": True}
    )
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1
    assert len(runs_client.get("/api/v1/runs?state=retired").json()) == 0
    assert len(runs_client.get("/api/v1/runs?state=trained").json()) == 2


def test_bulk_set_state_validates(runs_client):
    """Unknown target/filter state -> 400."""
    resp = runs_client.post("/api/v1/runs/bulk-state", json={"state": "bogus"})
    assert resp.status_code == 400


def test_bulk_set_state_retires_trained(runs_client):
    """Happy path: retire all trained runs, count reported, others untouched."""
    resp = runs_client.post(
        "/api/v1/runs/bulk-state",
        json={"state": "retired", "filter": {"current_state": "trained"}},
    )
    assert resp.status_code == 200
    assert resp.json()["updated"] == 2
    assert len(runs_client.get("/api/v1/runs?state=retired").json()) == 3
    assert len(runs_client.get("/api/v1/runs?state=trained").json()) == 0


# --- Cache-panel routes (/gc/sweep, /cache/purge-trash, /runs/{id}/cache) --------


def test_gc_sweep_dry_run(client, monkeypatch, tmp_path):
    """Dry-run sweep over an isolated empty results dir -> 200, 0 orphans, nothing moved."""
    (tmp_path / "cache" / "features").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    monkeypatch.setattr("stock_ml.api.routes.cache._results_dir", lambda: tmp_path)
    resp = client.post("/api/v1/gc/sweep", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["dry_run"] is True
    assert data["orphan_count"] == 0
    assert data["quarantined"] == 0


def test_purge_trash_removes_old_batches(client, monkeypatch, tmp_path):
    """Purge deletes trash batches older than the cutoff and reports freed bytes."""
    import os
    import time

    old_batch = tmp_path / "cache" / "_trash" / "batch_old"
    old_batch.mkdir(parents=True)
    (old_batch / "f.parquet").write_bytes(b"x" * 4096)
    stale = time.time() - 10 * 86400
    os.utime(old_batch, (stale, stale))

    monkeypatch.setattr("stock_ml.api.routes.cache._results_dir", lambda: tmp_path)
    resp = client.post("/api/v1/cache/purge-trash", json={"older_than_days": 7.0})
    assert resp.status_code == 200
    assert resp.json()["purged_dirs"] == 1
    assert not old_batch.exists()


def test_quarantine_run_cache_empty(runs_client):
    """Seeded run has no on-disk dir -> nothing to quarantine, still 200 (route not
    swallowed by the /runs/{run_id} catch-all)."""
    resp = runs_client.request("DELETE", "/api/v1/runs/r-trained-1/cache")
    assert resp.status_code == 200
    assert resp.json()["quarantined_cache"] == []


def test_quarantine_nonexistent_run_404(runs_client):
    """Unknown run id -> 404 from _resolve."""
    resp = runs_client.request("DELETE", "/api/v1/runs/does-not-exist/cache")
    assert resp.status_code == 404
