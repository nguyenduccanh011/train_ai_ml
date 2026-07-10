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
