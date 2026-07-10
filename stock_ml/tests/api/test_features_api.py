"""Tests for the read-only Feature Store API (/api/v1/features).

Feature definitions are authored in code (``src/features/catalog.py``) and seeded
one-way into the DB; the API only reads that mirror plus a stateless ``/validate``
checker. So these tests seed the catalog, then assert the read endpoints reflect it.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import stock_ml.db.models  # noqa: F401 — register all tables on Base.metadata
from stock_ml.api.main import app
from stock_ml.db.base import Base
from stock_ml.db.dependencies import get_db
from stock_ml.scripts.seed_features import seed_with_session


@pytest.fixture
def client(tmp_path):
    db_file = tmp_path / "test.db"
    sync_engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()

    # Seed the catalog into the DB mirror using a throwaway engine in its own loop
    # (avoids aiosqlite loop-affinity issues with the TestClient's request loop).
    async def _seed():
        seed_engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}")
        seed_maker = async_sessionmaker(seed_engine, expire_on_commit=False)
        async with seed_maker() as session:
            await seed_with_session(session)
        await seed_engine.dispose()

    asyncio.run(_seed())

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


def test_validate_endpoint(client):
    ok = client.post("/api/v1/features/validate", json={"expr": "RSI($close, 14)"}).json()
    assert ok["valid"] is True and ok["kind"] == "per_symbol"
    assert ok["deps"]["raw"] == ["close"]

    cs = client.post("/api/v1/features/validate", json={"expr": "CSRank(#ret_20d)"}).json()
    assert cs["kind"] == "cross_sectional" and cs["deps"]["features"] == ["ret_20d"]

    bad = client.post("/api/v1/features/validate", json={"expr": "RSI($close,"}).json()
    assert bad["valid"] is False and bad["error"]


def test_write_endpoints_are_gone(client):
    # The catalog is the single source of truth; the API must not create features.
    assert client.post(
        "/api/v1/features/definitions", json={"name": "x", "expr": "$close"}
    ).status_code == 405
    assert client.post(
        "/api/v1/features/sets", json={"name": "s", "featureIds": [1]}
    ).status_code == 405


def test_list_definitions_reflects_catalog(client):
    defs = client.get("/api/v1/features/definitions").json()
    names = {d["name"]: d for d in defs}
    assert "rsi_14" in names and "ret_20d" in names
    assert names["rsi_14"]["kind"] == "per_symbol"
    assert names["rsi_14"]["expr"] == "RSI($close, 14)"
    # rsi_14 belongs to several seeded sets → used by > 0
    assert names["rsi_14"]["usedByCount"] > 0
    # momentum_rank is cross-sectional
    assert names["momentum_rank"]["kind"] == "cross_sectional"

    filtered = client.get("/api/v1/features/definitions?kind=per_symbol&search=rsi").json()
    assert {d["name"] for d in filtered} == {"rsi_14", "rsi_7"}


def test_definition_detail(client):
    defs = {d["name"]: d for d in client.get("/api/v1/features/definitions").json()}
    fid = defs["rsi_14"]["id"]
    detail = client.get(f"/api/v1/features/definitions/{fid}").json()
    assert detail["expr"] == "RSI($close, 14)"
    assert detail["deps"]["raw"] == ["close"]
    assert len(detail["setIds"]) > 0
    assert detail["materialized"] is False


def test_sets_reflect_catalog(client):
    sets = {s["name"]: s for s in client.get("/api/v1/features/sets").json()}
    assert sets["basic_v1"]["memberCount"] == 8
    assert sets["leading_v2"]["memberCount"] == 37
    assert sets["leading_v3"]["memberCount"] == 56

    detail = client.get(f"/api/v1/features/sets/{sets['leading_v2']['id']}").json()
    member_names = [m["name"] for m in detail["members"]]
    assert len(member_names) == 37
    assert member_names[0] == "ret_1d"  # ordered by position


def test_materializations_empty(client):
    # Seeding writes definitions/sets only — no feature has been materialised yet.
    assert client.get("/api/v1/features/materializations").json() == []
