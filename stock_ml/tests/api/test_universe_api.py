"""Tests for universe management API endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from stock_ml.api.main import app
from stock_ml.db.base import Base
from stock_ml.db.dependencies import get_db


@pytest.fixture
def client(tmp_path):
    """Test client backed by a fresh file-based SQLite DB.

    The universe routes use the async ``get_db`` dependency (AsyncSession), so we
    override it with an async session maker. A file (not :memory:) is used so the
    schema created up-front is visible to the connections TestClient opens on its
    own event loop.
    """
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


class TestUniversesAPI:
    """Test universe management endpoints."""

    def test_list_universes_empty(self, client):
        """GET /universes should return empty list initially."""
        response = client.get("/api/v1/universes")
        assert response.status_code == 200
        data = response.json()
        assert "universes" in data
        assert len(data["universes"]) == 0

    def test_create_universe(self, client):
        """POST /universes should create new universe."""
        payload = {
            "slug": "test_set_1",
            "name": "Test Set 1",
            "market": "vn_stock",
            "description": "Test universe",
            "symbols": [{"symbol": "ACB"}, {"symbol": "BID"}],
        }
        response = client.post("/api/v1/universes", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["slug"] == "test_set_1"
        assert data["name"] == "Test Set 1"
        assert data["market"] == "vn_stock"
        assert data["symbol_count"] == 2

    def test_create_duplicate_slug_fails(self, client):
        """POST /universes should fail on duplicate slug."""
        payload1 = {
            "slug": "duplicate",
            "name": "First",
            "market": "vn_stock",
        }
        payload2 = {
            "slug": "duplicate",
            "name": "Second",
            "market": "vn_stock",
        }
        client.post("/api/v1/universes", json=payload1)
        response = client.post("/api/v1/universes", json=payload2)
        assert response.status_code == 409

    def test_get_universe(self, client):
        """GET /universes/{slug} should return universe with symbols."""
        payload = {
            "slug": "test_get",
            "name": "Test Get",
            "market": "vn_stock",
            "description": "Test",
            "symbols": [{"symbol": "ACB", "group": "bank"}],
        }
        client.post("/api/v1/universes", json=payload)
        response = client.get("/api/v1/universes/test_get")
        assert response.status_code == 200
        data = response.json()
        assert data["slug"] == "test_get"
        assert data["name"] == "Test Get"
        assert len(data["symbols"]) == 1
        assert data["symbols"][0]["symbol"] == "ACB"
        assert data["symbols"][0]["group"] == "bank"

    def test_get_nonexistent_universe_404(self, client):
        """GET /universes/{slug} should return 404 for nonexistent slug."""
        response = client.get("/api/v1/universes/nonexistent")
        assert response.status_code == 404

    def test_update_universe_metadata(self, client):
        """PUT /universes/{slug} should update metadata."""
        payload = {
            "slug": "test_update",
            "name": "Original",
            "market": "vn_stock",
        }
        client.post("/api/v1/universes", json=payload)

        update_payload = {
            "name": "Updated Name",
            "description": "New description",
            "is_locked": True,
        }
        response = client.put("/api/v1/universes/test_update", json=update_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["is_locked"] is True

        # Verify
        response = client.get("/api/v1/universes/test_update")
        assert response.json()["name"] == "Updated Name"

    def test_add_symbols(self, client):
        """POST /universes/{slug}/symbols should add symbols."""
        payload = {
            "slug": "test_add_symbols",
            "name": "Test",
            "market": "vn_stock",
        }
        client.post("/api/v1/universes", json=payload)

        add_payload = {
            "symbols": [
                {"symbol": "ACB", "group": "bank"},
                {"symbol": "BID", "group": "bank"},
            ]
        }
        response = client.post("/api/v1/universes/test_add_symbols/symbols", json=add_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["added_count"] == 2
        assert data["symbol_count"] == 2

    def test_add_symbols_deduped(self, client):
        """POST /universes/{slug}/symbols should ignore duplicates."""
        payload = {
            "slug": "test_dedup",
            "name": "Test",
            "market": "vn_stock",
            "symbols": [{"symbol": "ACB"}],
        }
        client.post("/api/v1/universes", json=payload)

        add_payload = {"symbols": [{"symbol": "ACB"}, {"symbol": "BID"}]}
        response = client.post("/api/v1/universes/test_dedup/symbols", json=add_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["added_count"] == 1  # Only BID added
        assert data["symbol_count"] == 2

    def test_remove_symbol(self, client):
        """DELETE /universes/{slug}/symbols/{symbol} should remove symbol."""
        payload = {
            "slug": "test_remove",
            "name": "Test",
            "market": "vn_stock",
            "symbols": [{"symbol": "ACB"}, {"symbol": "BID"}],
        }
        client.post("/api/v1/universes", json=payload)

        response = client.delete("/api/v1/universes/test_remove/symbols/ACB")
        assert response.status_code == 200
        data = response.json()
        assert data["removed"] == "ACB"
        assert data["symbol_count"] == 1

        # Verify
        response = client.get("/api/v1/universes/test_remove")
        assert len(response.json()["symbols"]) == 1

    def test_replace_symbols(self, client):
        """PUT /universes/{slug}/symbols should replace all symbols."""
        payload = {
            "slug": "test_replace",
            "name": "Test",
            "market": "vn_stock",
            "symbols": [{"symbol": "ACB"}, {"symbol": "BID"}],
        }
        client.post("/api/v1/universes", json=payload)

        replace_payload = {"symbols": [{"symbol": "FPT"}, {"symbol": "VNM"}]}
        response = client.put("/api/v1/universes/test_replace/symbols", json=replace_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol_count"] == 2

        # Verify old symbols gone, new symbols present
        response = client.get("/api/v1/universes/test_replace")
        symbols = [s["symbol"] for s in response.json()["symbols"]]
        assert "ACB" not in symbols
        assert "FPT" in symbols
        assert "VNM" in symbols

    def test_soft_delete_universe(self, client):
        """DELETE /universes/{slug} should soft-delete."""
        payload = {
            "slug": "test_delete",
            "name": "Test",
            "market": "vn_stock",
        }
        client.post("/api/v1/universes", json=payload)

        response = client.delete("/api/v1/universes/test_delete")
        assert response.status_code == 200

        # Should not appear in active-only list
        response = client.get("/api/v1/universes?active_only=true")
        slugs = [u["slug"] for u in response.json()["universes"]]
        assert "test_delete" not in slugs

    def test_delete_locked_fails(self, client):
        """DELETE /universes/{slug} should fail if is_locked."""
        payload = {
            "slug": "test_locked",
            "name": "Test",
            "market": "vn_stock",
        }
        client.post("/api/v1/universes", json=payload)

        # Lock it
        client.put("/api/v1/universes/test_locked", json={"is_locked": True})

        response = client.delete("/api/v1/universes/test_locked")
        assert response.status_code == 400

    def test_clone_universe(self, client):
        """POST /universes/{slug}/clone should clone universe."""
        payload = {
            "slug": "test_clone_source",
            "name": "Source",
            "market": "vn_stock",
            "description": "Original",
            "symbols": [{"symbol": "ACB"}, {"symbol": "BID"}],
        }
        client.post("/api/v1/universes", json=payload)

        clone_payload = {"new_slug": "test_clone_copy", "new_name": "Cloned"}
        response = client.post("/api/v1/universes/test_clone_source/clone", json=clone_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["slug"] == "test_clone_copy"

        # Verify symbols copied
        response = client.get("/api/v1/universes/test_clone_copy")
        assert len(response.json()["symbols"]) == 2

    def test_list_markets_endpoint(self, client):
        """GET /markets should list available markets."""
        response = client.get("/api/v1/markets")
        assert response.status_code == 200
        data = response.json()
        assert "markets" in data
        assert isinstance(data["markets"], list)
        assert len(data["markets"]) > 0

    def test_list_market_symbols_endpoint(self, client):
        """GET /markets/{market}/symbols should list symbols."""
        response = client.get("/api/v1/markets/vn_stock/symbols")
        assert response.status_code == 200
        data = response.json()
        assert data["market"] == "vn_stock"
        assert "symbols" in data
        assert "symbol_count" in data

    def test_list_market_symbols_invalid_market(self, client):
        """GET /markets/{market}/symbols should 404 for invalid market."""
        response = client.get("/api/v1/markets/invalid_market/symbols")
        assert response.status_code == 404

    def test_filter_by_market(self, client):
        """GET /universes?market=... should filter by market."""
        payload1 = {
            "slug": "vn_stock_set",
            "name": "VN Set",
            "market": "vn_stock",
        }
        payload2 = {
            "slug": "crypto_set",
            "name": "Crypto Set",
            "market": "crypto_spot",
        }
        client.post("/api/v1/universes", json=payload1)
        client.post("/api/v1/universes", json=payload2)

        response = client.get("/api/v1/universes?market=vn_stock")
        universes = response.json()["universes"]
        assert all(u["market"] == "vn_stock" for u in universes)
        assert len(universes) >= 1

    def test_version_increments_on_symbol_edit(self, client):
        """Universe version should increment when symbols change."""
        payload = {
            "slug": "test_version",
            "name": "Test",
            "market": "vn_stock",
        }
        response = client.post("/api/v1/universes", json=payload)
        initial_version = response.json()["version"]
        assert initial_version == 1

        add_payload = {"symbols": [{"symbol": "ACB"}]}
        client.post("/api/v1/universes/test_version/symbols", json=add_payload)

        response = client.get("/api/v1/universes/test_version")
        assert response.json()["version"] == 2
