"""Tests for the Model Lifecycle API server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def results_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    results = tmp_path / "results"
    (results / "leaderboard").mkdir(parents=True)
    (results / "experiments").mkdir(parents=True)
    (results / "cache" / "features" / "leading").mkdir(parents=True)
    (results / "cache" / "predictions").mkdir(parents=True)
    (tmp_path / "viz").mkdir()
    monkeypatch.setenv("STOCK_RESULTS_DIR", str(results))
    monkeypatch.setenv("STOCK_VIZ_DIR", str(tmp_path / "viz"))
    return results


def _write_run(results: Path, bundle: str, name: str, feat_key: str, pred_key: str) -> Path:
    run_dir = results / "experiments" / bundle / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions_meta.json").write_text(
        json.dumps(
            {
                "config_hash": "abc12345",
                "market": "vn_stock",
                "feature_set": "leading",
                "entry_model": "lightgbm",
                "exit_model_type": "none",
                "exit_model_enabled": False,
                "cache_keys": {"features": feat_key, "predictions": pred_key},
                "split": {"first_test_year": 2020, "last_test_year": 2024},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "ranking_row.json").write_text(
        json.dumps({"name": name, "composite_score": 1.0, "trade_count": 2}), encoding="utf-8"
    )
    (run_dir / "config.resolved.yaml").write_text(
        "market: vn_stock\nname: " + name + "\nstrategy: v22\n", encoding="utf-8"
    )
    (run_dir / "trades.csv").write_text(
        "symbol,entry_date,exit_date,pnl_pct,pnl,holding_days\n"
        "ACB,2020-01-02,2020-01-10,1.5,1.5,8\n"
        "FPT,2020-02-02,2020-02-10,-0.5,-0.5,8\n",
        encoding="utf-8",
    )
    return run_dir


@pytest.fixture
def client(results_env: Path):
    from fastapi.testclient import TestClient
    from scripts.api_server import _rebuild_leaderboard, create_app

    _write_run(results_env, "bundleA", "runX", "featX", "predX")
    _rebuild_leaderboard()
    return TestClient(create_app()), results_env


def test_list_runs(client) -> None:
    c, _ = client
    resp = c.get("/api/runs")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["state"] == "trained"
    assert rows[0]["cache_keys"]["features"] == "featX"


def test_patch_state_to_pinned(client) -> None:
    c, results = client
    run_id = c.get("/api/runs").json()[0]["run_id"]
    resp = c.patch("/api/runs", params={"run_id": run_id}, json={"state": "pinned"})
    assert resp.status_code == 200
    assert resp.json()["state"] == "pinned"
    # lifecycle.json written
    lc = results / "experiments" / "bundleA" / "runX" / "lifecycle.json"
    assert lc.exists()
    assert json.loads(lc.read_text())["state"] == "pinned"
    # reflected in list
    assert c.get("/api/runs").json()[0]["state"] == "pinned"


def test_pin_exports_to_dashboard_manifest(client, tmp_path: Path) -> None:
    c, _results = client
    viz = tmp_path / "viz"
    run_id = c.get("/api/runs").json()[0]["run_id"]
    c.patch("/api/runs", params={"run_id": run_id}, json={"state": "pinned"})
    # manifest contains exactly the pinned model
    manifest = json.loads((viz / "manifest.json").read_text())
    assert len(manifest["models"]) == 1
    vk = manifest["models"][0]["version_key"]
    assert vk.startswith("pin_")
    # per-symbol export dir created
    assert (viz / f"data_{vk}").is_dir()
    # unpin removes it from manifest and drops the data dir
    c.patch("/api/runs", params={"run_id": run_id}, json={"state": "trained"})
    manifest2 = json.loads((viz / "manifest.json").read_text())
    assert manifest2["models"] == []
    assert not (viz / f"data_{vk}").exists()


def test_patch_invalid_state_rejected(client) -> None:
    c, _ = client
    run_id = c.get("/api/runs").json()[0]["run_id"]
    resp = c.patch("/api/runs", params={"run_id": run_id}, json={"state": "bogus"})
    assert resp.status_code == 400


def test_delete_cache_quarantines(client) -> None:
    c, results = client
    # create cache files matching the run's keys
    feat = results / "cache" / "features" / "leading" / "featX.parquet"
    feat.write_text("x" * 100, encoding="utf-8")
    pred = results / "cache" / "predictions" / "predX.pkl"
    pred.write_text("x" * 50, encoding="utf-8")

    run_id = c.get("/api/runs").json()[0]["run_id"]
    resp = c.delete("/api/runs/cache", params={"run_id": run_id})
    assert resp.status_code == 200
    assert len(resp.json()["quarantined_cache"]) == 2
    assert not feat.exists()
    assert not pred.exists()
    # metrics row still present
    assert len(c.get("/api/runs").json()) == 1


def test_delete_run_removes_dir_and_row(client) -> None:
    c, results = client
    run_id = c.get("/api/runs").json()[0]["run_id"]
    resp = c.delete("/api/runs", params={"run_id": run_id})
    assert resp.status_code == 200
    assert not (results / "experiments" / "bundleA" / "runX").exists()
    assert c.get("/api/runs").json() == []


def test_gc_sweep_dry_run(client) -> None:
    c, results = client
    # orphan cache (key not referenced)
    orphan = results / "cache" / "features" / "leading" / "orphan_key.parquet"
    orphan.write_text("x" * 100, encoding="utf-8")
    resp = c.post("/api/gc/sweep", json={"apply": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["orphan_count"] >= 1
    assert orphan.exists()  # dry-run keeps file


def test_unknown_run_404(client) -> None:
    c, _ = client
    assert (
        c.patch("/api/runs", params={"run_id": "nope#123"}, json={"state": "pinned"}).status_code
        == 404
    )
    assert c.delete("/api/runs", params={"run_id": "nope#123"}).status_code == 404


def test_job_not_found(client) -> None:
    c, _ = client
    assert c.get("/api/jobs/deadbeef").status_code == 404
