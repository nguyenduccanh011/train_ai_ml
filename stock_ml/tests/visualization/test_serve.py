from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.leaderboard.loader import run_dir_to_row
from visualization.serve import create_app
from src.model_dashboard import DashboardStore

FIXTURES = Path(__file__).resolve().parents[1] / "leaderboard" / "fixtures"


def test_api_leaderboard_reads_from_db(tmp_path: Path):
    db_path = tmp_path / "dashboard.sqlite"
    store = DashboardStore(db_path)
    row = run_dir_to_row(FIXTURES / "rule_run", bundle="rule")
    store.upsert_row(row)

    app = create_app(db_path, bootstrap_from_disk=False)
    client = app.test_client()

    response = client.get("/api/leaderboard")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["source"] == "db"
    assert payload["rows"][0]["run_id"] == row.run_id


def test_api_models_bulk_action_and_audit(tmp_path: Path):
    db_path = tmp_path / "dashboard.sqlite"
    store = DashboardStore(db_path)
    row = run_dir_to_row(FIXTURES / "rule_run", bundle="rule")
    bundle = store.upsert_row(row)

    app = create_app(db_path, bootstrap_from_disk=False)
    client = app.test_client()

    models = client.get("/api/models?search=rule&sort_by=name&sort_dir=asc")
    assert models.status_code == 200
    models_payload = models.get_json()
    assert models_payload["source"] == "db"
    assert models_payload["rows"][0]["id"] == bundle.model.id

    bulk = client.post(
        "/api/models/bulk",
        json={
            "action": "archive",
            "model_ids": [bundle.model.id],
            "actor": "tester",
            "reason": "retention",
        },
    )
    assert bulk.status_code == 200
    bulk_payload = bulk.get_json()
    assert bulk_payload["results"][0]["action"] == "archive"

    detail = client.get(f"/api/models/{bundle.model.id}")
    assert detail.status_code == 200
    detail_payload = detail.get_json()
    assert detail_payload["model"]["status"] == "archived"

    audit = client.get(f"/api/models/{bundle.model.id}/audit?limit=20")
    assert audit.status_code == 200
    audit_payload = audit.get_json()
    actions = [row["action"] for row in audit_payload["rows"]]
    assert "archived" in actions
    assert "ingest" in actions
