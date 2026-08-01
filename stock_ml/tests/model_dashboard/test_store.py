import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stock_ml.src.leaderboard.loader import run_dir_to_row
from stock_ml.src.model_dashboard import DashboardStore

FIXTURES = Path(__file__).resolve().parents[1] / "leaderboard" / "fixtures"


def test_store_ingest_and_read_active_bundle(tmp_path: Path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    row = run_dir_to_row(FIXTURES / "v22_run")

    bundle = store.upsert_row(row)

    assert bundle.model.visible_in_dashboard is True
    assert store.list_models(status="active", visible_only=True)[0].id == bundle.model.id

    fetched = store.get_bundle(bundle.model.id)
    assert fetched is not None
    assert fetched.run.id == row.run_id
    assert len(fetched.artifacts) == len(bundle.artifacts)
    assert any(record.action == "ingest" for record in fetched.audit_log)


def test_store_transitions_toggle_visibility_and_restore(tmp_path: Path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    row = run_dir_to_row(FIXTURES / "rule_run", bundle="rule")
    bundle = store.upsert_row(row)

    archived = store.archive_model(bundle.model.id, actor="tester", reason="cleanup")
    assert archived.status == "archived"
    assert archived.visible_in_dashboard is False

    quarantined = store.quarantine_model(bundle.model.id, actor="tester", reason="leakage")
    assert quarantined.status == "quarantined"
    assert quarantined.visible_in_dashboard is False

    restored = store.restore_model(bundle.model.id, actor="tester")
    assert restored.status == "active"
    assert restored.visible_in_dashboard is True
    assert restored.retired_at is None
    assert restored.retired_reason is None


def test_store_purge_marks_artifacts_deleted(tmp_path: Path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    row = run_dir_to_row(FIXTURES / "rule_run", bundle="rule")
    bundle = store.upsert_row(row)
    store.archive_model(bundle.model.id, reason="retention")

    plan = store.purge_model(bundle.model.id, reason="retention")
    assert plan.model_id == bundle.model.id
    assert plan.run_id == bundle.run.id
    assert plan.artifact_paths

    purged = store.get_bundle(bundle.model.id)
    assert purged is not None
    assert purged.model.status == "purged"
    assert all(artifact.deleted_at is not None for artifact in purged.artifacts)
    assert any(record.action == "purge" for record in purged.audit_log)


def test_store_rejects_purge_without_archive(tmp_path: Path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    row = run_dir_to_row(FIXTURES / "rule_run", bundle="rule")
    bundle = store.upsert_row(row)

    with pytest.raises(ValueError, match="purge requires archived or quarantined model"):
        store.purge_model(bundle.model.id)


def test_store_exposes_leaderboard_rows_from_db(tmp_path: Path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    row = run_dir_to_row(FIXTURES / "v22_run")

    store.upsert_row(row)

    rows = store.list_leaderboard_rows(market=row.market, include_superseded=False)

    assert len(rows) == 1
    assert rows[0].run_id == row.run_id
    assert rows[0].composite_score == row.composite_score


def test_store_search_sort_and_audit_log(tmp_path: Path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    row_a = run_dir_to_row(FIXTURES / "v22_run")
    row_b = run_dir_to_row(FIXTURES / "rule_run", bundle="rule")

    bundle_a = store.upsert_row(row_a)
    store.upsert_row(row_b)

    models = store.list_models(sort_by="name", sort_dir="asc")
    assert [model.name for model in models] == sorted([row_a.run_name, row_b.run_name])

    audit_rows = store.list_audit_log(entity_type="model", entity_id=bundle_a.model.id, limit=5)
    assert audit_rows
    assert audit_rows[0].action == "ingest"
