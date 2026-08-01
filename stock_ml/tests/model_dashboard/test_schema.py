import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stock_ml.src.leaderboard.loader import run_dir_to_row
from stock_ml.src.model_dashboard.schema import (
    ARTIFACT_KIND_FILENAMES,
    DashboardBundle,
    artifact_root,
    canonical_artifact_path,
    canonical_run_dir,
    leaderboard_row_to_dashboard_bundle,
    model_id_for_row,
)

FIXTURES = Path(__file__).resolve().parents[1] / "leaderboard" / "fixtures"


def test_phase0_bundle_maps_leaderboard_row():
    row = run_dir_to_row(FIXTURES / "v22_run")
    bundle = leaderboard_row_to_dashboard_bundle(row, root=Path("/tmp/stock_ml"))

    assert isinstance(bundle, DashboardBundle)
    assert bundle.model.id == model_id_for_row(row)
    assert bundle.run.id == row.run_id
    assert (
        bundle.run.config_path
        == "/tmp/stock_ml/results/experiments/champions_2020_2025_fair/v22/config.resolved.yaml"
    )
    assert bundle.metrics_snapshot.run_id == row.run_id
    assert bundle.model.visible_in_dashboard is True
    assert [artifact.kind for artifact in bundle.artifacts] == list(ARTIFACT_KIND_FILENAMES)


def test_phase0_paths_follow_canonical_layout():
    row = run_dir_to_row(FIXTURES / "rule_run", bundle="rule")
    root = Path("/tmp/stock_ml")

    assert canonical_run_dir(row, root=root) == root / "results" / "experiments" / "rule"
    assert (
        canonical_artifact_path(row, "trades", root=root)
        == root / "results" / "experiments" / "rule" / "trades.csv"
    )
    assert (
        canonical_artifact_path(row, "unknown_kind", root=root)
        == root / "results" / "experiments" / "rule" / "unknown_kind.json"
    )
    assert artifact_root("models", root=root) == root / "models"


def test_phase0_schema_forbids_extra_fields():
    row = run_dir_to_row(FIXTURES / "rule_run", bundle="rule")
    bundle = leaderboard_row_to_dashboard_bundle(row)

    with pytest.raises(ValidationError):
        bundle.model.__class__(**{**bundle.model.model_dump(), "extra": 1})
