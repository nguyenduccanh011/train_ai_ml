"""Phase 0: Schema contract tests for LeaderboardRow.

Verifies:
- Valid rows parse correctly from fixture data
- Required fields are enforced
- CostProfile accepts both float and "unknown"
- TargetConfig handles null thresholds
- Golden snapshot rows all validate against schema
- Extra fields are rejected (extra="forbid")
- JSON Schema export works
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from src.leaderboard.schema import CostProfile, LeaderboardRow, TargetConfig, export_json_schema

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DIR = Path(__file__).parent / "golden"


def _make_valid_row(**overrides) -> dict:
    base = {
        "run_id": "bundle/run#abc12345",
        "bundle": "bundle",
        "run_name": "run",
        "config_hash": "abc12345def6",
        "generated_at": "2026-05-01T10:00:00",
        "superseded": False,
        "strategy": "v22",
        "feature_set": "leading_v2",
        "entry_model": "lightgbm",
        "exit_model_type": "lightgbm",
        "exit_model_enabled": True,
        "target": {
            "type": "trend_regime",
            "forward_window": 8,
            "gain_threshold": 0.06,
            "loss_threshold": 0.03,
        },
        "trades": 1000,
        "wr": 55.0,
        "avg_pnl": 3.5,
        "total_pnl": 3500.0,
        "pf": 2.5,
        "avg_hold": 10.0,
        "sharpe": 1.2,
        "max_drawdown": 200.0,
        "mdd_per_symbol": 20.0,
        "yearly_consistency": 0.6,
        "composite_score": 400.0,
        "score_mode": "live",
        "n_symbols": 61,
        "first_test_year": 2020,
        "last_test_year": 2025,
        "cost_profile": {
            "commission": "unknown",
            "tax": "unknown",
            "slippage": "unknown",
        },
        "fairness_group_key": "abc123sha1",
        "warnings": [],
    }
    base.update(overrides)
    return base


class TestLeaderboardRowBasic:
    def test_valid_row_parses(self):
        row = LeaderboardRow(**_make_valid_row())
        assert row.run_id == "bundle/run#abc12345"
        assert row.score_mode == "live"
        assert row.superseded is False

    def test_missing_required_field_fails(self):
        data = _make_valid_row()
        del data["composite_score"]
        with pytest.raises(ValidationError):
            LeaderboardRow(**data)

    def test_extra_field_rejected(self):
        data = _make_valid_row()
        data["unexpected_field"] = "value"
        with pytest.raises(ValidationError):
            LeaderboardRow(**data)

    def test_defaults_applied(self):
        row = LeaderboardRow(**_make_valid_row())
        assert row.superseded is False
        assert row.score_mode == "live"
        assert row.warnings == []

    def test_warnings_list(self):
        data = _make_valid_row(warnings=["cost_profile=unknown", "trades.csv missing"])
        row = LeaderboardRow(**data)
        assert len(row.warnings) == 2


class TestTargetConfig:
    def test_null_thresholds_allowed(self):
        target = TargetConfig(type="custom", forward_window=5)
        assert target.gain_threshold is None
        assert target.loss_threshold is None

    def test_with_thresholds(self):
        target = TargetConfig(
            type="trend_regime", forward_window=8, gain_threshold=0.06, loss_threshold=0.03
        )
        assert target.forward_window == 8


class TestCostProfile:
    def test_all_unknown(self):
        cp = CostProfile()
        assert cp.commission == "unknown"
        assert cp.tax == "unknown"
        assert cp.slippage == "unknown"

    def test_float_values(self):
        cp = CostProfile(commission=0.001, tax=0.002, slippage=0.0005)
        assert cp.commission == 0.001

    def test_mixed(self):
        cp = CostProfile(commission=0.001, tax="unknown", slippage="unknown")
        assert cp.commission == 0.001
        assert cp.tax == "unknown"

    def test_invalid_value_rejected(self):
        with pytest.raises(ValidationError):
            CostProfile(commission="invalid_string")


class TestGoldenSnapshot:
    def test_golden_rows_validate(self):
        golden_path = GOLDEN_DIR / "leaderboard.json"
        assert golden_path.exists(), f"Golden file missing: {golden_path}"
        rows_data = json.loads(golden_path.read_text())
        assert len(rows_data) >= 2
        for data in rows_data:
            row = LeaderboardRow(**data)
            assert row.composite_score > 0
            assert row.bundle != ""

    def test_golden_sorted_desc(self):
        golden_path = GOLDEN_DIR / "leaderboard.json"
        rows_data = json.loads(golden_path.read_text())
        scores = [r["composite_score"] for r in rows_data]
        assert scores == sorted(scores, reverse=True), "Golden rows must be sorted desc by score"


class TestJsonSchemaExport:
    def test_schema_export_returns_dict(self):
        schema = export_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "run_id" in schema["properties"]

    def test_schema_has_required_fields(self):
        schema = export_json_schema()
        required = schema.get("required", [])
        for field in ["run_id", "bundle", "composite_score", "trades", "target"]:
            assert field in required, f"Field '{field}' should be required in schema"

    def test_schema_write_to_file(self, tmp_path):
        out = tmp_path / "schema.json"
        export_json_schema(str(out))
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert "properties" in loaded
