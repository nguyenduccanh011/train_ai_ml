"""Unit tests for LegacyVersionAdapter and migrate_legacy (no I/O, no ML training)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from src.pipeline.legacy_adapter import (
    CHAMPION_VERSIONS,
    LEGACY_STRATEGY_MAP,
    LegacyRunResult,
    LegacyVersionAdapter,
    _trades_to_dataframe,
    list_all_legacy_versions,
    list_legacy_versions,
)

# ── LEGACY_STRATEGY_MAP coverage ────────────────────────────────────────────


def test_strategy_map_has_expected_keys():
    expected = {"v22", "v34", "v37a", "v37d", "v25", "v19_3", "v11", "v32"}
    assert expected <= set(LEGACY_STRATEGY_MAP)


def test_strategy_map_tuples_are_two_element():
    for key, val in LEGACY_STRATEGY_MAP.items():
        assert len(val) == 2, f"{key}: expected (module, func), got {val}"


def test_champion_versions_mostly_in_strategy_map():
    # 'rule' has a separate entry point (compare_rule_vs_model), not in LEGACY_STRATEGY_MAP
    missing_allowed = {"rule"}
    unexpected_missing = CHAMPION_VERSIONS - set(LEGACY_STRATEGY_MAP) - missing_allowed
    assert not unexpected_missing, (
        f"Champions missing from LEGACY_STRATEGY_MAP: {unexpected_missing}"
    )


# ── list helpers ────────────────────────────────────────────────────────────


def test_list_legacy_versions_excludes_champions():
    legacy_only = list_legacy_versions()
    for k in legacy_only:
        assert k not in CHAMPION_VERSIONS


def test_list_all_legacy_versions_includes_most_champions():
    # 'rule' is a champion but has its own entry point, not in LEGACY_STRATEGY_MAP
    all_keys = set(list_all_legacy_versions())
    expected_in_map = CHAMPION_VERSIONS - {"rule"}
    assert expected_in_map <= all_keys


def test_list_legacy_versions_sorted():
    result = list_legacy_versions()
    assert result == sorted(result)


# ── constructor validation ───────────────────────────────────────────────────


def test_unknown_version_raises():
    with pytest.raises(ValueError, match="Unknown legacy version"):
        LegacyVersionAdapter("v999_fake")


def test_champion_version_warns(capsys):
    """Champion versions should warn but still construct."""
    mock_cfg = {
        "feature_set": "leading_v2",
        "mods": {},
        "params": {},
        "exit_model": {"enabled": False},
    }
    with patch(
        "src.pipeline.legacy_adapter.LegacyVersionAdapter._load_model_cfg", return_value=mock_cfg
    ):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter = LegacyVersionAdapter("v22")
            assert len(w) == 1
            assert "champion" in str(w[0].message).lower()
        assert adapter.version_key == "v22"


# ── build_experiment_config ──────────────────────────────────────────────────


@pytest.fixture
def adapter_v25():
    mock_cfg = {
        "feature_set": "leading_v2",
        "model_type": "lightgbm",
        "mods": {"a": True, "b": True, "c": False},
        "params": {"peak_protect_strong_threshold": 0.12},
        "exit_model": {"enabled": True, "forward_window": 15, "loss_threshold": 0.05},
    }
    with patch(
        "src.pipeline.legacy_adapter.LegacyVersionAdapter._load_model_cfg", return_value=mock_cfg
    ):
        return LegacyVersionAdapter("v25")


def test_build_experiment_config_schema(adapter_v25):
    doc = adapter_v25.build_experiment_config()
    assert doc["name"] == "v25"
    assert doc["strategy"] == "v25"
    assert doc["components"]["features"] == "leading_v2"
    assert doc["components"]["entry_model"]["type"] == "lightgbm"
    assert doc["components"]["exit_model"]["enabled"] is True
    assert doc["mods"]["a"] is True
    assert doc["params"]["peak_protect_strong_threshold"] == 0.12


def test_build_experiment_config_target_defaults(adapter_v25):
    doc = adapter_v25.build_experiment_config()
    target = doc["components"]["target"]
    assert target["type"] == "trend_regime"
    assert "forward_window" in target
    assert "gain_threshold" in target


def test_build_experiment_config_target_override():
    mock_cfg = {
        "feature_set": "leading_v4",
        "target": {
            "type": "early_wave",
            "forward_window": 8,
            "gain_threshold": 0.06,
            "loss_threshold": 0.03,
            "classes": 3,
        },
        "mods": {},
        "params": {},
        "exit_model": {},
    }
    with patch(
        "src.pipeline.legacy_adapter.LegacyVersionAdapter._load_model_cfg", return_value=mock_cfg
    ):
        adapter = LegacyVersionAdapter("v34")
    doc = adapter.build_experiment_config()
    assert doc["components"]["target"]["type"] == "early_wave"
    assert doc["components"]["target"]["forward_window"] == 8


# ── _trades_to_dataframe ─────────────────────────────────────────────────────


def test_trades_to_dataframe_empty():
    df = _trades_to_dataframe([])
    assert df.empty


def test_trades_to_dataframe_column_order():
    trades = [
        {
            "symbol": "AAA",
            "entry_date": "2021-01-01",
            "exit_date": "2021-01-10",
            "entry_price": 10.0,
            "exit_price": 11.0,
            "pnl": 10.0,
            "exit_reason": "signal",
            "extra_col": 99,
        }
    ]
    df = _trades_to_dataframe(trades)
    assert list(df.columns[:7]) == [
        "symbol",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "pnl",
        "exit_reason",
    ]
    assert "extra_col" in df.columns


def test_trades_to_dataframe_partial_columns():
    trades = [{"symbol": "BBB", "pnl": 5.0}]
    df = _trades_to_dataframe(trades)
    assert "symbol" in df.columns
    assert "pnl" in df.columns


# ── LegacyRunResult ──────────────────────────────────────────────────────────


def test_legacy_run_result_n_trades():
    result = LegacyRunResult(name="v25", trades=[{"a": 1}, {"b": 2}])
    assert result.n_trades == 2


def test_legacy_run_result_empty():
    result = LegacyRunResult(name="v25")
    assert result.n_trades == 0
    assert result.trades_df.empty


# ── migrate_legacy helpers ───────────────────────────────────────────────────


def test_migrate_dry_run_prints(capsys, tmp_path):
    from scripts.migrate_legacy import migrate_version

    mock_cfg = {
        "feature_set": "leading_v2",
        "mods": {"a": True},
        "params": {},
        "exit_model": {"enabled": False},
        "name": "V25",
        "description": "test",
        "active": False,
    }
    with patch("scripts.migrate_legacy._load_model_cfg", return_value=mock_cfg):
        path = migrate_version("v25", dry_run=True)

    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    assert "v25" in captured.out
    assert not path.exists()


def test_migrate_writes_yaml(tmp_path):
    import yaml
    from scripts.migrate_legacy import migrate_version

    mock_cfg = {
        "feature_set": "leading_v2",
        "mods": {"a": True},
        "params": {"peak_protect_strong_threshold": 0.12},
        "exit_model": {"enabled": False},
        "name": "V25",
        "description": "test version",
        "active": False,
    }
    out_file = tmp_path / "v25.yaml"
    with patch("scripts.migrate_legacy._load_model_cfg", return_value=mock_cfg):
        returned_path = migrate_version("v25", output_path=out_file)

    assert returned_path == out_file
    assert out_file.exists()
    doc = yaml.safe_load(out_file.read_text(encoding="utf-8"))
    assert doc["name"] == "v25"
    assert doc["strategy"] == "v25"
    assert doc["components"]["features"] == "leading_v2"
    assert doc["params"]["peak_protect_strong_threshold"] == 0.12
