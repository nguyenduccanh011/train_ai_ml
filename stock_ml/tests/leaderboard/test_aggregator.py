import json
import shutil
from pathlib import Path

import pytest
from src.leaderboard.aggregator import (
    INDEX_JSON,
    LEADERBOARD_CSV,
    LEADERBOARD_JSON,
    SCHEMA_JSON,
    SUMMARY_JSON,
    _prepare_rows,
    append_or_update,
    rebuild_leaderboard,
    validate_leaderboard,
)
from src.leaderboard.hooks import append_or_update as hook_append_or_update
from src.leaderboard.schema import LeaderboardRow

FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixture(src_name: str, dest: Path) -> None:
    shutil.copytree(FIXTURES / src_name, dest)


def test_rebuild_writes_all_output_files(tmp_path):
    experiments = tmp_path / "experiments"
    _copy_fixture("rule_run", experiments / "bundle_a" / "rule_run")
    _copy_fixture("v22_run", experiments / "bundle_b" / "v22_run")

    rows = rebuild_leaderboard(experiments, tmp_path / "leaderboard")

    assert len(rows) == 2
    assert [row.composite_score for row in rows] == sorted(
        [row.composite_score for row in rows], reverse=True
    )
    output_dir = tmp_path / "leaderboard"
    for filename in [LEADERBOARD_JSON, LEADERBOARD_CSV, SUMMARY_JSON, INDEX_JSON, SCHEMA_JSON]:
        assert (output_dir / filename).exists()

    data = json.loads((output_dir / LEADERBOARD_JSON).read_text(encoding="utf-8"))
    summary = json.loads((output_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
    index = json.loads((output_dir / INDEX_JSON).read_text(encoding="utf-8"))

    assert len(data) == 2
    assert summary["row_count"] == 2
    assert set(index) == {row["run_id"] for row in data}


def test_append_or_update_replaces_same_run_id(tmp_path):
    append_or_update(FIXTURES / "rule_run", tmp_path, bundle="rule")
    append_or_update(FIXTURES / "rule_run", tmp_path, bundle="rule")

    rows = json.loads((tmp_path / LEADERBOARD_JSON).read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert rows[0]["run_id"] == "rule/rule#35a72911"


def test_validate_rejects_duplicate_run_ids(tmp_path):
    append_or_update(FIXTURES / "rule_run", tmp_path, bundle="rule")
    rows = validate_leaderboard(tmp_path / LEADERBOARD_JSON)
    duplicate = [row.model_dump(mode="json") for row in rows[:1]] * 2
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate run_id"):
        validate_leaderboard(duplicate_path)


def test_hook_append_or_update_writes_leaderboard_outputs(tmp_path):
    row = hook_append_or_update(FIXTURES / "rule_run", tmp_path, bundle="rule")

    rows = json.loads((tmp_path / LEADERBOARD_JSON).read_text(encoding="utf-8"))
    assert row.run_id == "rule/rule#35a72911"
    assert rows[0]["run_id"] == row.run_id


def _mock_row(*, run_id: str, schema: str, timeframe: str) -> LeaderboardRow:
    return LeaderboardRow(
        run_id=run_id,
        bundle="bundle_x",
        run_name=run_id.split("#")[0],
        config_hash=run_id.split("#")[1],
        generated_at="2026-05-06T00:00:00",
        superseded=False,
        market="vn_stock",
        market_family="vn_stock",
        currency="VND",
        pnl_mode="spot",
        schema=schema,
        timeframe=timeframe,
        strategy="v22",
        feature_set="leading",
        entry_model="random_forest",
        exit_model_type="lightgbm",
        exit_model_enabled=True,
        target={
            "type": "early_wave_v2",
            "forward_window": 21,
            "gain_threshold": 0.03,
            "loss_threshold": 0.015,
        },
        trades=1000,
        wr=70.0,
        avg_pnl=10.0,
        total_pnl=10000.0,
        pf=2.0,
        avg_hold=30.0,
        sharpe=1.2,
        max_drawdown=400.0,
        mdd_per_symbol=20.0,
        yearly_consistency=0.6,
        composite_score=600.0,
        score_mode="live",
        n_symbols=50,
        first_test_year=2020,
        last_test_year=2025,
        backtest_window_key="2020-2025",
        cost_profile={"commission": 0.001, "tax": 0.001, "slippage": 0.001},
        fairness_group_key="fair-key",
        warnings=[],
    )


def test_prepare_rows_prefers_known_schema_on_tie():
    unknown_row = _mock_row(
        run_id="bundle_x/run_unknown#11111111",
        schema="unknown",
        timeframe="unknown",
    )
    known_row = _mock_row(
        run_id="bundle_x/run_known#22222222",
        schema="ohlcv_daily",
        timeframe="1D",
    ).model_copy(update={"total_pnl": 10001.0})

    ranked = _prepare_rows([unknown_row, known_row])

    assert ranked[0].run_id == known_row.run_id


def test_rebuild_writes_market_family_split(tmp_path):
    experiments = tmp_path / "experiments"
    _copy_fixture("rule_run", experiments / "bundle_a" / "rule_run")

    rows = rebuild_leaderboard(experiments, tmp_path / "leaderboard")

    assert rows[0].market_family == "vn_stock"
    assert (tmp_path / "leaderboard" / "by_market_family" / "vn_stock" / LEADERBOARD_JSON).exists()


def test_csv_includes_market_family_columns(tmp_path):
    append_or_update(FIXTURES / "rule_run", tmp_path, bundle="rule")

    csv_text = (tmp_path / LEADERBOARD_CSV).read_text(encoding="utf-8")

    assert "market_family" in csv_text.splitlines()[0]
    assert "backtest_window_key" in csv_text.splitlines()[0]
    assert "same_timeframe_as_baseline" in csv_text.splitlines()[0]
