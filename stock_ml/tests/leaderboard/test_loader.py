from pathlib import Path

from src.leaderboard.loader import (
    COST_PROFILE_UNKNOWN_WARNING,
    MISSING_TRADES_WARNING,
    run_dir_to_row,
)
from src.leaderboard.schema import LeaderboardRow

FIXTURES = Path(__file__).parent / "fixtures"


def test_loader_rule_run():
    row = run_dir_to_row(FIXTURES / "rule_run", bundle="rule")

    assert isinstance(row, LeaderboardRow)
    assert row.run_id == "rule/rule#35a72911"
    assert row.bundle == "rule"
    assert row.run_name == "rule"
    assert row.strategy == "rule"
    assert row.feature_set == "leading_v2"
    assert row.entry_model == "rule"
    assert row.exit_model_type == "null"
    assert row.exit_model_enabled is False
    assert row.target.type == "early_wave"
    assert row.first_test_year == 2020
    assert row.last_test_year == 2025
    assert row.cost_profile.commission == 0.0015
    assert row.cost_profile.tax == 0.001
    assert row.cost_profile.slippage == 0.0
    assert COST_PROFILE_UNKNOWN_WARNING not in row.warnings


def test_loader_v22_run_uses_matrix_name_bundle():
    row = run_dir_to_row(FIXTURES / "v22_run")

    assert row.run_id == "champions_2020_2025_fair/v22#c43b4390"
    assert row.bundle == "champions_2020_2025_fair"
    assert row.run_name == "v22"
    assert row.entry_model == "lightgbm"
    assert row.exit_model_type == "lightgbm"
    assert row.exit_model_enabled is True
    assert row.target.type == "trend_regime"
    assert row.trades == 11
    assert row.n_symbols == 4


def test_loader_no_trades_fallback():
    row = run_dir_to_row(FIXTURES / "no_trades_run", bundle="rule")

    assert row.run_id == "rule/rule#35a72911"
    assert row.trades == 2585
    assert row.wr == 41.66
    assert row.avg_pnl == 2.948
    assert row.total_pnl == 7621.66
    assert row.pf == 2.051
    assert row.max_drawdown == 612.42
    assert row.mdd_per_symbol == 39.13
    assert row.yearly_consistency == 1.1748
    assert row.composite_score == 316.4
    assert row.n_symbols == 61
    assert MISSING_TRADES_WARNING in row.warnings


def test_loader_fairness_group_key_stable():
    row_a = run_dir_to_row(FIXTURES / "v22_run")
    row_b = run_dir_to_row(FIXTURES / "v22_run")

    assert row_a.fairness_group_key == row_b.fairness_group_key
    assert len(row_a.fairness_group_key) == 40
