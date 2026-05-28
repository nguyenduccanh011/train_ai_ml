from pathlib import Path

import pandas as pd
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
    assert row.backtest_window_key == "2020-2025"
    assert row.cost_profile.commission == 0.0015
    assert row.cost_profile.tax == 0.001
    assert row.cost_profile.slippage == 0.001
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


def test_loader_resolves_derivatives_30m_market_family(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "symbol": "VN30F1M",
                "entry_date": "2020-01-01 09:00:00",
                "exit_date": "2020-01-01 10:00:00",
                "pnl_pct": 1.0,
                "pnl": 100.0,
                "holding_days": 0.04,
            }
        ]
    ).to_csv(run_dir / "trades.csv", index=False)
    (run_dir / "predictions_meta.json").write_text(
        """
{
  "config_hash": "abc12345def0",
  "market": "vn_derivatives_30m",
  "currency": "VND",
  "pnl_mode": "futures_contract",
  "schema": "ohlcv_futures_1h",
  "timeframe": "30m",
  "entry_model": "random_forest",
  "exit_model_type": "lightgbm",
  "exit_model_enabled": true,
  "feature_set": "leading",
  "split": {"first_test_year": 2020, "last_test_year": 2025},
  "created_at": "2026-05-06T10:00:00"
}
""".strip(),
        encoding="utf-8",
    )
    (run_dir / "ranking_row.json").write_text(
        """
{
  "name": "sample_30m",
  "feature_set": "leading",
  "entry_model": "random_forest",
  "exit_model_type": "lightgbm",
  "exit_model_enabled": true
}
""".strip(),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.resolved.yaml").write_text(
        """
name: sample_30m
strategy: v22
market: vn_derivatives_30m
signals:
  features: leading
  target:
    type: early_wave_v2
    forward_window: 10
    gain_threshold: 0.008
    loss_threshold: 0.003
execution:
  currency: VND
  pnl_mode: futures_contract
split:
  first_test_year: 2020
  last_test_year: 2025
""".strip(),
        encoding="utf-8",
    )

    row = run_dir_to_row(run_dir, bundle="tmp_bundle")

    assert row.market == "vn_derivatives_30m"
    assert row.market_family == "vn_derivatives"
    assert row.timeframe == "30m"
    assert row.backtest_window_key == "2020-2025"


def test_loader_prefers_holding_days_column_over_date_delta(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    trades = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "entry_date": "2020-01-01",
                "exit_date": "2020-03-01",
                "pnl_pct": 10.0,
                "pnl": 100.0,
                "holding_days": 5,
            },
            {
                "symbol": "BBB",
                "entry_date": "2020-01-01",
                "exit_date": "2020-04-01",
                "pnl_pct": -5.0,
                "pnl": -50.0,
                "holding_days": 7,
            },
        ]
    )
    trades.to_csv(run_dir / "trades.csv", index=False)
    (run_dir / "predictions_meta.json").write_text(
        """
{
  "config_hash": "abc12345def0",
  "market": "vn_stock",
  "currency": "VND",
  "pnl_mode": "equity_spot",
  "schema": "ohlcv_daily",
  "timeframe": "1D",
  "entry_model": "random_forest",
  "exit_model_type": "lightgbm",
  "exit_model_enabled": true,
  "feature_set": "leading",
  "split": {"first_test_year": 2020, "last_test_year": 2025},
  "created_at": "2026-05-06T10:00:00"
}
""".strip(),
        encoding="utf-8",
    )
    (run_dir / "ranking_row.json").write_text(
        """
{
  "name": "sample_run",
  "feature_set": "leading",
  "entry_model": "random_forest",
  "exit_model_type": "lightgbm",
  "exit_model_enabled": true
}
""".strip(),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.resolved.yaml").write_text(
        """
name: sample_run
strategy: v22
market: vn_stock
signals:
  features: leading
  target:
    type: early_wave_v2
    forward_window: 21
    gain_threshold: 0.03
    loss_threshold: 0.015
execution:
  currency: VND
  pnl_mode: equity_spot
split:
  first_test_year: 2020
  last_test_year: 2025
""".strip(),
        encoding="utf-8",
    )

    row = run_dir_to_row(run_dir, bundle="tmp_bundle")

    # Date delta would be much larger (~75 days avg); loader should honor holding_days from CSV.
    assert row.avg_hold == 6.0
