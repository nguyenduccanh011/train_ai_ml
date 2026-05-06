from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.backtest.engine import backtest_unified
from src.market_profile import MarketProfile


def _df(close: list[float], low: list[float], high: list[float] | None = None) -> pd.DataFrame:
    n = len(close)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": close,
            "high": close if high is None else high,
            "low": low,
            "close": close,
            "volume": [2_000_000.0] * n,
            "symbol": ["BTCUSDT"] * n,
            "rsi_slope_5d": np.full(n, 5.0),
            "vol_surge_ratio": np.full(n, 1.2),
            "range_position_20d": np.full(n, 0.4),
            "dist_to_resistance": np.full(n, 0.05),
            "breakout_setup_score": np.full(n, 4.0),
            "bb_width_percentile": np.full(n, 0.5),
            "higher_lows_count": np.full(n, 3.0),
            "obv_price_divergence": np.zeros(n),
        }
    )


def test_liquidation_exits_long_when_intrabar_low_crosses_threshold() -> None:
    df = _df([100.0, 100.0, 101.0, 101.0, 101.0], [100.0, 100.0, 101.0, 79.0, 101.0])
    y_pred = np.array([1, 1, 1, 1])
    returns = np.array([0.0, 0.0, 0.01, 0.0, 0.0])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        leverage=5.0,
        maintenance_margin_rate=0.005,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["exit_reason"] == "liquidation"
    assert result["trades"][0]["pnl_pct"] == -19.5


def test_default_leverage_does_not_liquidate_spot_path() -> None:
    df = _df([100.0, 100.0, 101.0, 101.0, 101.0], [100.0, 100.0, 101.0, 79.0, 101.0])
    y_pred = np.array([1, 1, 1, 1])
    returns = np.array([0.0, 0.0, 0.01, 0.0, 0.0])

    result = backtest_unified(y_pred, returns, df, feature_cols=[])

    assert result["trades"][0]["exit_reason"] == "end"
    assert result["trades"][0]["pnl_pct"] >= 0


def test_linear_perp_funding_fee_reduces_equity() -> None:
    df = _df([100.0, 101.0, 102.0, 103.0, 104.0], [100.0, 101.0, 102.0, 103.0, 104.0])
    df["funding_rate"] = [0.0, 0.001, 0.001, 0.001, 0.001]
    y_pred = np.array([1, 1, 1, 1])
    returns = np.array([0.0, 0.01, 0.01, 0.01, 0.01])

    without_funding = backtest_unified(
        y_pred,
        returns,
        df.drop(columns=["funding_rate"]),
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )
    with_funding = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        funding_rate_column="funding_rate",
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert with_funding["equity_curve"][-1] < without_funding["equity_curve"][-1]


def test_inverse_perp_engine_uses_inverse_trade_return() -> None:
    df = _df([100.0, 100.0, 95.0, 90.0, 90.0], [100.0, 100.0, 95.0, 90.0, 90.0])
    y_pred = np.array([1, 1, 1, 1])
    returns = np.array([0.0, 0.0, -0.05, -0.0526315789, 0.0])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="inverse_perp",
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["pnl_pct"] > 0


def test_futures_contract_pnl_mode_runs() -> None:
    df = _df([1000.0, 1001.0, 1002.0, 1003.0, 1004.0], [1000.0, 1001.0, 1002.0, 1003.0, 1004.0])
    y_pred = np.array([1, 1, 1, 1])
    returns = np.array([0.0, 0.001, 0.001, 0.001, 0.001])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="futures_contract",
        contract_multiplier=100000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert "trades" in result
    assert "equity_curve" in result


def test_rollover_resets_entry_and_applies_cost() -> None:
    df = _df([1000.0, 1001.0, 1002.0, 1003.0, 1004.0], [1000.0, 1001.0, 1002.0, 1003.0, 1004.0])
    df["expiry_date"] = pd.to_datetime(
        ["2024-01-10", "2024-01-10", "2024-01-03", "2024-01-10", "2024-01-10"]
    )
    y_pred = np.array([1, 1, 1, 1])
    returns = np.array([0.0, 0.001, 0.001, 0.001, 0.001])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="futures_contract",
        contract_multiplier=1,
        expiry_date_column="expiry_date",
        roll_cost_rate=0.01,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert len(result["trades"]) == 1
    assert result["trades"][0]["pnl_pct"] == round((1004.0 / 1002.0 - 1) * 100, 2)
    assert result["equity_curve"][2] < 100_000_000 * 1.001 * 1.001


def test_rollover_applies_to_short_futures_position() -> None:
    df = _df([1000.0, 999.0, 998.0, 997.0, 996.0], [1000.0, 999.0, 998.0, 997.0, 996.0])
    df["expiry_date"] = pd.to_datetime(
        ["2024-01-10", "2024-01-10", "2024-01-03", "2024-01-10", "2024-01-10"]
    )
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, -0.001, -0.001, -0.001, -0.001])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="futures_contract",
        contract_multiplier=1,
        expiry_date_column="expiry_date",
        roll_cost_rate=0.01,
        short_enabled=True,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert len(result["trades"]) == 1
    assert result["trades"][0]["pnl_pct"] == round(-(996.0 / 998.0 - 1) * 100, 2)
    assert result["equity_curve"][2] < 100_000_000 * 1.001 * 1.001


def test_roll_volume_crossover_triggers_before_expiry() -> None:
    df = _df(
        [1000.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
        [1000.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
    )
    df["expiry_date"] = pd.to_datetime(["2024-01-06"] * len(df))
    df["volume"] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    df["next_volume"] = [0.0, 50.0, 60.0, 150.0, 200.0, 300.0]
    y_pred = np.array([1, 1, 1, 1, 1])
    returns = np.array([0.0, 0.0, 0.001, 0.001, 0.001, 0.001])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="futures_contract",
        contract_multiplier=1,
        expiry_date_column="expiry_date",
        roll_rule="volume_crossover",
        next_volume_column="next_volume",
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["entry_day"] == 3
    assert result["trades"][0]["pnl_pct"] == round((1004.0 / 1002.0 - 1) * 100, 2)


def test_roll_n_days_before_expiry() -> None:
    df = _df(
        [1000.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
        [1000.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
    )
    df["expiry_date"] = pd.to_datetime(["2024-01-06"] * len(df))
    y_pred = np.array([1, 1, 1, 1, 1])
    returns = np.array([0.0, 0.0, 0.001, 0.001, 0.001, 0.001])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="futures_contract",
        contract_multiplier=1,
        expiry_date_column="expiry_date",
        roll_rule="n_days_before_expiry",
        roll_days_before_expiry=2,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["entry_day"] == 3
    assert result["trades"][0]["pnl_pct"] == round((1004.0 / 1002.0 - 1) * 100, 2)


def test_roll_rule_none_keeps_old_expiry_behavior() -> None:
    df = _df(
        [1000.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
        [1000.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
    )
    df["expiry_date"] = pd.to_datetime(["2024-01-05"] * len(df))
    df["volume"] = [100.0] * len(df)
    df["next_volume"] = [1000.0] * len(df)
    y_pred = np.array([1, 1, 1, 1, 1])
    returns = np.array([0.0, 0.0, 0.001, 0.001, 0.001, 0.001])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="futures_contract",
        contract_multiplier=1,
        expiry_date_column="expiry_date",
        next_volume_column="next_volume",
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["entry_day"] == 4
    assert result["trades"][0]["pnl_pct"] == round((1004.0 / 1003.0 - 1) * 100, 2)


def test_roll_only_once_per_expiry_period() -> None:
    df = _df(
        [1000.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
        [1000.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
    )
    df["expiry_date"] = pd.to_datetime(["2024-01-06"] * len(df))
    df["volume"] = [100.0] * len(df)
    df["next_volume"] = [0.0, 120.0, 130.0, 140.0, 150.0, 160.0]
    y_pred = np.array([1, 1, 1, 1, 1])
    returns = np.array([0.0, 0.0, 0.001, 0.001, 0.001, 0.001])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="futures_contract",
        contract_multiplier=1,
        expiry_date_column="expiry_date",
        roll_rule="volume_crossover",
        next_volume_column="next_volume",
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["entry_day"] == 3
    assert result["trades"][0]["pnl_pct"] == round((1004.0 / 1002.0 - 1) * 100, 2)


def test_rollover_does_not_trigger_when_no_expiry_column() -> None:
    df = _df([1000.0, 1001.0, 1002.0, 1003.0, 1004.0], [1000.0, 1001.0, 1002.0, 1003.0, 1004.0])
    y_pred = np.array([1, 1, 1, 1])
    returns = np.array([0.0, 0.001, 0.001, 0.001, 0.001])

    without_rollover = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="futures_contract",
        contract_multiplier=1,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )
    with_missing_expiry = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="futures_contract",
        contract_multiplier=1,
        expiry_date_column="expiry_date",
        roll_cost_rate=0.01,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert with_missing_expiry["equity_curve"][-1] == without_rollover["equity_curve"][-1]


def test_rollover_config_is_ignored_for_non_futures_pnl_mode() -> None:
    df = _df([1000.0, 1001.0, 1002.0, 1003.0, 1004.0], [1000.0, 1001.0, 1002.0, 1003.0, 1004.0])
    df["expiry_date"] = pd.to_datetime(
        ["2024-01-10", "2024-01-10", "2024-01-03", "2024-01-10", "2024-01-10"]
    )
    y_pred = np.array([1, 1, 1, 1])
    returns = np.array([0.0, 0.001, 0.001, 0.001, 0.001])

    without_rollover = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )
    with_rollover_config = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        expiry_date_column="expiry_date",
        roll_cost_rate=0.01,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert with_rollover_config["equity_curve"][-1] == without_rollover["equity_curve"][-1]


def test_short_liquidation_triggers_on_high() -> None:
    df = _df(
        [100.0, 100.0, 99.0, 99.0, 99.0],
        [100.0, 100.0, 99.0, 99.0, 99.0],
        high=[100.0, 100.0, 99.0, 121.0, 99.0],
    )
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, 0.0, -0.01, 0.0, 0.0])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        leverage=5.0,
        maintenance_margin_rate=0.005,
        short_enabled=True,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["exit_reason"] == "liquidation"
    assert result["trades"][0]["pnl_pct"] == -19.5


def test_short_disabled_blocks_short_entry() -> None:
    df = _df([100.0, 99.0, 98.0, 97.0, 96.0], [100.0, 99.0, 98.0, 97.0, 96.0])
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, -0.01, -0.010101, -0.010204, -0.010309])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"] == []
    assert result["equity_curve"][-1] == 100_000_000


def test_short_funding_rate_is_credited_when_positive() -> None:
    df = _df([100.0, 99.0, 98.0, 97.0, 96.0], [100.0, 99.0, 98.0, 97.0, 96.0])
    df["funding_rate"] = [0.0, 0.001, 0.001, 0.001, 0.001]
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, -0.01, -0.010101, -0.010204, -0.010309])

    without_funding = backtest_unified(
        y_pred,
        returns,
        df.drop(columns=["funding_rate"]),
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )
    with_funding = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        funding_rate_column="funding_rate",
        short_enabled=True,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert with_funding["equity_curve"][-1] > without_funding["equity_curve"][-1]


def test_short_hard_stop_exits_when_price_rises() -> None:
    df = _df([100.0, 100.0, 109.0, 109.0, 109.0], [100.0, 100.0, 109.0, 109.0, 109.0])
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, 0.0, 0.09, 0.0, 0.0])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["exit_reason"] == "hard_stop"
    assert result["trades"][0]["pnl_pct"] < 0


def test_short_zombie_exit_exits_flat_trade_after_max_hold() -> None:
    df = _df([100.0] * 18, [100.0] * 18)
    y_pred = np.array([-1] * 17)
    returns = np.zeros(18)

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["exit_reason"] == "zombie_exit"


def test_short_position_size_controls_deployed_notional() -> None:
    df = _df([100.0, 100.0, 99.0, 98.0, 97.0], [100.0, 100.0, 99.0, 98.0, 97.0])
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, 0.0, -0.01, -0.010101, -0.010204])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        short_position_size=0.4,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["position_size"] == 0.4
    assert result["equity_curve"][2] == 100_400_000


def test_short_hard_cap_exits_profitable_short() -> None:
    df = _df([100.0, 100.0, 97.0, 95.0, 95.0], [100.0, 100.0, 97.0, 95.0, 95.0])
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, 0.0, -0.03, -0.020619, 0.0])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        short_hard_cap=0.04,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["exit_reason"] == "signal_hard_cap"
    assert result["trades"][0]["pnl_pct"] > 0


def test_short_fast_exit_takes_fast_profit() -> None:
    df = _df([100.0, 100.0, 99.0, 98.0, 96.0, 95.0], [100.0, 100.0, 99.0, 98.0, 96.0, 95.0])
    y_pred = np.array([-1, -1, -1, -1, -1])
    returns = np.array([0.0, 0.0, -0.01, -0.010101, -0.020408, -0.010417])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["exit_reason"] == "fast_exit_profit"
    assert result["trades"][0]["pnl_pct"] > 0


def test_short_borrow_rate_reduces_equity() -> None:
    df = _df([100.0, 99.0, 98.0, 97.0, 96.0], [100.0, 99.0, 98.0, 97.0, 96.0])
    df["borrow_rate"] = [0.0, 0.001, 0.001, 0.001, 0.001]
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, -0.01, -0.010101, -0.010204, -0.010309])

    without_borrow = backtest_unified(
        y_pred,
        returns,
        df.drop(columns=["borrow_rate"]),
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )
    with_borrow = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        borrow_rate_column="borrow_rate",
        short_enabled=True,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert with_borrow["equity_curve"][-1] < without_borrow["equity_curve"][-1]


def test_short_squeeze_exit_on_volume_spike() -> None:
    df = _df([100.0, 100.0, 99.0, 103.0, 103.0], [100.0, 100.0, 99.0, 103.0, 103.0])
    df["open"] = [100.0, 100.0, 99.0, 99.0, 103.0]
    df["volume"] = [1_000_000.0, 1_000_000.0, 1_000_000.0, 20_000_000.0, 1_000_000.0]
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, 0.0, -0.01, 0.040404, 0.0])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        short_squeeze_exit=True,
        short_squeeze_vol_mult=3.0,
        short_squeeze_price_pct=0.03,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["exit_reason"] == "short_squeeze"


def test_short_squeeze_exit_not_triggered_below_threshold() -> None:
    df = _df([100.0, 100.0, 99.0, 102.0, 101.0], [100.0, 100.0, 99.0, 102.0, 101.0])
    df["open"] = [100.0, 100.0, 99.0, 100.0, 101.0]
    df["volume"] = [1_000_000.0, 1_000_000.0, 1_000_000.0, 20_000_000.0, 1_000_000.0]
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, 0.0, -0.01, 0.030303, -0.009804])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        short_squeeze_exit=True,
        short_squeeze_vol_mult=3.0,
        short_squeeze_price_pct=0.04,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["exit_reason"] != "short_squeeze"


def test_short_borrow_available_blocks_entry() -> None:
    df = _df([100.0, 99.0, 98.0, 97.0, 96.0], [100.0, 99.0, 98.0, 97.0, 96.0])
    df["borrow_available"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, -0.01, -0.010101, -0.010204, -0.010309])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        borrow_available_column="borrow_available",
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"] == []
    assert result["equity_curve"][-1] == 100_000_000


def test_short_borrow_recalled_exits_position() -> None:
    df = _df([100.0, 99.0, 98.0, 97.0, 96.0], [100.0, 99.0, 98.0, 97.0, 96.0])
    df["borrow_available"] = [1.0, 1.0, 1.0, 0.0, 0.0]
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, -0.01, -0.010101, -0.010204, -0.010309])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        borrow_available_column="borrow_available",
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["exit_reason"] == "borrow_recalled"


def test_short_max_notional_caps_position_size() -> None:
    df = _df([100.0, 100.0, 99.0, 98.0, 97.0], [100.0, 100.0, 99.0, 98.0, 97.0])
    y_pred = np.array([-1, -1, -1, -1])
    returns = np.array([0.0, 0.0, -0.01, -0.010101, -0.010204])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        short_position_size=0.8,
        max_short_notional=0.3,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["position_size"] == 0.3
    assert np.isclose(result["equity_curve"][2], 100_300_000)


def test_short_cooldown_uses_signed_pnl() -> None:
    df = _df([100.0, 100.0, 90.0, 90.0, 90.0, 90.0], [100.0, 100.0, 90.0, 90.0, 90.0, 90.0])
    y_pred = np.array([-1, 0, 1, 1, 1])
    returns = np.array([0.0, 0.0, -0.10, 0.0, 0.0, 0.0])

    result = backtest_unified(
        y_pred,
        returns,
        df,
        feature_cols=[],
        pnl_mode="linear_usdt_perp",
        short_enabled=True,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
    )

    assert result["trades"][0]["pnl_pct"] > 0
    assert len(result["trades"]) == 1


def test_market_profile_rejects_unsupported_pnl_mode() -> None:
    with pytest.raises(ValueError, match="unsupported pnl_mode"):
        MarketProfile.model_validate({"name": "bad_market", "execution": {"pnl_mode": "spot"}})
