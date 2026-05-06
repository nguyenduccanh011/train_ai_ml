import numpy as np
import pandas as pd
from src.backtest.portfolio_engine import backtest_portfolio


def _df(close, returns=None, **extra):
    n = len(close)
    data = {
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": [1000] * n,
        "return_1d": returns if returns is not None else [0.0] * n,
    }
    data.update(extra)
    return pd.DataFrame(data)


def test_portfolio_equity_consolidation():
    predictions = {
        "AAA": np.array([0, 1, 1, 0]),
        "BBB": np.array([0, 0, 1, 0]),
    }
    returns = {
        "AAA": np.array([0.0, 0.0, 0.10, 0.0]),
        "BBB": np.array([0.0, 0.0, 0.05, 0.0]),
    }
    dfs = {
        "AAA": _df([100, 100, 110, 110]),
        "BBB": _df([50, 50, 52.5, 52.5]),
    }

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=10,
        maintenance_margin_rate=0.01,
        position_size=0.4,
        margin_mode="cross",
    )

    assert result["final_equity"] > 1000
    assert result["total_return_pct"] > 0
    assert len(result["trades"]) == 2


def test_cross_margin_no_new_entry_when_margin_exhausted():
    predictions = {
        "AAA": np.array([0, 1, 1, 1]),
        "BBB": np.array([0, 0, 1, 1]),
    }
    returns = {
        "AAA": np.zeros(4),
        "BBB": np.zeros(4),
    }
    dfs = {
        "AAA": _df([100, 100, 100, 100]),
        "BBB": _df([50, 50, 50, 50]),
    }

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=1,
        maintenance_margin_rate=0.0,
        margin_mode="cross",
    )

    opened_symbols = {trade["entry_symbol"] for trade in result["trades"]}
    assert opened_symbols == {"AAA"}


def test_cross_margin_liquidation_triggers():
    predictions = {"AAA": np.array([0, 1, 1, 1])}
    returns = {"AAA": np.array([0.0, 0.0, -0.80, 0.0])}
    dfs = {"AAA": _df([100, 100, 20, 20])}

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=10,
        maintenance_margin_rate=0.30,
        liquidation_fee=0.0,
        margin_mode="cross",
    )

    assert any(trade["exit_reason"] == "cross_margin_liquidation" for trade in result["trades"])


def test_max_total_short_notional_cap():
    predictions = {
        "AAA": np.array([0, -1, -1]),
        "BBB": np.array([0, -1, -1]),
    }
    returns = {
        "AAA": np.zeros(3),
        "BBB": np.zeros(3),
    }
    dfs = {
        "AAA": _df([100, 100, 100]),
        "BBB": _df([50, 50, 50]),
    }

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=10,
        maintenance_margin_rate=0.01,
        short_enabled=True,
        short_position_size=0.4,
        max_total_short_notional=5000,
        margin_mode="cross",
    )

    assert sum(1 for trade in result["trades"] if trade["entry_symbol"] in {"AAA", "BBB"}) == 1


def test_borrow_recall_forces_short_exit():
    predictions = {"AAA": np.array([0, -1, -1, -1])}
    returns = {"AAA": np.zeros(4)}
    dfs = {
        "AAA": _df(
            [100, 100, 100, 100],
            borrow_available=[1, 1, 0, 0],
        )
    }

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=10,
        maintenance_margin_rate=0.01,
        short_enabled=True,
        borrow_available_column="borrow_available",
        margin_mode="cross",
    )

    assert result["trades"][0]["exit_reason"] == "borrow_recalled"


def test_portfolio_short_hard_stop_exit():
    predictions = {"AAA": np.array([0, -1, -1, -1])}
    returns = {"AAA": np.array([0.0, 0.0, 0.10, 0.0])}
    dfs = {"AAA": _df([100, 100, 110, 110])}

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=1,
        maintenance_margin_rate=0.0,
        position_size=1.0,
        short_enabled=True,
        margin_mode="cross",
    )

    assert result["trades"][0]["exit_reason"] == "hard_stop"


def test_portfolio_short_hard_cap_exit():
    predictions = {"AAA": np.array([0, -1, -1, -1])}
    returns = {"AAA": np.array([0.0, 0.0, -0.05, 0.0])}
    dfs = {"AAA": _df([100, 100, 95, 95])}

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=1,
        maintenance_margin_rate=0.0,
        short_enabled=True,
        short_hard_cap=0.04,
        margin_mode="cross",
    )

    assert result["trades"][0]["exit_reason"] == "signal_hard_cap"


def test_portfolio_short_zombie_exit():
    n = 17
    predictions = {"AAA": np.array([0] + [-1] * (n - 1))}
    returns = {"AAA": np.zeros(n)}
    dfs = {"AAA": _df([100] * n)}

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=1,
        maintenance_margin_rate=0.0,
        short_enabled=True,
        margin_mode="cross",
    )

    assert result["trades"][0]["exit_reason"] == "zombie_exit"


def test_portfolio_short_squeeze_exit():
    close = [100] * 5 + [106, 106]
    open_ = [100] * 7
    volume = [1000] * 5 + [5000, 5000]
    predictions = {"AAA": np.array([0] + [-1] * 6)}
    returns = {"AAA": np.zeros(7)}
    dfs = {"AAA": _df(close, open=open_, volume=volume)}

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=1,
        maintenance_margin_rate=0.0,
        short_enabled=True,
        short_squeeze_exit=True,
        short_squeeze_vol_mult=3.0,
        short_squeeze_price_pct=0.05,
        margin_mode="cross",
    )

    assert result["trades"][0]["exit_reason"] == "short_squeeze"


def test_portfolio_short_atr_position_sizing_changes_position_size():
    predictions = {
        "AAA": np.array([0, -1, 0]),
        "BBB": np.array([0, -1, 0]),
    }
    returns = {
        "AAA": np.zeros(3),
        "BBB": np.zeros(3),
    }
    dfs = {
        "AAA": _df([100, 100, 100], atr=[10, 10, 10]),
        "BBB": _df([100, 100, 100], atr=[1, 1, 1]),
    }

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=1,
        maintenance_margin_rate=0.0,
        short_enabled=True,
        atr_position_sizing=True,
        atr_risk_target=0.005,
        margin_mode="cross",
    )

    sizes = {trade["entry_symbol"]: trade["position_size"] for trade in result["trades"]}
    assert np.isclose(sizes["AAA"], 0.05)
    assert np.isclose(sizes["BBB"], 0.5)


def test_portfolio_futures_rollover_resets_entry_and_applies_cost():
    predictions = {"VN30F1M": np.array([0, 1, 1, 1, 1])}
    returns = {"VN30F1M": np.array([0.0, 0.0, 0.001, 0.001, 0.001])}
    dfs = {
        "VN30F1M": _df(
            [1000.0, 1000.0, 1001.0, 1002.0, 1003.0],
            expiry_date=pd.to_datetime(
                ["2024-01-10", "2024-01-10", "2024-01-03", "2024-01-10", "2024-01-10"]
            ),
        )
    }

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=1,
        maintenance_margin_rate=0.0,
        pnl_mode="futures_contract",
        expiry_date_column="expiry_date",
        roll_cost_rate=0.01,
        margin_mode="cross",
    )

    assert result["trades"][0]["entry_day"] == 2
    assert result["trades"][0]["pnl_pct"] == round((1003.0 / 1001.0 - 1) * 100, 2)
    assert result["equity_curve"][2] < 1001.0


def test_portfolio_volume_crossover_roll():
    predictions = {"VN30F1M": np.array([0, 1, 1, 1, 1, 1])}
    returns = {"VN30F1M": np.array([0.0, 0.0, 0.001, 0.001, 0.001, 0.001])}
    dfs = {
        "VN30F1M": _df(
            [1000.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
            expiry_date=pd.to_datetime(["2024-01-06"] * 6),
            volume=[100.0] * 6,
            next_volume=[0.0, 50.0, 60.0, 150.0, 200.0, 300.0],
        )
    }

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=1,
        maintenance_margin_rate=0.0,
        pnl_mode="futures_contract",
        expiry_date_column="expiry_date",
        roll_rule="volume_crossover",
        next_volume_column="next_volume",
        margin_mode="cross",
    )

    assert result["trades"][0]["entry_day"] == 3
    assert result["trades"][0]["pnl_pct"] == round((1004.0 / 1002.0 - 1) * 100, 2)


def test_portfolio_rollover_config_ignored_for_non_futures():
    predictions = {"BTCUSDT": np.array([0, 1, 1, 1])}
    returns = {"BTCUSDT": np.array([0.0, 0.0, 0.001, 0.001])}
    dfs = {
        "BTCUSDT": _df(
            [1000.0, 1000.0, 1001.0, 1002.0],
            expiry_date=pd.to_datetime(["2024-01-10", "2024-01-10", "2024-01-03", "2024-01-10"]),
        )
    }

    result = backtest_portfolio(
        predictions,
        returns,
        dfs,
        initial_capital=1000,
        commission=0.0,
        tax=0.0,
        slippage=0.0,
        leverage=1,
        maintenance_margin_rate=0.0,
        pnl_mode="linear_usdt_perp",
        expiry_date_column="expiry_date",
        roll_cost_rate=0.01,
        margin_mode="cross",
    )

    assert result["trades"][0]["entry_day"] == 1
    assert result["final_equity"] == 1002
