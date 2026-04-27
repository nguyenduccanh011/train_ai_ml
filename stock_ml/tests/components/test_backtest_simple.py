"""Unit tests for SimpleLongBacktester."""

from __future__ import annotations

import pandas as pd
import pytest
from src.components.backtest import SimpleLongBacktester
from src.components.base import Action


def _df(prices: list[float], symbol: str = "TEST") -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="D"),
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1000.0] * n,
            "symbol": [symbol] * n,
        }
    )


class TestSimpleLongBacktester:
    def test_single_trade(self) -> None:
        df = _df([100.0, 105.0, 110.0])
        actions = [
            Action(bar_idx=0, type="enter_long", reason="buy"),
            Action(bar_idx=2, type="exit", reason="sell"),
        ]
        trades = SimpleLongBacktester().run(actions, df)
        assert len(trades) == 1
        t = trades[0]
        assert t.entry_price == 100.0
        assert t.exit_price == 110.0
        assert t.holding_days == 2
        # (110/100 - 1)*100 - (0.0015+0.001)*2*100 = 10 - 0.5 = 9.5
        assert t.pnl_pct == 9.5
        assert t.entry_reason == "buy"
        assert t.exit_reason == "sell"
        assert t.symbol == "TEST"
        assert t.entry_date == "2020-01-01"
        assert t.exit_date == "2020-01-03"

    def test_multiple_trades(self) -> None:
        df = _df([100.0, 110.0, 100.0, 105.0])
        actions = [
            Action(bar_idx=0, type="enter_long", reason="b1"),
            Action(bar_idx=1, type="exit", reason="s1"),
            Action(bar_idx=2, type="enter_long", reason="b2"),
            Action(bar_idx=3, type="exit", reason="s2"),
        ]
        trades = SimpleLongBacktester().run(actions, df)
        assert len(trades) == 2
        assert [t.entry_price for t in trades] == [100.0, 100.0]
        assert [t.exit_price for t in trades] == [110.0, 105.0]

    def test_dangling_enter_dropped(self) -> None:
        df = _df([100.0, 110.0, 120.0])
        actions = [Action(bar_idx=0, type="enter_long", reason="b")]
        trades = SimpleLongBacktester().run(actions, df)
        assert trades == []

    def test_hold_actions_skipped(self) -> None:
        df = _df([100.0, 105.0, 110.0])
        actions = [
            Action(bar_idx=0, type="enter_long", reason="b"),
            Action(bar_idx=1, type="hold", reason="h"),
            Action(bar_idx=2, type="exit", reason="s"),
        ]
        trades = SimpleLongBacktester().run(actions, df)
        assert len(trades) == 1
        assert trades[0].holding_days == 2

    def test_double_enter_ignored(self) -> None:
        df = _df([100.0, 110.0, 105.0])
        actions = [
            Action(bar_idx=0, type="enter_long", reason="b1"),
            Action(bar_idx=1, type="enter_long", reason="b2"),
            Action(bar_idx=2, type="exit", reason="s"),
        ]
        trades = SimpleLongBacktester().run(actions, df)
        assert len(trades) == 1
        assert trades[0].entry_price == 100.0  # first enter wins

    def test_exit_without_position_skipped(self) -> None:
        df = _df([100.0, 105.0])
        actions = [Action(bar_idx=0, type="exit", reason="s")]
        trades = SimpleLongBacktester().run(actions, df)
        assert trades == []

    def test_pnl_round_2_decimals(self) -> None:
        df = _df([100.0, 100.333])
        actions = [
            Action(bar_idx=0, type="enter_long", reason="b"),
            Action(bar_idx=1, type="exit", reason="s"),
        ]
        trades = SimpleLongBacktester().run(actions, df)
        # (100.333/100-1)*100 - 0.5 = 0.333 - 0.5 = -0.167 → round 2 = -0.17
        assert trades[0].pnl_pct == -0.17
        assert trades[0].exit_price == 100.33

    def test_custom_fees(self) -> None:
        df = _df([100.0, 110.0])
        actions = [
            Action(bar_idx=0, type="enter_long", reason="b"),
            Action(bar_idx=1, type="exit", reason="s"),
        ]
        bt = SimpleLongBacktester(commission=0.0, tax=0.0)
        trades = bt.run(actions, df)
        assert trades[0].pnl_pct == 10.0  # no fees

    def test_date_column_fallback(self) -> None:
        df = _df([100.0, 110.0])
        df = df.rename(columns={"timestamp": "date"})
        actions = [
            Action(bar_idx=0, type="enter_long", reason="b"),
            Action(bar_idx=1, type="exit", reason="s"),
        ]
        trades = SimpleLongBacktester().run(actions, df)
        assert len(trades) == 1
        assert trades[0].entry_date == "2020-01-01"

    def test_missing_date_column_raises(self) -> None:
        df = pd.DataFrame({"close": [100.0], "symbol": ["X"]})
        with pytest.raises(KeyError, match="timestamp"):
            SimpleLongBacktester().run([], df)
