from __future__ import annotations

import pandas as pd
import pytest
from src.components.backtest import LegacyBacktestAdapter, SimpleLongBacktester
from src.components.base import Action, Trade


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


def _assert_trade_invariants(trades: list[Trade]) -> None:
    for trade in trades:
        assert pd.Timestamp(trade.entry_date) <= pd.Timestamp(trade.exit_date)
        assert trade.holding_days >= 0
        assert trade.entry_reason
        assert trade.exit_reason
        assert trade.symbol == "TEST"

    for prev, curr in zip(trades, trades[1:], strict=False):
        assert pd.Timestamp(prev.exit_date) <= pd.Timestamp(curr.entry_date)


def test_entry_date_lte_exit_date() -> None:
    df = _df([100.0, 101.0, 102.0, 103.0])
    actions = [
        Action(bar_idx=0, type="enter_long", reason="entry"),
        Action(bar_idx=3, type="exit", reason="exit"),
    ]

    trades = SimpleLongBacktester().run(actions, df)

    _assert_trade_invariants(trades)


def test_no_overlapping_trades() -> None:
    df = _df([100.0, 101.0, 102.0, 103.0, 104.0])
    actions = [
        Action(bar_idx=0, type="enter_long", reason="entry_1"),
        Action(bar_idx=2, type="exit", reason="exit_1"),
        Action(bar_idx=3, type="enter_long", reason="entry_2"),
        Action(bar_idx=4, type="exit", reason="exit_2"),
    ]

    trades = SimpleLongBacktester().run(actions, df)

    assert len(trades) == 2
    _assert_trade_invariants(trades)


@pytest.mark.parametrize(
    ("actions", "expected_count"),
    [
        ([], 0),
        ([Action(bar_idx=0, type="exit", reason="orphan_exit")], 0),
        ([Action(bar_idx=0, type="enter_long", reason="dangling_entry")], 0),
        (
            [
                Action(bar_idx=0, type="enter_long", reason="entry_1"),
                Action(bar_idx=1, type="enter_long", reason="ignored_entry"),
                Action(bar_idx=2, type="exit", reason="exit_1"),
            ],
            1,
        ),
        (
            [
                Action(bar_idx=0, type="enter_long", reason="entry_1"),
                Action(bar_idx=1, type="hold", reason="hold"),
                Action(bar_idx=2, type="exit", reason="exit_1"),
                Action(bar_idx=3, type="exit", reason="ignored_exit"),
                Action(bar_idx=4, type="enter_long", reason="entry_2"),
                Action(bar_idx=5, type="exit", reason="exit_2"),
            ],
            2,
        ),
    ],
)
def test_trade_count_invariants(actions: list[Action], expected_count: int) -> None:
    df = _df([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

    trades = SimpleLongBacktester().run(actions, df)

    assert len(trades) == expected_count
    _assert_trade_invariants(trades)


def test_legacy_adapter_converts_backtest_unified_trade_dict() -> None:
    df = _df([100.0, 102.0, 105.0])
    result = {
        "trades": [
            {
                "entry_day": 0,
                "exit_day": 2,
                "holding_days": 2,
                "pnl_pct": 5.0,
                "exit_reason": "exit_model",
                "exit_date": "2020-01-03",
                "entry_date": "2020-01-01",
                "entry_symbol": "TEST",
                "entry_score": 4,
            }
        ]
    }

    trades = LegacyBacktestAdapter().convert(result, df_test=df)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.entry_date == "2020-01-01"
    assert trade.exit_date == "2020-01-03"
    assert trade.entry_price == 100.0
    assert trade.exit_price == 105.0
    assert trade.pnl_pct == 5.0
    assert trade.holding_days == 2
    assert trade.entry_reason == "legacy_entry"
    assert trade.exit_reason == "exit_model"
    assert trade.symbol == "TEST"
    assert trade.metadata["entry_score"] == 4
