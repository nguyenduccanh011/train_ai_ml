"""Rule baseline runner — Phase 2.3a end-to-end test bed for FusionStack + Backtester."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.components.backtest import SimpleLongBacktester
from src.components.base import Action, BarContext
from src.components.fusion.stack import FusionStack
from src.components.fusion.strategies.rule_signal import RuleSignalEntry, RuleSignalExit
from src.data.loader import DataLoader

if TYPE_CHECKING:
    from src.components.base import Trade


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ma20 and macd_hist columns inline (matches legacy backtest_rule)."""
    out = df.copy()
    close = out["close"]
    out["ma20"] = close.rolling(20, min_periods=10).mean()
    ema_fast = close.ewm(span=12).mean()
    ema_slow = close.ewm(span=26).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9).mean()
    out["macd_hist"] = macd_line - signal_line
    return out


def _run_symbol(df: pd.DataFrame, stack: FusionStack) -> list[Trade]:
    """Run fusion stack across bars for a single symbol → trades."""
    actions: list[Action] = []
    in_position = False

    # Entry/exit signal preset to 1 so FusionStack runs the entry/exit_override
    # layer at every bar; the rule strategies inspect df_test themselves.
    for i in range(26, len(df)):
        ctx = BarContext(
            bar_idx=i,
            df_test=df,
            entry_signal=0 if in_position else 1,
            entry_proba=None,
            exit_signal=1 if in_position else 0,
            exit_proba=None,
            position=_make_dummy_position(df, i) if in_position else None,
            config={},
        )
        outcome = stack.run(ctx)
        action = outcome.action
        if action.type == "enter_long":
            in_position = True
            actions.append(action)
        elif action.type == "exit":
            in_position = False
            actions.append(action)
        # type == "hold" → skip (Backtester ignores holds)

    return SimpleLongBacktester().run(actions, df)


def _make_dummy_position(df: pd.DataFrame, _bar_idx: int):  # noqa: ANN202
    """Cheap Position stub — FusionStack only checks `is None` for in_position state."""
    from src.components.base import Position

    return Position(
        symbol=str(df["symbol"].iloc[0]) if "symbol" in df.columns else "",
        entry_idx=0,
        entry_date=pd.Timestamp("1970-01-01"),
        entry_price=0.0,
    )


def run_rule_baseline(
    symbols: list[str],
    data_dir: str,
    *,
    first_test_year: int = 2020,
    min_bars: int = 50,
) -> list[Trade]:
    """Reproduce legacy rule baseline using FusionStack + SimpleLongBacktester.

    Mirrors `_run_rule_backtest_fair` in run_pipeline.py:407-439.
    """
    loader = DataLoader(data_dir)
    raw_df = loader.load_all(symbols=symbols)
    date_col = "timestamp" if "timestamp" in raw_df.columns else "date"
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], utc=True)

    stack = FusionStack([RuleSignalEntry(), RuleSignalExit()])

    all_trades: list[Trade] = []
    for sym in symbols:
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_test = sym_data[sym_data[date_col].dt.year >= first_test_year].reset_index(drop=True)
        if len(sym_test) < min_bars:
            continue
        sym_test = _compute_indicators(sym_test)
        all_trades.extend(_run_symbol(sym_test, stack))

    return all_trades


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Serialize trades into a DataFrame matching legacy CSV schema."""
    rows = [
        {
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "holding_days": t.holding_days,
            "pnl_pct": t.pnl_pct,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "symbol": t.symbol,
        }
        for t in trades
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "entry_date",
            "exit_date",
            "holding_days",
            "pnl_pct",
            "entry_price",
            "exit_price",
            "exit_reason",
            "symbol",
        ],
    )
