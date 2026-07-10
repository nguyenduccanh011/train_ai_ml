"""Time-stepped portfolio executor tests (Phase 3).

Deterministic price paths so NAV outcomes are checkable by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.backtest.engine import CostModel, EngineConfig  # noqa: E402
from src.backtest.stats import equity_stats  # noqa: E402
from src.execution import run_portfolio_backtest  # noqa: E402

NO_COST = CostModel(commission=0.0, tax=0.0, slippage=0.0)


def _flat_ohlcv(prices: dict[str, list[float]], start: str = "2020-01-01") -> pd.DataFrame:
    """OHLC all equal to the given close path (no intrabar range)."""
    frames = []
    for sym, closes in prices.items():
        dates = pd.bdate_range(start=start, periods=len(closes))
        arr = np.array(closes, dtype=float)
        frames.append(
            pd.DataFrame(
                {"symbol": sym, "date": dates, "open": arr, "high": arr, "low": arr, "close": arr}
            )
        )
    return pd.concat(frames, ignore_index=True)


def _targets(rows: list[tuple]) -> pd.DataFrame:
    # rows: (date, symbol, weight)
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "symbol": s, "target_weight": w} for d, s, w in rows]
    )


def test_full_allocation_tracks_price_return_no_cost():
    # One name, +10% over the hold. Enter day0 (fill open day1), exit weight->0 at day4.
    closes = [100, 100, 110, 110, 110, 110]
    ohlcv = _flat_ohlcv({"A": closes})
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    targets = _targets([(dates[0], "A", 1.0), (dates[4], "A", 0.0)])

    cfg = EngineConfig(cost=NO_COST, hard_stop_pct=None, max_hold_bars=999)
    trades, equity = run_portfolio_backtest(targets, ohlcv, cfg, initial_capital=1_000.0)

    # Bought at open day1 (=100), price rose to 110 -> +10% on full capital.
    assert equity["nav"].iloc[-1] == 1100.0
    assert len(trades) == 1
    assert trades[0].side == 1
    assert abs(trades[0].pnl_pct - 0.10) < 1e-9
    assert equity_stats(equity)["total_return"] > 0.09


def test_hard_stop_forces_exit():
    # Price collapses; hard_stop at -8% must close the position.
    closes = [100, 100, 100, 80, 80, 80]
    ohlcv = _flat_ohlcv({"A": closes})
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    targets = _targets([(dates[0], "A", 1.0)])  # never told to exit by signal

    cfg = EngineConfig(cost=NO_COST, hard_stop_pct=-0.08, min_hold_bars=1, max_hold_bars=999)
    trades, equity = run_portfolio_backtest(targets, ohlcv, cfg, initial_capital=1_000.0)

    assert len(trades) == 1
    assert trades[0].exit_reason == "hard_stop"


def test_no_lookahead_fills_next_open():
    # Signal at day0 close; fill must be at day1 open (=105), not day0 (=100).
    df = pd.DataFrame(
        {
            "symbol": "A",
            "date": pd.bdate_range("2020-01-01", periods=4),
            "open": [100.0, 105.0, 105.0, 105.0],
            "high": [100.0, 105.0, 105.0, 105.0],
            "low": [100.0, 105.0, 105.0, 105.0],
            "close": [100.0, 105.0, 105.0, 105.0],
        }
    )
    dates = pd.bdate_range("2020-01-01", periods=4)
    targets = _targets([(dates[0], "A", 1.0), (dates[2], "A", 0.0)])
    cfg = EngineConfig(cost=NO_COST, hard_stop_pct=None, max_hold_bars=999)
    trades, _ = run_portfolio_backtest(targets, ohlcv=df, cfg=cfg, initial_capital=1_000.0)
    assert len(trades) == 1
    assert trades[0].entry_price == 105.0  # filled at day1 open, no same-bar lookahead


def test_market_neutral_opens_both_sides():
    closes_a = [100, 100, 120, 120]  # winner
    closes_b = [100, 100, 80, 80]  # loser
    ohlcv = _flat_ohlcv({"A": closes_a, "B": closes_b})
    dates = pd.bdate_range("2020-01-01", periods=4)
    targets = _targets([(dates[0], "A", 0.5), (dates[0], "B", -0.5)])
    cfg = EngineConfig(cost=NO_COST, hard_stop_pct=None, max_hold_bars=999)
    trades, equity = run_portfolio_backtest(targets, ohlcv, cfg, initial_capital=1_000.0)

    sides = {t.symbol: t.side for t in trades}
    assert sides["A"] == 1 and sides["B"] == -1
    # Long winner (+20% on 0.5) and short loser (+20% on 0.5) -> NAV up.
    assert equity["nav"].iloc[-1] > 1000.0
