"""Time-stepped, capital-aware portfolio executor.

Consumes a TargetWeightFrame ([date, symbol, target_weight, ...]) plus OHLCV and walks
the calendar day by day. Decisions are made at the close of day *t* and filled at the
open of day *t+1* (a one-day pending-order queue) so there is no look-ahead.

Per day:
  1. fill orders queued yesterday at today's open (open/resize/close positions, pay costs)
  2. mark the book to market at today's close -> equity point
  3. decide today's orders for tomorrow:
       - per-position risk exits (hard_stop via today's low, max_hold via hold bars)
       - on rebalance dates, move the book toward the target weights

The equity (NAV) curve is the source of truth for performance; per-trade `pnl_pct`
is informational and computed from slippage-adjusted fill prices net of round-trip cost.
"""

from __future__ import annotations

import pandas as pd

from stock_ml.src.backtest.defaults import DEFAULT_INITIAL_CAPITAL
from stock_ml.src.backtest.engine import CostModel, EngineConfig, Trade, _ensure_ohlcv
from stock_ml.src.execution.state import PortfolioPosition, PortfolioState


def _pivot(bars: pd.DataFrame, col: str) -> dict:
    return bars.pivot_table(index="date", columns="symbol", values=col, aggfunc="last").to_dict(
        "index"
    )


class PortfolioExecutor:
    def __init__(self, cfg: EngineConfig | None = None, initial_capital: float = DEFAULT_INITIAL_CAPITAL):
        if cfg is None:
            raise ValueError(
                "PortfolioExecutor requires an explicit EngineConfig — no silent default. "
                "Declare max_hold_bars and hard_stop_pct (hard_stop_pct=None disables the stop)."
            )
        self.cfg = cfg
        self.initial_capital = float(initial_capital)

    def run(self, targets: pd.DataFrame, ohlcv: pd.DataFrame) -> tuple[list[Trade], pd.DataFrame]:
        bars = _ensure_ohlcv(ohlcv)
        opens = _pivot(bars, "open")
        lows = _pivot(bars, "low")
        closes = _pivot(bars, "close")
        calendar = sorted(closes.keys())

        tgt = targets.copy()
        tgt["date"] = pd.to_datetime(tgt["date"])
        target_by_date: dict = {
            d: dict(zip(g["symbol"], g["target_weight"]))
            for d, g in tgt.groupby("date", sort=True)
        }
        rebalance_dates = set(target_by_date.keys())
        if not rebalance_dates:
            return [], pd.DataFrame(columns=["date", "nav", "cash", "gross_exposure", "net_exposure"])

        # Only walk from the first decision date onward.
        first = min(rebalance_dates)
        calendar = [d for d in calendar if d >= first]

        state = PortfolioState(cash=self.initial_capital)
        cost = self.cfg.cost
        trades: list[Trade] = []
        equity: list[dict] = []
        pending: dict = {}  # symbol -> {"target_shares", "reason", "signal_date", "weight"}
        suspended: set[str] = set()  # names risk-exited in the current rebalance cycle

        for i, d in enumerate(calendar):
            open_px = opens.get(d, {})
            low_px = lows.get(d, {})
            close_px = closes.get(d, {})

            # 1) Execute pending orders at today's open.
            for sym, order in pending.items():
                px = open_px.get(sym)
                if px is None or px != px:  # missing / NaN bar -> cannot fill today
                    continue
                self._apply_order(state, trades, sym, order, float(px), d, i, cost)
            pending = {}

            # 2) Mark to market at close, record equity.
            nav = state.nav(close_px)
            equity.append(
                {
                    "date": d,
                    "nav": nav,
                    "cash": state.cash,
                    "gross_exposure": state.gross_exposure(close_px),
                    "net_exposure": state.net_exposure(close_px),
                }
            )

            if i >= len(calendar) - 1:
                break  # no next open to fill into; positions closed below

            # 3a) Rebalance decision (target shares sized at today's close as reference).
            new_orders: dict = {}
            if d in rebalance_dates:
                suspended.clear()
                desired = target_by_date[d]
                symbols = set(desired) | set(state.positions)
                for sym in symbols:
                    ref = close_px.get(sym)
                    if ref is None or ref != ref:
                        continue
                    w = float(desired.get(sym, 0.0))
                    target_shares = (w * nav) / ref if ref else 0.0
                    new_orders[sym] = {
                        "target_shares": target_shares,
                        "reason": "signal",
                        "signal_date": d,
                        "weight": w,
                    }

            # 3b) Per-position risk exits override toward a flat target.
            for sym, pos in state.positions.items():
                hold = i - pos.entry_idx
                reason = self._risk_exit(pos, low_px.get(sym), hold)
                if reason:
                    new_orders[sym] = {
                        "target_shares": 0.0,
                        "reason": reason,
                        "signal_date": d,
                        "weight": 0.0,
                    }
                    suspended.add(sym)

            # Suspended names stay flat until the next rebalance clears them.
            for sym in suspended:
                if sym in new_orders and new_orders[sym]["reason"] == "signal":
                    new_orders[sym]["target_shares"] = 0.0
                    new_orders[sym]["weight"] = 0.0

            pending = new_orders

        # Force-close any remaining book at the last close (end_of_data).
        self._force_close_all(state, trades, calendar[-1], len(calendar) - 1, closes.get(calendar[-1], {}), cost)

        equity_df = pd.DataFrame(equity, columns=["date", "nav", "cash", "gross_exposure", "net_exposure"])
        return trades, equity_df

    def _risk_exit(self, pos: PortfolioPosition, low: float | None, hold: int) -> str | None:
        cfg = self.cfg
        if "hard_stop" in cfg.exit_priority and cfg.hard_stop_pct is not None and hold >= cfg.min_hold_bars:
            if low is not None and low == low:
                mtm = pos.side * (low / pos.entry_price - 1.0)
                if mtm <= cfg.hard_stop_pct:
                    return "hard_stop"
        if "max_hold" in cfg.exit_priority and hold >= cfg.max_hold_bars:
            return "max_hold"
        return None

    def _apply_order(
        self,
        state: PortfolioState,
        trades: list[Trade],
        sym: str,
        order: dict,
        fill_open: float,
        date: pd.Timestamp,
        idx: int,
        cost: CostModel,
    ) -> None:
        target = order["target_shares"]
        pos = state.positions.get(sym)
        current = pos.shares if pos else 0.0

        # Sign flip closes the old leg first, then opens the new one.
        if pos and current != 0 and target != 0 and (current > 0) != (target > 0):
            self._close(state, trades, sym, fill_open, date, order["reason"], cost)
            pos = None
            current = 0.0

        delta = target - current
        if abs(delta) < 1e-12:
            return

        if delta > 0:  # buying
            fill = cost.fill_buy(fill_open)
            state.cash -= delta * fill + abs(delta) * fill_open * cost.commission
        else:  # selling
            fill = cost.fill_sell(fill_open)
            state.cash += (-delta) * fill - abs(delta) * fill_open * (cost.commission + cost.tax)

        if pos is None:  # opening fresh
            state.positions[sym] = PortfolioPosition(
                symbol=sym,
                entry_date=date,
                entry_signal_date=order["signal_date"],
                entry_price=fill,
                shares=target,
                entry_weight=order["weight"],
                entry_idx=idx,
            )
        elif abs(target) < 1e-12:  # full close
            self._record_trade(trades, pos, fill, date, order["reason"], cost)
            del state.positions[sym]
        else:  # resize (same side): weighted-average cost basis if adding
            if abs(target) > abs(current):
                pos.entry_price = (pos.entry_price * current + fill * delta) / target
            pos.shares = target

    def _close(self, state, trades, sym, fill_open, date, reason, cost):
        pos = state.positions.get(sym)
        if pos is None:
            return
        if pos.shares > 0:
            fill = cost.fill_sell(fill_open)
            state.cash += pos.shares * fill - abs(pos.shares) * fill_open * (cost.commission + cost.tax)
        else:
            fill = cost.fill_buy(fill_open)
            state.cash -= (-pos.shares) * fill + abs(pos.shares) * fill_open * cost.commission
        self._record_trade(trades, pos, fill, date, reason, cost)
        del state.positions[sym]

    def _force_close_all(self, state, trades, date, idx, close_px, cost):
        for sym in list(state.positions):
            px = close_px.get(sym)
            if px is None or px != px:
                continue
            pos = state.positions[sym]
            if pos.shares > 0:
                fill = cost.fill_sell(float(px))
                state.cash += pos.shares * fill - abs(pos.shares) * float(px) * (cost.commission + cost.tax)
            else:
                fill = cost.fill_buy(float(px))
                state.cash -= (-pos.shares) * fill + abs(pos.shares) * float(px) * cost.commission
            self._record_trade(trades, pos, fill, date, "end_of_data", cost)
            del state.positions[sym]

    @staticmethod
    def _record_trade(trades, pos: PortfolioPosition, exit_fill: float, exit_date, reason, cost):
        gross = pos.side * (exit_fill / pos.entry_price - 1.0)
        net = gross - cost.round_trip_cost()
        trades.append(
            Trade(
                symbol=pos.symbol,
                entry_date=pos.entry_date,
                entry_price=float(pos.entry_price),
                exit_date=exit_date,
                exit_price=float(exit_fill),
                holding_days=int((pd.Timestamp(exit_date) - pd.Timestamp(pos.entry_date)).days),
                pnl_pct=float(net),
                exit_reason=reason,
                entry_signal_date=pos.entry_signal_date,
                side=pos.side,
                weight=float(pos.entry_weight),
                notional=float(abs(pos.shares) * pos.entry_price),
            )
        )


def run_portfolio_backtest(
    targets: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: EngineConfig | None = None,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> tuple[list[Trade], pd.DataFrame]:
    """Convenience wrapper. Returns (trades, equity_curve_df)."""
    return PortfolioExecutor(cfg, initial_capital).run(targets, ohlcv)
