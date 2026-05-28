"""Entry and exit execution — T open based on frozen T-1 signals."""

from __future__ import annotations

import pandas as pd

from src.backtest.engine import CostModel, EngineConfig
from src.live_sim.signals import FrozenSignalSet
from src.live_sim.state import ClosedTrade, Position, SimState


class EntryExecutor:
    """Execute buy entries at T OPEN from frozen T-1 signals."""

    def __init__(self, cost_model: CostModel, engine_cfg: EngineConfig):
        self.cost = cost_model
        self.engine_cfg = engine_cfg

    def execute(
        self,
        frozen: FrozenSignalSet,
        bars_today: pd.DataFrame,
        state: SimState,
    ) -> list[Position]:
        """Execute buy signals from frozen set.

        Args:
            frozen: FrozenSignalSet generated at T-1
            bars_today: OHLCV for T (must have date == frozen.for_execution_date)
            state: current SimState (checked for open positions)

        Returns:
            list of newly opened Position objects

        Raises:
            ValueError: if frozen.for_execution_date != bars_today date
        """
        if bars_today.empty:
            raise ValueError(f"bars_today is empty on {frozen.for_execution_date}")

        unique_dates = bars_today["date"].unique()
        if len(unique_dates) != 1:
            raise ValueError(
                f"bars_today must contain single date, got {len(unique_dates)}: {unique_dates}"
            )

        bars_date = pd.to_datetime(unique_dates[0]).normalize()
        exec_date = frozen.for_execution_date.normalize()

        if bars_date != exec_date:
            raise ValueError(
                f"frozen.for_execution_date ({exec_date}) != bars_today date ({bars_date})"
            )

        entries = []
        for sym in frozen.buys():
            if state.has_position(sym):
                continue

            sym_bar = bars_today[bars_today["symbol"] == sym]
            if sym_bar.empty:
                continue

            open_price = float(sym_bar.iloc[0]["open"])
            entry_price = self.cost.fill_buy(open_price)

            pos = Position(
                symbol=sym,
                entry_date=pd.Timestamp(frozen.for_execution_date),
                entry_price=entry_price,
                entry_signal_date=pd.Timestamp(frozen.generated_at),
                hold_bars=0,
            )
            entries.append(pos)

        return entries


class ExitEvaluator:
    """Evaluate and execute exit conditions for open positions at T."""

    def __init__(self, cost_model: CostModel, engine_cfg: EngineConfig):
        self.cost = cost_model
        self.engine_cfg = engine_cfg

    def evaluate(
        self,
        state: SimState,
        frozen: FrozenSignalSet,
        bars_today: pd.DataFrame,
    ) -> list[ClosedTrade]:
        """Evaluate exit conditions for all open positions.

        Priority: hard_stop > exit_signal > max_hold

        Args:
            state: SimState with open_positions
            frozen: FrozenSignalSet (for exit signals)
            bars_today: OHLCV for T

        Returns:
            list of ClosedTrade objects
        """
        if bars_today.empty:
            raise ValueError("bars_today is empty")

        exits = []
        for sym, pos in list(state.open_positions.items()):
            sym_bar = bars_today[bars_today["symbol"] == sym]
            if sym_bar.empty:
                continue

            reason = self._evaluate_exit_reason(pos, sym_bar, frozen, state.current_date)

            if reason is not None:
                sym_bar_row = sym_bar.iloc[0]
                exit_price = self.cost.fill_sell(float(sym_bar_row["open"]))
                gross = exit_price / pos.entry_price - 1.0
                net = gross - self.cost.round_trip_cost()
                holding_days = (pd.Timestamp(sym_bar_row["date"]) - pos.entry_date).days

                trade = ClosedTrade(
                    symbol=sym,
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    exit_date=pd.Timestamp(sym_bar_row["date"]),
                    exit_price=exit_price,
                    holding_days=max(0, holding_days),
                    pnl_pct=float(net),
                    exit_reason=reason,
                    entry_signal_date=pos.entry_signal_date,
                    signal_hash="",  # will be filled by reporter
                )
                exits.append(trade)
                state.close_position(sym)

        return exits

    def _evaluate_exit_reason(
        self,
        pos: Position,
        sym_bar: pd.DataFrame,
        frozen: FrozenSignalSet,
        current_date: pd.Timestamp | None,
    ) -> str | None:
        """Determine exit reason by priority: hard_stop > signal > max_hold."""
        if (
            self.engine_cfg.hard_stop_pct is not None
            and pos.hold_bars >= self.engine_cfg.min_hold_bars
        ):
            low = float(sym_bar.iloc[0]["low"])
            mtm_low = low / pos.entry_price - 1.0
            if mtm_low <= self.engine_cfg.hard_stop_pct:
                return "hard_stop"

        if (
            self.engine_cfg.signal_exit_enabled
            and pos.symbol in frozen.sells()
            and pos.hold_bars >= self.engine_cfg.min_hold_bars
        ):
            return "signal"

        if pos.hold_bars >= self.engine_cfg.max_hold_bars:
            return "max_hold"

        return None
