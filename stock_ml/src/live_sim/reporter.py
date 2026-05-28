"""Reporter — daily logs and final summary."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.backtest.stats import per_year_stats
from src.evaluation.scoring import calc_metrics, composite_score
from src.live_sim.signals import FrozenSignalSet
from src.live_sim.state import ClosedTrade, Position, SimState


class Reporter:
    """Write daily logs, signal logs, and final summary."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.daily_log_path = self.out_dir / "live_sim_daily_log.csv"
        self.signal_log_path = self.out_dir / "signals_log.csv"
        self.trades_path = self.out_dir / "trades.csv"

    def write_signal_log(self, frozen: FrozenSignalSet) -> None:
        """Append frozen signals to log."""
        rows = []
        for sym, signal in frozen.signals.items():
            rows.append(
                {
                    "generated_at": frozen.generated_at.isoformat(),
                    "for_execution_date": frozen.for_execution_date.isoformat(),
                    "symbol": sym,
                    "signal": signal,
                    "integrity_hash": frozen.integrity_hash[:8],
                }
            )

        if not rows:
            return

        df = pd.DataFrame(rows)
        if not self.signal_log_path.exists():
            df.to_csv(self.signal_log_path, index=False)
        else:
            df.to_csv(self.signal_log_path, mode="a", header=False, index=False)

    def write_day(
        self,
        date: pd.Timestamp,
        entries: list[Position],
        exits: list[ClosedTrade],
        frozen: FrozenSignalSet,
        state: SimState,
    ) -> None:
        """Append daily summary."""
        realized_pnl = sum(t.pnl_pct for t in exits)

        row = {
            "date": date.isoformat(),
            "n_entries": len(entries),
            "n_exits": len(exits),
            "n_open_positions": state.n_open(),
            "realized_pnl_today": realized_pnl,
            "signal_hash": frozen.integrity_hash[:8],
            "n_buy_signals": frozen.n_buy,
            "n_sell_signals": frozen.n_sell,
            "n_neutral_signals": frozen.n_neutral,
        }

        df = pd.DataFrame([row])
        if not self.daily_log_path.exists():
            df.to_csv(self.daily_log_path, index=False)
        else:
            df.to_csv(self.daily_log_path, mode="a", header=False, index=False)

    def write_trades_csv(self, state: SimState) -> None:
        """Write all closed trades to CSV."""
        if not state.closed_trades:
            trades_df = pd.DataFrame(
                columns=[
                    "symbol",
                    "entry_date",
                    "entry_price",
                    "exit_date",
                    "exit_price",
                    "holding_days",
                    "pnl_pct",
                    "exit_reason",
                    "entry_signal_date",
                ]
            )
        else:
            trades_df = pd.DataFrame(
                {
                    "symbol": [t.symbol for t in state.closed_trades],
                    "entry_date": [t.entry_date for t in state.closed_trades],
                    "entry_price": [t.entry_price for t in state.closed_trades],
                    "exit_date": [t.exit_date for t in state.closed_trades],
                    "exit_price": [t.exit_price for t in state.closed_trades],
                    "holding_days": [t.holding_days for t in state.closed_trades],
                    "pnl_pct": [t.pnl_pct for t in state.closed_trades],
                    "exit_reason": [t.exit_reason for t in state.closed_trades],
                    "entry_signal_date": [t.entry_signal_date for t in state.closed_trades],
                }
            )

        trades_df.to_csv(self.trades_path, index=False)

    def write_summary(
        self, state: SimState, sim_config: object, sim_start: str, sim_end: str
    ) -> dict:
        """Write final summary and return dict."""
        self.write_trades_csv(state)

        trades_list = [
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date.isoformat(),
                "exit_date": t.exit_date.isoformat(),
                "pnl_pct": t.pnl_pct,
                "holding_days": t.holding_days,
            }
            for t in state.closed_trades
        ]

        metrics = (
            calc_metrics(trades_list)
            if trades_list
            else {
                "trades": 0,
                "wr": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "pf": 0.0,
                "max_loss": 0.0,
                "avg_hold": 0.0,
            }
        )
        composite = composite_score(metrics, trades_list) if trades_list else 0.0

        trades_df = pd.read_csv(self.trades_path) if self.trades_path.exists() else pd.DataFrame()

        daily_log_df = (
            pd.read_csv(self.daily_log_path) if self.daily_log_path.exists() else pd.DataFrame()
        )
        yearly_df = per_year_stats(trades_df) if not trades_df.empty else pd.DataFrame()

        if yearly_df is not None and not yearly_df.empty:
            yearly_df.to_csv(self.out_dir / "yearly_stats.csv", index=False)

        summary = {
            "name": "live_sim",
            "sim_period": {"start": sim_start, "end": sim_end},
            "n_symbols": len(set(t["symbol"] for t in trades_list)) if trades_list else 0,
            "n_trades": int(metrics["trades"]),
            "metrics": {
                "total_pnl": float(metrics["total_pnl"]),
                "avg_pnl": float(metrics["avg_pnl"]),
                "win_rate": float(metrics["wr"]),
                "profit_factor": float(metrics["pf"]),
                "avg_hold_days": float(metrics["avg_hold"]),
            },
            "composite_score": float(composite),
            "outputs": {
                "trades_csv": str(self.trades_path),
                "daily_log_csv": str(self.daily_log_path),
                "signals_log_csv": str(self.signal_log_path),
                "yearly_stats_csv": str(self.out_dir / "yearly_stats.csv"),
            },
            "generated_at": datetime.now(UTC).isoformat(),
        }

        summary_path = self.out_dir / "summary_live_sim.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )

        return summary
