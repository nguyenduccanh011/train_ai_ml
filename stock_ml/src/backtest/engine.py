"""Leakage-safe backtest engine.

Rules (strict):
  1. Signal at bar `t` triggers entry at OPEN of bar `t+1` (next-bar fill).
  2. Exit signal at bar `t` (or hold expiry, or stop) triggers exit at OPEN of
     bar `t+1` — never the same bar as the decision.
  3. Costs (per side, multiplicative on price): slippage, commission.
     Sell-side tax applied on exit only. Default values come from
     `src.backtest.defaults.DEFAULT_TRADING_COST`.
  4. One open trade per symbol at a time (no pyramiding).
  5. Signals dataframe must come from out-of-sample predictions; the engine
     itself is symmetric and has no opinion on fold boundaries — the splitter
     and target module enforce no-leakage upstream.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import pandas as pd

from src.backtest.defaults import DEFAULT_TRADING_COST


@dataclass
class CostModel:
    commission: float = DEFAULT_TRADING_COST["commission"]
    tax: float = DEFAULT_TRADING_COST["tax"]
    slippage: float = DEFAULT_TRADING_COST["slippage"]

    def fill_buy(self, raw_price: float) -> float:
        return raw_price * (1.0 + self.slippage)

    def fill_sell(self, raw_price: float) -> float:
        return raw_price * (1.0 - self.slippage)

    def round_trip_cost(self) -> float:
        """Sum of fee/tax fractions applied to gross pnl (per-side commission ×2 + tax)."""
        return 2.0 * self.commission + self.tax


@dataclass
class EngineConfig:
    max_hold_bars: int = 20
    min_hold_bars: int = 1
    hard_stop_pct: float | None = -0.08  # close trade if mark-to-market drops below this
    signal_exit_enabled: bool = True  # whether to use model's sell signal (-1) as exit trigger
    exit_priority: list[str] = field(
        default_factory=lambda: ["hard_stop", "signal", "max_hold"]
    )  # order of exit conditions to check
    cost: CostModel = field(default_factory=CostModel)


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float  # effective fill price (after slippage)
    exit_date: pd.Timestamp
    exit_price: float  # effective fill price (after slippage)
    holding_days: int
    pnl_pct: float  # net of all costs, expressed as fraction (0.05 == +5%)
    exit_reason: str  # 'signal' | 'max_hold' | 'hard_stop' | 'end_of_data'
    entry_signal_date: pd.Timestamp  # the bar whose signal triggered entry (entry_date-1)


def _ensure_signals(signals: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "signal"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"signals missing columns: {sorted(missing)}")
    out = signals.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["signal"] = out["signal"].astype(int)
    return out


def _ensure_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv missing columns: {sorted(missing)}")
    out = ohlcv.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _run_symbol(
    sym: str,
    bars: pd.DataFrame,
    sig_map: dict[pd.Timestamp, int],
    cfg: EngineConfig,
) -> list[Trade]:
    trades: list[Trade] = []
    n = len(bars)
    if n < 2:
        return trades

    dates = bars["date"].to_numpy()
    opens = bars["open"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)

    in_pos = False
    entry_idx = -1
    entry_signal_idx = -1
    entry_fill = 0.0

    i = 0
    while i < n - 1:  # need i+1 for next-bar fill
        sig = sig_map.get(pd.Timestamp(dates[i]), 0)
        if not in_pos:
            if sig > 0:
                entry_signal_idx = i
                entry_idx = i + 1
                entry_fill = cfg.cost.fill_buy(opens[entry_idx])
                in_pos = True
                i = entry_idx
                continue
            i += 1
            continue

        # In position. Check exit conditions in configured priority order.
        hold_bars = i - entry_idx
        reason: str | None = None

        for exit_rule in cfg.exit_priority:
            if exit_rule == "hard_stop":
                if cfg.hard_stop_pct is not None and hold_bars >= cfg.min_hold_bars:
                    mtm_low = lows[i] / entry_fill - 1.0
                    if mtm_low <= cfg.hard_stop_pct:
                        reason = "hard_stop"
                        break

            elif exit_rule == "signal":
                if cfg.signal_exit_enabled and sig < 0 and hold_bars >= cfg.min_hold_bars:
                    reason = "signal"
                    break

            elif exit_rule == "max_hold":
                if hold_bars >= cfg.max_hold_bars:
                    reason = "max_hold"
                    break

        if reason is not None:
            exit_idx = min(i + 1, n - 1)
            exit_fill = cfg.cost.fill_sell(opens[exit_idx])
            gross = exit_fill / entry_fill - 1.0
            net = gross - cfg.cost.round_trip_cost()
            trades.append(
                Trade(
                    symbol=sym,
                    entry_date=pd.Timestamp(dates[entry_idx]),
                    entry_price=float(entry_fill),
                    exit_date=pd.Timestamp(dates[exit_idx]),
                    exit_price=float(exit_fill),
                    holding_days=int(exit_idx - entry_idx),
                    pnl_pct=float(net),
                    exit_reason=reason,
                    entry_signal_date=pd.Timestamp(dates[entry_signal_idx]),
                )
            )
            in_pos = False
            i = exit_idx
            continue

        i += 1

    if in_pos:
        exit_idx = n - 1
        exit_fill = cfg.cost.fill_sell(closes[exit_idx])  # forced close on last bar
        gross = exit_fill / entry_fill - 1.0
        net = gross - cfg.cost.round_trip_cost()
        trades.append(
            Trade(
                symbol=sym,
                entry_date=pd.Timestamp(dates[entry_idx]),
                entry_price=float(entry_fill),
                exit_date=pd.Timestamp(dates[exit_idx]),
                exit_price=float(exit_fill),
                holding_days=int(exit_idx - entry_idx),
                pnl_pct=float(net),
                exit_reason="end_of_data",
                entry_signal_date=pd.Timestamp(dates[entry_signal_idx]),
            )
        )
    return trades


def run_backtest(
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: EngineConfig | None = None,
) -> list[Trade]:
    """Execute backtest. Signals and OHLCV must be aligned on (symbol, date)."""
    cfg = cfg or EngineConfig()
    sig = _ensure_signals(signals)
    bars = _ensure_ohlcv(ohlcv)

    trades: list[Trade] = []
    for sym, g in bars.groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        sym_sig = sig[sig["symbol"] == sym]
        sig_map = dict(zip(sym_sig["date"], sym_sig["signal"]))
        trades.extend(_run_symbol(str(sym), g, sig_map, cfg))
    return trades


def trades_to_dataframe(trades: Iterable[Trade]) -> pd.DataFrame:
    rows = [asdict(t) for t in trades]
    if not rows:
        return pd.DataFrame(
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
    df = pd.DataFrame(rows)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["entry_signal_date"] = pd.to_datetime(df["entry_signal_date"])
    return df
