from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .defaults import DEFAULT_PARAMS
from .pnl import get_pnl_calculator

HARD_STOP = 0.08
ZOMBIE_BARS = 14


def _format_trade_time(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.time() == pd.Timestamp(ts.date()).time():
        return ts.date().isoformat()
    return ts.isoformat()


@dataclass
class PositionState:
    symbol: str
    position: int
    position_size: float
    entry_equity: float
    entry_close: float
    entry_day: int
    entry_date: str
    notional: float
    max_equity: float
    peak_portfolio_equity: float
    cumulative_pnl: float = 0.0
    hold_days: int = 0
    last_roll_expiry: Any = None


@dataclass
class PortfolioState:
    total_equity: float
    maintenance_margin_rate: float
    leverage: float
    positions: dict[str, PositionState] = field(default_factory=dict)

    @property
    def total_notional(self) -> float:
        return sum(abs(pos.notional) for pos in self.positions.values())

    @property
    def total_initial_margin(self) -> float:
        return self.total_notional / max(self.leverage, 1.0)

    @property
    def total_maintenance_margin(self) -> float:
        return self.total_notional * self.maintenance_margin_rate

    @property
    def available_margin(self) -> float:
        return self.total_equity - self.total_initial_margin

    @property
    def total_short_notional(self) -> float:
        return sum(abs(pos.notional) for pos in self.positions.values() if pos.position == -1)


def _to_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(np.isnan(arr), 0.0, arr)


def _dates(df: pd.DataFrame, n: int) -> np.ndarray:
    if "date" in df.columns:
        return pd.to_datetime(df["date"]).to_numpy()
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"]).to_numpy()
    return np.arange(n)


def _close(df: pd.DataFrame, n: int) -> np.ndarray:
    if "close" in df.columns:
        return _to_array(df["close"].to_numpy())
    return np.ones(n)


def _price_column(df: pd.DataFrame, column: str, n: int) -> np.ndarray:
    if column in df.columns:
        return _to_array(df[column].to_numpy())
    return _close(df, n)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(window, min_periods=1).mean().to_numpy(dtype=float)


def _column_or_zeros(df: pd.DataFrame, column: str | None, n: int) -> np.ndarray:
    if column and column in df.columns:
        return _to_array(df[column].to_numpy())
    return np.zeros(n)


def _position_pnl_pct(pnl_calc, pos: PositionState, exit_price: float, slippage: float) -> float:
    pnl_pct = pnl_calc.trade_return(pos.entry_close, exit_price, slippage) * 100
    if pos.position == -1:
        pnl_pct = -pnl_pct
    return pnl_pct


def _should_roll(
    i: int,
    pos: PositionState,
    signal: int,
    expiry_dates: np.ndarray | None,
    dates: np.ndarray,
    roll_rule: str | None,
    roll_days_before_expiry: int,
    next_volume: np.ndarray,
    current_volume: np.ndarray,
    next_oi: np.ndarray,
    current_oi: np.ndarray,
) -> bool:
    if expiry_dates is None or signal != pos.position:
        return False
    expiry = pd.Timestamp(expiry_dates[i])
    if pd.isna(expiry):
        return False
    if pos.last_roll_expiry is not None and expiry == pd.Timestamp(pos.last_roll_expiry):
        return False
    date = pd.Timestamp(dates[i])
    if date >= expiry:
        return True
    if roll_rule == "volume_crossover":
        return next_volume[i] > current_volume[i]
    if roll_rule == "oi_crossover":
        return next_oi[i] > current_oi[i]
    if roll_rule == "n_days_before_expiry":
        return (expiry.normalize() - date.normalize()).days <= roll_days_before_expiry
    return False


def _trade_record(
    pnl_calc,
    pos: PositionState,
    i: int,
    exit_date: Any,
    exit_price: float,
    exit_reason: str,
    slippage: float,
) -> dict[str, Any]:
    pnl_pct = _position_pnl_pct(pnl_calc, pos, exit_price, slippage)
    max_profit_pct = 0.0
    if pos.entry_equity > 0:
        max_profit_pct = (pos.max_equity - pos.entry_equity) / pos.entry_equity * 100
    return {
        "entry_day": pos.entry_day,
        "exit_day": i,
        "holding_days": i - pos.entry_day,
        "pnl_pct": round(pnl_pct, 2),
        "max_profit_pct": round(max_profit_pct, 2),
        "exit_reason": exit_reason,
        "exit_date": _format_trade_time(exit_date),
        "entry_date": pos.entry_date,
        "entry_symbol": pos.symbol,
        "position_size": pos.position_size,
    }


def backtest_portfolio(
    predictions: dict[str, np.ndarray],
    returns: dict[str, np.ndarray],
    dfs: dict[str, pd.DataFrame],
    feature_cols: list[str] | None = None,
    **config,
) -> dict[str, Any]:
    cfg = {**DEFAULT_PARAMS, **config}
    initial_capital = float(cfg["initial_capital"])
    commission = float(cfg["commission"])
    tax = float(cfg["tax"])
    slippage = float(cfg["slippage"])
    leverage = float(cfg["leverage"])
    maintenance_margin_rate = float(cfg["maintenance_margin_rate"])
    liquidation_fee = float(cfg["liquidation_fee"])
    contract_multiplier = float(cfg["contract_multiplier"])
    short_enabled = bool(cfg["short_enabled"])
    short_hard_cap = cfg.get("short_hard_cap")
    short_squeeze_exit = bool(cfg["short_squeeze_exit"])
    short_squeeze_vol_mult = float(cfg["short_squeeze_vol_mult"])
    short_squeeze_price_pct = float(cfg["short_squeeze_price_pct"])
    fast_exit_weak = float(cfg["fast_exit_weak"])
    atr_position_sizing = bool(cfg["atr_position_sizing"])
    atr_risk_target = float(cfg["atr_risk_target"])
    max_total_short_notional = cfg.get("max_total_short_notional")
    max_total_short_notional = (
        float(max_total_short_notional) if max_total_short_notional is not None else None
    )
    max_short_notional = cfg.get("max_short_notional")
    max_short_notional = float(max_short_notional) if max_short_notional is not None else None
    roll_rule = cfg["roll_rule"]
    roll_days_before_expiry = int(cfg["roll_days_before_expiry"])
    pnl_calc = get_pnl_calculator(cfg["pnl_mode"])

    symbols = [sym for sym in predictions if sym in returns and sym in dfs]
    if not symbols:
        return {
            "equity_curve": np.array([], dtype=float),
            "symbol_equity": {},
            "trades": [],
            "total_return_pct": 0.0,
            "final_equity": round(initial_capital),
        }

    lengths = [min(len(predictions[sym]), len(returns[sym]), len(dfs[sym])) for sym in symbols]
    n = min(lengths)
    if n == 0:
        return {
            "equity_curve": np.array([], dtype=float),
            "symbol_equity": {sym: np.array([], dtype=float) for sym in symbols},
            "trades": [],
            "total_return_pct": 0.0,
            "final_equity": round(initial_capital),
        }

    pred_arrays = {sym: np.asarray(predictions[sym], dtype=int)[:n] for sym in symbols}
    ret_arrays = {sym: _to_array(returns[sym])[:n] for sym in symbols}
    close_arrays = {sym: _close(dfs[sym], n)[:n] for sym in symbols}
    open_arrays = {sym: _price_column(dfs[sym], "open", n)[:n] for sym in symbols}
    volume_arrays = {sym: _column_or_zeros(dfs[sym], "volume", n)[:n] for sym in symbols}
    next_volume_arrays = {
        sym: _column_or_zeros(dfs[sym], cfg["next_volume_column"], n)[:n] for sym in symbols
    }
    oi_arrays = {sym: _column_or_zeros(dfs[sym], "open_interest", n)[:n] for sym in symbols}
    next_oi_arrays = {
        sym: _column_or_zeros(dfs[sym], cfg["next_oi_column"], n)[:n] for sym in symbols
    }
    avg_volume20_arrays = {sym: _rolling_mean(volume_arrays[sym], 20)[:n] for sym in symbols}
    atr_arrays = {sym: _column_or_zeros(dfs[sym], "atr", n)[:n] for sym in symbols}
    date_arrays = {sym: _dates(dfs[sym], n)[:n] for sym in symbols}
    funding_arrays = {
        sym: _column_or_zeros(dfs[sym], cfg["funding_rate_column"], n)[:n] for sym in symbols
    }
    borrow_arrays = {
        sym: _column_or_zeros(dfs[sym], cfg["borrow_rate_column"], n)[:n] for sym in symbols
    }
    borrow_available = {
        sym: _column_or_zeros(dfs[sym], cfg["borrow_available_column"], n)[:n]
        if cfg["borrow_available_column"]
        else None
        for sym in symbols
    }
    expiry_arrays = {
        sym: pd.to_datetime(dfs[sym][cfg["expiry_date_column"]]).to_numpy()[:n]
        if cfg["pnl_mode"] == "futures_contract"
        and cfg["expiry_date_column"]
        and cfg["expiry_date_column"] in dfs[sym].columns
        else None
        for sym in symbols
    }

    state = PortfolioState(initial_capital, maintenance_margin_rate, leverage)
    equity_curve = np.zeros(n)
    equity_curve[0] = initial_capital
    symbol_equity = {sym: np.zeros(n) for sym in symbols}
    trades: list[dict[str, Any]] = []

    for i in range(1, n):
        prev_equity = state.total_equity
        bar_pnl = 0.0
        financing_cost = 0.0
        for sym, pos in state.positions.items():
            signed_ret = ret_arrays[sym][i] if pos.position == 1 else -ret_arrays[sym][i]
            pnl_change = pos.notional * pnl_calc.bar_return(signed_ret, 1.0)
            bar_pnl += pnl_change
            if funding_arrays[sym][i] != 0:
                signed_funding = (
                    funding_arrays[sym][i] if pos.position == 1 else -funding_arrays[sym][i]
                )
                financing_cost += abs(pos.notional) * signed_funding
            if pos.position == -1 and borrow_arrays[sym][i] != 0:
                financing_cost += abs(pos.notional) * borrow_arrays[sym][i]
            pos.hold_days += 1
            pos.cumulative_pnl += pnl_change
            pos.max_equity = max(pos.max_equity, pos.entry_equity + pos.cumulative_pnl)

        state.total_equity = prev_equity + bar_pnl - financing_cost

        while state.positions and state.total_equity < state.total_maintenance_margin:
            worst_sym, worst_pos = min(
                state.positions.items(),
                key=lambda item: _position_pnl_pct(
                    pnl_calc, item[1], close_arrays[item[0]][i], slippage
                ),
            )
            exit_notional = abs(worst_pos.notional)
            cost = (
                pnl_calc.exit_cost(exit_notional, commission, tax, slippage)
                + exit_notional * liquidation_fee
            )
            state.total_equity -= cost
            trades.append(
                _trade_record(
                    pnl_calc,
                    worst_pos,
                    i,
                    date_arrays[worst_sym][i],
                    close_arrays[worst_sym][i],
                    "cross_margin_liquidation",
                    slippage,
                )
            )
            del state.positions[worst_sym]

        for sym in symbols:
            signal = pred_arrays[sym][i]
            pos = state.positions.get(sym)
            if pos is not None:
                exit_reason = "signal" if signal == 0 else None
                expiry_dates = expiry_arrays[sym]
                roll_happened = _should_roll(
                    i,
                    pos,
                    signal,
                    expiry_dates,
                    date_arrays[sym],
                    roll_rule,
                    roll_days_before_expiry,
                    next_volume_arrays[sym],
                    volume_arrays[sym],
                    next_oi_arrays[sym],
                    oi_arrays[sym],
                )
                if roll_happened:
                    roll_notional = abs(pos.notional)
                    roll_cost_fn = getattr(pnl_calc, "compute_roll_cost", None)
                    roll_cost = (
                        roll_cost_fn(roll_notional, float(cfg["roll_cost_rate"]))
                        if roll_cost_fn
                        else 0.0
                    )
                    state.total_equity -= (
                        pnl_calc.exit_cost(roll_notional, commission, tax, slippage)
                        + roll_cost
                        + pnl_calc.entry_cost(roll_notional, commission, slippage)
                    )
                    pos.entry_close = close_arrays[sym][i]
                    pos.entry_day = i
                    pos.entry_date = _format_trade_time(date_arrays[sym][i])
                    pos.entry_equity = pos.notional / max(leverage, 1.0) - roll_cost
                    pos.max_equity = pos.entry_equity
                    pos.cumulative_pnl = 0.0
                    pos.hold_days = 0
                    pos.last_roll_expiry = expiry_dates[i]
                    continue
                if pos.position == -1:
                    close_now = close_arrays[sym][i]
                    open_now = open_arrays[sym][i]
                    price_cur_ret = (pos.entry_close / close_now - 1) if close_now > 0 else 0.0
                    portfolio_ret = (
                        (state.total_equity - pos.peak_portfolio_equity) / pos.peak_portfolio_equity
                        if pos.peak_portfolio_equity > 0
                        else 0.0
                    )
                    if portfolio_ret <= -HARD_STOP:
                        exit_reason = "hard_stop"
                    elif short_hard_cap is not None and price_cur_ret >= float(short_hard_cap):
                        exit_reason = "signal_hard_cap"
                    elif pos.hold_days >= 3 and price_cur_ret >= abs(fast_exit_weak):
                        exit_reason = "fast_exit_profit"
                    elif pos.hold_days >= ZOMBIE_BARS and price_cur_ret < 0.01:
                        exit_reason = "zombie_exit"
                    elif short_squeeze_exit:
                        avg_volume_now = avg_volume20_arrays[sym][i]
                        vol_ratio = (
                            volume_arrays[sym][i] / avg_volume_now if avg_volume_now > 0 else 0.0
                        )
                        bar_ret = (close_now / open_now - 1) if open_now > 0 else 0.0
                        if (
                            vol_ratio >= short_squeeze_vol_mult
                            and bar_ret >= short_squeeze_price_pct
                        ):
                            exit_reason = "short_squeeze"
                    if borrow_available[sym] is not None and borrow_available[sym][i] <= 0:
                        exit_reason = exit_reason or "borrow_recalled"
                if exit_reason:
                    exit_notional = abs(pos.notional)
                    state.total_equity -= pnl_calc.exit_cost(
                        exit_notional, commission, tax, slippage
                    )
                    trades.append(
                        _trade_record(
                            pnl_calc,
                            pos,
                            i,
                            date_arrays[sym][i],
                            close_arrays[sym][i],
                            exit_reason,
                            slippage,
                        )
                    )
                    del state.positions[sym]
                continue

            if signal == 0:
                continue
            direction = 1 if signal > 0 else -1
            if direction == -1:
                if not short_enabled:
                    continue
                if borrow_available[sym] is not None and borrow_available[sym][i] <= 0:
                    continue

            position_size = float(
                cfg["short_position_size"]
                if direction == -1 and cfg["short_position_size"] is not None
                else cfg.get("position_size", 1.0)
            )
            if direction == -1 and atr_position_sizing:
                atr_now = atr_arrays[sym][i]
                close_now = close_arrays[sym][i]
                if atr_now > 0 and close_now > 0:
                    position_size = atr_risk_target / (atr_now / close_now)
            position_size = max(0.0, min(position_size, 1.0))
            margin_equity = state.total_equity * position_size
            notional = margin_equity * max(leverage, 1.0) * contract_multiplier
            if direction == -1 and max_short_notional is not None and notional > max_short_notional:
                notional = max_short_notional
                margin_equity = notional / max(leverage, 1.0)
                position_size = (
                    margin_equity / state.total_equity if state.total_equity > 0 else 0.0
                )
            if notional <= 0:
                continue
            required_margin = notional / max(leverage, 1.0)
            if state.available_margin < required_margin:
                continue
            if direction == -1 and max_total_short_notional is not None:
                if state.total_short_notional + notional > max_total_short_notional:
                    continue

            entry_cost = pnl_calc.entry_cost(notional, commission, slippage)
            if state.available_margin < required_margin + entry_cost:
                continue
            state.total_equity -= entry_cost
            entry_equity = margin_equity - entry_cost
            state.positions[sym] = PositionState(
                symbol=sym,
                position=direction,
                position_size=position_size,
                entry_equity=entry_equity,
                entry_close=close_arrays[sym][i],
                entry_day=i,
                entry_date=_format_trade_time(date_arrays[sym][i]),
                notional=notional,
                max_equity=entry_equity,
                peak_portfolio_equity=state.total_equity,
            )

        equity_curve[i] = state.total_equity
        for sym in symbols:
            pos = state.positions.get(sym)
            symbol_equity[sym][i] = state.total_equity * pos.position_size if pos else 0.0

    for sym, pos in list(state.positions.items()):
        exit_notional = abs(pos.notional)
        state.total_equity -= pnl_calc.exit_cost(exit_notional, commission, tax, slippage)
        trades.append(
            _trade_record(
                pnl_calc,
                pos,
                n - 1,
                date_arrays[sym][n - 1],
                close_arrays[sym][n - 1],
                "end",
                slippage,
            )
        )

    equity_curve[-1] = state.total_equity
    return {
        "equity_curve": equity_curve,
        "symbol_equity": symbol_equity,
        "trades": trades,
        "total_return_pct": round((state.total_equity / initial_capital - 1) * 100, 2),
        "final_equity": round(state.total_equity),
    }
