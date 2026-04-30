from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.components.base import BarContext, Position, Trade
from src.components.fusion.helpers import (
    compute_v19_indicators,
    detect_trend_strength,
    get_regime_adapter,
)
from src.components.fusion.strategies.core import (
    AdaptiveTrailing,
    AtrStopLoss,
    FastExitLossLegacy,
    HardStopExit,
    MaCrossHybridExit,
    MinHoldProtection,
    ModelBExit,
    PeakProtectDist,
    PeakProtectEma8Streak,
    ProfitLock,
    SignalHardCapExit,
    ZombieExit,
)
from src.components.fusion.strategies.entry import V19EntryCascade
from src.components.fusion.strategies.hold import V19SignalHoldGuard
from src.components.runners import _sim_utils

_atr_stop = _sim_utils.atr_stop
_format_date = _sim_utils.format_date
_track_result = _sim_utils.track_result

if TYPE_CHECKING:
    from src.components.fusion.base import FusionStrategy


DEFAULT_V19_MODS: dict[str, bool] = {
    "a": True,
    "b": True,
    "c": False,
    "d": False,
    "e": True,
    "f": True,
    "g": True,
    "h": True,
    "i": True,
    "j": True,
}

V19_TRADE_COLUMNS: list[str] = [
    "entry_day",
    "exit_day",
    "holding_days",
    "pnl_pct",
    "exit_reason",
    "exit_date",
    "entry_wp",
    "entry_dp",
    "entry_rs",
    "entry_vs",
    "entry_bs",
    "entry_hl",
    "entry_od",
    "entry_bb",
    "entry_score",
    "entry_date",
    "entry_symbol",
    "position_size",
    "entry_trend",
    "quick_reentry",
    "breakout_entry",
    "vshape_entry",
    "entry_ret_5d",
    "entry_drop20d",
    "entry_dist_sma20",
    "entry_profile",
    "entry_choppy_regime",
    "symbol",
    "max_profit_pct",
]


def _base_ctx(
    *,
    i: int,
    df: pd.DataFrame,
    pred: int,
    position: Position | None,
    ind: dict[str, Any],
    mods: dict[str, bool],
    trend: str,
    regime_cfg: dict[str, Any],
    trade_state: dict[str, Any] | None = None,
    entry_state: dict[str, Any] | None = None,
    exit_signal_override: int | None = None,
) -> BarContext:
    return BarContext(
        bar_idx=i,
        df_test=df,
        entry_signal=pred,
        entry_proba=None,
        exit_signal=exit_signal_override
        if exit_signal_override is not None
        else (1 if pred == 0 else 0),
        exit_proba=None,
        position=position,
        config={
            "indicators": ind,
            "mods": mods,
            "trend": trend,
            "regime_cfg": regime_cfg,
            "trade_state": trade_state or {},
            "entry_state": entry_state or {},
        },
    )


def _run_exit_sequence(
    strategies: list[FusionStrategy],
    ctx: BarContext,
    counters: Counter[str],
) -> str | None:
    ts = ctx.config["trade_state"]
    for strat in strategies:
        res = strat.apply(ctx)
        _track_result(res, counters)
        if res.metadata.get("hybrid_block_trailing"):
            ts["hybrid_block_trailing"] = True
        if res.action == "exit":
            return res.reason
    return None


def _make_force_exit_strategies(enable_model_b_exit: bool = False) -> list[FusionStrategy]:
    strategies: list[FusionStrategy] = [HardStopExit()]
    if enable_model_b_exit:
        strategies.append(ModelBExit())
    strategies.extend(
        [
            SignalHardCapExit(),
            FastExitLossLegacy(),
            AtrStopLoss(),
            PeakProtectDist(),
        ]
    )
    return strategies


def _make_continuation_exit_strategies() -> list[FusionStrategy]:
    return [
        PeakProtectEma8Streak(),
        MaCrossHybridExit(),
        AdaptiveTrailing(),
        ProfitLock(),
        ZombieExit(),
    ]


def _build_prediction_cache(
    symbols: list[str],
    *,
    device: str,
    feature_set: str = "leading",
) -> list[dict[str, Any]]:
    from run_pipeline import _build_predictions

    target_cfg = {
        "type": "trend_regime",
        "trend_method": "dual_ma",
        "short_window": 5,
        "long_window": 20,
        "classes": 3,
    }
    return _build_predictions(
        symbols,
        feature_set,
        target_cfg,
        device,
        model_type="lightgbm",
        exit_model_cfg={"forward_window": 15, "loss_threshold": 0.05},
    )


def run_v19_3(
    symbols: list[str],
    data_dir: str,
    *,
    mods: dict[str, bool] | None = None,
    first_test_year: int = 2020,
    last_test_year: int = 2025,
    train_years: int = 4,
    device: str = "cpu",
    prediction_cache: list[dict[str, Any]] | None = None,
    initial_capital: float = 100_000_000,
    commission: float = 0.0015,
    tax: float = 0.001,
    record_trades: bool = True,
    enable_model_b_exit: bool = False,
) -> list[Trade]:
    del data_dir, first_test_year, last_test_year, train_years
    active_mods = {**DEFAULT_V19_MODS, **(mods or {})}
    cache = prediction_cache or _build_prediction_cache(symbols, device=device)

    all_trades: list[Trade] = []
    for item in cache:
        if str(item["symbol"]) not in symbols:
            continue
        all_trades.extend(
            _run_cache_item(
                item,
                mods=active_mods,
                initial_capital=initial_capital,
                commission=commission,
                tax=tax,
                record_trades=record_trades,
                enable_model_b_exit=enable_model_b_exit,
            )
        )
    return all_trades


def _run_cache_item(
    item: dict[str, Any],
    *,
    mods: dict[str, bool],
    initial_capital: float,
    commission: float,
    tax: float,
    record_trades: bool,
    enable_model_b_exit: bool = False,
) -> list[Trade]:  # noqa: PLR0912, PLR0915, C901
    y_pred = np.asarray(item["y_pred"])
    y_pred_exit = (
        np.asarray(item["y_pred_exit"])
        if enable_model_b_exit and item.get("y_pred_exit") is not None
        else None
    )
    returns = np.asarray(item["returns"])
    df = item["sym_test_df"].reset_index(drop=True)
    symbol = str(item["symbol"])

    n = len(y_pred)
    equity = np.zeros(n)
    equity[0] = initial_capital

    ind = compute_v19_indicators(df, mod_e=mods.get("e", True))
    entry_strategy = V19EntryCascade()
    min_hold = MinHoldProtection()
    signal_hold_guard = V19SignalHoldGuard()
    force_exit_strategies = _make_force_exit_strategies(enable_model_b_exit)
    continuation_exit_strategies = _make_continuation_exit_strategies()

    trades: list[Trade] = []
    counters: Counter[str] = Counter()
    position: Position | None = None
    cooldown_remaining = 0
    last_exit_price = 0.0
    last_exit_reason = ""
    last_exit_bar = -999

    close = ind["close"]
    dates = ind["dates"]

    for i in range(1, n):
        pred = int(y_pred[i - 1])
        exit_signal_val: int | None = int(y_pred_exit[i - 1]) if y_pred_exit is not None else None
        ret = float(returns[i]) if not np.isnan(returns[i]) else 0.0
        raw_signal = 1 if pred == 1 else 0
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        trend = detect_trend_strength(ind, i)
        regime_cfg = get_regime_adapter(symbol, ind, i, trend)

        if position is None:
            entry_state = {
                "cooldown_remaining": cooldown_remaining,
                "last_exit_price": last_exit_price,
                "last_exit_reason": last_exit_reason,
                "last_exit_bar": last_exit_bar,
                "prev_pred": int(y_pred[i - 2]) if i >= 2 else 0,
                "trend": trend,
            }
            ctx = _base_ctx(
                i=i,
                df=df,
                pred=pred,
                position=None,
                ind=ind,
                mods=mods,
                trend=trend,
                regime_cfg=regime_cfg,
                entry_state=entry_state,
                exit_signal_override=exit_signal_val,
            )
            res = entry_strategy.apply(ctx)
            _track_result(res, counters)
            if res.action == "enter":
                size = float(res.metadata.get("size", 1.0))
                deploy = equity[i - 1] * size
                cost = deploy * commission
                entry_equity = deploy - cost
                position = Position(
                    symbol=symbol,
                    entry_idx=i,
                    entry_date=pd.Timestamp(dates[i]),
                    entry_price=float(close[i]),
                    size=size,
                    holding_days=0,
                    entry_close=float(close[i]),
                    entry_equity=entry_equity,
                    max_equity_in_trade=entry_equity,
                    max_price_in_trade=float(close[i]),
                    metadata=dict(res.metadata.get("entry_features", {})),
                    strategy_state={
                        "consecutive_exit_signals": 0,
                        "consecutive_below_ema8": 0,
                    },
                )
                equity[i] = equity[i - 1] - cost
            else:
                equity[i] = equity[i - 1]
            continue

        projected = equity[i - 1] * (1 + ret * position.size)
        position.max_equity_in_trade = max(position.max_equity_in_trade, projected)
        if close[i] > position.max_price_in_trade:
            position.max_price_in_trade = float(close[i])

        cum_ret = (
            (projected - position.entry_equity) / position.entry_equity
            if position.entry_equity > 0
            else 0.0
        )
        max_profit = (
            (position.max_equity_in_trade - position.entry_equity) / position.entry_equity
            if position.entry_equity > 0
            else 0.0
        )
        price_max_profit = (
            position.max_price_in_trade / position.entry_close - 1
            if position.entry_close > 0
            else 0.0
        )
        price_cur_ret = close[i] / position.entry_close - 1 if position.entry_close > 0 else 0.0

        trade_state = {
            "raw_signal": raw_signal,
            "cum_ret": cum_ret,
            "max_profit": max_profit,
            "price_max_profit": price_max_profit,
            "price_cur_ret": price_cur_ret,
            "hold_days": position.holding_days,
            "trend": trend,
            "atr_stop": _atr_stop(ind, i),
            "consecutive_exit_signals": int(
                position.strategy_state.get("consecutive_exit_signals", 0)
            ),
        }
        ctx = _base_ctx(
            i=i,
            df=df,
            pred=pred,
            position=position,
            ind=ind,
            mods=mods,
            trend=trend,
            regime_cfg=regime_cfg,
            trade_state=trade_state,
            exit_signal_override=exit_signal_val,
        )

        pending_exit_reason: str | None = "signal" if raw_signal == 0 else None
        reason = _run_exit_sequence(force_exit_strategies, ctx, counters)
        if reason is not None:
            pending_exit_reason = reason
        elif raw_signal == 1:
            reason = _run_exit_sequence(continuation_exit_strategies, ctx, counters)
            if reason is not None:
                pending_exit_reason = reason

        if pending_exit_reason is not None:
            trade_state["pending_exit_reason"] = pending_exit_reason
            min_hold_res = min_hold.apply(ctx)
            _track_result(min_hold_res, counters)
            if min_hold_res.action == "keep_position":
                pending_exit_reason = None
            elif pending_exit_reason == "signal":
                guard_res = signal_hold_guard.apply(ctx)
                _track_result(guard_res, counters)
                new_consec = guard_res.metadata.get("consecutive_exit_signals")
                if isinstance(new_consec, int):
                    position.strategy_state["consecutive_exit_signals"] = new_consec
                if guard_res.action == "keep_position":
                    pending_exit_reason = None

        if pending_exit_reason is None:
            equity[i] = projected
            position.holding_days += 1
            continue

        cost = equity[i - 1] * position.size * (commission + tax)
        pnl_pct_now = (
            (close[i] / position.entry_close - 1) * 100 if position.entry_close > 0 else 0.0
        )
        cooldown_remaining = 5 if pnl_pct_now < -5 else 3
        last_exit_price = float(close[i])
        last_exit_reason = pending_exit_reason
        last_exit_bar = i

        if record_trades and position.entry_equity > 0:
            trades.append(
                _make_trade(
                    position=position,
                    symbol=symbol,
                    exit_idx=i,
                    exit_date=dates[i],
                    exit_price=float(close[i]),
                    exit_reason=pending_exit_reason,
                    max_profit_pct=round(max_profit * 100, 2),
                )
            )
        equity[i] = projected - cost
        position = None

    if position is not None and position.entry_equity > 0 and record_trades:
        trades.append(
            _make_trade(
                position=position,
                symbol=symbol,
                exit_idx=n - 1,
                exit_date=dates[-1],
                exit_price=float(close[-1]),
                exit_reason="end",
                max_profit_pct=None,
            )
        )

    return trades


def _make_trade(
    *,
    position: Position,
    symbol: str,
    exit_idx: int,
    exit_date: object,
    exit_price: float,
    exit_reason: str,
    max_profit_pct: float | None,
) -> Trade:
    pnl_pct = (exit_price / position.entry_close - 1) * 100 if position.entry_close > 0 else 0.0
    metadata = dict(position.metadata)
    metadata.update(
        {
            "entry_day": position.entry_idx,
            "exit_day": exit_idx,
            "max_profit_pct": max_profit_pct,
        }
    )
    return Trade(
        entry_date=_format_date(position.entry_date),
        exit_date=_format_date(exit_date),
        entry_price=round(position.entry_close, 2),
        exit_price=round(exit_price, 2),
        pnl_pct=round(pnl_pct, 2),
        holding_days=exit_idx - position.entry_idx,
        entry_reason="v19_3",
        exit_reason=exit_reason,
        symbol=symbol,
        metadata=metadata,
    )


def trades_to_v19_3_dataframe(trades: list[Trade]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        meta = trade.metadata
        row = {
            "entry_day": meta.get("entry_day"),
            "exit_day": meta.get("exit_day"),
            "holding_days": trade.holding_days,
            "pnl_pct": trade.pnl_pct,
            "exit_reason": trade.exit_reason,
            "exit_date": trade.exit_date,
            "entry_wp": meta.get("entry_wp"),
            "entry_dp": meta.get("entry_dp"),
            "entry_rs": meta.get("entry_rs"),
            "entry_vs": meta.get("entry_vs"),
            "entry_bs": meta.get("entry_bs"),
            "entry_hl": meta.get("entry_hl"),
            "entry_od": meta.get("entry_od"),
            "entry_bb": meta.get("entry_bb"),
            "entry_score": meta.get("entry_score"),
            "entry_date": meta.get("entry_date", trade.entry_date),
            "entry_symbol": meta.get("entry_symbol", trade.symbol),
            "position_size": meta.get("position_size"),
            "entry_trend": meta.get("entry_trend"),
            "quick_reentry": meta.get("quick_reentry"),
            "breakout_entry": meta.get("breakout_entry"),
            "vshape_entry": meta.get("vshape_entry"),
            "entry_ret_5d": meta.get("entry_ret_5d"),
            "entry_drop20d": meta.get("entry_drop20d"),
            "entry_dist_sma20": meta.get("entry_dist_sma20"),
            "entry_profile": meta.get("entry_profile"),
            "entry_choppy_regime": meta.get("entry_choppy_regime"),
            "symbol": trade.symbol,
            "max_profit_pct": meta.get("max_profit_pct"),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=V19_TRADE_COLUMNS)
