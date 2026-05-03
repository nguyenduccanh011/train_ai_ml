from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.backtest.defaults import DEFAULT_PARAMS
from src.backtest.indicators import (
    compute_indicators,
)
from src.backtest.indicators import (
    detect_trend_strength as detect_v22_trend_strength,
)
from src.backtest.indicators import (
    get_regime_adapter as get_v22_regime_adapter,
)
from src.components.base import BarContext, Position, Trade
from src.components.fusion.helpers import (
    compute_v19_indicators,
)
from src.components.fusion.helpers import (
    detect_trend_strength as detect_v19_trend_strength,
)
from src.components.fusion.helpers import (
    get_regime_adapter as get_v19_regime_adapter,
)
from src.components.fusion.strategies import build_exit_strategies
from src.components.fusion.strategies.core import (
    AdaptiveTrailing,
    AtrStopLoss,
    ExitModelExit,
    FastExitLossLegacy,
    HardStopExit,
    MaCrossHybridExit,
    MinHoldProtection,
    PeakProtectDist,
    PeakProtectEma8Streak,
    ProfitLock,
    SignalHardCapExit,
    V22FastExit,
    V22HardCap,
    ZombieExit,
)
from src.components.fusion.strategies.entry import V19EntryCascade
from src.components.fusion.strategies.hold import LongHorizonCarry, V19SignalHoldGuard
from src.components.runners import _sim_utils

_atr_stop = _sim_utils.atr_stop
_format_date = _sim_utils.format_date
_track_result = _sim_utils.track_result

if TYPE_CHECKING:
    from src.components.fusion.base import FusionStrategy
    from src.pipeline.config import StrategyV3Config


DEFAULT_FUSION_MODS: dict[str, bool] = {
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

DEFAULT_V19_MODS = DEFAULT_FUSION_MODS
DEFAULT_V22_MODS = DEFAULT_FUSION_MODS

V22_DEFAULTS: dict[str, Any] = {
    **DEFAULT_PARAMS,
    "v22_mode": True,
    "v22_fast_exit_skip_strong": True,
    "v22_fast_exit_vol_confirm": True,
    "v22_fast_exit_threshold_hb": -0.07,
    "v22_fast_exit_threshold_std": -0.05,
    "v22_adaptive_hard_cap": True,
    "v22_hard_cap_mult_hb": 3.0,
    "v22_hard_cap_mult_std": 2.5,
    "v22_hard_cap_floor": 0.12,
    "v22_hard_cap_floor_hb": 0.15,
}

V22_FORCE_EXIT_STRATEGY_NAMES: tuple[str, ...] = ("hard_stop_exit", "v22_hard_cap")
V22_ACTIVE_EXIT_STRATEGY_NAMES: tuple[str, ...] = (
    "v22_fast_exit",
    "atr_stop_loss",
    "peak_protect_dist",
    "peak_protect_ema8_streak",
    "ma_cross_hybrid_exit",
    "adaptive_trailing",
    "profit_lock",
    "zombie_exit",
)

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

V22_TRADE_COLUMNS: list[str] = [
    "entry_day",
    "exit_day",
    "holding_days",
    "pnl_pct",
    "max_profit_pct",
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
]


@dataclass(frozen=True)
class FusionRunnerDef:
    version_key: str
    entry_reason: str
    feature_set_default: str
    indicator_fn: Callable[..., dict[str, Any]]
    detect_trend_fn: Callable[..., str]
    regime_adapter_fn: Callable[..., dict[str, Any]]
    force_exit_factory: Callable[[bool, StrategyV3Config | None], list[FusionStrategy]]
    active_exit_factory: Callable[[bool, StrategyV3Config | None], list[FusionStrategy]]
    defaults: dict[str, Any] = field(default_factory=dict)
    include_params_in_ctx: bool = False
    include_symbol_profile: bool = False
    include_strong_uptrend: bool = False
    include_atr_ratio_now: bool = False
    use_long_horizon_carry: bool = False
    patch_open_indicator: bool = False


def _atr_ratio(ind: dict[str, Any], i: int) -> float:
    atr14 = ind["atr14"]
    close = ind["close"]
    if close[i] > 0 and not np.isnan(atr14[i]):
        return float(atr14[i] / close[i])
    return 0.03


def _base_ctx(
    *,
    defn: FusionRunnerDef,
    i: int,
    df: pd.DataFrame,
    pred: int,
    position: Position | None,
    ind: dict[str, Any],
    mods: dict[str, bool],
    params: dict[str, Any],
    trend: str,
    regime_cfg: dict[str, Any],
    trade_state: dict[str, Any] | None = None,
    entry_state: dict[str, Any] | None = None,
    exit_signal_override: int | None = None,
) -> BarContext:
    config = {
        "indicators": ind,
        "mods": mods,
        "trend": trend,
        "regime_cfg": regime_cfg,
        "trade_state": trade_state or {},
        "entry_state": entry_state or {},
    }
    if defn.include_params_in_ctx:
        config["params"] = params

    kwargs: dict[str, Any] = {}
    if defn.include_symbol_profile:
        kwargs["symbol_profile"] = str(regime_cfg.get("profile", "balanced"))

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
        config=config,
        **kwargs,
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


def _make_v19_force_exit_strategies(
    enable_exit_model: bool = False,
    strategy_v3: StrategyV3Config | None = None,
) -> list[FusionStrategy]:
    if strategy_v3 is not None and strategy_v3.force_exit_rules:
        rule_names = list(strategy_v3.force_exit_rules)
        if enable_exit_model and "exit_model" not in rule_names:
            rule_names.insert(1, "exit_model")
        if not enable_exit_model:
            rule_names = [name for name in rule_names if name != "exit_model"]
        return build_exit_strategies(rule_names)

    strategies: list[FusionStrategy] = [HardStopExit()]
    if enable_exit_model:
        strategies.append(ExitModelExit())
    strategies.extend(
        [
            SignalHardCapExit(),
            FastExitLossLegacy(),
            AtrStopLoss(),
            PeakProtectDist(),
        ]
    )
    return strategies


def _make_v19_active_exit_strategies(
    enable_exit_model: bool = False,
    strategy_v3: StrategyV3Config | None = None,
) -> list[FusionStrategy]:
    del enable_exit_model
    if strategy_v3 is not None and strategy_v3.active_exit_rules:
        return build_exit_strategies(strategy_v3.active_exit_rules)
    return [
        PeakProtectEma8Streak(),
        MaCrossHybridExit(),
        AdaptiveTrailing(),
        ProfitLock(),
        ZombieExit(),
    ]


def _make_v22_force_exit_strategies(
    enable_exit_model: bool = False,
    strategy_v3: StrategyV3Config | None = None,
) -> list[FusionStrategy]:
    del enable_exit_model
    if strategy_v3 is not None and strategy_v3.force_exit_rules:
        return build_exit_strategies(strategy_v3.force_exit_rules)
    return [
        HardStopExit(),
        V22HardCap(),
    ]


def _make_v22_active_exit_strategies(
    enable_exit_model: bool = False,
    strategy_v3: StrategyV3Config | None = None,
) -> list[FusionStrategy]:
    if strategy_v3 is not None and strategy_v3.active_exit_rules:
        rule_names = list(strategy_v3.active_exit_rules)
        if enable_exit_model and "exit_model" not in rule_names:
            rule_names.insert(0, "exit_model")
        if not enable_exit_model:
            rule_names = [name for name in rule_names if name != "exit_model"]
        return build_exit_strategies(rule_names)

    strategies: list[FusionStrategy] = []
    if enable_exit_model:
        strategies.append(ExitModelExit())
    strategies.extend(
        [
            V22FastExit(),
            AtrStopLoss(),
            PeakProtectDist(),
            PeakProtectEma8Streak(),
            MaCrossHybridExit(),
            AdaptiveTrailing(),
            ProfitLock(),
            ZombieExit(),
        ]
    )
    return strategies


def build_fusion_prediction_cache(
    defn: FusionRunnerDef,
    symbols: list[str],
    *,
    device: str,
    feature_set: str | None = None,
) -> list[dict[str, Any]]:
    from src.pipeline.build_predictions import _build_predictions

    target_cfg = {
        "type": "trend_regime",
        "trend_method": "dual_ma",
        "short_window": 5,
        "long_window": 20,
        "classes": 3,
    }
    return _build_predictions(
        symbols,
        feature_set or defn.feature_set_default,
        target_cfg,
        device,
        model_type="lightgbm",
        exit_model_cfg={"forward_window": 15, "loss_threshold": 0.05},
    )


def run_fusion(
    defn: FusionRunnerDef,
    symbols: list[str],
    data_dir: str,
    *,
    mods: dict[str, bool] | None = None,
    params: dict[str, Any] | None = None,
    first_test_year: int = 2020,
    last_test_year: int = 2025,
    train_years: int = 4,
    device: str = "cpu",
    prediction_cache: list[dict[str, Any]] | None = None,
    initial_capital: float = 100_000_000,
    commission: float = 0.0015,
    tax: float = 0.001,
    record_trades: bool = True,
    enable_exit_model: bool = False,
    strategy_v3: StrategyV3Config | None = None,
) -> list[Trade]:
    del data_dir, first_test_year, last_test_year, train_years
    active_mods = {**DEFAULT_FUSION_MODS, **(mods or {})}
    active_params = {**defn.defaults, **(params or {})}
    cache = prediction_cache or build_fusion_prediction_cache(defn, symbols, device=device)

    force_exit_strategies = defn.force_exit_factory(enable_exit_model, strategy_v3)
    active_exit_strategies = defn.active_exit_factory(enable_exit_model, strategy_v3)

    all_trades: list[Trade] = []
    for item in cache:
        if str(item["symbol"]) not in symbols:
            continue
        all_trades.extend(
            _run_cache_item(
                defn,
                item,
                mods=active_mods,
                params=active_params,
                initial_capital=initial_capital,
                commission=commission,
                tax=tax,
                record_trades=record_trades,
                enable_exit_model=enable_exit_model,
                force_exit_strategies=force_exit_strategies,
                active_exit_strategies=active_exit_strategies,
            )
        )
    return all_trades


def _trend(defn: FusionRunnerDef, ind: dict[str, Any], i: int) -> str:
    if defn.version_key == "v19_3":
        return defn.detect_trend_fn(ind, i)
    return defn.detect_trend_fn(i, ind)


def _regime(
    defn: FusionRunnerDef, symbol: str, ind: dict[str, Any], i: int, trend: str
) -> dict[str, Any]:
    if defn.version_key == "v19_3":
        return defn.regime_adapter_fn(symbol, ind, i, trend)
    return defn.regime_adapter_fn(i, trend, ind, patch_symbol_tuning=False)


def _run_cache_item(
    defn: FusionRunnerDef,
    item: dict[str, Any],
    *,
    mods: dict[str, bool],
    params: dict[str, Any],
    initial_capital: float,
    commission: float,
    tax: float,
    record_trades: bool,
    enable_exit_model: bool = False,
    force_exit_strategies: list[FusionStrategy] | None = None,
    active_exit_strategies: list[FusionStrategy] | None = None,
) -> list[Trade]:  # noqa: PLR0912, PLR0915, C901
    y_pred = np.asarray(item["y_pred"])
    y_pred_exit = (
        np.asarray(item["y_pred_exit"])
        if enable_exit_model and item.get("y_pred_exit") is not None
        else None
    )
    returns = np.asarray(item["returns"])
    df = item["sym_test_df"].reset_index(drop=True)
    symbol = str(item["symbol"])

    n = min(len(y_pred), len(returns), len(df))
    if y_pred_exit is not None:
        n = min(n, len(y_pred_exit))
    equity = np.zeros(n)
    equity[0] = initial_capital

    ind = defn.indicator_fn(df, mod_e=mods.get("e", True))
    if defn.patch_open_indicator:
        ind["open"] = ind["opn"]
    entry_strategy = V19EntryCascade()
    min_hold = MinHoldProtection()
    signal_hold_guard = V19SignalHoldGuard()
    long_horizon_carry = LongHorizonCarry() if defn.use_long_horizon_carry else None
    if force_exit_strategies is None:
        force_exit_strategies = defn.force_exit_factory(enable_exit_model, None)
    if active_exit_strategies is None:
        active_exit_strategies = defn.active_exit_factory(enable_exit_model, None)

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

        trend = _trend(defn, ind, i)
        regime_cfg = _regime(defn, symbol, ind, i, trend)

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
                defn=defn,
                i=i,
                df=df,
                pred=pred,
                position=None,
                ind=ind,
                mods=mods,
                params=params,
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
        if defn.include_strong_uptrend:
            trade_state["strong_uptrend"] = trend == "strong"
        if defn.include_atr_ratio_now:
            trade_state["atr_ratio_now"] = _atr_ratio(ind, i)

        ctx = _base_ctx(
            defn=defn,
            i=i,
            df=df,
            pred=pred,
            position=position,
            ind=ind,
            mods=mods,
            params=params,
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
            reason = _run_exit_sequence(active_exit_strategies, ctx, counters)
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

        if pending_exit_reason is not None and long_horizon_carry is not None:
            trade_state["pending_exit_reason"] = pending_exit_reason
            carry_res = long_horizon_carry.apply(ctx)
            _track_result(carry_res, counters)
            if carry_res.action == "keep_position":
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
                    defn=defn,
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
                defn=defn,
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
    defn: FusionRunnerDef,
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
        entry_reason=defn.entry_reason,
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


def trades_to_v22_dataframe(trades: list[Trade]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        meta = trade.metadata
        row = {
            "entry_day": meta.get("entry_day"),
            "exit_day": meta.get("exit_day"),
            "holding_days": trade.holding_days,
            "pnl_pct": trade.pnl_pct,
            "max_profit_pct": meta.get("max_profit_pct"),
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
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=V22_TRADE_COLUMNS)


FUSION_RUNNER_DEFS: dict[str, FusionRunnerDef] = {
    "v19_3": FusionRunnerDef(
        version_key="v19_3",
        entry_reason="v19_3",
        feature_set_default="leading",
        indicator_fn=compute_v19_indicators,
        detect_trend_fn=detect_v19_trend_strength,
        regime_adapter_fn=get_v19_regime_adapter,
        force_exit_factory=_make_v19_force_exit_strategies,
        active_exit_factory=_make_v19_active_exit_strategies,
    ),
    "v22": FusionRunnerDef(
        version_key="v22",
        entry_reason="v22",
        feature_set_default="leading_v2",
        indicator_fn=compute_indicators,
        detect_trend_fn=detect_v22_trend_strength,
        regime_adapter_fn=get_v22_regime_adapter,
        force_exit_factory=_make_v22_force_exit_strategies,
        active_exit_factory=_make_v22_active_exit_strategies,
        defaults=V22_DEFAULTS,
        include_params_in_ctx=True,
        include_symbol_profile=True,
        include_strong_uptrend=True,
        include_atr_ratio_now=True,
        use_long_horizon_carry=True,
        patch_open_indicator=True,
    ),
    "v22_with_exit_model": FusionRunnerDef(
        version_key="v22",
        entry_reason="v22",
        feature_set_default="leading_v2",
        indicator_fn=compute_indicators,
        detect_trend_fn=detect_v22_trend_strength,
        regime_adapter_fn=get_v22_regime_adapter,
        force_exit_factory=_make_v22_force_exit_strategies,
        active_exit_factory=_make_v22_active_exit_strategies,
        defaults=V22_DEFAULTS,
        include_params_in_ctx=True,
        include_symbol_profile=True,
        include_strong_uptrend=True,
        include_atr_ratio_now=True,
        use_long_horizon_carry=True,
        patch_open_indicator=True,
    ),
}


__all__ = [
    "DEFAULT_V19_MODS",
    "DEFAULT_V22_MODS",
    "FUSION_RUNNER_DEFS",
    "FusionRunnerDef",
    "V19_TRADE_COLUMNS",
    "V22_ACTIVE_EXIT_STRATEGY_NAMES",
    "V22_DEFAULTS",
    "V22_FORCE_EXIT_STRATEGY_NAMES",
    "V22_TRADE_COLUMNS",
    "build_fusion_prediction_cache",
    "run_fusion",
    "trades_to_v19_3_dataframe",
    "trades_to_v22_dataframe",
]
