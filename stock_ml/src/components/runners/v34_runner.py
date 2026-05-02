from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pandas as pd

from src.components.base import Trade
from src.config_loader import get_model_config

V34_TARGET: dict[str, Any] = {
    "type": "early_wave",
    "forward_window": 8,
    "short_window": 8,
    "long_window": 20,
    "gain_threshold": 0.06,
    "loss_threshold": 0.03,
    "classes": 3,
}

V34_TRADE_COLUMNS: list[str] = [
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
    "exit_trend",
    "exit_dist_sma20",
    "exit_ret_5d",
    "exit_rsi14",
    "exit_macd_hist",
    "price_max_profit_pct",
    "exit_above_sma20",
    "exit_above_ema8",
    "exit_vol_ratio",
    "symbol",
]


def _exit_model_cfg(model_cfg: dict[str, Any]) -> dict[str, Any] | None:
    exit_model_cfg = model_cfg.get("exit_model") or {}
    return exit_model_cfg if exit_model_cfg.get("enabled") else None


def _build_prediction_cache(
    symbols: list[str],
    *,
    device: str,
    feature_set: str | None = None,
) -> list[dict[str, Any]]:
    from src.pipeline.build_predictions import _build_predictions

    model_cfg = get_model_config("v34")
    return _build_predictions(
        symbols,
        feature_set or model_cfg.get("feature_set", "leading_v4"),
        model_cfg.get("target", V34_TARGET),
        device,
        model_type=model_cfg.get("model_type"),
        exit_model_cfg=_exit_model_cfg(model_cfg),
    )


def _mod_kwargs(mods: dict[str, bool]) -> dict[str, bool]:
    return {
        "mod_a": mods.get("a", True),
        "mod_b": mods.get("b", True),
        "mod_c": mods.get("c", False),
        "mod_d": mods.get("d", False),
        "mod_e": mods.get("e", True),
        "mod_f": mods.get("f", True),
        "mod_g": mods.get("g", True),
        "mod_h": mods.get("h", True),
        "mod_i": mods.get("i", True),
        "mod_j": mods.get("j", True),
    }


def run_v34(
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
) -> list[Trade]:
    del data_dir, first_test_year, last_test_year, train_years
    model_cfg = get_model_config("v34")
    active_mods = {**model_cfg.get("mods", {}), **(mods or {})}
    active_params = {
        **model_cfg.get("params", {}),
        **(params or {}),
        "initial_capital": initial_capital,
        "commission": commission,
        "tax": tax,
        "record_trades": record_trades,
    }
    cache = (
        prediction_cache
        if prediction_cache is not None
        else _build_prediction_cache(symbols, device=device)
    )
    return _run_v34_lineage_cache(
        symbols,
        cache,
        backtest_fn=_backtest_v34,
        mods=active_mods,
        params=active_params,
        entry_reason="v34",
        enable_exit_model=enable_exit_model,
    )


def _backtest_v34(
    y_pred: Any,
    returns: Any,
    df_test: Any,
    feature_cols: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    from src.components.runners.lineage_backtests import backtest_v34

    return backtest_v34(y_pred, returns, df_test, feature_cols, **kwargs)


def _run_v34_lineage_cache(
    symbols: list[str],
    cache: list[dict[str, Any]],
    *,
    backtest_fn: Callable[..., dict[str, Any]],
    mods: dict[str, bool],
    params: dict[str, Any],
    entry_reason: str,
    enable_exit_model: bool = False,
) -> list[Trade]:
    symbol_set = set(symbols)
    all_trades: list[Trade] = []
    for item in cache:
        if str(item["symbol"]) not in symbol_set:
            continue
        all_trades.extend(
            _run_cache_item(
                item,
                backtest_fn=backtest_fn,
                mods=mods,
                params=params,
                entry_reason=entry_reason,
                enable_exit_model=enable_exit_model,
            )
        )
    return all_trades


def _run_cache_item(
    item: dict[str, Any],
    *,
    backtest_fn: Callable[..., dict[str, Any]],
    mods: dict[str, bool],
    params: dict[str, Any],
    entry_reason: str,
    enable_exit_model: bool = False,
) -> list[Trade]:
    sig_params = set(inspect.signature(backtest_fn).parameters)
    mod_kwargs = {k: v for k, v in _mod_kwargs(mods).items() if k in sig_params}
    extra: dict[str, Any] = {}
    if enable_exit_model and item.get("y_pred_exit") is not None and "y_pred_exit" in sig_params:
        extra["y_pred_exit"] = item["y_pred_exit"]
    result = backtest_fn(
        item["y_pred"],
        item["returns"],
        item["sym_test_df"],
        item["feature_cols"],
        **mod_kwargs,
        **params,
        **extra,
    )

    trades: list[Trade] = []
    for raw_trade in result["trades"]:
        trade_data = dict(raw_trade)
        trade_data["symbol"] = str(item["symbol"])
        trades.append(_trade_from_legacy(trade_data, entry_reason=entry_reason))
    return trades


def _trade_from_legacy(trade_data: dict[str, Any], *, entry_reason: str) -> Trade:
    return Trade(
        entry_date=str(trade_data.get("entry_date", "")),
        exit_date=str(trade_data.get("exit_date", "")),
        entry_price=0.0,
        exit_price=0.0,
        pnl_pct=float(trade_data.get("pnl_pct", 0.0)),
        holding_days=int(trade_data.get("holding_days", 0)),
        entry_reason=entry_reason,
        exit_reason=str(trade_data.get("exit_reason", "")),
        symbol=str(trade_data.get("symbol", "")),
        metadata=trade_data,
    )


def trades_to_v34_dataframe(trades: list[Trade | dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        if isinstance(trade, Trade):
            meta = trade.metadata
            row = {col: meta.get(col) for col in V34_TRADE_COLUMNS}
            row["holding_days"] = meta.get("holding_days", trade.holding_days)
            row["pnl_pct"] = meta.get("pnl_pct", trade.pnl_pct)
            row["exit_reason"] = meta.get("exit_reason", trade.exit_reason)
            row["exit_date"] = meta.get("exit_date", trade.exit_date)
            row["symbol"] = meta.get("symbol", trade.symbol)
        else:
            row = {col: trade.get(col) for col in V34_TRADE_COLUMNS}
        rows.append(row)
    return pd.DataFrame(rows, columns=V34_TRADE_COLUMNS)
