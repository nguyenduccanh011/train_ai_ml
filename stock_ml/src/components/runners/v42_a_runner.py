from __future__ import annotations

from typing import Any

import pandas as pd

from src.components.base import Trade
from src.components.runners.v34_runner import (
    V34_TARGET,
    _exit_model_cfg,
    _run_v34_lineage_cache,
    trades_to_v34_dataframe,
)
from src.config_loader import get_model_config


def _build_prediction_cache(
    symbols: list[str],
    *,
    device: str,
    feature_set: str | None = None,
) -> list[dict[str, Any]]:
    from run_pipeline import _build_predictions

    model_cfg = get_model_config("v42_a")
    return _build_predictions(
        symbols,
        feature_set or model_cfg.get("feature_set", "leading_v4"),
        model_cfg.get("target", V34_TARGET),
        device,
        model_type=model_cfg.get("model_type"),
        exit_model_cfg=_exit_model_cfg(model_cfg),
    )


def run_v42_a(
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
) -> list[Trade]:
    del data_dir, first_test_year, last_test_year, train_years
    model_cfg = get_model_config("v42_a")
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
        backtest_fn=_backtest_v42_a,
        mods=active_mods,
        params=active_params,
        entry_reason="v42_a",
    )


def _backtest_v42_a(
    y_pred: Any,
    returns: Any,
    df_test: Any,
    feature_cols: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    from experiments.run_v42 import backtest_v42

    return backtest_v42(y_pred, returns, df_test, feature_cols, **kwargs)


def trades_to_v42_a_dataframe(trades: list[Trade | dict[str, Any]]) -> pd.DataFrame:
    return trades_to_v34_dataframe(trades)
