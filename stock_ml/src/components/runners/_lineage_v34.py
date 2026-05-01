from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.components.base import Trade
from src.components.runners.v34_runner import (
    V34_TARGET,
    _exit_model_cfg,
    _run_v34_lineage_cache,
    trades_to_v34_dataframe,
)
from src.config_loader import get_model_config
from src.pipeline.config import StrategyV3Config


@dataclass(frozen=True)
class RunnerDef:
    version_key: str
    backtest_fn: Callable[..., dict[str, Any]]
    entry_reason: str
    feature_set_default: str = "leading_v4"
    target_default: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(V34_TARGET))


def build_prediction_cache(
    defn: RunnerDef,
    symbols: list[str],
    *,
    device: str,
    feature_set: str | None = None,
) -> list[dict[str, Any]]:
    from src.pipeline.build_predictions import _build_predictions

    model_cfg = get_model_config(defn.version_key)
    return _build_predictions(
        symbols,
        feature_set or model_cfg.get("feature_set", defn.feature_set_default),
        model_cfg.get("target", defn.target_default),
        device,
        model_type=model_cfg.get("model_type"),
        exit_model_cfg=_exit_model_cfg(model_cfg),
        model_extras=model_cfg.get("entry_model", {}).get("extras", {}),
    )


def run_lineage(
    defn: RunnerDef,
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
    enable_model_b_exit: bool = False,
    strategy_v3: StrategyV3Config | None = None,
) -> list[Trade]:
    del data_dir, first_test_year, last_test_year, train_years
    model_cfg = get_model_config(defn.version_key)
    strategy_mods = strategy_v3.mods if strategy_v3 is not None else {}
    strategy_params = strategy_v3.params if strategy_v3 is not None else {}
    active_mods = {**model_cfg.get("mods", {}), **strategy_mods, **(mods or {})}
    active_params = {
        **model_cfg.get("params", {}),
        **strategy_params,
        **(params or {}),
        "initial_capital": initial_capital,
        "commission": commission,
        "tax": tax,
        "record_trades": record_trades,
    }
    cache = (
        prediction_cache
        if prediction_cache is not None
        else build_prediction_cache(defn, symbols, device=device)
    )
    return _run_v34_lineage_cache(
        symbols,
        cache,
        backtest_fn=defn.backtest_fn,
        mods=active_mods,
        params=active_params,
        entry_reason=defn.entry_reason,
        enable_model_b_exit=enable_model_b_exit,
    )


__all__ = [
    "RunnerDef",
    "build_prediction_cache",
    "run_lineage",
    "trades_to_v34_dataframe",
]
