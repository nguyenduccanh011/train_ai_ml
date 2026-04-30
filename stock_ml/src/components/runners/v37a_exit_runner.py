from __future__ import annotations

from typing import Any

import pandas as pd

from src.components.base import Trade
from src.components.runners._lineage_v34 import run_lineage
from src.components.runners.runner_registry import RUNNER_DEFS
from src.components.runners.v34_runner import trades_to_v34_dataframe

_DEF = RUNNER_DEFS["v37a_exit"]


def run_v37a_exit(
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
) -> list[Trade]:
    return run_lineage(
        _DEF,
        symbols,
        data_dir,
        mods=mods,
        params=params,
        first_test_year=first_test_year,
        last_test_year=last_test_year,
        train_years=train_years,
        device=device,
        prediction_cache=prediction_cache,
        initial_capital=initial_capital,
        commission=commission,
        tax=tax,
        record_trades=record_trades,
        enable_model_b_exit=enable_model_b_exit,
    )


def trades_to_v37a_exit_dataframe(trades: list[Trade | dict[str, Any]]) -> pd.DataFrame:
    return trades_to_v34_dataframe(trades)
