"""Shared data contracts between the three backtest tiers.

    Alpha  ->  Portfolio construction  ->  Execution / risk

This module intentionally imports nothing from the tiers themselves so it can be a
dependency of all of them without creating import cycles.

Frames (plain pandas DataFrames, so they round-trip to CSV/JSON):

- AlphaFrame        : [symbol, date, score]   (+ optional side_hint, model_id, fold_label)
                      `score` is the CANONICAL continuous alpha — never discretized here.
- TargetWeightFrame : [date, symbol, target_weight, side, rank, gated, score]
                      Portfolio output: the *desired* book per date. Execution turns the
                      delta versus the current book into orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

ALPHA_REQUIRED = ("symbol", "date", "score")
TARGET_WEIGHT_COLUMNS = ("date", "symbol", "target_weight", "side", "rank", "gated", "score")


class AlphaSpec:
    """Validation for the Alpha -> Portfolio contract."""

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in ALPHA_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(
                f"AlphaFrame missing required columns {missing}; got {list(df.columns)}"
            )
        return df


@dataclass
class TargetPosition:
    """One desired position for a (date, symbol) — the record form of a TargetWeightFrame row."""

    date: datetime
    symbol: str
    target_weight: float
    side: int = 0  # sign(target_weight): +1 long, -1 short, 0 flat
    rank: int | None = None
    gated: bool = False
    score: float = 0.0


@dataclass
class EquityPoint:
    """One mark-to-market snapshot of the book (time-stepped execution)."""

    date: datetime
    nav: float
    cash: float
    gross_exposure: float
    net_exposure: float
