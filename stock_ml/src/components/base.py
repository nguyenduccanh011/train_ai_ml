from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

ActionType = Literal["enter_long", "exit", "hold"]
FusionActionType = Literal["pass", "skip_entry", "enter", "exit", "modify_hold"]


@dataclass(slots=True)
class Position:
    symbol: str
    entry_idx: int
    entry_date: pd.Timestamp
    entry_price: float
    size: float = 1.0
    holding_days: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl_pct: float
    holding_days: int
    entry_reason: str
    exit_reason: str
    symbol: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Action:
    bar_idx: int
    type: ActionType
    size: float = 1.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BarContext:
    bar_idx: int
    df_test: pd.DataFrame
    entry_signal: int
    entry_proba: np.ndarray | None
    exit_signal: int | None
    exit_proba: np.ndarray | None
    position: Position | None
    config: dict[str, Any]
    symbol_profile: str | None = None


@dataclass(slots=True)
class FusionResult:
    action: FusionActionType
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
