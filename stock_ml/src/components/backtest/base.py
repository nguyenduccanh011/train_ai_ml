from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import pandas as pd

from src.components.base import Action, Trade


class Backtester(Protocol):
    def run(
        self,
        actions: Iterable[Action],
        df_test: pd.DataFrame,
        *,
        initial_cash: float = 100.0,
        fee_pct: float = 0.001,
    ) -> list[Trade]:
        """Convert actions to closed trades."""
