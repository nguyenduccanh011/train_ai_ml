from __future__ import annotations

from typing import Protocol

import pandas as pd

from src.components.base import Trade


class Evaluator(Protocol):
    def calc_metrics(self, trades: list[Trade]) -> dict[str, float]:
        """Calculate summary metrics from trade list."""

    def composite_score(self, metrics: dict[str, float], trades: list[Trade]) -> float:
        """Calculate final composite score."""

    def per_symbol_breakdown(self, trades: list[Trade]) -> pd.DataFrame:
        """Return per-symbol metric breakdown."""

    def per_year_breakdown(self, trades: list[Trade]) -> pd.DataFrame:
        """Return per-year metric breakdown."""
