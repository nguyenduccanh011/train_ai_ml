"""Year-based walk-forward splitter — strict no-leakage.

Definition for test year `Y` with `train_years=N`, `gap_days=G`:
  - train = rows with date in [Y-N, Y) **trimmed at the end by `G` calendar days**
  - test  = rows with date in [Y, Y+test_years)

The `gap_days` trim is critical: forward-looking targets (e.g. 5-bar forward
return) computed at the tail of `train` would otherwise peek into `test`.
Choose `gap_days >= target.forward_window * timeframe_days` (e.g. 15 bars × 1d
plus safety margin = 25 days).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitWindow:
    test_year: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive

    @property
    def label(self) -> str:
        return f"test_{self.test_year}"


@dataclass
class YearSplitter:
    train_years: int = 4
    test_years: int = 1
    gap_days: int = 25
    first_test_year: int = 2020
    last_test_year: int = 2025

    def __post_init__(self) -> None:
        if self.train_years < 1:
            raise ValueError("train_years must be >= 1")
        if self.test_years < 1:
            raise ValueError("test_years must be >= 1")
        if self.gap_days < 0:
            raise ValueError("gap_days must be >= 0")
        if self.first_test_year > self.last_test_year:
            raise ValueError("first_test_year must be <= last_test_year")

    def windows(self) -> list[SplitWindow]:
        out: list[SplitWindow] = []
        for y in range(self.first_test_year, self.last_test_year + 1, self.test_years):
            test_start = pd.Timestamp(year=y, month=1, day=1)
            test_end = pd.Timestamp(year=y + self.test_years, month=1, day=1)
            train_end_inclusive = test_start - pd.Timedelta(days=self.gap_days)
            train_end = train_end_inclusive + pd.Timedelta(days=1)  # exclusive
            train_start = pd.Timestamp(year=y - self.train_years, month=1, day=1)
            out.append(
                SplitWindow(
                    test_year=y,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
        return out

    def split(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> Iterator[tuple[SplitWindow, pd.DataFrame, pd.DataFrame]]:
        if date_col not in df.columns:
            raise ValueError(f"DataFrame missing '{date_col}' column")
        dates = pd.to_datetime(df[date_col])
        for w in self.windows():
            train_mask = (dates >= w.train_start) & (dates < w.train_end)
            test_mask = (dates >= w.test_start) & (dates < w.test_end)
            yield w, df.loc[train_mask].copy(), df.loc[test_mask].copy()
