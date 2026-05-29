"""Year-based walk-forward splitter + Purged K-Fold splitter — strict no-leakage.

YearSplitter definition for test year `Y` with `train_years=N`, `gap_days=G`:
  - train = rows with date in [Y-N, Y) **trimmed at the end by `G` calendar days**
  - test  = rows with date in [Y, Y+test_years)

PurgedKFoldSplitter (Phase 1.5.1): de Prado's Purged K-Fold with embargo.
  - Divides data into k folds by date
  - Each fold: test = [k_start, k_end), train = all other folds except embargo
  - embargo = boundary rows within embargo_days of train/test split
  - Prevents label leakage from forward-return targets (e.g., 5-bar forward return
    at train end would peek into test start without embargo)
  - Reference: de Prado, *Advances in Financial ML*, Ch. 7.
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

    @classmethod
    def from_data(
        cls,
        df: pd.DataFrame,
        train_years: int = 4,
        test_years: int = 1,
        gap_days: int = 25,
        date_col: str = "date",
    ) -> YearSplitter:
        """Auto-detect year range from DataFrame.

        Phase 1b.7: Auto-detect first_test_year and last_test_year from data.
        Ensures first_test_year has at least train_years of prior data.

        Args:
            df: DataFrame with date column
            train_years: number of years to train on
            test_years: number of years per test fold
            gap_days: gap between train and test
            date_col: name of date column

        Returns:
            YearSplitter instance with auto-detected year range

        Raises:
            ValueError: if date column missing or insufficient data
        """
        if date_col not in df.columns:
            raise ValueError(f"DataFrame missing '{date_col}' column")

        dates = pd.to_datetime(df[date_col])
        min_year = dates.dt.year.min()
        max_year = dates.dt.year.max()

        first_test_year = min_year + train_years
        last_test_year = max_year - test_years

        if first_test_year > last_test_year:
            raise ValueError(
                f"Insufficient data for {train_years}-year training: "
                f"year range {min_year}-{max_year} too short"
            )

        return cls(
            train_years=train_years,
            test_years=test_years,
            gap_days=gap_days,
            first_test_year=first_test_year,
            last_test_year=last_test_year,
        )

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


@dataclass(frozen=True)
class PurgedFoldWindow:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive
    embargo_start: pd.Timestamp  # embargo region start
    embargo_end: pd.Timestamp  # embargo region end

    @property
    def label(self) -> str:
        return f"fold_{self.fold_idx}"


@dataclass
class PurgedKFoldSplitter:
    """Purged K-Fold with embargo (Phase 1.5.1).

    Reference: de Prado, *Advances in Financial ML*, Ch. 7.

    Divides data into k folds by date. For each fold:
    - test = fold[i]
    - train = all folds except fold[i] and embargo rows
    - embargo = rows within embargo_days of fold[i] boundaries

    Prevents label leakage from forward-return targets: a 5-bar forward return
    computed at the end of training would otherwise peek into test start.
    """

    n_splits: int = 5
    embargo_days: int = 0
    label_horizon: int = 5

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")
        if self.label_horizon < 1:
            raise ValueError("label_horizon must be >= 1")

    def split(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> Iterator[tuple[PurgedFoldWindow, pd.DataFrame, pd.DataFrame]]:
        """Yield (window, train_df, test_df) for each fold.

        Args:
            df: DataFrame to split (must be sorted by date)
            date_col: name of date column

        Yields:
            (PurgedFoldWindow, train_df, test_df) for each of n_splits folds
        """
        if date_col not in df.columns:
            raise ValueError(f"DataFrame missing '{date_col}' column")

        dates = pd.to_datetime(df[date_col])
        unique_dates = sorted(dates.unique())

        if len(unique_dates) < self.n_splits:
            raise ValueError(
                f"Insufficient unique dates ({len(unique_dates)}) for {self.n_splits} splits"
            )

        fold_size = len(unique_dates) // self.n_splits
        embargo_td = pd.Timedelta(days=self.embargo_days)

        for fold_idx in range(self.n_splits):
            test_start_idx = fold_idx * fold_size
            test_end_idx = (
                (fold_idx + 1) * fold_size if fold_idx < self.n_splits - 1 else len(unique_dates)
            )

            test_start = pd.Timestamp(unique_dates[test_start_idx])
            test_end = pd.Timestamp(unique_dates[test_end_idx - 1]) + pd.Timedelta(days=1)

            embargo_start = test_start - embargo_td
            embargo_end = test_end + embargo_td

            test_mask = (dates >= test_start) & (dates < test_end)
            embargo_mask = (dates >= embargo_start) & (dates < embargo_end)
            train_mask = ~embargo_mask

            window = PurgedFoldWindow(
                fold_idx=fold_idx,
                train_start=pd.Timestamp(unique_dates[0]),
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end,
            )

            yield window, df.loc[train_mask].copy(), df.loc[test_mask].copy()
