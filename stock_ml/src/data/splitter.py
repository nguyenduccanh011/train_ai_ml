"""
Walk-forward and time-based data splitting.
Supports rolling window, expanding window, and simple train/test split.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Generator, Dict, Any
from dataclasses import dataclass


@dataclass
class SplitWindow:
    """A single train/test split window."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    label: str  # e.g. "train_2015-2018_test_2019"

    def __repr__(self):
        return (
            f"Window {self.window_id}: "
            f"Train [{self.train_start.date()} → {self.train_end.date()}] "
            f"Test [{self.test_start.date()} → {self.test_end.date()}]"
        )


class WalkForwardSplitter:
    """
    Walk-forward splitter for time series data.

    Supports:
    - rolling: fixed-size training window slides forward
    - expanding: training window grows, always starts from the same date
    """

    def __init__(
        self,
        method: str = "walk_forward",
        train_years: int = 4,
        test_years: int = 1,
        gap_days: int = 0,
        first_test_year: int = 2019,
        last_test_year: int = 2025,
    ):
        self.method = method
        self.train_years = train_years
        self.test_years = test_years
        self.gap_days = gap_days
        self.first_test_year = first_test_year
        self.last_test_year = last_test_year

    def get_windows(self) -> List[SplitWindow]:
        """Generate all walk-forward windows."""
        windows = []
        window_id = 0

        for test_year in range(self.first_test_year, self.last_test_year + 1):
            test_start = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
            test_end = pd.Timestamp(f"{test_year + self.test_years - 1}-12-31", tz="UTC")

            if self.method == "expanding":
                # Always start from the earliest data
                train_start = pd.Timestamp(
                    f"{self.first_test_year - self.train_years}-01-01", tz="UTC"
                )
            else:
                # Rolling: fixed window size
                train_start = pd.Timestamp(
                    f"{test_year - self.train_years}-01-01", tz="UTC"
                )

            train_end = test_start - pd.Timedelta(days=self.gap_days + 1)

            label = (
                f"train_{train_start.year}-{train_end.year}"
                f"_test_{test_year}"
            )

            windows.append(SplitWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                label=label,
            ))
            window_id += 1

        return windows

    def split(
        self,
        df: pd.DataFrame,
        time_col: str = "timestamp",
    ) -> Generator[Tuple[SplitWindow, pd.DataFrame, pd.DataFrame], None, None]:
        """
        Yield (window, train_df, test_df) for each walk-forward window.

        Args:
            df: DataFrame with a timestamp column
            time_col: name of the timestamp column
        """
        windows = self.get_windows()

        for window in windows:
            train_mask = (
                (df[time_col] >= window.train_start)
                & (df[time_col] <= window.train_end)
            )
            test_mask = (
                (df[time_col] >= window.test_start)
                & (df[time_col] <= window.test_end)
            )

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                continue

            yield window, train_df, test_df

    def summary(self) -> str:
        """Print a summary of all windows."""
        windows = self.get_windows()
        lines = [f"Walk-Forward Split ({self.method}): {len(windows)} windows"]
        for w in windows:
            lines.append(f"  {w}")
        return "\n".join(lines)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WalkForwardSplitter":
        """Create splitter from config dict."""
        split_cfg = config.get("split", config)
        return cls(
            method=split_cfg.get("method", "walk_forward"),
            train_years=split_cfg.get("train_years", 4),
            test_years=split_cfg.get("test_years", 1),
            gap_days=split_cfg.get("gap_days", 0),
            first_test_year=split_cfg.get("first_test_year", 2019),
            last_test_year=split_cfg.get("last_test_year", 2025),
        )
