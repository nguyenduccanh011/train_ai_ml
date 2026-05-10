"""WalkForwardWalker — thin wrapper around legacy WalkForwardSplitter using SplitConfig."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import pandas as pd

from src.data.splitter import SplitWindow, WalkForwardSplitter

if TYPE_CHECKING:
    from src.pipeline.config import SplitConfig


def build_splitter(split_cfg: SplitConfig) -> WalkForwardSplitter:
    return WalkForwardSplitter(
        method=split_cfg.method,
        train_years=split_cfg.train_years,
        test_years=split_cfg.test_years,
        gap_days=split_cfg.gap_days,
        first_test_year=split_cfg.first_test_year,
        last_test_year=split_cfg.last_test_year,
    )


def walk_forward(
    df: pd.DataFrame,
    split_cfg: SplitConfig,
    time_col: str = "timestamp",
) -> Generator[tuple[SplitWindow, pd.DataFrame, pd.DataFrame], None, None]:
    """Yield (window, train_df, test_df) for each walk-forward fold."""
    splitter = build_splitter(split_cfg)
    yield from splitter.split(df, time_col=time_col)
