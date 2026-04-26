from __future__ import annotations

from typing import Protocol

import pandas as pd


class FeatureBlock(Protocol):
    """One unit of feature computation."""

    name: str
    requires: list[str]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return an enriched DataFrame."""

    def get_feature_names(self) -> list[str]:
        """Return feature columns produced by this block."""
