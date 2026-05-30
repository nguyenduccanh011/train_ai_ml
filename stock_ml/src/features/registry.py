"""Feature set registry — manages feature definitions and builders.

Feature sets are named collections of features with a builder function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.features.basic import FEATURE_COLS as BASIC_COLS
from src.features.basic import add_features as add_basic_features
from src.features.leading_v2 import FEATURE_COLS as LEADING_V2_COLS
from src.features.leading_v2 import add_features as add_leading_v2_features
from src.features.leading_v3 import leading_v3_features

LEADING_V3_COLS: list[str] = LEADING_V2_COLS + [
    # Group A: Cross-sectional rank (5)
    "momentum_rank",
    "volatility_rank",
    "volume_rank",
    "rsi_rank",
    "price_strength_rank",
    # Group B: Sector-relative metrics (6)
    "return_vs_sector",
    "momentum_vs_sector",
    "volume_vs_sector",
    "volatility_vs_sector",
    "strength_vs_sector",
    "beta_to_sector",
    # Group C: Market regime interaction (4)
    "market_trend",
    "market_volatility_regime",
    "regime_interaction_momentum",
    "regime_interaction_strength",
    # Group D: Liquidity filters (4)
    "volume_20d_avg",
    "volume_rank_20d",
    "price_level",
    "volume_stability",
]


@dataclass
class FeatureSet:
    """Named feature set with columns and builder function."""

    name: str
    columns: list[str]
    builder: Callable[[pd.DataFrame], pd.DataFrame]


_REGISTRY: dict[str, FeatureSet] = {}


def _build_basic_v1(df: pd.DataFrame) -> pd.DataFrame:
    """Builder for basic_v1 feature set."""
    return add_basic_features(df)


def register_feature_set(
    name: str, columns: list[str], builder: Callable[[pd.DataFrame], pd.DataFrame]
) -> None:
    """Register a new feature set."""
    _REGISTRY[name] = FeatureSet(name=name, columns=columns, builder=builder)


def get_feature_cols(feature_set: str) -> list[str]:
    """Get list of feature column names for a feature set.

    Args:
        feature_set: key in registry (e.g., 'basic_v1', 'leading_v2')

    Returns:
        list of column names

    Raises:
        KeyError: if feature_set not found
    """
    if feature_set not in _REGISTRY:
        raise KeyError(f"Unknown feature set: {feature_set}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[feature_set].columns


def apply_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Apply feature builder to dataframe.

    Args:
        df: input DataFrame with [symbol, date, open, high, low, close, volume]
        feature_set: key in registry

    Returns:
        DataFrame with feature columns added

    Raises:
        KeyError: if feature_set not found
    """
    if feature_set not in _REGISTRY:
        raise KeyError(f"Unknown feature set: {feature_set}. Available: {sorted(_REGISTRY.keys())}")
    fs = _REGISTRY[feature_set]
    return fs.builder(df)


def _build_leading_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Builder for leading_v2 feature set."""
    return add_leading_v2_features(df)


def _build_leading_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Builder for leading_v3 feature set."""
    return leading_v3_features(df)


register_feature_set("basic_v1", BASIC_COLS, _build_basic_v1)
register_feature_set(
    "leading_v2",
    LEADING_V2_COLS,
    _build_leading_v2,
)
register_feature_set(
    "leading_v3",
    LEADING_V3_COLS,
    _build_leading_v3,
)
