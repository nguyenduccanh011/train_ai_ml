"""Target registry — manages label definitions for models.

Targets define how to compute labels {-1, 0, 1} from raw OHLCV data.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from src.targets.forward import ForwardReturnTarget
from src.targets.forward_regression import ForwardReturnRegressionTarget


class TargetProtocol(Protocol):
    """Target interface — applies labels to DataFrame."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'target' column with {-1, 0, 1, NaN} labels.

        Args:
            df: DataFrame with [symbol, date, close] columns

        Returns:
            DataFrame with 'target' column added
        """
        ...


class TrendRegimeTarget:
    """Trend regime target — labels based on SMA crossover.

    Buy (1): short SMA > long SMA and close above long SMA
    Sell (-1): short SMA < long SMA and close below long SMA
    Neutral (0): otherwise
    """

    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trend regime labels.

        Args:
            df: DataFrame with [symbol, date, close] columns

        Returns:
            DataFrame with 'target' column added
        """
        out = df.copy()

        def _labels_per_symbol(g):
            g = g.copy()
            close = g["close"]
            sma_short = close.rolling(self.short_window, min_periods=1).mean()
            sma_long = close.rolling(self.long_window, min_periods=1).mean()

            target = pd.Series(0, index=g.index, dtype="int8")
            buy_mask = (sma_short > sma_long) & (close > sma_long)
            sell_mask = (sma_short < sma_long) & (close < sma_long)

            target[buy_mask] = 1
            target[sell_mask] = -1
            g["target"] = target
            return g

        out = out.groupby("symbol", group_keys=False).apply(_labels_per_symbol)
        return out


_REGISTRY: dict[str, type] = {
    "forward_return": ForwardReturnTarget,
    "forward_return_regression": ForwardReturnRegressionTarget,
    "trend_regime": TrendRegimeTarget,
}


def build_target(config: dict) -> TargetProtocol:
    """Build a target by type and config.

    Args:
        config: dict with 'type' key + type-specific kwargs
            forward_return: {type, horizon, gain_threshold, loss_threshold}
            trend_regime: {type, short_window, long_window}

    Returns:
        Target instance implementing TargetProtocol

    Raises:
        KeyError: if config['type'] not in registry
        TypeError: if required kwargs missing
    """
    config = dict(config)
    target_type = config.pop("type")

    if target_type not in _REGISTRY:
        raise KeyError(f"Unknown target type: {target_type}. Available: {sorted(_REGISTRY.keys())}")

    target_class = _REGISTRY[target_type]
    return target_class(**config)
