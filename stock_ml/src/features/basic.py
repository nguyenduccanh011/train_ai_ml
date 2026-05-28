"""Basic, leakage-safe feature set.

Every feature at row `t` depends only on values at rows `<= t`. No `shift(-k)`,
no future windows. Per-symbol computation (group by symbol then transform).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = [
    "ret_1d",
    "ret_5d",
    "sma_5_ratio",
    "sma_20_ratio",
    "rsi_14",
    "volume_ratio_20",
    "high_low_pct",
    "atr_14_ratio",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _features_one(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    close = g["close"]
    high = g["high"]
    low = g["low"]
    volume = g["volume"]

    g["ret_1d"] = close.pct_change(1)
    g["ret_5d"] = close.pct_change(5)
    sma_5 = close.rolling(5, min_periods=5).mean()
    sma_20 = close.rolling(20, min_periods=20).mean()
    g["sma_5_ratio"] = close / sma_5 - 1.0
    g["sma_20_ratio"] = close / sma_20 - 1.0
    g["rsi_14"] = _rsi(close, 14)
    vol_sma_20 = volume.rolling(20, min_periods=20).mean()
    g["volume_ratio_20"] = volume / vol_sma_20.replace(0.0, np.nan)
    g["high_low_pct"] = (high - low) / close.replace(0.0, np.nan)
    g["atr_14_ratio"] = _atr(high, low, close, 14) / close.replace(0.0, np.nan)
    return g


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add FEATURE_COLS to df (must contain symbol, date, open, high, low, close, volume).

    Leaves rows with insufficient warmup as NaN — caller drops or imputes.
    """
    if "symbol" not in df.columns:
        raise ValueError("df must contain 'symbol' column")
    out = df.groupby("symbol", group_keys=False).apply(_features_one)
    return out.reset_index(drop=True)
