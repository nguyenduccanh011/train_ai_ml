"""Extended feature set with 35 per-symbol features — leading_v2.

All features are leakage-safe: at row `t`, features depend only on values at rows `<= t`.
Computed per-symbol (group by symbol, no cross-sectional rank).

Blocks:
  ohlcv_basic (5), moving_averages (5), momentum (5), trend (3), volatility (4),
  volume_advanced (4), market_structure (3), exhaustion (4), volatility_regime (3)

Total: 35 features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = [
    # ohlcv_basic (5)
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "close_to_open",
    # moving_averages (5)
    "sma_5_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "ema_10_ratio",
    "sma5_cross_sma20",
    # momentum (5)
    "rsi_14",
    "rsi_7",
    "macd_line",
    "macd_hist",
    "roc_10",
    # trend (3)
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    # volatility (4)
    "atr_14_ratio",
    "bb_width_20",
    "bb_pct_20",
    "realized_vol_10",
    # volume_advanced (4)
    "volume_ratio_5",
    "volume_ratio_20",
    "obv_slope_10",
    "mfi_14",
    # market_structure (3)
    "dist_52w_high",
    "dist_52w_low",
    "high_low_pct_5d",
    # exhaustion (4)
    "upper_wick_ratio",
    "lower_wick_ratio",
    "body_ratio",
    "high_low_pct",
    # volatility_regime (3)
    "atr_regime",
    "bb_squeeze",
    "vol_percentile_60",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI (14-bar exponential moving average of gains/losses)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX and directional indicators (+DI, -DI)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = (
        100
        * pd.Series(plus_dm).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        / (atr + 1e-8)
    )
    minus_di = (
        100
        * pd.Series(minus_dm).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        / (atr + 1e-8)
    )

    di_diff = (plus_di - minus_di).abs()
    di_sum = plus_di + minus_di
    di_ratio = di_diff / (di_sum.replace(0.0, np.nan) + 1e-8)
    adx = di_ratio.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() * 100

    return adx, plus_di, minus_di


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """MACD line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_hist


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def _mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """Money Flow Index."""
    tp = (high + low + close) / 3.0
    mf = tp * volume
    pos_mf = mf.where(tp.diff() > 0, 0.0)
    neg_mf = mf.where(tp.diff() < 0, 0.0)
    pos_mf_sum = pos_mf.rolling(period, min_periods=period).sum()
    neg_mf_sum = neg_mf.rolling(period, min_periods=period).sum()
    mfi = 100.0 - (100.0 / (1.0 + pos_mf_sum / (neg_mf_sum.replace(0.0, np.nan) + 1e-8)))
    return mfi.fillna(50.0)


def _features_one(g: pd.DataFrame) -> pd.DataFrame:
    """Compute all 35 features for one symbol group. Leakage-safe (per-symbol only)."""
    g = g.sort_values("date").copy()
    close = g["close"]
    high = g["high"]
    low = g["low"]
    open_ = g["open"]
    volume = g["volume"]

    # ohlcv_basic
    g["ret_1d"] = close.pct_change(1)
    g["ret_5d"] = close.pct_change(5)
    g["ret_10d"] = close.pct_change(10)
    g["ret_20d"] = close.pct_change(20)
    g["close_to_open"] = close / open_.replace(0.0, np.nan) - 1.0

    # moving_averages
    sma_5 = close.rolling(5, min_periods=5).mean()
    sma_20 = close.rolling(20, min_periods=20).mean()
    sma_50 = close.rolling(50, min_periods=50).mean()
    g["sma_5_ratio"] = close / sma_5.replace(0.0, np.nan) - 1.0
    g["sma_20_ratio"] = close / sma_20.replace(0.0, np.nan) - 1.0
    g["sma_50_ratio"] = close / sma_50.replace(0.0, np.nan) - 1.0
    ema_10 = close.ewm(span=10, adjust=False).mean()
    g["ema_10_ratio"] = close / ema_10.replace(0.0, np.nan) - 1.0
    g["sma5_cross_sma20"] = sma_5 / sma_20.replace(0.0, np.nan) - 1.0

    # momentum
    g["rsi_14"] = _rsi(close, 14)
    g["rsi_7"] = _rsi(close, 7)
    macd_line, macd_hist = _macd(close)
    g["macd_line"] = macd_line
    g["macd_hist"] = macd_hist
    g["roc_10"] = close / close.shift(10).replace(0.0, np.nan) - 1.0

    # trend
    adx, plus_di, minus_di = _adx(high, low, close, 14)
    g["adx_14"] = adx
    g["plus_di_14"] = plus_di
    g["minus_di_14"] = minus_di

    # volatility
    atr_14 = _atr(high, low, close, 14)
    g["atr_14_ratio"] = atr_14 / close.replace(0.0, np.nan)
    bb_mid = sma_20
    bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    g["bb_width_20"] = (bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)
    bb_range = bb_upper - bb_lower
    g["bb_pct_20"] = (close - bb_lower) / bb_range.replace(0.0, np.nan)
    g["realized_vol_10"] = close.pct_change(1).rolling(10, min_periods=10).std()

    # volume_advanced
    vol_sma_5 = volume.rolling(5, min_periods=5).mean()
    vol_sma_20 = volume.rolling(20, min_periods=20).mean()
    g["volume_ratio_5"] = volume / vol_sma_5.replace(0.0, np.nan)
    g["volume_ratio_20"] = volume / vol_sma_20.replace(0.0, np.nan)
    obv = _obv(close, volume)
    obv_change = obv.diff(10)
    obv_abs = obv.abs()
    g["obv_slope_10"] = obv_change / obv_abs.replace(0.0, np.nan)
    g["mfi_14"] = _mfi(high, low, close, volume, 14)

    # market_structure
    rolling_high_252 = close.rolling(252, min_periods=252).max()
    rolling_low_252 = close.rolling(252, min_periods=252).min()
    g["dist_52w_high"] = close / rolling_high_252.replace(0.0, np.nan) - 1.0
    g["dist_52w_low"] = close / rolling_low_252.replace(0.0, np.nan) - 1.0
    rolling_max_5 = high.rolling(5, min_periods=5).max()
    rolling_min_5 = low.rolling(5, min_periods=5).min()
    g["high_low_pct_5d"] = (rolling_max_5 - rolling_min_5) / close.replace(0.0, np.nan)

    # exhaustion (includes high_low_pct from basic_v1)
    range_hl = high - low
    g["upper_wick_ratio"] = (
        high - pd.concat([open_, close], axis=1).max(axis=1)
    ) / range_hl.replace(0.0, np.nan)
    g["lower_wick_ratio"] = (
        pd.concat([open_, close], axis=1).min(axis=1) - low
    ) / range_hl.replace(0.0, np.nan)
    g["body_ratio"] = (open_ - close).abs() / range_hl.replace(0.0, np.nan)
    g["high_low_pct"] = (high - low) / close.replace(0.0, np.nan)

    # volatility_regime
    atr_mean_50 = atr_14.rolling(50, min_periods=50).mean()
    g["atr_regime"] = atr_14 / atr_mean_50.replace(0.0, np.nan) - 1.0
    bb_width = bb_upper - bb_lower
    bb_width_mean_50 = bb_width.rolling(50, min_periods=50).mean()
    g["bb_squeeze"] = bb_width / bb_width_mean_50.replace(0.0, np.nan) - 1.0
    realized_vol = close.pct_change(1).rolling(10, min_periods=10).std()
    vol_rank = realized_vol.rolling(60, min_periods=60).rank()
    g["vol_percentile_60"] = vol_rank / 60.0

    return g


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leading_v2 features to df.

    Args:
        df: DataFrame with [symbol, date, open, high, low, close, volume]

    Returns:
        DataFrame with all 35 feature columns added (NaN for warmup period)
    """
    if "symbol" not in df.columns:
        raise ValueError("df must contain 'symbol' column")
    out = df.groupby("symbol", group_keys=False).apply(_features_one)
    return out.reset_index(drop=True)
