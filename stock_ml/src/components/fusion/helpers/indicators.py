"""V19 indicator pre-compute — exact byte-parity with legacy backtest_v19_3."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_FEAT_NAMES: tuple[str, ...] = (
    "rsi_slope_5d",
    "vol_surge_ratio",
    "range_position_20d",
    "dist_to_resistance",
    "breakout_setup_score",
    "bb_width_percentile",
    "higher_lows_count",
    "obv_price_divergence",
)
_FEAT_DEFAULTS: dict[str, float] = {
    "rsi_slope_5d": 0.0,
    "vol_surge_ratio": 1.0,
    "range_position_20d": 0.5,
    "dist_to_resistance": 0.05,
    "breakout_setup_score": 0.0,
    "bb_width_percentile": 0.5,
    "higher_lows_count": 0.0,
    "obv_price_divergence": 0.0,
}


def compute_v19_indicators(df_test: pd.DataFrame, *, mod_e: bool = True) -> dict[str, Any]:
    """Compute v19_3 indicator dict from a test DataFrame.

    Mirrors legacy.backtest_v19_3 lines 2454-2568 verbatim — same arithmetic,
    same min_periods, same edge-case handling. Returns a dict of ndarrays plus
    feat_arrays (alpha-feature columns) and dates/symbols arrays.
    """
    n = len(df_test)
    close = df_test["close"].values if "close" in df_test.columns else np.ones(n)
    opn = df_test["open"].values if "open" in df_test.columns else close
    high = df_test["high"].values if "high" in df_test.columns else close
    low = df_test["low"].values if "low" in df_test.columns else close
    volume = df_test["volume"].values if "volume" in df_test.columns else np.ones(n)

    sma10 = pd.Series(close).rolling(10, min_periods=3).mean().values
    sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
    sma50 = pd.Series(close).rolling(50, min_periods=10).mean().values
    ema8 = pd.Series(close).ewm(span=8, min_periods=4).mean().values
    ema12 = pd.Series(close).ewm(span=12, min_periods=8).mean().values
    ema26 = pd.Series(close).ewm(span=26, min_periods=15).mean().values
    macd_line = ema12 - ema26
    macd_signal = pd.Series(macd_line).ewm(span=9, min_periods=5).mean().values
    macd_hist = macd_line - macd_signal

    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().values

    local_low_20 = pd.Series(close).rolling(20, min_periods=5).min().values
    avg_vol20 = pd.Series(volume).rolling(20, min_periods=5).mean().values

    ret_5d = np.zeros(n)
    if n > 5:
        base_5d = close[:-5]
        ret_5d[5:] = np.where(base_5d > 0, close[5:] / base_5d - 1, 0)
    ret_20d = np.zeros(n)
    if n > 20:
        base_20d = close[:-20]
        ret_20d[20:] = np.where(base_20d > 0, close[20:] / base_20d - 1, 0)
    ret_60d = np.zeros(n)
    if n > 60:
        base_60d = close[:-60]
        ret_60d[60:] = np.where(base_60d > 0, close[60:] / base_60d - 1, 0)

    dist_sma20 = np.where((~np.isnan(sma20)) & (sma20 > 0), close / sma20 - 1, 0)
    roll_high_20 = pd.Series(close).rolling(20, min_periods=5).max().values
    drop_from_peak_20 = np.where(roll_high_20 > 0, close / roll_high_20 - 1, 0)

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=5).mean().values
    avg_loss = pd.Series(loss).rolling(14, min_periods=5).mean().values
    rs_arr = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi14 = 100 - 100 / (1 + rs_arr)

    stabilized_sideways = np.zeros(n, dtype=bool)
    for i in range(n):
        for bars in range(5, 11):
            start = i - bars + 1
            if start < 0:
                continue
            band = np.max(high[start : i + 1]) - np.min(low[start : i + 1])
            ref = close[i] if close[i] > 0 else 1.0
            if band / ref < 0.05:
                stabilized_sideways[i] = True
                break

    consolidation_breakout = np.zeros(n, dtype=bool)
    for i in range(10, n):
        prev_high = np.max(high[i - 10 : i])
        prev_low = np.min(low[i - 10 : i])
        ref = close[i - 1] if close[i - 1] > 0 else close[i]
        tight_range = ((prev_high - prev_low) / ref) < 0.08 if ref > 0 else False
        vol_ok = volume[i] > 1.2 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else False
        if tight_range and close[i] > prev_high and vol_ok:
            consolidation_breakout[i] = True

    secondary_breakout = np.zeros(n, dtype=bool)
    if mod_e:
        for i in range(10, n):
            prev_high = np.max(high[i - 10 : i])
            prev_low = np.min(low[i - 10 : i])
            ref = close[i - 1] if close[i - 1] > 0 else close[i]
            tight_range = ((prev_high - prev_low) / ref) < 0.10 if ref > 0 else False
            uptrend_macro = (
                not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i]
            )
            vol_threshold = 1.1 if uptrend_macro else 1.2
            vol_ok = (
                volume[i] > vol_threshold * avg_vol20[i] if not np.isnan(avg_vol20[i]) else False
            )
            max_high_5d = np.max(high[max(0, i - 5) : i])
            breakout_5d = close[i] > max_high_5d
            if tight_range and breakout_5d and vol_ok and uptrend_macro:
                secondary_breakout[i] = True

    vshape_bypass = np.zeros(n, dtype=bool)
    for i in range(15, n):
        had_deep_drop = False
        for j in range(max(0, i - 5), i + 1):
            if drop_from_peak_20[j] <= -0.15:
                had_deep_drop = True
                break
        if not had_deep_drop:
            continue
        rng = high[i] - low[i]
        if rng <= 0:
            continue
        bullish = close[i] > opn[i] + rng * 0.5 and close[i] > close[i - 1]
        if not bullish:
            continue
        oversold = (not np.isnan(rsi14[i - 1]) and rsi14[i - 1] < 35) or drop_from_peak_20[
            i
        ] <= -0.18
        if not oversold:
            continue
        if np.isnan(avg_vol20[i]) or volume[i] < 1.3 * avg_vol20[i]:
            continue
        for k in range(i, min(n, i + 5)):
            vshape_bypass[k] = True

    days_above_ma20 = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(sma20[i]) and close[i] > sma20[i]:
            days_above_ma20[i] = days_above_ma20[i - 1] + 1
        else:
            days_above_ma20[i] = 0

    rolling_high_250 = pd.Series(close).rolling(250, min_periods=20).max().values
    dist_from_52w_high = np.where(rolling_high_250 > 0, (close / rolling_high_250), 1.0)

    feat_arrays: dict[str, np.ndarray] = {}
    for fn in _FEAT_NAMES:
        if fn in df_test.columns:
            arr = df_test[fn].values.copy()
            arr = np.where(np.isnan(arr), _FEAT_DEFAULTS[fn], arr)
            feat_arrays[fn] = arr
        else:
            feat_arrays[fn] = np.full(n, _FEAT_DEFAULTS[fn])

    date_col = (
        "date"
        if "date" in df_test.columns
        else ("timestamp" if "timestamp" in df_test.columns else None)
    )
    dates = df_test[date_col].values if date_col else np.arange(n)
    symbols = df_test["symbol"].values if "symbol" in df_test.columns else np.array(["?"] * n)

    return {
        "n": n,
        "close": close,
        "open": opn,
        "high": high,
        "low": low,
        "volume": volume,
        "sma10": sma10,
        "sma20": sma20,
        "sma50": sma50,
        "ema8": ema8,
        "ema12": ema12,
        "ema26": ema26,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "atr14": atr14,
        "local_low_20": local_low_20,
        "avg_vol20": avg_vol20,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "ret_60d": ret_60d,
        "dist_sma20": dist_sma20,
        "roll_high_20": roll_high_20,
        "drop_from_peak_20": drop_from_peak_20,
        "rsi14": rsi14,
        "stabilized_sideways": stabilized_sideways,
        "consolidation_breakout": consolidation_breakout,
        "secondary_breakout": secondary_breakout,
        "vshape_bypass": vshape_bypass,
        "days_above_ma20": days_above_ma20,
        "dist_from_52w_high": dist_from_52w_high,
        "feat_arrays": feat_arrays,
        "dates": dates,
        "symbols": symbols,
    }
