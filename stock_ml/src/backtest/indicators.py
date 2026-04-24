import numpy as np
import pandas as pd

from .defaults import FEATURE_NAMES, FEATURE_DEFAULTS, SYMBOL_PROFILES


def extract_features(df_test, n):
    feat_arrays = {}
    for fn in FEATURE_NAMES:
        if fn in df_test.columns:
            arr = df_test[fn].values.copy()
            arr = np.where(np.isnan(arr), FEATURE_DEFAULTS[fn], arr)
            feat_arrays[fn] = arr
        else:
            feat_arrays[fn] = np.full(n, FEATURE_DEFAULTS[fn])
    return feat_arrays


def compute_indicators(df_test, mod_e=True):
    n = len(df_test)
    feat_arrays = extract_features(df_test, n)

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
    ret_2d = np.zeros(n)
    if n > 2:
        base_2d = close[:-2]
        ret_2d[2:] = np.where(base_2d > 0, close[2:] / base_2d - 1, 0)
    ret_3d = np.zeros(n)
    if n > 3:
        base_3d = close[:-3]
        ret_3d[3:] = np.where(base_3d > 0, close[3:] / base_3d - 1, 0)
    ret_5d = np.zeros(n)
    if n > 5:
        base_5d = close[:-5]
        ret_5d[5:] = np.where(base_5d > 0, close[5:] / base_5d - 1, 0)
    # Wave acceleration: ret_2d relative to ret_5d — positive = wave accelerating
    # early wave: ret_2d > 0 but ret_5d still small (< 3%)
    ret_acceleration = ret_2d - ret_5d / 2.5  # simplified slope proxy
    days_since_low_10 = np.zeros(n, dtype=int)
    for _i in range(1, n):
        lo10_start = max(0, _i - 10)
        lo10_idx = lo10_start + int(np.argmin(close[lo10_start:_i + 1]))
        days_since_low_10[_i] = _i - lo10_idx

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
    loss_arr = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=5).mean().values
    avg_loss_r = pd.Series(loss_arr).rolling(14, min_periods=5).mean().values
    rs_arr = np.where(avg_loss_r > 0, avg_gain / avg_loss_r, 100)
    rsi14 = 100 - 100 / (1 + rs_arr)

    stabilized_sideways = np.zeros(n, dtype=bool)
    for i in range(n):
        for bars in range(5, 11):
            start = i - bars + 1
            if start < 0:
                continue
            band = np.max(high[start:i + 1]) - np.min(low[start:i + 1])
            ref = close[i] if close[i] > 0 else 1.0
            if band / ref < 0.05:
                stabilized_sideways[i] = True
                break

    consolidation_breakout = np.zeros(n, dtype=bool)
    for i in range(10, n):
        prev_high = np.max(high[i - 10:i])
        prev_low = np.min(low[i - 10:i])
        ref = close[i - 1] if close[i - 1] > 0 else close[i]
        tight_range = ((prev_high - prev_low) / ref) < 0.08 if ref > 0 else False
        vol_ok = volume[i] > 1.2 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else False
        if tight_range and close[i] > prev_high and vol_ok:
            consolidation_breakout[i] = True

    secondary_breakout = np.zeros(n, dtype=bool)
    if mod_e:
        for i in range(10, n):
            prev_high = np.max(high[i - 10:i])
            prev_low = np.min(low[i - 10:i])
            ref = close[i - 1] if close[i - 1] > 0 else close[i]
            tight_range = ((prev_high - prev_low) / ref) < 0.10 if ref > 0 else False
            uptrend_macro = (not np.isnan(sma20[i]) and not np.isnan(sma50[i])
                             and sma20[i] > sma50[i])
            vol_threshold = 1.1 if uptrend_macro else 1.2
            vol_ok = volume[i] > vol_threshold * avg_vol20[i] if not np.isnan(avg_vol20[i]) else False
            max_high_5d = np.max(high[max(0, i - 5):i])
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
        oversold = (not np.isnan(rsi14[i - 1]) and rsi14[i - 1] < 35) or drop_from_peak_20[i] <= -0.18
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

    sma100 = pd.Series(close).rolling(100, min_periods=20).mean().values
    days_above_sma50 = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(sma50[i]) and close[i] > sma50[i]:
            days_above_sma50[i] = days_above_sma50[i - 1] + 1
        else:
            days_above_sma50[i] = 0

    rule_signal = np.zeros(n, dtype=int)
    for i in range(20, n):
        if (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and
            close[i] > sma20[i] and sma20[i] > sma50[i] and
            rsi14[i] > 50 and rsi14[i] < 80 and
            not np.isnan(avg_vol20[i]) and volume[i] > 0.8 * avg_vol20[i]):
            rule_signal[i] = 1

    rule_consecutive = np.zeros(n, dtype=int)
    for i in range(1, n):
        if rule_signal[i] == 1:
            rule_consecutive[i] = rule_consecutive[i - 1] + 1
        else:
            rule_consecutive[i] = 0

    rolling_high_250 = pd.Series(close).rolling(250, min_periods=20).max().values
    dist_from_52w_high = np.where(rolling_high_250 > 0, (close / rolling_high_250), 1.0)

    atr_ratio_arr = np.where((close > 0) & (~np.isnan(atr14)), atr14 / close, 0.03)
    atr_pctile = pd.Series(atr_ratio_arr).rolling(60, min_periods=20).rank(pct=True).values

    date_col = "date" if "date" in df_test.columns else ("timestamp" if "timestamp" in df_test.columns else None)
    dates = df_test[date_col].values if date_col else np.arange(n)
    symbols = df_test["symbol"].values if "symbol" in df_test.columns else ["?"] * n

    return {
        "n": n,
        "close": close, "opn": opn, "high": high, "low": low, "volume": volume,
        "sma10": sma10, "sma20": sma20, "sma50": sma50, "sma100": sma100,
        "ema8": ema8, "ema12": ema12, "ema26": ema26,
        "macd_line": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist,
        "atr14": atr14,
        "local_low_20": local_low_20, "avg_vol20": avg_vol20,
        "ret_2d": ret_2d, "ret_3d": ret_3d,
        "ret_acceleration": ret_acceleration, "days_since_low_10": days_since_low_10,
        "ret_5d": ret_5d, "ret_20d": ret_20d, "ret_60d": ret_60d,
        "dist_sma20": dist_sma20,
        "roll_high_20": roll_high_20, "drop_from_peak_20": drop_from_peak_20,
        "rsi14": rsi14,
        "stabilized_sideways": stabilized_sideways,
        "consolidation_breakout": consolidation_breakout,
        "secondary_breakout": secondary_breakout,
        "vshape_bypass": vshape_bypass,
        "days_above_ma20": days_above_ma20,
        "days_above_sma50": days_above_sma50,
        "rule_signal": rule_signal,
        "rule_consecutive": rule_consecutive,
        "rolling_high_250": rolling_high_250,
        "dist_from_52w_high": dist_from_52w_high,
        "atr_ratio_arr": atr_ratio_arr,
        "atr_pctile": atr_pctile,
        "dates": dates, "symbols": symbols,
        "feat_arrays": feat_arrays,
    }


def detect_trend_strength(i, ind):
    if i < 1:
        return "weak"
    sma20 = ind["sma20"]
    sma50 = ind["sma50"]
    close = ind["close"]
    macd_line = ind["macd_line"]
    days_above_ma20 = ind["days_above_ma20"]
    dist_from_52w_high = ind["dist_from_52w_high"]

    ma20_ok = not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i]
    price_above = close[i] > sma20[i] if not np.isnan(sma20[i]) else False
    macd_pos = macd_line[i] > 0
    days_ab = days_above_ma20[i]
    near_high = dist_from_52w_high[i] > 0.90
    score = sum([ma20_ok, price_above, macd_pos, days_ab >= 10, days_ab >= 20, near_high])
    if score >= 4:
        return "strong"
    elif score >= 2:
        return "moderate"
    return "weak"


def get_regime_adapter(i, trend, ind, patch_symbol_tuning=False):
    n = ind["n"]
    symbols = ind["symbols"]
    atr14 = ind["atr14"]
    close = ind["close"]
    ret_20d = ind["ret_20d"]
    feat_arrays = ind["feat_arrays"]

    sym = str(symbols[i]) if i < n else "?"
    profile = SYMBOL_PROFILES.get(sym, "balanced")
    bb_i = feat_arrays["bb_width_percentile"][i]
    low_vol = bb_i < 0.35
    weak_move = abs(ret_20d[i]) < 0.05
    choppy_regime = low_vol and weak_move and trend == "weak"
    atr_ratio = (atr14[i] / close[i]) if (i < n and close[i] > 0 and not np.isnan(atr14[i])) else 0.03

    params = {
        "profile": profile, "dp_floor": 0.020, "ret5_hot": 0.060,
        "size_mult": 1.0, "base_confirm_bars": 3, "exit_score_threshold": 2.0,
        "choppy_regime": choppy_regime,
    }
    if profile == "high_beta":
        params.update({"dp_floor": 0.015, "ret5_hot": 0.090, "size_mult": 0.98,
                       "base_confirm_bars": 2, "exit_score_threshold": 2.35})
    elif profile == "bank":
        params.update({"dp_floor": 0.020, "ret5_hot": 0.070, "size_mult": 0.92,
                       "base_confirm_bars": 3, "exit_score_threshold": 2.2})
    elif profile == "defensive":
        params.update({"dp_floor": 0.025, "ret5_hot": 0.050, "size_mult": 0.85,
                       "base_confirm_bars": 3, "exit_score_threshold": 1.8})
    elif profile == "momentum":
        params.update({"dp_floor": 0.018, "ret5_hot": 0.080, "size_mult": 0.92,
                       "base_confirm_bars": 2, "exit_score_threshold": 2.2})
    if choppy_regime:
        params["dp_floor"] += 0.004; params["size_mult"] *= 0.65
        params["base_confirm_bars"] += 1; params["exit_score_threshold"] += 0.4
    if atr_ratio > 0.040:
        params["size_mult"] *= 0.85; params["exit_score_threshold"] += 0.25
    if atr_ratio > 0.055:
        params["size_mult"] *= 0.90; params["base_confirm_bars"] += 1
        params["exit_score_threshold"] += 0.20
    if trend == "strong":
        params["dp_floor"] = max(0.012, params["dp_floor"] - 0.003)
        params["ret5_hot"] += 0.01; params["exit_score_threshold"] += 0.2
    if sym in ("HPG", "VND"):
        params["size_mult"] *= 0.86; params["exit_score_threshold"] += 0.15
        if atr_ratio > 0.045:
            params["base_confirm_bars"] += 1
    if patch_symbol_tuning:
        if sym == "REE":
            params["exit_score_threshold"] += 0.6
        elif sym == "AAS":
            params["disable_profit_lock_in_strong"] = True
        elif sym == "MBB":
            params["pp_sensitivity_bonus"] = 0.02
    return params
