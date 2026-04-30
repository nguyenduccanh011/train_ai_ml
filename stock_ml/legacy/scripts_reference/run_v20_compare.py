"""
V20: V19.3 + Reduced confirm_bars in strong/moderate trend.
=============================================================
Only change from V19.3: confirm_bars reduced for faster entry confirmation
  - Strong trend: confirm_bars = 1 (was 2-3)
  - Moderate trend: confirm_bars = min(current, 2)
Backtest showed +16.1% total PnL improvement over V19.3 baseline.
"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v7_compare import summarize
from run_v19_3_compare import backtest_v19_3
from compare_rule_vs_model import backtest_rule


def backtest_v20(y_pred, returns, df_test, feature_cols,
                 initial_capital=100_000_000, commission=0.0015, tax=0.001,
                 record_trades=True,
                 mod_a=True, mod_b=True, mod_c=False, mod_d=False,
                 mod_e=True, mod_f=True, mod_g=True, mod_h=True,
                 mod_i=True, mod_j=True):
    """V20: V19.3 + reduced confirm_bars for strong/moderate trend."""
    n = len(y_pred)
    equity = np.zeros(n)
    equity[0] = initial_capital
    position = 0
    trades = []
    current_entry_day = 0
    entry_equity = 0
    max_equity_in_trade = 0
    max_price_in_trade = 0
    hold_days = 0
    position_size = 1.0
    consecutive_exit_signals = 0
    consecutive_below_ema8 = 0

    MIN_HOLD = 6
    ZOMBIE_BARS = 14
    EXIT_CONFIRM = 3
    PROFIT_LOCK_THRESHOLD = 0.12
    PROFIT_LOCK_MIN = 0.06
    HARD_STOP = 0.08
    ATR_MULT = 1.8
    COOLDOWN_AFTER_BIG_LOSS = 5
    QUICK_REENTRY_WINDOW = 3
    STRONG_TREND_TRAIL_MULT = 0.45
    SIGNAL_HARD_CAP = 0.12

    cooldown_remaining = 0
    last_exit_price = 0
    last_exit_reason = ""
    last_exit_bar = -999
    entry_close = 0

    feat_names = ["rsi_slope_5d", "vol_surge_ratio", "range_position_20d",
                  "dist_to_resistance", "breakout_setup_score", "bb_width_percentile",
                  "higher_lows_count", "obv_price_divergence"]
    defaults = {"rsi_slope_5d": 0, "vol_surge_ratio": 1.0, "range_position_20d": 0.5,
                "dist_to_resistance": 0.05, "breakout_setup_score": 0, "bb_width_percentile": 0.5,
                "higher_lows_count": 0, "obv_price_divergence": 0}
    feat_arrays = {}
    for fn in feat_names:
        if fn in df_test.columns:
            arr = df_test[fn].values.copy()
            arr = np.where(np.isnan(arr), defaults[fn], arr)
            feat_arrays[fn] = arr
        else:
            feat_arrays[fn] = np.full(n, defaults[fn])

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
    loss_arr = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=5).mean().values
    avg_loss = pd.Series(loss_arr).rolling(14, min_periods=5).mean().values
    rs_arr = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
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
            uptrend_macro = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i])
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

    rolling_high_250 = pd.Series(close).rolling(250, min_periods=20).max().values
    dist_from_52w_high = np.where(rolling_high_250 > 0, (close / rolling_high_250), 1.0)

    date_col = "date" if "date" in df_test.columns else ("timestamp" if "timestamp" in df_test.columns else None)
    dates = df_test[date_col].values if date_col else np.arange(n)
    symbols = df_test["symbol"].values if "symbol" in df_test.columns else ["?"] * n

    def gf(name, idx):
        return feat_arrays[name][idx] if idx < n else defaults.get(name, 0)

    def detect_trend_strength(i):
        if i < 1:
            return "weak"
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

    symbol_profiles = {
        "ACB": "bank", "BID": "bank", "MBB": "bank", "TCB": "bank",
        "AAV": "high_beta", "AAS": "high_beta", "SSI": "high_beta", "VND": "high_beta",
        "DGC": "momentum", "HPG": "momentum", "VIC": "momentum",
        "FPT": "defensive", "REE": "defensive", "VNM": "defensive",
    }

    def get_regime_adapter(i, trend):
        sym = str(symbols[i]) if i < n else "?"
        profile = symbol_profiles.get(sym, "balanced")
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
            params.update({"dp_floor": 0.015, "ret5_hot": 0.090, "size_mult": 0.98, "base_confirm_bars": 2, "exit_score_threshold": 2.35})
        elif profile == "bank":
            params.update({"dp_floor": 0.020, "ret5_hot": 0.070, "size_mult": 0.92, "base_confirm_bars": 3, "exit_score_threshold": 2.2})
        elif profile == "defensive":
            params.update({"dp_floor": 0.025, "ret5_hot": 0.050, "size_mult": 0.85, "base_confirm_bars": 3, "exit_score_threshold": 1.8})
        elif profile == "momentum":
            params.update({"dp_floor": 0.018, "ret5_hot": 0.080, "size_mult": 0.92, "base_confirm_bars": 2, "exit_score_threshold": 2.2})

        if choppy_regime:
            params["dp_floor"] += 0.004
            params["size_mult"] *= 0.65
            params["base_confirm_bars"] += 1
            params["exit_score_threshold"] += 0.4

        if atr_ratio > 0.040:
            params["size_mult"] *= 0.85
            params["exit_score_threshold"] += 0.25
        if atr_ratio > 0.055:
            params["size_mult"] *= 0.90
            params["base_confirm_bars"] += 1
            params["exit_score_threshold"] += 0.20
        if trend == "strong":
            params["dp_floor"] = max(0.012, params["dp_floor"] - 0.003)
            params["ret5_hot"] += 0.01
            params["exit_score_threshold"] += 0.2

        if sym in ("HPG", "VND"):
            params["size_mult"] *= 0.86
            params["exit_score_threshold"] += 0.15
            if atr_ratio > 0.045:
                params["base_confirm_bars"] += 1

        # ═══ V20 CHANGE: Reduce confirm_bars in strong/moderate trend ═══
        if trend == "strong":
            params["base_confirm_bars"] = 1
        elif trend == "moderate":
            params["base_confirm_bars"] = min(params["base_confirm_bars"], 2)

        return params

    entry_features = {}
    n_vshape_entries = 0
    n_peak_protect = 0
    n_fast_loss_cut = 0
    n_secondary_breakout = 0
    n_bear_blocked = 0
    n_chop_blocked = 0
    n_confirmed_exit_blocked = 0
    n_trend_carry_saved = 0
    n_v18_relaxed_ret5_entries = 0
    n_v18_relaxed_dp_entries = 0
    n_v18_signal_quality_saves = 0
    n_v19_alpha_blocked = 0
    n_v19_overheat_entries = 0
    n_v19_exit_quality_saved = 0
    n_signal_hard_cap = 0
    n_fast_exit_loss = 0
    n_time_decay_exit = 0

    for i in range(1, n):
        pred = int(y_pred[i - 1])
        ret = returns[i] if not np.isnan(returns[i]) else 0
        raw_signal = 1 if pred == 1 else 0
        new_position = raw_signal
        exit_reason = "signal"

        wp = gf("range_position_20d", i)
        dp = gf("dist_to_resistance", i)
        rs = gf("rsi_slope_5d", i)
        vs = gf("vol_surge_ratio", i)
        bs = gf("breakout_setup_score", i)
        hl = gf("higher_lows_count", i)
        od = gf("obv_price_divergence", i)
        bb = gf("bb_width_percentile", i)

        trend = detect_trend_strength(i)
        regime_cfg = get_regime_adapter(i, trend)
        dp_floor = regime_cfg["dp_floor"]
        ret5_hot = regime_cfg["ret5_hot"]

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        quick_reentry = False
        breakout_entry = False
        vshape_entry = False

        if new_position == 0 and position == 0 and last_exit_reason == "trailing_stop":
            bars_since_exit = i - last_exit_bar
            if (bars_since_exit <= QUICK_REENTRY_WINDOW and trend in ("strong", "moderate")
                    and macd_line[i] > 0 and not np.isnan(sma20[i]) and close[i] > sma20[i]):
                new_position = 1
                quick_reentry = True

        bo_quality_ok = True
        if mod_f:
            macd_pos = macd_hist[i] > 0
            bullish = close[i] > opn[i]
            heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
            bo_quality_ok = macd_pos and bullish and heavy_vol

        if new_position == 0 and position == 0 and consolidation_breakout[i] and bo_quality_ok:
            new_position = 1
            breakout_entry = True

        if mod_e and new_position == 0 and position == 0 and secondary_breakout[i] and bo_quality_ok:
            new_position = 1
            breakout_entry = True
            n_secondary_breakout += 1

        if mod_a and new_position == 0 and position == 0 and vshape_bypass[i]:
            if not np.isnan(ema8[i]) and close[i] >= ema8[i] * 0.99:
                new_position = 1
                vshape_entry = True
                n_vshape_entries += 1

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if cooldown_remaining > 0:
                new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if last_exit_price > 0 and last_exit_reason != "trailing_stop":
                price_diff = abs(close[i] / last_exit_price - 1)
                if price_diff < 0.03:
                    new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry and not breakout_entry and not vshape_entry:
            prev_pred = int(y_pred[i - 2]) if i >= 2 else 0
            if bs >= 4 and vs > 1.2:
                pass
            elif trend == "strong" and rs > 0:
                pass
            elif prev_pred != 1:
                new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3 and not breakout_entry:
                        new_position = 0

        strong_breakout_context = (trend == "strong" and (bs >= 3 or vs > 1.5 or breakout_entry))
        entry_alpha_ok = True

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            near_sma_support = (not np.isnan(sma20[i]) and close[i] <= sma20[i] * 1.02 and close[i] >= sma20[i] * 0.97)
            near_local_low = (not np.isnan(local_low_20[i]) and close[i] <= local_low_20[i] * 1.05)
            in_uptrend_macro = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i])

            if trend == "strong":
                min_score = 1
            elif (near_sma_support or near_local_low) and in_uptrend_macro:
                min_score = 2
            elif in_uptrend_macro and rs > 0:
                min_score = 2
            else:
                min_score = 3

            if entry_score < min_score and not breakout_entry:
                entry_alpha_ok = False
            if wp > 0.9 and rs <= 0 and bs < 2 and trend != "strong" and not breakout_entry:
                entry_alpha_ok = False
            if bb > 0.85 and bs < 2 and entry_score < 4 and trend != "strong" and not breakout_entry:
                entry_alpha_ok = False
            if entry_alpha_ok:
                if wp > 0.78 and bb < 0.35 and trend == "weak" and not breakout_entry:
                    entry_alpha_ok = False
            if entry_alpha_ok and dp < dp_floor:
                if entry_score < 4 and not strong_breakout_context:
                    entry_alpha_ok = False
                elif entry_score < 4 and strong_breakout_context:
                    n_v18_relaxed_dp_entries += 1

        if new_position == 1 and position == 0 and not vshape_entry:
            if ret_5d[i] > ret5_hot and not strong_breakout_context:
                entry_alpha_ok = False
            elif ret_5d[i] > ret5_hot and strong_breakout_context:
                n_v18_relaxed_ret5_entries += 1

        if new_position == 1 and position == 0 and not vshape_entry and entry_alpha_ok:
            if drop_from_peak_20[i] <= -0.15 and not stabilized_sideways[i]:
                entry_alpha_ok = False

        if new_position == 1 and position == 0 and entry_alpha_ok:
            vol_floor = 0.7 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else 0
            if vol_floor > 0 and volume[i] < vol_floor:
                entry_alpha_ok = False

        if mod_g and new_position == 1 and position == 0 and not vshape_entry and entry_alpha_ok:
            sma20_below_50 = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] < sma50[i])
            close_below_50 = (not np.isnan(sma50[i]) and close[i] < sma50[i])
            deep_60d_loss = ret_60d[i] < -0.10
            if sma20_below_50 and close_below_50 and deep_60d_loss:
                entry_alpha_ok = False
                n_bear_blocked += 1

        if mod_j and new_position == 1 and position == 0 and not vshape_entry and not breakout_entry and entry_alpha_ok:
            ma_flat = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and abs(sma20[i] / sma50[i] - 1) < 0.02)
            weak_momo = abs(ret_20d[i]) < 0.06
            narrow_vol = bb < 0.45
            weak_trend = trend == "weak"
            if ma_flat and weak_momo and narrow_vol and weak_trend:
                entry_alpha_ok = False
                n_chop_blocked += 1

        if new_position == 1 and position == 0 and not entry_alpha_ok:
            new_position = 0
            n_v19_alpha_blocked += 1

        # Position sizing (V19.3 binary)
        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            atr_ratio = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03

            if vshape_entry:
                position_size = 0.50
            elif trend == "strong" and entry_score >= 4:
                position_size = 0.95
            elif trend == "strong" and entry_score >= 3:
                position_size = 0.90
            elif trend == "moderate" and entry_score >= 3:
                position_size = 0.50
            elif trend == "weak":
                position_size = 0.30
            else:
                position_size = 0.50

            if atr_ratio > 0.055:
                position_size = min(position_size, 0.35)
            elif atr_ratio > 0.040:
                position_size = min(position_size, 0.50)

            if trend == "weak":
                position_size = min(position_size, 0.40)
            elif trend == "moderate":
                position_size = min(position_size, 0.70)

            if close[i] <= opn[i] and not vshape_entry:
                position_size *= 0.75

            if ret_5d[i] > ret5_hot:
                position_size = min(position_size, 0.40)
                n_v19_overheat_entries += 1

            position_size *= regime_cfg["size_mult"]
            position_size = max(0.25, min(position_size, 1.0))

        # EXIT LOGIC
        if position == 1:
            projected = equity[i - 1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            if close[i] > max_price_in_trade:
                max_price_in_trade = close[i]
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0
            price_max_profit = (max_price_in_trade / entry_close - 1) if entry_close > 0 else 0
            price_cur_ret = (close[i] / entry_close - 1) if entry_close > 0 else 0

            in_uptrend = rs > 0 and hl >= 2
            strong_uptrend = trend == "strong"

            if not np.isnan(atr14[i]) and close[i] > 0:
                atr_stop = ATR_MULT * atr14[i] / close[i]
                atr_stop = max(0.025, min(atr_stop, 0.06))
            else:
                atr_stop = 0.04

            if cum_ret <= -HARD_STOP:
                new_position = 0
                exit_reason = "hard_stop"
            elif price_cur_ret <= -SIGNAL_HARD_CAP:
                new_position = 0
                exit_reason = "signal_hard_cap"
                n_signal_hard_cap += 1
            elif price_cur_ret < -0.05 and hold_days > 3:
                new_position = 0
                exit_reason = "fast_exit_loss"
                n_fast_exit_loss += 1
            elif (price_cur_ret < -0.03 and hold_days > 2
                  and macd_hist[i] < 0 and (not np.isnan(ema8[i]) and close[i] < ema8[i])):
                new_position = 0
                exit_reason = "fast_exit_loss"
                n_fast_exit_loss += 1
            elif cum_ret <= -atr_stop:
                new_position = 0
                exit_reason = "stop_loss"
            elif (mod_c and hold_days < 5 and cum_ret < -0.03
                  and macd_hist[i] < 0 and close[i] < opn[i]):
                new_position = 0
                exit_reason = "fast_loss_cut"
                n_fast_loss_cut += 1
            elif mod_b and price_max_profit >= 0.20:
                price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
                bearish_candle = close[i] < opn[i]
                if price_below_sma10 and heavy_vol and bearish_candle:
                    new_position = 0
                    exit_reason = "peak_protect_dist"
                    n_peak_protect += 1

            if mod_b and new_position == 1 and price_max_profit >= 0.15:
                if not np.isnan(ema8[i]) and close[i] < ema8[i]:
                    consecutive_below_ema8 += 1
                else:
                    consecutive_below_ema8 = 0
                if consecutive_below_ema8 >= 2 and price_cur_ret < price_max_profit * 0.75:
                    new_position = 0
                    exit_reason = "peak_protect_ema"
                    n_peak_protect += 1

            if new_position == 1 and strong_uptrend and cum_ret > 0.05 and max_profit > 0.08:
                macd_bearish = macd_hist[i] < 0 and macd_hist[i - 1] >= 0 if i > 0 else False
                price_below_ma20 = close[i] < sma20[i] if not np.isnan(sma20[i]) else False
                if macd_bearish and price_below_ma20:
                    new_position = 0
                    exit_reason = "hybrid_exit"
                elif price_below_ma20 and cum_ret < max_profit * 0.5:
                    new_position = 0
                    exit_reason = "hybrid_exit"
                else:
                    new_position = 1
            elif new_position == 1 and max_profit > 0.03:
                if max_profit > 0.25:
                    trail_pct = 0.18
                elif max_profit > 0.15:
                    trail_pct = 0.25
                elif max_profit > 0.08:
                    trail_pct = 0.40
                else:
                    trail_pct = 0.65
                if strong_uptrend:
                    trail_pct *= STRONG_TREND_TRAIL_MULT
                elif trend == "moderate":
                    trail_pct *= 0.7
                giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                if giveback >= trail_pct:
                    new_position = 0
                    exit_reason = "trailing_stop"

            if new_position == 1 and max_profit >= PROFIT_LOCK_THRESHOLD:
                if cum_ret < PROFIT_LOCK_MIN:
                    if not strong_uptrend:
                        new_position = 0
                        exit_reason = "profit_lock"

            if new_position == 1 and hold_days >= ZOMBIE_BARS and cum_ret < 0.01:
                if not strong_uptrend:
                    new_position = 0
                    exit_reason = "zombie_exit"

            if new_position == 0 and exit_reason not in (
                    "stop_loss", "hard_stop", "hybrid_exit", "peak_protect_dist",
                    "peak_protect_ema", "fast_loss_cut", "signal_hard_cap",
                    "fast_exit_loss") and hold_days < MIN_HOLD:
                if cum_ret > -atr_stop:
                    new_position = 1

            if new_position == 0 and exit_reason == "signal":
                if mod_h:
                    below_ma20 = (not np.isnan(sma20[i]) and close[i] < sma20[i])
                    below_ma50 = (not np.isnan(sma50[i]) and close[i] < sma50[i])
                    old_bearish_confirm = (below_ma20 and macd_hist[i] < 0) or below_ma50
                    heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.4 * avg_vol20[i])
                    bearish_candle = close[i] < opn[i]
                    macd_falling = macd_hist[i] < macd_hist[i - 1] if i > 0 else False
                    below_ema8 = (not np.isnan(ema8[i]) and close[i] < ema8[i] * 0.997)
                    weak_rebound = (ret_5d[i] < 0.01 and rs <= 0)

                    bearish_score = 0.0
                    bearish_score += 2.0 if below_ma50 else 0.0
                    bearish_score += 1.0 if below_ma20 else 0.0
                    bearish_score += 1.0 if macd_hist[i] < -0.03 else 0.0
                    bearish_score += 0.8 if (macd_hist[i] < 0 and macd_falling) else 0.0
                    bearish_score += 0.8 if (bearish_candle and heavy_vol) else 0.0
                    bearish_score += 0.7 if below_ema8 else 0.0
                    bearish_score += 0.5 if weak_rebound else 0.0

                    score_threshold = regime_cfg["exit_score_threshold"]
                    if cum_ret > 0.06 and max_profit > 0.10 and trend == "strong":
                        score_threshold += 0.7
                    if hold_days < 7 and cum_ret > -0.02:
                        score_threshold += 0.4
                    if cum_ret < -0.03:
                        score_threshold -= 0.4
                    if hold_days > 15 and cum_ret < 0.03:
                        score_threshold *= 0.60
                        n_time_decay_exit += 1
                    elif hold_days > 10 and cum_ret < 0.01:
                        score_threshold *= 0.75

                    bearish_confirm = bearish_score >= score_threshold
                    if old_bearish_confirm and not bearish_confirm:
                        n_v18_signal_quality_saves += 1
                        n_v19_exit_quality_saved += 1
                    if not bearish_confirm:
                        new_position = 1
                        n_confirmed_exit_blocked += 1

            if new_position == 0 and exit_reason == "signal":
                if cum_ret < 0:
                    confirm_bars = 0
                elif mod_d:
                    if cum_ret < 0:
                        confirm_bars = 1
                    elif max_profit > 0 and cum_ret < max_profit * 0.6:
                        confirm_bars = 1
                    else:
                        confirm_bars = EXIT_CONFIRM
                else:
                    confirm_bars = regime_cfg["base_confirm_bars"]
                    if cum_ret < -0.03:
                        confirm_bars = max(1, confirm_bars - 1)
                if raw_signal == 0:
                    consecutive_exit_signals += 1
                else:
                    consecutive_exit_signals = 0
                if consecutive_exit_signals < confirm_bars:
                    new_position = 1
                else:
                    consecutive_exit_signals = 0

            if new_position == 0 and exit_reason == "signal":
                if cum_ret > 0.03 and trend == "strong":
                    new_position = 1

            if mod_i and new_position == 0 and exit_reason == "signal":
                still_supported = (not np.isnan(sma20[i]) and close[i] >= sma20[i] * 0.99)
                trend_ok = trend in ("strong", "moderate")
                if cum_ret > 0.03 and max_profit > 0.06 and trend_ok and still_supported and macd_hist[i] > -0.02:
                    new_position = 1
                    n_trend_carry_saved += 1

        else:
            consecutive_exit_signals = 0
            consecutive_below_ema8 = 0

        # EXECUTE
        cost = 0
        if new_position != position:
            if new_position == 1:
                deploy = equity[i - 1] * position_size
                cost = deploy * commission
                entry_equity = deploy - cost
                max_equity_in_trade = entry_equity
                current_entry_day = i
                hold_days = 0
                consecutive_exit_signals = 0
                consecutive_below_ema8 = 0
                entry_close = close[i]
                max_price_in_trade = close[i]
                entry_features = {
                    "entry_wp": wp, "entry_dp": dp, "entry_rs": rs,
                    "entry_vs": vs, "entry_bs": bs, "entry_hl": hl,
                    "entry_od": od, "entry_bb": bb,
                    "entry_score": sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2]),
                    "entry_date": str(dates[i])[:10],
                    "entry_symbol": str(symbols[i]),
                    "position_size": position_size,
                    "entry_trend": trend,
                    "quick_reentry": quick_reentry,
                    "breakout_entry": breakout_entry,
                    "vshape_entry": vshape_entry,
                    "entry_ret_5d": round(ret_5d[i] * 100, 2),
                    "entry_drop20d": round(drop_from_peak_20[i] * 100, 2),
                    "entry_dist_sma20": round(dist_sma20[i] * 100, 2),
                    "entry_profile": regime_cfg["profile"],
                    "entry_choppy_regime": regime_cfg["choppy_regime"],
                }
            else:
                cost = equity[i - 1] * position_size * (commission + tax)
                pnl_pct_now = (close[i] / entry_close - 1) * 100 if entry_close > 0 else 0
                if pnl_pct_now < -5:
                    cooldown_remaining = COOLDOWN_AFTER_BIG_LOSS
                else:
                    cooldown_remaining = 3

                last_exit_price = close[i]
                last_exit_reason = exit_reason
                last_exit_bar = i

                if record_trades and entry_equity > 0:
                    pnl_pct = (close[i] / entry_close - 1) * 100 if entry_close > 0 else 0
                    max_pnl_pct = (max_equity_in_trade - entry_equity) / entry_equity * 100 if entry_equity > 0 else 0
                    trades.append({
                        "entry_day": current_entry_day, "exit_day": i,
                        "holding_days": i - current_entry_day,
                        "pnl_pct": round(pnl_pct, 2),
                        "max_profit_pct": round(max_pnl_pct, 2),
                        "exit_reason": exit_reason,
                        "exit_date": str(dates[i])[:10],
                        **entry_features,
                    })
                entry_equity = 0
                max_equity_in_trade = 0
                max_price_in_trade = 0
                position_size = 1.0

        if position == 1:
            equity[i] = equity[i - 1] * (1 + ret * position_size) - cost
            hold_days += 1
        else:
            equity[i] = equity[i - 1] - cost
        position = new_position

    if position == 1 and entry_equity > 0 and record_trades:
        pnl_pct = (close[-1] / entry_close - 1) * 100 if entry_close > 0 else 0
        trades.append({
            "entry_day": current_entry_day, "exit_day": n - 1,
            "holding_days": n - 1 - current_entry_day,
            "pnl_pct": round(pnl_pct, 2), "exit_reason": "end",
            "exit_date": str(dates[-1])[:10], **entry_features,
        })

    return {
        "equity_curve": equity, "trades": trades,
        "total_return_pct": round((equity[-1] / initial_capital - 1) * 100, 2),
        "final_equity": round(equity[-1]),
        "n_vshape_entries": n_vshape_entries,
        "n_peak_protect": n_peak_protect,
        "n_fast_loss_cut": n_fast_loss_cut,
        "n_secondary_breakout": n_secondary_breakout,
        "n_bear_blocked": n_bear_blocked,
        "n_chop_blocked": n_chop_blocked,
        "n_confirmed_exit_blocked": n_confirmed_exit_blocked,
        "n_trend_carry_saved": n_trend_carry_saved,
        "n_v18_relaxed_ret5_entries": n_v18_relaxed_ret5_entries,
        "n_v18_relaxed_dp_entries": n_v18_relaxed_dp_entries,
        "n_v18_signal_quality_saves": n_v18_signal_quality_saves,
        "n_v19_alpha_blocked": n_v19_alpha_blocked,
        "n_v19_overheat_entries": n_v19_overheat_entries,
        "n_v19_exit_quality_saved": n_v19_exit_quality_saved,
        "n_signal_hard_cap": n_signal_hard_cap,
        "n_fast_exit_loss": n_fast_exit_loss,
        "n_time_decay_exit": n_time_decay_exit,
    }
