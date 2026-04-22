"""
V26 Experiments: Test each proposed improvement individually and in combinations.
Measures delta vs V25 baseline to decide what goes into V26.

Improvements being tested:
  A) Adaptive hard_cap with wider ATR multipliers for volatile stocks
  B) Relaxed entry for confirmed strong trends (bypass prev_pred requirement)
  C) Skip choppy regime trades entirely
  D) Extended hold for winning trades in strong trends
  E) Stronger rule ensemble (rule_signal 3+ bars = force entry)
  F) Minimum position threshold (skip low-conviction trades)
  G) Score-5 penalty (reduce position for overfit high-score entries)
  H) Wider hard_cap confirm bars in strong trend
"""
import sys, os, copy, time
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v19_1_compare import run_test as run_test_base, run_rule_test, calc_metrics
from run_v25 import backtest_v25, comp_score

# ============================================================
# V26 CANDIDATE BACKTEST — extends v25 with new patches
# ============================================================
def backtest_v26(y_pred, returns, df_test, feature_cols,
                 initial_capital=100_000_000, commission=0.0015, tax=0.001,
                 record_trades=True,
                 mod_a=True, mod_b=True, mod_c=False, mod_d=False,
                 mod_e=True, mod_f=True, mod_g=True, mod_h=True,
                 mod_i=True, mod_j=True,
                 fast_exit_strong=-0.08,
                 fast_exit_moderate=-0.06,
                 fast_exit_weak=-0.04,
                 fast_exit_hb_buffer=0.02,
                 peak_protect_strong_threshold=0.12,
                 peak_protect_normal_threshold=0.20,
                 hard_cap_weak=-0.10,
                 hard_cap_moderate_mult=2.0,
                 hard_cap_strong_mult=3.0,
                 hard_cap_strong_floor=0.15,
                 hard_cap_moderate_floor=0.12,
                 time_decay_bars=20,
                 time_decay_mult=0.50,
                 patch_smart_hardcap=True,
                 patch_pp_restore=True,
                 patch_long_horizon=True,
                 patch_symbol_tuning=True,
                 patch_rule_ensemble=True,
                 patch_noise_filter=False,
                 patch_adaptive_hardcap=False,
                 patch_pp_2of3=False,
                 # ---- V26 NEW PATCHES ----
                 v26_wider_hardcap=False,
                 v26_relaxed_entry=False,
                 v26_skip_choppy=False,
                 v26_extended_hold=False,
                 v26_strong_rule_ensemble=False,
                 v26_min_position=False,
                 v26_score5_penalty=False,
                 v26_hardcap_confirm_strong=False,
                 # ---- V27 NEW PATCHES ----
                 v27_selective_choppy=False,
                 v27_hardcap_two_step=False,
                 v27_rule_priority=False,
                 v27_dynamic_score5_penalty=False,
                 v27_trend_persistence_hold=False,
                 ):
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

    hard_cap_pending_bars = 0
    HARD_CAP_CONFIRM_STRONG = 2 if v26_hardcap_confirm_strong else 1
    HARD_CAP_CONFIRM_MODERATE = 1
    HARD_CAP_CONFIRM_WEAK = 0

    pp_pending_bars = 0

    MIN_HOLD = 6
    ZOMBIE_BARS = 14
    PROFIT_LOCK_THRESHOLD = 0.12
    PROFIT_LOCK_MIN = 0.06
    HARD_STOP = 0.08
    ATR_MULT = 1.8
    COOLDOWN_AFTER_BIG_LOSS = 5
    QUICK_REENTRY_WINDOW = 3
    STRONG_TREND_TRAIL_MULT = 0.45

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

    # V26: consecutive rule signal count
    rule_consecutive = np.zeros(n, dtype=int)
    for i in range(1, n):
        if rule_signal[i] == 1:
            rule_consecutive[i] = rule_consecutive[i-1] + 1
        else:
            rule_consecutive[i] = 0

    rolling_high_250 = pd.Series(close).rolling(250, min_periods=20).max().values
    dist_from_52w_high = np.where(rolling_high_250 > 0, (close / rolling_high_250), 1.0)

    atr_ratio_arr = np.where((close > 0) & (~np.isnan(atr14)), atr14 / close, 0.03)

    # V26: ATR percentile for adaptive sizing
    atr_pctile = pd.Series(atr_ratio_arr).rolling(60, min_periods=20).rank(pct=True).values

    date_col = "date" if "date" in df_test.columns else ("timestamp" if "timestamp" in df_test.columns else None)
    dates = df_test[date_col].values if date_col else np.arange(n)
    symbols = df_test["symbol"].values if "symbol" in df_test.columns else ["?"] * n

    def gf(name, idx):
        return feat_arrays[name][idx] if idx < n else defaults.get(name, 0)

    symbol_profiles = {
        "ACB": "bank", "BID": "bank", "MBB": "bank", "TCB": "bank",
        "AAV": "high_beta", "AAS": "high_beta", "SSI": "high_beta", "VND": "high_beta",
        "DGC": "momentum", "HPG": "momentum", "VIC": "momentum",
        "FPT": "defensive", "REE": "defensive", "VNM": "defensive",
    }
    rule_priority_symbols = {"AAA", "SSN", "TEG", "GAS", "PLX", "IJC", "DQC"}
    score5_risky_symbols = {"AAA", "IJC", "ITC", "VHM", "TEG", "QBS", "KMR", "SSN", "PLX"}

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

    entry_features = {}
    counters = defaultdict(int)

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
        sym = str(symbols[i]) if i < n else "?"
        profile = symbol_profiles.get(sym, "balanced")

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # === ENTRY LOGIC ===
        quick_reentry = False
        breakout_entry = False
        vshape_entry = False

        if new_position == 0 and position == 0 and last_exit_reason == "trailing_stop":
            bars_since_exit = i - last_exit_bar
            if (bars_since_exit <= QUICK_REENTRY_WINDOW and trend in ("strong", "moderate") and
                macd_line[i] > 0 and not np.isnan(sma20[i]) and close[i] > sma20[i]):
                new_position = 1; quick_reentry = True

        bo_quality_ok = True
        if mod_f:
            bo_quality_ok = (macd_hist[i] > 0 and close[i] > opn[i] and
                            not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])

        if new_position == 0 and position == 0 and consolidation_breakout[i] and bo_quality_ok:
            new_position = 1; breakout_entry = True

        if mod_e and new_position == 0 and position == 0 and secondary_breakout[i] and bo_quality_ok:
            new_position = 1; breakout_entry = True; counters["secondary_bo"] += 1

        if mod_a and new_position == 0 and position == 0 and vshape_bypass[i]:
            if not np.isnan(ema8[i]) and close[i] >= ema8[i] * 0.99:
                new_position = 1; vshape_entry = True; counters["vshape"] += 1

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if cooldown_remaining > 0:
                new_position = 0
        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if last_exit_price > 0 and last_exit_reason != "trailing_stop":
                if abs(close[i] / last_exit_price - 1) < 0.03:
                    new_position = 0

        # V26-B: Relaxed entry for confirmed strong trends
        if new_position == 1 and position == 0 and not quick_reentry and not breakout_entry and not vshape_entry:
            prev_pred = int(y_pred[i - 2]) if i >= 2 else 0
            if bs >= 4 and vs > 1.2: pass
            elif trend == "strong" and rs > 0: pass
            elif v26_relaxed_entry and trend == "strong" and rule_consecutive[i] >= 3:
                pass  # V26: allow entry when rule has been signaling 3+ bars
            elif prev_pred != 1: new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3 and not breakout_entry: new_position = 0

        strong_breakout_context = (trend == "strong" and (bs >= 3 or vs > 1.5 or breakout_entry))
        entry_alpha_ok = True
        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            near_sma_support = (not np.isnan(sma20[i]) and close[i] <= sma20[i] * 1.02 and close[i] >= sma20[i] * 0.97)
            near_local_low = (not np.isnan(local_low_20[i]) and close[i] <= local_low_20[i] * 1.05)
            in_uptrend_macro = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i])
            if trend == "strong": min_score = 1
            elif (near_sma_support or near_local_low) and in_uptrend_macro: min_score = 2
            elif in_uptrend_macro and rs > 0: min_score = 2
            else: min_score = 3
            if entry_score < min_score and not breakout_entry: entry_alpha_ok = False
            if wp > 0.9 and rs <= 0 and bs < 2 and trend != "strong" and not breakout_entry: entry_alpha_ok = False
            if bb > 0.85 and bs < 2 and entry_score < 4 and trend != "strong" and not breakout_entry: entry_alpha_ok = False
            if entry_alpha_ok and wp > 0.78 and bb < 0.35 and trend == "weak" and not breakout_entry: entry_alpha_ok = False
            if entry_alpha_ok and dp < dp_floor:
                if entry_score < 4 and not strong_breakout_context: entry_alpha_ok = False

        if new_position == 1 and position == 0 and not vshape_entry:
            if ret_5d[i] > ret5_hot and not strong_breakout_context: entry_alpha_ok = False
        if new_position == 1 and position == 0 and not vshape_entry and entry_alpha_ok:
            if drop_from_peak_20[i] <= -0.15 and not stabilized_sideways[i]: entry_alpha_ok = False
        if new_position == 1 and position == 0 and entry_alpha_ok:
            vol_floor = 0.7 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else 0
            if vol_floor > 0 and volume[i] < vol_floor: entry_alpha_ok = False
        if mod_g and new_position == 1 and position == 0 and not vshape_entry and entry_alpha_ok:
            if (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] < sma50[i] and
                close[i] < sma50[i] and ret_60d[i] < -0.10):
                entry_alpha_ok = False; counters["bear_blocked"] += 1
        if mod_j and new_position == 1 and position == 0 and not vshape_entry and not breakout_entry and entry_alpha_ok:
            if (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and
                abs(sma20[i] / sma50[i] - 1) < 0.02 and abs(ret_20d[i]) < 0.06 and
                bb < 0.45 and trend == "weak"):
                entry_alpha_ok = False; counters["chop_blocked"] += 1

        # V26-C: Skip choppy regime entirely
        if v26_skip_choppy and new_position == 1 and position == 0 and entry_alpha_ok:
            if regime_cfg["choppy_regime"]:
                entry_alpha_ok = False; counters["v26_choppy_skipped"] += 1

        # V27: selective choppy filter (keep only high-quality setups in choppy regime)
        if (not v26_skip_choppy) and v27_selective_choppy and new_position == 1 and position == 0 and entry_alpha_ok:
            if regime_cfg["choppy_regime"]:
                vol_ok = (not np.isnan(avg_vol20[i]) and volume[i] >= 0.95 * avg_vol20[i])
                quality_ok = (
                    breakout_entry or
                    (trend in ("strong", "moderate") and bs >= 3 and vs > 1.15 and vol_ok) or
                    (rule_consecutive[i] >= 3 and rs > 0 and vol_ok)
                )
                if not quality_ok:
                    entry_alpha_ok = False
                    counters["v27_choppy_low_quality_skipped"] += 1
                else:
                    counters["v27_choppy_quality_kept"] += 1

        if patch_noise_filter and new_position == 1 and position == 0 and entry_alpha_ok:
            entry_score_nf = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            if trend == "weak" and entry_score_nf < 3 and ret_5d[i] > 0.03:
                entry_alpha_ok = False; counters["noise_filtered"] += 1

        if new_position == 1 and position == 0 and not entry_alpha_ok:
            new_position = 0; counters["alpha_blocked"] += 1

        # Rule ensemble
        if patch_rule_ensemble:
            ml_buy = (new_position == 1 and position == 0)
            rule_buy = (rule_signal[i] == 1)
            if rule_buy and not ml_buy and position == 0 and trend == "strong":
                new_position = 1
                position_size = 0.30
                counters["rule_only_entry"] += 1

        # V26-E: Stronger rule ensemble — force entry when rule signals 3+ consecutive bars
        if v26_strong_rule_ensemble and not patch_rule_ensemble:
            ml_buy = (new_position == 1 and position == 0)
            if not ml_buy and position == 0 and rule_consecutive[i] >= 3:
                if trend in ("strong", "moderate"):
                    new_position = 1
                    position_size = 0.35
                    counters["v26_strong_rule_entry"] += 1
        elif v26_strong_rule_ensemble and patch_rule_ensemble:
            if position == 0 and new_position == 0 and rule_consecutive[i] >= 3:
                if trend == "moderate":
                    new_position = 1
                    position_size = 0.30
                    counters["v26_strong_rule_moderate"] += 1

        # V27: rule-priority entry on symbols where rule historically dominates
        if v27_rule_priority and position == 0 and new_position == 0:
            if sym in rule_priority_symbols and rule_consecutive[i] >= 2 and trend in ("strong", "moderate"):
                new_position = 1
                position_size = 0.35 if trend == "moderate" else 0.40
                counters["v27_rule_priority_entry"] += 1

        # Position sizing
        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            atr_ratio = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03
            if vshape_entry: position_size = 0.50
            elif trend == "strong" and entry_score >= 4: position_size = 0.95
            elif trend == "strong" and entry_score >= 3: position_size = 0.90
            elif trend == "moderate" and entry_score >= 3: position_size = 0.50
            elif trend == "weak": position_size = 0.30
            else: position_size = 0.50
            if atr_ratio > 0.055: position_size = min(position_size, 0.35)
            elif atr_ratio > 0.040: position_size = min(position_size, 0.50)
            if trend == "weak": position_size = min(position_size, 0.40)
            elif trend == "moderate": position_size = min(position_size, 0.70)
            if close[i] <= opn[i] and not vshape_entry: position_size *= 0.75
            if ret_5d[i] > ret5_hot: position_size = min(position_size, 0.40)
            position_size *= regime_cfg["size_mult"]

            # V26-G: Score-5 penalty
            if v26_score5_penalty and entry_score == 5:
                position_size *= 0.75
                counters["v26_score5_penalized"] += 1

            # V27: dynamic score-5 penalty by regime/symbol risk
            if v27_dynamic_score5_penalty and entry_score == 5:
                score5_penalty = 1.0
                if sym in score5_risky_symbols:
                    score5_penalty *= 0.70
                if trend == "weak":
                    score5_penalty *= 0.85
                if bb < 0.40 and trend != "strong":
                    score5_penalty *= 0.90
                if rule_consecutive[i] >= 3 and trend in ("strong", "moderate"):
                    score5_penalty = max(score5_penalty, 0.80)
                if score5_penalty < 0.999:
                    position_size *= score5_penalty
                    counters["v27_score5_dynamic_penalized"] += 1

            position_size = max(0.25, min(position_size, 1.0))

            # V26-F: Minimum position threshold
            if v26_min_position and position_size < 0.28:
                new_position = 0; counters["v26_min_pos_blocked"] += 1

        # === EXIT LOGIC ===
        if position == 1:
            projected = equity[i - 1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            if close[i] > max_price_in_trade: max_price_in_trade = close[i]
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0
            price_max_profit = (max_price_in_trade / entry_close - 1) if entry_close > 0 else 0
            price_cur_ret = (close[i] / entry_close - 1) if entry_close > 0 else 0
            strong_uptrend = trend == "strong"
            atr_ratio_now = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03

            if not np.isnan(atr14[i]) and close[i] > 0:
                atr_stop = ATR_MULT * atr14[i] / close[i]
                atr_stop = max(0.025, min(atr_stop, 0.06))
            else:
                atr_stop = 0.04

            # 0) HARD STOP
            if cum_ret <= -HARD_STOP:
                new_position = 0; exit_reason = "hard_stop"

            # === HARD CAP LOGIC ===
            elif patch_smart_hardcap:
                if trend == "weak":
                    if price_cur_ret <= hard_cap_weak:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        hard_cap_pending_bars = 0
                        counters["signal_hard_cap"] += 1
                elif trend == "moderate":
                    if v26_wider_hardcap:
                        cap = max(hard_cap_moderate_floor * 1.3, hard_cap_moderate_mult * 1.3 * atr_ratio_now)
                    elif patch_adaptive_hardcap:
                        cap = max(0.10 if atr_ratio_now < 0.025 else hard_cap_moderate_floor,
                                  hard_cap_moderate_mult * atr_ratio_now)
                    else:
                        cap = max(hard_cap_moderate_floor, hard_cap_moderate_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        hard_cap_pending_bars += 1
                        hc_confirm_bars = 1 + HARD_CAP_CONFIRM_MODERATE
                        if v27_hardcap_two_step and profile in ("high_beta", "momentum"):
                            hc_confirm_bars += 1
                        if hard_cap_pending_bars >= hc_confirm_bars:
                            new_position = 0; exit_reason = "signal_hard_cap"
                            hard_cap_pending_bars = 0
                            counters["signal_hard_cap"] += 1
                    else:
                        hard_cap_pending_bars = 0
                else:  # strong
                    if v26_wider_hardcap:
                        cap = max(hard_cap_strong_floor * 1.4, hard_cap_strong_mult * 1.4 * atr_ratio_now)
                        if atr_ratio_now > 0.04:
                            cap = max(0.25, 4.0 * atr_ratio_now)
                        elif profile == "high_beta":
                            cap = max(0.22, 4.0 * atr_ratio_now)
                    elif patch_adaptive_hardcap:
                        if atr_ratio_now > 0.04:
                            cap = max(0.20, 3.5 * atr_ratio_now)
                        elif profile == "high_beta":
                            cap = max(0.18, 3.5 * atr_ratio_now)
                        else:
                            cap = max(hard_cap_moderate_floor, hard_cap_strong_mult * atr_ratio_now)
                    else:
                        if profile == "high_beta":
                            cap = max(0.18, 3.5 * atr_ratio_now)
                        else:
                            cap = max(hard_cap_moderate_floor, hard_cap_strong_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        hard_cap_pending_bars += 1
                        hc_confirm_bars = 1 + HARD_CAP_CONFIRM_STRONG
                        if v27_hardcap_two_step and profile in ("high_beta", "momentum"):
                            hc_confirm_bars += 1
                        if hard_cap_pending_bars >= hc_confirm_bars:
                            new_position = 0; exit_reason = "signal_hard_cap"
                            hard_cap_pending_bars = 0
                            counters["signal_hard_cap"] += 1
                    else:
                        hard_cap_pending_bars = 0
            else:
                if trend == "weak":
                    if price_cur_ret <= hard_cap_weak:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["signal_hard_cap"] += 1
                elif trend == "moderate":
                    cap = max(hard_cap_moderate_floor, hard_cap_moderate_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["signal_hard_cap"] += 1
                else:
                    if profile == "high_beta":
                        cap = max(hard_cap_strong_floor, hard_cap_strong_mult * atr_ratio_now)
                    else:
                        cap = max(hard_cap_moderate_floor, hard_cap_strong_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["signal_hard_cap"] += 1

            # Fast exit loss
            if new_position == 1:
                hb_buf = fast_exit_hb_buffer if profile == "high_beta" else 0.0
                if trend == "strong":
                    ft = fast_exit_strong - hb_buf
                elif trend == "moderate":
                    ft = fast_exit_moderate - hb_buf
                else:
                    ft = fast_exit_weak - hb_buf
                mt = ft + 0.02

                do_fast_exit = False
                if price_cur_ret < ft and hold_days > 3:
                    do_fast_exit = True
                elif (price_cur_ret < mt and hold_days > 2 and
                      macd_hist[i] < 0 and not np.isnan(ema8[i]) and close[i] < ema8[i]):
                    do_fast_exit = True

                if do_fast_exit:
                    new_position = 0; exit_reason = "fast_exit_loss"
                    counters["fast_exit_loss"] += 1

            # ATR stop
            if new_position == 1 and cum_ret <= -atr_stop:
                new_position = 0; exit_reason = "stop_loss"

            # Peak protection
            if new_position == 1 and mod_b:
                if patch_pp_restore:
                    if price_max_profit >= 0.25:
                        pp_threshold_active = 0.10
                    elif strong_uptrend:
                        pp_threshold_active = peak_protect_strong_threshold
                    else:
                        pp_threshold_active = peak_protect_normal_threshold

                    pp_bonus = regime_cfg.get("pp_sensitivity_bonus", 0)
                    pp_threshold_active = max(0.08, pp_threshold_active - pp_bonus)

                    if price_max_profit >= pp_threshold_active:
                        price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                        if price_below_sma10:
                            pp_pending_bars += 1
                            heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.3 * avg_vol20[i])
                            if pp_pending_bars >= 2 or heavy_vol:
                                new_position = 0; exit_reason = "peak_protect_dist"
                                pp_pending_bars = 0
                                counters["peak_protect"] += 1
                        else:
                            pp_pending_bars = 0
                elif patch_pp_2of3:
                    pp_threshold = peak_protect_strong_threshold if strong_uptrend else peak_protect_normal_threshold
                    if price_max_profit >= pp_threshold:
                        price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                        heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
                        bearish_candle = close[i] < opn[i]
                        checks = sum([price_below_sma10, heavy_vol, bearish_candle])
                        if checks >= 2:
                            new_position = 0; exit_reason = "peak_protect_dist"
                            counters["peak_protect"] += 1
                else:
                    pp_threshold = peak_protect_strong_threshold if strong_uptrend else peak_protect_normal_threshold
                    if price_max_profit >= pp_threshold:
                        price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                        heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
                        bearish_candle = close[i] < opn[i]
                        if price_below_sma10 and heavy_vol and bearish_candle:
                            new_position = 0; exit_reason = "peak_protect_dist"
                            counters["peak_protect"] += 1

            # EMA8 peak protect
            if mod_b and new_position == 1 and position == 1:
                if price_max_profit >= 0.15:
                    if not np.isnan(ema8[i]) and close[i] < ema8[i]:
                        consecutive_below_ema8 += 1
                    else:
                        consecutive_below_ema8 = 0
                    if consecutive_below_ema8 >= 2 and price_cur_ret < price_max_profit * 0.75:
                        new_position = 0; exit_reason = "peak_protect_ema"
                        counters["peak_protect"] += 1

            # Hybrid exit
            if new_position == 1 and strong_uptrend and cum_ret > 0.05 and max_profit > 0.08:
                macd_bearish = macd_hist[i] < 0 and macd_hist[i - 1] >= 0 if i > 0 else False
                price_below_ma20 = close[i] < sma20[i] if not np.isnan(sma20[i]) else False
                if macd_bearish and price_below_ma20:
                    new_position = 0; exit_reason = "hybrid_exit"
                elif price_below_ma20 and cum_ret < max_profit * 0.5:
                    new_position = 0; exit_reason = "hybrid_exit"

            # Adaptive trailing
            elif new_position == 1 and max_profit > 0.03:
                if max_profit > 0.25: trail_pct = 0.18
                elif max_profit > 0.15: trail_pct = 0.25
                elif max_profit > 0.08: trail_pct = 0.40
                else: trail_pct = 0.65
                if strong_uptrend: trail_pct *= STRONG_TREND_TRAIL_MULT
                elif trend == "moderate": trail_pct *= 0.7
                giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                if giveback >= trail_pct:
                    new_position = 0; exit_reason = "trailing_stop"

            # Profit lock
            if new_position == 1 and max_profit >= PROFIT_LOCK_THRESHOLD:
                if cum_ret < PROFIT_LOCK_MIN and not strong_uptrend:
                    if patch_symbol_tuning and regime_cfg.get("disable_profit_lock_in_strong") and trend == "strong":
                        pass
                    else:
                        new_position = 0; exit_reason = "profit_lock"

            # Zombie
            if new_position == 1 and hold_days >= ZOMBIE_BARS and cum_ret < 0.01:
                if not strong_uptrend:
                    new_position = 0; exit_reason = "zombie_exit"

            # V26-D: Extended hold for winning trades in strong trends
            extended_min_hold = MIN_HOLD
            if v26_extended_hold and strong_uptrend and cum_ret > 0.05:
                extended_min_hold = 12
            if v27_trend_persistence_hold and strong_uptrend and cum_ret > 0.03:
                trend_supported = (
                    (not np.isnan(sma20[i]) and close[i] >= sma20[i] * 0.995) and
                    (macd_hist[i] > -0.01)
                )
                if trend_supported:
                    extended_min_hold = max(extended_min_hold, 10)
                    counters["v27_persistence_hold"] += 1

            # Min hold
            if new_position == 0 and exit_reason not in (
                    "stop_loss", "hard_stop", "hybrid_exit", "peak_protect_dist",
                    "peak_protect_ema", "fast_loss_cut", "signal_hard_cap",
                    "fast_exit_loss") and hold_days < extended_min_hold:
                if cum_ret > -atr_stop: new_position = 1

            # Confirmed signal exit
            if new_position == 0 and exit_reason == "signal" and mod_h:
                below_ma20 = (not np.isnan(sma20[i]) and close[i] < sma20[i])
                below_ma50 = (not np.isnan(sma50[i]) and close[i] < sma50[i])
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
                if cum_ret > 0.06 and max_profit > 0.10 and trend == "strong": score_threshold += 0.7
                if hold_days < 7 and cum_ret > -0.02: score_threshold += 0.4
                if cum_ret < -0.03: score_threshold -= 0.4
                if hold_days > time_decay_bars and cum_ret < 0.02:
                    score_threshold *= time_decay_mult
                    counters["time_decay_exit"] += 1
                elif hold_days > 15 and cum_ret < 0.03:
                    score_threshold *= 0.60
                elif hold_days > 10 and cum_ret < 0.01:
                    score_threshold *= 0.75
                if bearish_score < score_threshold:
                    new_position = 1; counters["confirmed_exit_blocked"] += 1

            if new_position == 0 and exit_reason == "signal":
                if cum_ret < 0: confirm_bars = 0
                else:
                    confirm_bars = regime_cfg["base_confirm_bars"]
                    if cum_ret < -0.03: confirm_bars = max(1, confirm_bars - 1)
                if raw_signal == 0: consecutive_exit_signals += 1
                else: consecutive_exit_signals = 0
                if consecutive_exit_signals < confirm_bars: new_position = 1
                else: consecutive_exit_signals = 0

            if new_position == 0 and exit_reason == "signal":
                if cum_ret > 0.03 and trend == "strong": new_position = 1

            if mod_i and new_position == 0 and exit_reason == "signal":
                still_supported = (not np.isnan(sma20[i]) and close[i] >= sma20[i] * 0.99)
                trend_ok = trend in ("strong", "moderate")
                if (cum_ret > 0.03 and max_profit > 0.06 and trend_ok and still_supported and
                    macd_hist[i] > -0.02):
                    new_position = 1; counters["trend_carry_saved"] += 1

            # Long-horizon carry
            if patch_long_horizon:
                if (new_position == 0 and exit_reason in ("signal", "peak_protect_dist", "peak_protect_ema",
                                                           "profit_lock", "trailing_stop")):
                    long_horizon_regime = (
                        ret_60d[i] > 0.30 and
                        not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and not np.isnan(sma100[i]) and
                        sma20[i] > sma50[i] > sma100[i] and
                        days_above_sma50[i] >= 20 and
                        cum_ret > 0.15
                    )
                    if long_horizon_regime:
                        hard_breakdown = (
                            (close[i] < sma50[i] * 0.97) or
                            (i >= 3 and macd_hist[i] < 0 and macd_hist[i-1] < 0 and macd_hist[i-2] < 0)
                        )
                        if not hard_breakdown:
                            new_position = 1
                            counters["long_horizon_carry"] += 1

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
                current_entry_day = i; hold_days = 0
                consecutive_exit_signals = 0; consecutive_below_ema8 = 0
                hard_cap_pending_bars = 0; pp_pending_bars = 0
                entry_close = close[i]; max_price_in_trade = close[i]
                entry_features = {
                    "entry_wp": wp, "entry_dp": dp, "entry_rs": rs,
                    "entry_vs": vs, "entry_bs": bs, "entry_hl": hl,
                    "entry_od": od, "entry_bb": bb,
                    "entry_score": sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2]),
                    "entry_date": str(dates[i])[:10], "entry_symbol": str(symbols[i]),
                    "position_size": position_size, "entry_trend": trend,
                    "quick_reentry": quick_reentry, "breakout_entry": breakout_entry,
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
                cooldown_remaining = COOLDOWN_AFTER_BIG_LOSS if pnl_pct_now < -5 else 3
                last_exit_price = close[i]; last_exit_reason = exit_reason; last_exit_bar = i
                if record_trades and entry_equity > 0:
                    pnl_pct = (close[i] / entry_close - 1) * 100 if entry_close > 0 else 0
                    max_pnl_pct = (max_equity_in_trade - entry_equity) / entry_equity * 100
                    trades.append({
                        "entry_day": current_entry_day, "exit_day": i,
                        "holding_days": i - current_entry_day,
                        "pnl_pct": round(pnl_pct, 2), "max_profit_pct": round(max_pnl_pct, 2),
                        "exit_reason": exit_reason, "exit_date": str(dates[i])[:10],
                        **entry_features,
                    })
                entry_equity = 0; max_equity_in_trade = 0; max_price_in_trade = 0; position_size = 1.0
                hard_cap_pending_bars = 0; pp_pending_bars = 0

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
        **{f"n_{k}": v for k, v in counters.items()},
    }


# ============================================================
# MAIN: Run experiments
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    DEFAULT_SYMBOLS = (
        "ACB,AAS,AAV,ACV,BCG,BCM,BID,BSR,BVH,CTG,DCM,DGC,DIG,DPM,EIB,"
        "FPT,FRT,GAS,GEX,GMD,HCM,HDB,HDG,HPG,HSG,KBC,KDH,LPB,MBB,MSN,"
        "MWG,NKG,NLG,NT2,NVL,OCB,PC1,PDR,PLX,PNJ,POW,PVD,PVS,REE,SAB,"
        "SBT,SHB,SSI,STB,TCB,TPB,VCB,VCI,VDS,VHM,VIC,VJC,VND,VNM,VPB,VTP"
    )

    SYMBOLS = args.symbols.strip() if args.symbols.strip() else DEFAULT_SYMBOLS
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    print("=" * 150)
    print("V26 EXPERIMENTS: Testing improvements individually and in combinations")
    print("=" * 150)

    # V25 baseline patches (all V24 patches ON, V25 new patches OFF)
    V25_BASE = dict(
        patch_smart_hardcap=True, patch_pp_restore=True,
        patch_long_horizon=True, patch_symbol_tuning=True,
        patch_rule_ensemble=True,
        patch_noise_filter=False, patch_adaptive_hardcap=False, patch_pp_2of3=False,
    )

    V26_OFF = dict(
        v26_wider_hardcap=False, v26_relaxed_entry=False,
        v26_skip_choppy=False, v26_extended_hold=False,
        v26_strong_rule_ensemble=False, v26_min_position=False,
        v26_score5_penalty=False, v26_hardcap_confirm_strong=False,
    )

    # ---- PHASE 1: V25 Baseline ----
    print("\n  [Phase 1] V25 Baseline...")
    def make_v25_fn():
        def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
            return backtest_v25(y_pred, returns, df_test, feature_cols,
                               peak_protect_strong_threshold=0.12,
                               **V25_BASE, **kwargs)
        return bt_fn

    t_v25 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=make_v25_fn())
    m_v25 = calc_metrics(t_v25)
    print(f"    V25 Baseline: {m_v25['trades']} trades, TotalPnL={m_v25['total_pnl']:+.1f}%, "
          f"WR={m_v25['wr']:.1f}%, PF={m_v25['pf']:.2f}, AvgPnL={m_v25['avg_pnl']:+.2f}%")

    # Rule baseline
    t_rule = run_rule_test(SYMBOLS)
    m_rule = calc_metrics(t_rule)
    print(f"    Rule:         {m_rule['trades']} trades, TotalPnL={m_rule['total_pnl']:+.1f}%")

    # ---- PHASE 2: Individual improvements ----
    print("\n  [Phase 2] Individual V26 improvements...")

    experiments = [
        # Individual
        ("A: Wider hardcap",         {**V26_OFF, "v26_wider_hardcap": True}),
        ("B: Relaxed entry",         {**V26_OFF, "v26_relaxed_entry": True}),
        ("C: Skip choppy",           {**V26_OFF, "v26_skip_choppy": True}),
        ("D: Extended hold",         {**V26_OFF, "v26_extended_hold": True}),
        ("E: Strong rule ensemble",  {**V26_OFF, "v26_strong_rule_ensemble": True}),
        ("F: Min position (0.28)",   {**V26_OFF, "v26_min_position": True}),
        ("G: Score-5 penalty",       {**V26_OFF, "v26_score5_penalty": True}),
        ("H: HC confirm strong+1",   {**V26_OFF, "v26_hardcap_confirm_strong": True}),
        # Combinations of 2
        ("A+C",                      {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True}),
        ("A+D",                      {**V26_OFF, "v26_wider_hardcap": True, "v26_extended_hold": True}),
        ("A+H",                      {**V26_OFF, "v26_wider_hardcap": True, "v26_hardcap_confirm_strong": True}),
        ("B+E",                      {**V26_OFF, "v26_relaxed_entry": True, "v26_strong_rule_ensemble": True}),
        ("C+F",                      {**V26_OFF, "v26_skip_choppy": True, "v26_min_position": True}),
        ("C+G",                      {**V26_OFF, "v26_skip_choppy": True, "v26_score5_penalty": True}),
        # Combinations of 3
        ("A+C+D",                    {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True, "v26_extended_hold": True}),
        ("A+C+H",                    {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True, "v26_hardcap_confirm_strong": True}),
        ("A+D+H",                    {**V26_OFF, "v26_wider_hardcap": True, "v26_extended_hold": True, "v26_hardcap_confirm_strong": True}),
        ("B+C+E",                    {**V26_OFF, "v26_relaxed_entry": True, "v26_skip_choppy": True, "v26_strong_rule_ensemble": True}),
        ("A+C+F",                    {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True, "v26_min_position": True}),
        # Big combos
        ("A+C+D+H",                  {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True, "v26_extended_hold": True, "v26_hardcap_confirm_strong": True}),
        ("A+B+C+D+H",               {**V26_OFF, "v26_wider_hardcap": True, "v26_relaxed_entry": True, "v26_skip_choppy": True, "v26_extended_hold": True, "v26_hardcap_confirm_strong": True}),
        ("A+B+C+E+H",               {**V26_OFF, "v26_wider_hardcap": True, "v26_relaxed_entry": True, "v26_skip_choppy": True, "v26_strong_rule_ensemble": True, "v26_hardcap_confirm_strong": True}),
        ("A+C+D+F+G+H",             {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True, "v26_extended_hold": True, "v26_min_position": True, "v26_score5_penalty": True, "v26_hardcap_confirm_strong": True}),
        ("ALL",                      {k: True for k in V26_OFF}),
        ("ALL except F",             {**{k: True for k in V26_OFF}, "v26_min_position": False}),
        ("ALL except G",             {**{k: True for k in V26_OFF}, "v26_score5_penalty": False}),
    ]

    print(f"\n  {'Config':<30} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} "
          f"{'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7} | {'vs V25':>9} {'vs Rule':>9}")
    print("  " + "-" * 145)

    # Print V25 baseline
    cs25 = comp_score(m_v25)
    print(f"  {'V25-BASELINE':<30} | {m_v25['trades']:>5} {m_v25['wr']:>5.1f}% {m_v25['avg_pnl']:>+7.2f}% "
          f"{m_v25['total_pnl']:>+9.1f}% {m_v25['pf']:>5.2f} {m_v25['max_loss']:>+7.1f}% {m_v25['avg_hold']:>5.1f}d | "
          f"{cs25:>6.0f} |      --- {'':>9}")
    print("  " + "-" * 145)

    results = {}
    for label, v26_cfg in experiments:
        t0 = time.time()
        print(f"    Running {label}...", end="", flush=True)

        def make_fn(cfg):
            def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
                return backtest_v26(y_pred, returns, df_test, feature_cols,
                                   peak_protect_strong_threshold=0.12,
                                   **V25_BASE, **cfg, **kwargs)
            return bt_fn

        trades = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                               backtest_fn=make_fn(v26_cfg))
        m = calc_metrics(trades)
        results[label] = (m, trades, v26_cfg)

        cs = comp_score(m)
        dt = time.time() - t0
        delta_v25 = m['total_pnl'] - m_v25['total_pnl']
        delta_rule = m['total_pnl'] - m_rule['total_pnl']
        marker = " +++" if delta_v25 > 200 else " ++" if delta_v25 > 50 else " +" if delta_v25 > 0 else " ---" if delta_v25 < -200 else " --" if delta_v25 < -50 else ""
        print(f"\r  {label:<30} | {m['trades']:>5} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>5.1f}d | "
              f"{cs:>6.0f} | {delta_v25:>+8.1f}% {delta_rule:>+8.1f}%{marker}")

    # ---- PHASE 3: Ranking ----
    print(f"\n{'='*150}")
    print("  RANKING BY COMPOSITE SCORE (higher = better)")
    print(f"{'='*150}")

    scored = []
    for name, (m, trades, cfg) in results.items():
        scored.append((comp_score(m), name, m, cfg))
    scored.sort(reverse=True)

    for rank, (sc, name, m, cfg) in enumerate(scored, 1):
        delta = m['total_pnl'] - m_v25['total_pnl']
        marker = " <<<" if rank <= 3 else ""
        print(f"  #{rank:>2} {sc:>7.0f}  {name:<30} TotPnL={m['total_pnl']:>+9.1f}% "
              f"PF={m['pf']:>5.2f} WR={m['wr']:>5.1f}% MaxLoss={m['max_loss']:>+7.1f}% "
              f"vs_V25={delta:>+8.1f}%{marker}")

    # Best config
    best_score, best_name, best_m, best_cfg = scored[0]
    print(f"\n  BEST: {best_name}")
    print(f"    Composite: {best_score:.0f} (V25 baseline: {cs25:.0f}, delta: {best_score-cs25:+.0f})")
    print(f"    TotalPnL:  {best_m['total_pnl']:+.1f}% (vs V25: {best_m['total_pnl']-m_v25['total_pnl']:+.1f}%)")
    print(f"    PF:        {best_m['pf']:.2f}")
    print(f"    WR:        {best_m['wr']:.1f}%")
    active_patches = [k for k, v in best_cfg.items() if v]
    print(f"    Patches:   {active_patches}")

    # ---- Improvement summary ----
    print(f"\n{'='*150}")
    print("  INDIVIDUAL IMPROVEMENT DELTA vs V25")
    print(f"{'='*150}")
    for name in ["A: Wider hardcap", "B: Relaxed entry", "C: Skip choppy",
                  "D: Extended hold", "E: Strong rule ensemble",
                  "F: Min position (0.28)", "G: Score-5 penalty", "H: HC confirm strong+1"]:
        if name in results:
            m, _, _ = results[name]
            dt = m['total_pnl'] - m_v25['total_pnl']
            dwr = m['wr'] - m_v25['wr']
            dpf = m['pf'] - m_v25['pf']
            verdict = "KEEP" if dt > 0 and dpf >= 0 else "KEEP(marginal)" if dt > 0 else "SKIP" if dt < -50 else "NEUTRAL"
            print(f"  {name:<30} dTotPnL={dt:>+8.1f}%  dWR={dwr:>+5.2f}%  dPF={dpf:>+5.3f}  => {verdict}")

    # Save best trades
    if scored:
        _, best_name_2, _, _ = scored[0]
        _, best_trades, _ = results[best_name_2]
        df_best = pd.DataFrame(best_trades)
        if len(df_best) > 0:
            df_best.to_csv(os.path.join(OUT, "trades_v26_best.csv"), index=False)
            print(f"\n  Saved {len(df_best)} V26-best trades to results/trades_v26_best.csv")

    print(f"\n{'='*150}")
    print("  DONE")
    print(f"{'='*150}")
