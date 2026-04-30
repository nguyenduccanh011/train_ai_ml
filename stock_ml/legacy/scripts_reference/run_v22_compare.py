"""
V22 Breakthrough Upgrade from V19.3.
=====================================
Six independent improvement modules (can toggle each):
  fix_A: Smart fast_exit_loss - skip in strong trend with MACD+, raise threshold for high_beta
  fix_B: ATR-adaptive signal_hard_cap instead of fixed -12%
  fix_C: Continuous position sizing (replace binary), raise floor
  fix_D: Profile-specific trend detection for bank/defensive
  fix_E: Early cycle detector - relax filters after bear market exit
  fix_F: Hybrid rule-model fallback entries
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
from run_v19_1_compare import run_test as run_test_base, run_rule_test, calc_metrics
from run_v19_3_compare import backtest_v19_3


def backtest_v22(y_pred, returns, df_test, feature_cols,
                 initial_capital=100_000_000, commission=0.0015, tax=0.001,
                 record_trades=True,
                 mod_a=True, mod_b=True, mod_c=False, mod_d=False,
                 mod_e=True, mod_f=True, mod_g=True, mod_h=True,
                 mod_i=True, mod_j=True,
                 # V22 fixes
                 fix_A=False, fix_B=False, fix_C=False,
                 fix_D=False, fix_E=False, fix_F=False):
    """V22: V19.3 base + selective breakthrough fixes."""
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
    sma200 = pd.Series(close).rolling(200, min_periods=50).mean().values
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

    rolling_high_250 = pd.Series(close).rolling(250, min_periods=20).max().values
    dist_from_52w_high = np.where(rolling_high_250 > 0, (close / rolling_high_250), 1.0)

    # ═══ fix_E: Early cycle detection ═══
    early_cycle = np.zeros(n, dtype=bool)
    if fix_E:
        for i in range(1, n):
            if np.isnan(sma200[i]) or np.isnan(sma50[i]):
                continue
            crossed_up_200 = False
            for j in range(max(1, i - 60), i + 1):
                if not np.isnan(sma200[j]) and j > 0:
                    if close[j] > sma200[j] and close[j - 1] <= sma200[j - 1]:
                        crossed_up_200 = True
                        break
            if crossed_up_200 and close[i] > sma200[i]:
                early_cycle[i] = True

    # ═══ fix_F: Rule-based entry conditions (precompute) ═══
    rule_entry_signal = np.zeros(n, dtype=bool)
    if fix_F:
        for i in range(20, n):
            if np.isnan(sma20[i]) or np.isnan(sma50[i]):
                continue
            uptrend = sma20[i] > sma50[i] and close[i] > sma20[i]
            bo = close[i] > np.max(high[max(0, i - 20):i]) if i > 0 else False
            vol_ok = (not np.isnan(avg_vol20[i]) and volume[i] > 1.3 * avg_vol20[i])
            bullish = close[i] > opn[i]
            if uptrend and bo and vol_ok and bullish:
                rule_entry_signal[i] = True

    date_col = "date" if "date" in df_test.columns else ("timestamp" if "timestamp" in df_test.columns else None)
    dates = df_test[date_col].values if date_col else np.arange(n)
    symbols = df_test["symbol"].values if "symbol" in df_test.columns else ["?"] * n

    def gf(name, idx):
        return feat_arrays[name][idx] if idx < n else defaults.get(name, 0)

    def detect_trend_strength(i):
        if i < 1:
            return "weak"
        sym = str(symbols[i]) if i < n else "?"

        # ═══ fix_D: Profile-specific trend detection ═══
        if fix_D:
            profile = symbol_profiles.get(sym, "balanced")
            if profile == "bank":
                ma20_ok = not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i]
                price_above = close[i] > sma20[i] if not np.isnan(sma20[i]) else False
                vol_confirm = (not np.isnan(avg_vol20[i]) and volume[i] > 0.9 * avg_vol20[i])
                macd_pos = macd_line[i] > 0
                days_ab = days_above_ma20[i]
                ema_align = (not np.isnan(ema8[i]) and not np.isnan(sma20[i]) and ema8[i] > sma20[i])
                score = sum([ma20_ok, price_above, macd_pos, vol_confirm, ema_align,
                            days_ab >= 15, days_ab >= 25])
                if score >= 5:
                    return "strong"
                elif score >= 3:
                    return "moderate"
                return "weak"
            elif profile == "defensive":
                ma20_ok = not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i]
                price_above = close[i] > sma20[i] if not np.isnan(sma20[i]) else False
                macd_pos = macd_line[i] > 0
                days_ab = days_above_ma20[i]
                low_vol = feat_arrays["bb_width_percentile"][i] < 0.5
                score = sum([ma20_ok, price_above, macd_pos, days_ab >= 15, days_ab >= 30, low_vol])
                if score >= 5:
                    return "strong"
                elif score >= 3:
                    return "moderate"
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
            "profile": profile,
            "dp_floor": 0.020,
            "ret5_hot": 0.060,
            "size_mult": 1.0,
            "base_confirm_bars": 3,
            "exit_score_threshold": 2.0,
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
    n_fixA_saved = 0
    n_fixB_saved = 0
    n_fixE_relaxed = 0
    n_fixF_hybrid = 0

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

        # ═══════════════════════════════════════
        # ENTRY LOGIC
        # ═══════════════════════════════════════
        quick_reentry = False
        breakout_entry = False
        vshape_entry = False
        hybrid_entry = False

        if new_position == 0 and position == 0 and last_exit_reason == "trailing_stop":
            bars_since_exit = i - last_exit_bar
            if (bars_since_exit <= QUICK_REENTRY_WINDOW and
                trend in ("strong", "moderate") and
                macd_line[i] > 0 and
                not np.isnan(sma20[i]) and close[i] > sma20[i]):
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

        # ═══ fix_F: Hybrid rule-model fallback entry ═══
        if fix_F and new_position == 0 and position == 0 and not quick_reentry and not vshape_entry:
            if rule_entry_signal[i] and trend in ("strong", "moderate"):
                new_position = 1
                hybrid_entry = True
                n_fixF_hybrid += 1

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry and not hybrid_entry:
            if cooldown_remaining > 0:
                new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry and not hybrid_entry:
            if last_exit_price > 0 and last_exit_reason != "trailing_stop":
                price_diff = abs(close[i] / last_exit_price - 1)
                if price_diff < 0.03:
                    new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry and not breakout_entry and not vshape_entry and not hybrid_entry:
            prev_pred = int(y_pred[i - 2]) if i >= 2 else 0
            if bs >= 4 and vs > 1.2:
                pass
            elif trend == "strong" and rs > 0:
                pass
            elif prev_pred != 1:
                new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry and not hybrid_entry:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3 and not breakout_entry:
                        new_position = 0

        strong_breakout_context = (
            trend == "strong" and (bs >= 3 or vs > 1.5 or breakout_entry)
        )
        entry_alpha_ok = True

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry and not hybrid_entry:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            near_sma_support = (not np.isnan(sma20[i]) and
                               close[i] <= sma20[i] * 1.02 and
                               close[i] >= sma20[i] * 0.97)
            near_local_low = (not np.isnan(local_low_20[i]) and
                             close[i] <= local_low_20[i] * 1.05)
            in_uptrend_macro = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and
                               sma20[i] > sma50[i])

            if trend == "strong":
                min_score = 1
            elif (near_sma_support or near_local_low) and in_uptrend_macro:
                min_score = 2
            elif in_uptrend_macro and rs > 0:
                min_score = 2
            else:
                min_score = 3

            # ═══ fix_E: Relax entry in early cycle ═══
            if fix_E and early_cycle[i]:
                min_score = max(1, min_score - 1)
                n_fixE_relaxed += 1

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

        if new_position == 1 and position == 0 and not vshape_entry and not hybrid_entry:
            if ret_5d[i] > ret5_hot and not strong_breakout_context:
                entry_alpha_ok = False
            elif ret_5d[i] > ret5_hot and strong_breakout_context:
                n_v18_relaxed_ret5_entries += 1

        if new_position == 1 and position == 0 and not vshape_entry and entry_alpha_ok and not hybrid_entry:
            if drop_from_peak_20[i] <= -0.15 and not stabilized_sideways[i]:
                entry_alpha_ok = False

        if new_position == 1 and position == 0 and entry_alpha_ok and not hybrid_entry:
            vol_floor = 0.7 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else 0
            if vol_floor > 0 and volume[i] < vol_floor:
                entry_alpha_ok = False

        if mod_g and new_position == 1 and position == 0 and not vshape_entry and entry_alpha_ok and not hybrid_entry:
            sma20_below_50 = (not np.isnan(sma20[i]) and not np.isnan(sma50[i])
                              and sma20[i] < sma50[i])
            close_below_50 = (not np.isnan(sma50[i]) and close[i] < sma50[i])
            deep_60d_loss = ret_60d[i] < -0.10
            if sma20_below_50 and close_below_50 and deep_60d_loss:
                entry_alpha_ok = False
                n_bear_blocked += 1

        if mod_j and new_position == 1 and position == 0 and not vshape_entry and not breakout_entry and entry_alpha_ok and not hybrid_entry:
            ma_flat = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and
                       abs(sma20[i] / sma50[i] - 1) < 0.02)
            weak_momo = abs(ret_20d[i]) < 0.06
            narrow_vol = bb < 0.45
            weak_trend = trend == "weak"
            if ma_flat and weak_momo and narrow_vol and weak_trend:
                entry_alpha_ok = False
                n_chop_blocked += 1

        if new_position == 1 and position == 0 and not entry_alpha_ok and not hybrid_entry:
            new_position = 0
            n_v19_alpha_blocked += 1

        # ═══ POSITION SIZING ═══
        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            atr_ratio = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03

            if fix_C:
                # ═══ fix_C: Continuous sizing with higher floor ═══
                if vshape_entry:
                    position_size = 0.50
                elif hybrid_entry:
                    position_size = 0.50
                elif trend == "strong":
                    position_size = 0.55 + 0.40 * min(entry_score / 5.0, 1.0)
                elif trend == "moderate":
                    position_size = 0.45 + 0.25 * min(entry_score / 5.0, 1.0)
                else:
                    position_size = 0.35 + 0.15 * min(entry_score / 5.0, 1.0)

                if atr_ratio > 0.055:
                    position_size = min(position_size, 0.40)
                elif atr_ratio > 0.040:
                    position_size = min(position_size, 0.55)

                if trend == "weak":
                    position_size = min(position_size, 0.45)
                elif trend == "moderate":
                    position_size = min(position_size, 0.70)
            else:
                # Original V19.3 binary sizing
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

        # ═══════════════════════════════════════
        # EXIT LOGIC
        # ═══════════════════════════════════════
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

            atr_ratio_now = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03

            # 0) HARD STOP
            if cum_ret <= -HARD_STOP:
                new_position = 0
                exit_reason = "hard_stop"

            # 0.5) SIGNAL HARD CAP
            elif fix_B:
                # ═══ fix_B: ATR-adaptive signal_hard_cap ═══
                adaptive_cap = max(0.12, 2.5 * atr_ratio_now)
                if profile == "high_beta":
                    adaptive_cap = max(0.15, 3.0 * atr_ratio_now)
                if price_cur_ret <= -adaptive_cap:
                    new_position = 0
                    exit_reason = "signal_hard_cap"
                    n_signal_hard_cap += 1
                elif not fix_B and price_cur_ret <= -SIGNAL_HARD_CAP:
                    new_position = 0
                    exit_reason = "signal_hard_cap"
                    n_signal_hard_cap += 1
            elif price_cur_ret <= -SIGNAL_HARD_CAP:
                new_position = 0
                exit_reason = "signal_hard_cap"
                n_signal_hard_cap += 1

            # 0.6) FAST EXIT LOSS
            if new_position == 1:
                do_fast_exit = False

                if fix_A:
                    # ═══ fix_A: Smart fast_exit_loss ═══
                    # Skip fast exit in strong trend when MACD still positive and price above SMA20
                    trend_still_healthy = (strong_uptrend and macd_line[i] > 0
                                          and not np.isnan(sma20[i]) and close[i] > sma20[i] * 0.97)
                    # Raise threshold for high_beta from -5% to -7%
                    fast_threshold = -0.07 if profile == "high_beta" else -0.05
                    moderate_threshold = -0.04 if profile == "high_beta" else -0.03

                    if price_cur_ret < fast_threshold and hold_days > 3:
                        if not trend_still_healthy:
                            do_fast_exit = True
                        else:
                            n_fixA_saved += 1
                    elif (price_cur_ret < moderate_threshold and hold_days > 2
                          and macd_hist[i] < 0
                          and (not np.isnan(ema8[i]) and close[i] < ema8[i])):
                        if not trend_still_healthy:
                            do_fast_exit = True
                        else:
                            n_fixA_saved += 1
                else:
                    # Original V19.2 fast exit
                    if price_cur_ret < -0.05 and hold_days > 3:
                        do_fast_exit = True
                    elif (price_cur_ret < -0.03 and hold_days > 2
                          and macd_hist[i] < 0
                          and (not np.isnan(ema8[i]) and close[i] < ema8[i])):
                        do_fast_exit = True

                if do_fast_exit:
                    new_position = 0
                    exit_reason = "fast_exit_loss"
                    n_fast_exit_loss += 1

            # 1) ATR stop loss
            if new_position == 1 and cum_ret <= -atr_stop:
                new_position = 0
                exit_reason = "stop_loss"

            # MODULE C: FAST LOSS CUT
            if (new_position == 1 and mod_c and hold_days < 5 and cum_ret < -0.03
                  and macd_hist[i] < 0 and close[i] < opn[i]):
                new_position = 0
                exit_reason = "fast_loss_cut"
                n_fast_loss_cut += 1

            # MODULE B: PROFIT-PEAK PROTECTION
            if new_position == 1 and mod_b and price_max_profit >= 0.20:
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

            # Hybrid exit
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

            # Adaptive trailing
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

            # Profit lock
            if new_position == 1 and max_profit >= PROFIT_LOCK_THRESHOLD:
                if cum_ret < PROFIT_LOCK_MIN:
                    if not strong_uptrend:
                        new_position = 0
                        exit_reason = "profit_lock"

            # Zombie exit
            if new_position == 1 and hold_days >= ZOMBIE_BARS and cum_ret < 0.01:
                if not strong_uptrend:
                    new_position = 0
                    exit_reason = "zombie_exit"

            # Min hold
            if new_position == 0 and exit_reason not in (
                    "stop_loss", "hard_stop", "hybrid_exit", "peak_protect_dist",
                    "peak_protect_ema", "fast_loss_cut", "signal_hard_cap",
                    "fast_exit_loss") and hold_days < MIN_HOLD:
                if cum_ret > -atr_stop:
                    new_position = 1

            # Exit confirmation (Module H)
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

            # MODULE I: TREND-CARRY OVERRIDE
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
                    "hybrid_entry": hybrid_entry,
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
        "n_fast_exit_loss": n_fast_exit_loss,
        "n_signal_hard_cap": n_signal_hard_cap,
        "n_time_decay_exit": n_time_decay_exit,
        "n_fixA_saved": n_fixA_saved,
        "n_fixB_saved": n_fixB_saved,
        "n_fixE_relaxed": n_fixE_relaxed,
        "n_fixF_hybrid": n_fixF_hybrid,
    }


SYMBOLS = "ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC"


def run_v22_variant(fix_flags, label=""):
    """Run a V22 variant with specific fix flags."""
    fA, fB, fC, fD, fE, fF = fix_flags

    def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        return backtest_v22(y_pred, returns, df_test, feature_cols,
                           fix_A=fA, fix_B=fB, fix_C=fC,
                           fix_D=fD, fix_E=fE, fix_F=fF,
                           **kwargs)

    trades = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                           backtest_fn=bt_fn)
    return trades


if __name__ == "__main__":
    print("=" * 130)
    print("V22 BREAKTHROUGH UPGRADE - SYSTEMATIC TESTING")
    print("=" * 130)

    # ═══════════════════════════════════════
    # PHASE 1: Test each fix individually
    # ═══════════════════════════════════════
    print("\n" + "=" * 130)
    print("PHASE 1: INDIVIDUAL FIX TESTING (each fix alone vs V19.3 baseline)")
    print("=" * 130)

    # Baseline V19.3
    print("\n  Running V19.3 baseline...")
    trades_base = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                                backtest_fn=backtest_v19_3)
    m_base = calc_metrics(trades_base)

    print("  Running Rule baseline...")
    trades_rule = run_rule_test(SYMBOLS)
    m_rule = calc_metrics(trades_rule)

    individual_fixes = {
        "V19.3 baseline":  (False, False, False, False, False, False),
        "fix_A only":      (True,  False, False, False, False, False),
        "fix_B only":      (False, True,  False, False, False, False),
        "fix_C only":      (False, False, True,  False, False, False),
        "fix_D only":      (False, False, False, True,  False, False),
        "fix_E only":      (False, False, False, False, True,  False),
        "fix_F only":      (False, False, False, False, False, True),
    }

    results = {}
    print(f"\n  {'Config':<25} | {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} "
          f"{'MaxLoss':>8} {'AvgHold':>8} | {'vs Base':>9} {'vs Rule':>9}")
    print("  " + "-" * 120)

    # Print baselines
    print(f"  {'V19.3 baseline':<25} | {m_base['trades']:>4} {m_base['wr']:>5.1f}% {m_base['avg_pnl']:>+7.2f}% "
          f"{m_base['total_pnl']:>+9.1f}% {m_base['pf']:>5.2f} {m_base['max_loss']:>+7.1f}% {m_base['avg_hold']:>6.1f}d | "
          f"{'---':>9} {m_base['total_pnl']-m_rule['total_pnl']:>+8.1f}%")
    print(f"  {'Rule':<25} | {m_rule['trades']:>4} {m_rule['wr']:>5.1f}% {m_rule['avg_pnl']:>+7.2f}% "
          f"{m_rule['total_pnl']:>+9.1f}% {m_rule['pf']:>5.2f} {m_rule['max_loss']:>+7.1f}% {m_rule['avg_hold']:>6.1f}d | "
          f"{m_rule['total_pnl']-m_base['total_pnl']:>+8.1f}% {'---':>9}")
    print("  " + "-" * 120)

    results["V19.3 baseline"] = (m_base, trades_base)
    results["Rule"] = (m_rule, trades_rule)

    for name, flags in individual_fixes.items():
        if name == "V19.3 baseline":
            continue
        print(f"  Running {name}...", end="", flush=True)
        trades = run_v22_variant(flags, name)
        m = calc_metrics(trades)
        results[name] = (m, trades)
        delta_base = m["total_pnl"] - m_base["total_pnl"]
        delta_rule = m["total_pnl"] - m_rule["total_pnl"]
        marker = " ***" if delta_base > 30 else (" ++" if delta_base > 10 else ("" if delta_base > -10 else " --"))
        print(f"\r  {name:<25} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d | "
              f"{delta_base:>+8.1f}% {delta_rule:>+8.1f}%{marker}")

    # ═══════════════════════════════════════
    # PHASE 2: Test promising combinations
    # ═══════════════════════════════════════
    print("\n\n" + "=" * 130)
    print("PHASE 2: COMBINATION TESTING")
    print("=" * 130)

    # Identify positive fixes
    positive_fixes = []
    for name, (m, _) in results.items():
        if name in ("V19.3 baseline", "Rule"):
            continue
        if m["total_pnl"] > m_base["total_pnl"]:
            positive_fixes.append(name)

    print(f"\n  Positive individual fixes: {positive_fixes}")

    combos = {
        "A+B":          (True,  True,  False, False, False, False),
        "A+C":          (True,  False, True,  False, False, False),
        "A+B+C":        (True,  True,  True,  False, False, False),
        "A+B+D":        (True,  True,  False, True,  False, False),
        "A+B+C+D":      (True,  True,  True,  True,  False, False),
        "A+B+E":        (True,  True,  False, False, True,  False),
        "A+B+C+E":      (True,  True,  True,  False, True,  False),
        "A+B+C+D+E":    (True,  True,  True,  True,  True,  False),
        "A+B+F":        (True,  True,  False, False, False, True),
        "A+B+C+F":      (True,  True,  True,  False, False, True),
        "A+B+C+D+F":    (True,  True,  True,  True,  False, True),
        "A+B+C+E+F":    (True,  True,  True,  False, True,  True),
        "ALL (A-F)":    (True,  True,  True,  True,  True,  True),
    }

    print(f"\n  {'Config':<25} | {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} "
          f"{'MaxLoss':>8} {'AvgHold':>8} | {'vs Base':>9} {'vs Rule':>9}")
    print("  " + "-" * 120)

    for name, flags in combos.items():
        print(f"  Running {name}...", end="", flush=True)
        trades = run_v22_variant(flags, name)
        m = calc_metrics(trades)
        results[name] = (m, trades)
        delta_base = m["total_pnl"] - m_base["total_pnl"]
        delta_rule = m["total_pnl"] - m_rule["total_pnl"]
        marker = " ***" if delta_base > 100 else (" ++" if delta_base > 30 else ("" if delta_base > -30 else " --"))
        print(f"\r  {name:<25} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d | "
              f"{delta_base:>+8.1f}% {delta_rule:>+8.1f}%{marker}")

    # ═══════════════════════════════════════
    # PHASE 3: Find best combination
    # ═══════════════════════════════════════
    print("\n\n" + "=" * 130)
    print("PHASE 3: RANKING & SELECTION")
    print("=" * 130)

    # Composite score: balanced between total PnL, PF, WR, MaxLoss
    ranked = []
    for name, (m, trades) in results.items():
        if name in ("V19.3 baseline", "Rule"):
            continue
        score = m["total_pnl"] * 0.4 + m["pf"] * 200 + m["wr"] * 5 - abs(m["max_loss"]) * 3
        ranked.append((score, name, m))

    ranked.sort(reverse=True)

    print(f"\n  {'Rank':>4} {'Config':<25} {'TotPnL':>10} {'PF':>6} {'WR':>6} {'MaxLoss':>8} {'Score':>8}")
    print("  " + "-" * 80)
    for rank, (score, name, m) in enumerate(ranked[:15], 1):
        marker = " <-- V22" if rank == 1 else ""
        print(f"  {rank:>4} {name:<25} {m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['wr']:>5.1f}% "
              f"{m['max_loss']:>+7.1f}% {score:>7.1f}{marker}")

    # ═══════════════════════════════════════
    # PHASE 4: Per-symbol analysis of best combo
    # ═══════════════════════════════════════
    best_name = ranked[0][1]
    best_m, best_trades = results[best_name]

    print(f"\n\n" + "=" * 130)
    print(f"PHASE 4: PER-SYMBOL ANALYSIS - BEST = {best_name}")
    print("=" * 130)

    df_best = pd.DataFrame(best_trades)
    df_base = pd.DataFrame(trades_base)
    df_rule = pd.DataFrame(trades_rule)

    print(f"\n  {'Sym':<6}| {'V19.3 Tot':>10} | {'Best Tot':>10} | {'Rule Tot':>10} | {'Best-V19.3':>11} | {'Best-Rule':>10} | {'Winner':>8}")
    print("  " + "-" * 90)

    syms = sorted(set(df_best["symbol"].unique()) | set(df_base["symbol"].unique()) | set(df_rule["symbol"].unique()))
    tot_best = tot_base = tot_rule = 0
    for sym in syms:
        mb = calc_metrics(df_base[df_base["symbol"] == sym].to_dict("records")) if sym in df_base["symbol"].values else {"total_pnl": 0}
        mt = calc_metrics(df_best[df_best["symbol"] == sym].to_dict("records")) if sym in df_best["symbol"].values else {"total_pnl": 0}
        mr = calc_metrics(df_rule[df_rule["symbol"] == sym].to_dict("records")) if sym in df_rule["symbol"].values else {"total_pnl": 0}
        tot_best += mt["total_pnl"]
        tot_base += mb["total_pnl"]
        tot_rule += mr["total_pnl"]
        d_base = mt["total_pnl"] - mb["total_pnl"]
        d_rule = mt["total_pnl"] - mr["total_pnl"]
        winner = "Best" if d_base > 0 and d_rule > 0 else ("Best>V19" if d_base > 0 else ("Worse" if d_base < -10 else "~same"))
        print(f"  {sym:<6}| {mb['total_pnl']:>+9.1f}% | {mt['total_pnl']:>+9.1f}% | {mr['total_pnl']:>+9.1f}% | "
              f"{d_base:>+10.1f}% | {d_rule:>+9.1f}% | {winner:>8}")
    print("  " + "-" * 90)
    print(f"  {'TOTAL':<6}| {tot_base:>+9.1f}% | {tot_best:>+9.1f}% | {tot_rule:>+9.1f}% | "
          f"{tot_best-tot_base:>+10.1f}% | {tot_best-tot_rule:>+9.1f}%")

    # Exit reason analysis
    print(f"\n  EXIT REASON ANALYSIS ({best_name}):")
    for reason, grp in df_best.groupby("exit_reason"):
        wins = len(grp[grp["pnl_pct"] > 0])
        total = grp["pnl_pct"].sum()
        avg = grp["pnl_pct"].mean()
        print(f"    {reason:<25}: {len(grp):>4} trades ({wins}W), WR={wins/len(grp)*100:>5.1f}%, "
              f"avg={avg:>+6.2f}%, total={total:>+8.1f}%")

    # By trend
    print(f"\n  BY TREND ({best_name}):")
    if "entry_trend" in df_best.columns:
        for trend in ["strong", "moderate", "weak"]:
            tt = df_best[df_best["entry_trend"] == trend]
            if len(tt) == 0:
                continue
            m = calc_metrics(tt.to_dict("records"))
            print(f"    {trend:<12}: {m['trades']:>4} trades, WR={m['wr']:>5.1f}%, avg={m['avg_pnl']:>+6.2f}%, "
                  f"total={m['total_pnl']:>+8.1f}%, PF={m['pf']:>5.2f}")

    print("\n" + "=" * 130)
    print(f"VERDICT: Best V22 config = {best_name}")
    print(f"  Total PnL: {best_m['total_pnl']:>+.1f}% (vs V19.3: {best_m['total_pnl']-m_base['total_pnl']:>+.1f}%, vs Rule: {best_m['total_pnl']-m_rule['total_pnl']:>+.1f}%)")
    print(f"  PF: {best_m['pf']:.2f}, WR: {best_m['wr']:.1f}%, MaxLoss: {best_m['max_loss']:+.1f}%")
    print("=" * 130)
