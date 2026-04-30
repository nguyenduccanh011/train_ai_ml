"""
V19.2 Signal Exit Overhaul.
===========================
Built from V19.1 with focus on ROOT CAUSE #1: Signal exit losses.
Changes:
1) SIGNAL_HARD_CAP = -12% (was -28.6% possible via signal exit)
2) FAST EXIT when losing: cum_ret < -5% and hold > 3d -> exit immediately
3) TIME DECAY: hold > 15d and cum_ret < 3% -> reduce threshold 40%
4) No confirm bars when losing (cum_ret < 0)
5) Weaker strong-trend override: only override if cum_ret > 3% (was > 0%)
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
from run_v11_compare import backtest_v11
from run_v17_compare import backtest_v17
from run_v18_compare import backtest_v18
from run_v19_compare import backtest_v19
from compare_rule_vs_model import backtest_rule


def backtest_v19_2(y_pred, returns, df_test, feature_cols,
                 initial_capital=100_000_000, commission=0.0015, tax=0.001,
                 record_trades=True,
                 mod_a=True,    # V-shape entry
                 mod_b=True,    # Profit-peak protection
                 mod_c=False,   # Fast loss cut (disabled)
                 mod_d=False,   # Adaptive exit confirm (disabled)
                 mod_e=True,    # Secondary breakout
                 mod_f=True,    # BO Quality Filter
                 mod_g=True,    # Bear Regime Defense
                 mod_h=True,    # Confirmed signal exit
                 mod_i=True,    # Trend-carry override
                 mod_j=True):   # Anti-chop entry filter
    """V19.2: Signal exit overhaul - hard cap losses + fast exit + time decay."""
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
    consecutive_below_ema8 = 0  # Module B

    # V11 params
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
    SIGNAL_HARD_CAP = 0.12  # V19.2: Max loss from signal exit = -12%

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

    # RSI for module A
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

    # ═══ MODULE E: Secondary breakout scanner (looser criteria) ═══
    secondary_breakout = np.zeros(n, dtype=bool)
    if mod_e:
        for i in range(10, n):
            prev_high = np.max(high[i - 10:i])
            prev_low = np.min(low[i - 10:i])
            ref = close[i - 1] if close[i - 1] > 0 else close[i]
            # Looser: range up to 10%
            tight_range = ((prev_high - prev_low) / ref) < 0.10 if ref > 0 else False
            uptrend_macro = (not np.isnan(sma20[i]) and not np.isnan(sma50[i])
                             and sma20[i] > sma50[i])
            # Lower volume bar if uptrend confirmed
            vol_threshold = 1.1 if uptrend_macro else 1.2
            vol_ok = volume[i] > vol_threshold * avg_vol20[i] if not np.isnan(avg_vol20[i]) else False
            # Use max_high_5d instead of prev_high (10d)
            max_high_5d = np.max(high[max(0, i - 5):i])
            breakout_5d = close[i] > max_high_5d
            if tight_range and breakout_5d and vol_ok and uptrend_macro:
                secondary_breakout[i] = True

    # ═══════════════════════════════════════
    # MODULE A: V-shape detection (precompute)
    # ═══════════════════════════════════════
    # A V-shape "confirmed reversal" bar at index i requires:
    #   1) Within last 5 bars, drop_from_peak_20 was <= -0.15 (deep drop)
    #   2) Prev RSI14 was < 35 OR drop > 18%
    #   3) Bullish reversal candle: close > open + range*0.5 AND close > prev_close
    #   4) Volume > 1.3x avg20
    # If confirmed, the next 5 bars (including bar i) get a bypass flag.
    vshape_bypass = np.zeros(n, dtype=bool)
    for i in range(15, n):
        # Check deep drop in past 5 bars
        had_deep_drop = False
        for j in range(max(0, i - 5), i + 1):
            if drop_from_peak_20[j] <= -0.15:
                had_deep_drop = True
                break
        if not had_deep_drop:
            continue
        # Reversal candle
        rng = high[i] - low[i]
        if rng <= 0:
            continue
        bullish = close[i] > opn[i] + rng * 0.5 and close[i] > close[i - 1]
        if not bullish:
            continue
        # Oversold or deeper drop
        oversold = (not np.isnan(rsi14[i - 1]) and rsi14[i - 1] < 35) or drop_from_peak_20[i] <= -0.18
        if not oversold:
            continue
        # Volume confirmation
        if np.isnan(avg_vol20[i]) or volume[i] < 1.3 * avg_vol20[i]:
            continue
        # Mark next 5 bars bypassable
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

        # Targeted drawdown control for historically unstable symbols.
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

        # ═══════════════════════════════════════
        # ENTRY LOGIC
        # ═══════════════════════════════════════
        quick_reentry = False
        breakout_entry = False
        vshape_entry = False

        if new_position == 0 and position == 0 and last_exit_reason == "trailing_stop":
            bars_since_exit = i - last_exit_bar
            if (bars_since_exit <= QUICK_REENTRY_WINDOW and
                trend in ("strong", "moderate") and
                macd_line[i] > 0 and
                not np.isnan(sma20[i]) and close[i] > sma20[i]):
                new_position = 1
                quick_reentry = True

        # ═══ MODULE F: BO Quality precondition ═══
        # Only allow breakout entries when MACD positive + bullish candle + heavy volume
        bo_quality_ok = True
        if mod_f:
            macd_pos = macd_hist[i] > 0
            bullish = close[i] > opn[i]
            heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
            bo_quality_ok = macd_pos and bullish and heavy_vol

        if new_position == 0 and position == 0 and consolidation_breakout[i] and bo_quality_ok:
            new_position = 1
            breakout_entry = True

        # ═══ MODULE E: Secondary breakout entry ═══
        if mod_e and new_position == 0 and position == 0 and secondary_breakout[i] and bo_quality_ok:
            new_position = 1
            breakout_entry = True
            n_secondary_breakout += 1

        # ═══ MODULE A: V-shape entry override ═══
        if mod_a and new_position == 0 and position == 0 and vshape_bypass[i]:
            # Override ML signal: enter on V-shape bypass even without raw_signal
            # but require basic trend support: close > EMA8 to confirm reversal continuing
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

        strong_breakout_context = (
            trend == "strong" and (bs >= 3 or vs > 1.5 or breakout_entry)
        )
        entry_alpha_ok = True

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
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

        # ret_5d & falling-knife — MODULE A bypasses these
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

        # ═══ MODULE G: BEAR REGIME DEFENSE ═══
        # Block all non-V-shape entries when in bear regime
        if mod_g and new_position == 1 and position == 0 and not vshape_entry and entry_alpha_ok:
            sma20_below_50 = (not np.isnan(sma20[i]) and not np.isnan(sma50[i])
                              and sma20[i] < sma50[i])
            close_below_50 = (not np.isnan(sma50[i]) and close[i] < sma50[i])
            deep_60d_loss = ret_60d[i] < -0.10
            if sma20_below_50 and close_below_50 and deep_60d_loss:
                entry_alpha_ok = False
                n_bear_blocked += 1

        # ═══ MODULE J: ANTI-CHOP ENTRY FILTER ═══
        # Block weak non-breakout entries in sideways/chop regimes.
        if mod_j and new_position == 1 and position == 0 and not vshape_entry and not breakout_entry and entry_alpha_ok:
            ma_flat = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and
                       abs(sma20[i] / sma50[i] - 1) < 0.02)
            weak_momo = abs(ret_20d[i]) < 0.06
            narrow_vol = bb < 0.45
            weak_trend = trend == "weak"
            if ma_flat and weak_momo and narrow_vol and weak_trend:
                entry_alpha_ok = False
                n_chop_blocked += 1

        if new_position == 1 and position == 0 and not entry_alpha_ok:
            new_position = 0
            n_v19_alpha_blocked += 1

        # Position sizing (separate layer after alpha pass)
        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            if vshape_entry:
                position_size = 0.6  # V-shape = early entry, smaller size
            elif trend == "strong" and entry_score >= 3:
                position_size = 1.0
            elif dp < 0.025:
                position_size = 0.5
            elif bb > 0.7:
                position_size = 0.7
            elif trend == "strong":
                position_size = 0.9
            else:
                position_size = 1.0

            if close[i] <= opn[i] and not vshape_entry:
                position_size *= 0.7

            if dist_sma20[i] > 0.05 and position_size > 0.5:
                position_size = 0.5
            if ret_5d[i] > ret5_hot:
                # Keep alpha but reduce size aggressively in overheat bars.
                position_size = min(position_size, 0.45 if not strong_breakout_context else 0.60)
                n_v19_overheat_entries += 1
            if dp < dp_floor:
                position_size = min(position_size, 0.55)
            position_size *= regime_cfg["size_mult"]
            position_size = max(0.30, min(position_size, 1.0))

        # ═══════════════════════════════════════
        # EXIT LOGIC (V11 base + Module B)
        # ═══════════════════════════════════════
        if position == 1:
            projected = equity[i - 1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            if close[i] > max_price_in_trade:
                max_price_in_trade = close[i]
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0
            # Price-based max for Module B (independent of position_size scaling)
            price_max_profit = (max_price_in_trade / entry_close - 1) if entry_close > 0 else 0
            price_cur_ret = (close[i] / entry_close - 1) if entry_close > 0 else 0

            in_uptrend = rs > 0 and hl >= 2
            strong_uptrend = trend == "strong"

            if not np.isnan(atr14[i]) and close[i] > 0:
                atr_stop = ATR_MULT * atr14[i] / close[i]
                atr_stop = max(0.025, min(atr_stop, 0.06))
            else:
                atr_stop = 0.04

            # 0) HARD STOP
            if cum_ret <= -HARD_STOP:
                new_position = 0
                exit_reason = "hard_stop"

            # 0.5) V19.2: SIGNAL HARD CAP using price return (not equity-weighted)
            elif price_cur_ret <= -SIGNAL_HARD_CAP:
                new_position = 0
                exit_reason = "signal_hard_cap"
                n_signal_hard_cap += 1

            # 0.6) V19.2: FAST EXIT when price drops > 5% and held > 3 days
            elif price_cur_ret < -0.05 and hold_days > 3:
                new_position = 0
                exit_reason = "fast_exit_loss"
                n_fast_exit_loss += 1

            # 0.7) V19.2: FAST EXIT - moderate price loss with bearish confirmation
            elif (price_cur_ret < -0.03 and hold_days > 2
                  and macd_hist[i] < 0
                  and (not np.isnan(ema8[i]) and close[i] < ema8[i])):
                new_position = 0
                exit_reason = "fast_exit_loss"
                n_fast_exit_loss += 1

            # 1) ATR stop loss
            elif cum_ret <= -atr_stop:
                new_position = 0
                exit_reason = "stop_loss"

            # ═══ MODULE C: FAST LOSS CUT ═══
            # In first 5 bars, exit immediately if losing with bearish confirmation
            elif (mod_c and hold_days < 5 and cum_ret < -0.03
                  and macd_hist[i] < 0 and close[i] < opn[i]):
                new_position = 0
                exit_reason = "fast_loss_cut"
                n_fast_loss_cut += 1

            # ═══ MODULE B: PROFIT-PEAK PROTECTION ═══
            # Independent layer — protects realized profit when peak reached
            elif mod_b and price_max_profit >= 0.20:
                # Aggressive protect: close < SMA10 + heavy volume = distribution
                price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
                bearish_candle = close[i] < opn[i]
                if price_below_sma10 and heavy_vol and bearish_candle:
                    new_position = 0
                    exit_reason = "peak_protect_dist"
                    n_peak_protect += 1

            # Module B tier 2: 15% peak + 2 consecutive close < EMA8
            if mod_b and new_position == 1 and price_max_profit >= 0.15:
                if not np.isnan(ema8[i]) and close[i] < ema8[i]:
                    consecutive_below_ema8 += 1
                else:
                    consecutive_below_ema8 = 0
                if consecutive_below_ema8 >= 2 and price_cur_ret < price_max_profit * 0.75:
                    new_position = 0
                    exit_reason = "peak_protect_ema"
                    n_peak_protect += 1

            # Hybrid exit (V11 strong-trend rule-based)
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

            # Adaptive trailing (V11 original — NO time-decay)
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

            # Min hold (Module B/C exits override min hold)
            if new_position == 0 and exit_reason not in (
                    "stop_loss", "hard_stop", "hybrid_exit", "peak_protect_dist",
                    "peak_protect_ema", "fast_loss_cut", "signal_hard_cap",
                    "fast_exit_loss") and hold_days < MIN_HOLD:
                if cum_ret > -atr_stop:
                    new_position = 1

            # Exit confirmation (Module D adaptive + Module H confirmed signal exit)
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
                    # V19.2: TIME DECAY using price return
                    if hold_days > 15 and price_cur_ret < 0.03:
                        score_threshold *= 0.60
                        n_time_decay_exit += 1
                    elif hold_days > 10 and price_cur_ret < 0.01:
                        score_threshold *= 0.75

                    bearish_confirm = bearish_score >= score_threshold
                    if old_bearish_confirm and not bearish_confirm:
                        n_v18_signal_quality_saves += 1
                        n_v19_exit_quality_saved += 1
                    if not bearish_confirm:
                        new_position = 1
                        n_confirmed_exit_blocked += 1

            if new_position == 0 and exit_reason == "signal":
                # V19.2: NO confirm bars when price is losing
                if price_cur_ret < 0:
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

            # Strong trend override (V19.2: only if price is meaningfully profitable)
            if new_position == 0 and exit_reason == "signal":
                if price_cur_ret > 0.03 and trend == "strong":
                    new_position = 1

            # ═══ MODULE I: TREND-CARRY OVERRIDE ═══
            # Keep profitable trades alive when trend health is still acceptable.
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


def run_test(symbols_str, mod_a, mod_b, mod_c=False, mod_d=False, mod_e=False,
             mod_f=False, mod_g=False, mod_h=False, mod_i=False, mod_j=False,
             backtest_fn=backtest_v19_2):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    config = {
        "data": {"data_dir": data_dir},
        "split": {"method": "walk_forward", "train_years": 4, "test_years": 1,
                  "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025},
        "target": {"type": "trend_regime", "trend_method": "dual_ma",
                   "short_window": 5, "long_window": 20, "classes": 3},
    }

    pick = [s.strip() for s in symbols_str.split(",")]
    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    raw_df = loader.load_all(symbols=pick)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    all_trades = []
    for window, train_df, test_df in splitter.split(df):
        model = build_model("lightgbm")
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        for sym in test_df["symbol"].unique():
            if sym not in pick:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values

            r = backtest_fn(y_pred, rets, sym_test, feature_cols,
                            mod_a=mod_a, mod_b=mod_b,
                            mod_c=mod_c, mod_d=mod_d, mod_e=mod_e,
                            mod_f=mod_f, mod_g=mod_g,
                            mod_h=mod_h, mod_i=mod_i, mod_j=mod_j)
            for t in r["trades"]:
                t["symbol"] = sym
            all_trades.extend(r["trades"])

    return all_trades


def run_rule_test(symbols_str):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    pick = [s.strip() for s in symbols_str.split(",")]
    loader = DataLoader(data_dir)
    symbols = [s for s in pick if s in loader.symbols]
    raw_df = loader.load_all(symbols=symbols)
    all_trades = []
    for sym in symbols:
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])
        sym_test = sym_data[sym_data[date_col] >= "2020-01-01"].reset_index(drop=True)
        if len(sym_test) < 50:
            continue
        trades = backtest_rule(sym_test)
        for t in trades:
            t["symbol"] = sym
        all_trades.extend(trades)
    return all_trades


def calc_metrics(trades):
    if not trades:
        return {"trades": 0, "wr": 0, "avg_pnl": 0, "total_pnl": 0, "pf": 0, "max_loss": 0, "avg_hold": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = wins / n * 100
    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    total_pnl = sum(t["pnl_pct"] for t in trades)
    gp = sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0)
    gl = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))
    pf = gp / gl if gl > 0 else 99
    max_loss = min(t["pnl_pct"] for t in trades)
    avg_hold = np.mean([t.get("holding_days", 0) for t in trades])
    return {"trades": n, "wr": wr, "avg_pnl": avg_pnl, "total_pnl": total_pnl, "pf": pf,
            "max_loss": max_loss, "avg_hold": avg_hold}



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", type=str,
                        default="ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC")
    args = parser.parse_args()

    print("=" * 132)
    print(f"V19.1 COMPARISON (V11/V17/V18/V19/V19.1 + Rule) - Symbols: {args.pick}")
    print("=" * 132)
    print("H=Confirmed signal exit   I=Trend-carry override   J=Anti-chop entry filter")
    print()

    # a, b, c, d, e, f, g, h, i, j
    configs = [
        ("V11 baseline               ", backtest_v17, False, False, False, False, False, False, False, False, False, False),
        ("V17 full (A+B+E+F+G+H+I+J) ", backtest_v17, True,  True,  False, False, True,  True,  True,  True,  True,  True),
        ("V18 adaptive (entry+exit)  ", backtest_v18, True,  True,  False, False, True,  True,  True,  True,  True,  True),
        ("V19 adaptive+regime        ", backtest_v19, True,  True,  False, False, True,  True,  True,  True,  True,  True),
        ("V19.1 risk-tuned           ", backtest_v19_1, True,  True,  False, False, True,  True,  True,  True,  True,  True),
    ]

    print(f"{'Config':<31} | {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} "
          f"{'MaxLoss':>8} {'AvgHold':>8}")
    print("-" * 132)

    results = {}
    for name, fn, ma, mb, mc, md, me, mf, mg, mh, mi, mj in configs:
        trades = run_test(args.pick, ma, mb, mc, md, me, mf, mg, mh, mi, mj, backtest_fn=fn)
        m = calc_metrics(trades)
        results[name] = (m, trades)
        print(f"{name:<31} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d")

    rule_trades = run_rule_test(args.pick)
    m_rule = calc_metrics(rule_trades)
    results["Rule-based                 "] = (m_rule, rule_trades)
    print(f"{'Rule-based':<31} | {m_rule['trades']:>4} {m_rule['wr']:>5.1f}% {m_rule['avg_pnl']:>+7.2f}% "
          f"{m_rule['total_pnl']:>+9.1f}% {m_rule['pf']:>5.2f} {m_rule['max_loss']:>+7.1f}% {m_rule['avg_hold']:>6.1f}d")

    print("\n" + "=" * 132)
    print("PER-SYMBOL: V11 vs V17 vs V18 vs V19 vs V19.1 vs Rule")
    print("=" * 132)

    t11 = results["V11 baseline               "][1]
    t17 = results["V17 full (A+B+E+F+G+H+I+J) "][1]
    t18 = results["V18 adaptive (entry+exit)  "][1]
    t19 = results["V19 adaptive+regime        "][1]
    t191 = results["V19.1 risk-tuned           "][1]
    trule = results["Rule-based                 "][1]
    b11, b17, b18, b19, b191, br = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for t in t11: b11[t["symbol"]].append(t)
    for t in t17: b17[t["symbol"]].append(t)
    for t in t18: b18[t["symbol"]].append(t)
    for t in t19: b19[t["symbol"]].append(t)
    for t in t191: b191[t["symbol"]].append(t)
    for t in trule: br[t["symbol"]].append(t)

    print(f"{'Sym':<6}| {'V11 Tot':>9} {'V11 PF':>6} | {'V17 Tot':>9} {'V17 PF':>6} | "
          f"{'V18 Tot':>9} {'V18 PF':>6} | {'V19 Tot':>9} {'V19 PF':>6} | "
          f"{'V19.1 Tot':>10} {'V19.1 PF':>8} | {'Rule Tot':>9} {'Rule PF':>7} | {'V19.1-V19':>10} {'V19.1-Rule':>11}")
    print("-" * 132)
    T11 = T17 = T18 = T19 = T191 = TR = 0
    for sym in sorted(set(b11.keys()) | set(b17.keys()) | set(b18.keys()) | set(b19.keys()) | set(b191.keys()) | set(br.keys())):
        m11 = calc_metrics(b11.get(sym, []))
        m17 = calc_metrics(b17.get(sym, []))
        m18 = calc_metrics(b18.get(sym, []))
        m19 = calc_metrics(b19.get(sym, []))
        m191 = calc_metrics(b191.get(sym, []))
        mr = calc_metrics(br.get(sym, []))
        T11 += m11["total_pnl"]; T17 += m17["total_pnl"]; T18 += m18["total_pnl"]; T19 += m19["total_pnl"]; T191 += m191["total_pnl"]; TR += mr["total_pnl"]
        print(f"{sym:<6}| {m11['total_pnl']:>+8.1f}% {m11['pf']:>5.2f} | "
              f"{m17['total_pnl']:>+8.1f}% {m17['pf']:>5.2f} | "
              f"{m18['total_pnl']:>+8.1f}% {m18['pf']:>5.2f} | "
              f"{m19['total_pnl']:>+8.1f}% {m19['pf']:>5.2f} | "
              f"{m191['total_pnl']:>+9.1f}% {m191['pf']:>7.2f} | "
              f"{mr['total_pnl']:>+8.1f}% {mr['pf']:>6.2f} | "
              f"{m191['total_pnl']-m19['total_pnl']:>+9.1f}% {m191['total_pnl']-mr['total_pnl']:>+10.1f}%")
    print("-" * 132)
    print(f"{'TOT':<6}| {T11:>+8.1f}% {'':>6} | {T17:>+8.1f}% {'':>6} | "
          f"{T18:>+8.1f}% {'':>6} | {T19:>+8.1f}% {'':>6} | {T191:>+9.1f}% {'':>8} | {TR:>+8.1f}% {'':>7} | {T191-T19:>+9.1f}% {T191-TR:>+10.1f}%")
