"""
V13 — Improvements over V12 based on trade analysis
=====================================================
Key changes from V12:
1. ATR_MULT 1.8→2.2 (reduce stop_loss frequency)
2. HARD_STOP 0.08→0.10 (more room)
3. PROFIT_LOCK_THRESHOLD 0.12→0.20, PROFIT_LOCK_MIN 0.06→0.10
4. Time-decay only when RSI<50 or MACD negative (not unconditional)
5. Relaxed entry: cooldown 5→3, consecutive pred not required in moderate trend
6. Wider trailing in strong trend (STRONG_TREND_TRAIL_MULT 0.45→0.35)
7. Profit floor: only trigger when lost >70% of peak (was 65%)
8. ZOMBIE_BARS 14→11
9. MIN_HOLD 6→5
"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v7_compare import summarize
from run_v11_compare import backtest_v11
from run_v12_compare import backtest_v12


def backtest_v13(y_pred, returns, df_test, feature_cols,
                 initial_capital=100_000_000, commission=0.0015, tax=0.001,
                 record_trades=True):
    """V13: V12 + wider stops, relaxed entries, smarter trailing."""
    n = len(y_pred)
    equity = np.zeros(n)
    equity[0] = initial_capital
    position = 0
    trades = []
    current_entry_day = 0
    entry_equity = 0
    max_equity_in_trade = 0
    hold_days = 0
    position_size = 1.0
    consecutive_exit_signals = 0

    # === V13 TUNED PARAMETERS ===
    MIN_HOLD = 5              # was 6
    ZOMBIE_BARS = 11          # was 14
    EXIT_CONFIRM = 3
    PROFIT_LOCK_THRESHOLD = 0.20  # was 0.12
    PROFIT_LOCK_MIN = 0.10        # was 0.06
    HARD_STOP = 0.10              # was 0.08
    ATR_MULT = 2.2                # was 1.8

    COOLDOWN_AFTER_BIG_LOSS = 3   # was 5
    QUICK_REENTRY_WINDOW = 3
    STRONG_TREND_TRAIL_MULT = 0.35  # was 0.45 (wider = hold longer)

    cooldown_remaining = 0
    last_exit_price = 0
    last_exit_reason = ""
    last_exit_bar = -999
    entry_close = 0
    last_new_high_bar = 0
    max_price_in_trade = 0

    # Feature arrays
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

    sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
    sma50 = pd.Series(close).rolling(50, min_periods=10).mean().values
    ema12 = pd.Series(close).ewm(span=12, min_periods=8).mean().values
    ema26 = pd.Series(close).ewm(span=26, min_periods=15).mean().values
    macd_line = ema12 - ema26
    macd_signal = pd.Series(macd_line).ewm(span=9, min_periods=5).mean().values
    macd_hist = macd_line - macd_signal

    rsi_14 = np.full(n, 50.0)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss_arr = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=5).mean().values
    avg_loss = pd.Series(loss_arr).rolling(14, min_periods=5).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_val = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
        rsi_14 = 100 - 100 / (1 + rs_val)
    rsi_14 = np.where(np.isnan(rsi_14), 50, rsi_14)

    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().values

    local_low_20 = pd.Series(close).rolling(20, min_periods=5).min().values
    avg_vol20 = pd.Series(volume).rolling(20, min_periods=5).mean().values
    ret_5d = np.zeros(n)
    if n > 5:
        base_5d = close[:-5]
        ret_5d[5:] = np.where(base_5d > 0, close[5:] / base_5d - 1, 0)
    dist_sma20 = np.where((~np.isnan(sma20)) & (sma20 > 0), close / sma20 - 1, 0)
    roll_high_20 = pd.Series(close).rolling(20, min_periods=5).max().values
    drop_from_peak_20 = np.where(roll_high_20 > 0, close / roll_high_20 - 1, 0)

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

    days_above_ma20 = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(sma20[i]) and close[i] > sma20[i]:
            days_above_ma20[i] = days_above_ma20[i - 1] + 1
        else:
            days_above_ma20[i] = 0

    rolling_high_250 = pd.Series(close).rolling(250, min_periods=20).max().values
    dist_from_52w_high = np.where(rolling_high_250 > 0, (close / rolling_high_250), 1.0)

    # V13 NEW: Momentum reentry detection
    momentum_surge = np.zeros(n, dtype=bool)
    for i in range(2, n):
        above_sma20 = not np.isnan(sma20[i]) and close[i] > sma20[i]
        was_below = not np.isnan(sma20[i-1]) and close[i-1] <= sma20[i-1]
        vol_surge = volume[i] > 1.5 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else False
        macd_pos = macd_line[i] > 0
        if above_sma20 and was_below and vol_surge and macd_pos:
            momentum_surge[i] = True

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

    entry_features = {}

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

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # ═══════════════════════════════════════
        # ENTRY LOGIC (V13: relaxed)
        # ═══════════════════════════════════════
        quick_reentry = False
        breakout_entry = False
        momentum_entry = False

        # Quick reentry after trailing stop
        if new_position == 0 and position == 0 and last_exit_reason == "trailing_stop":
            bars_since_exit = i - last_exit_bar
            if (bars_since_exit <= QUICK_REENTRY_WINDOW and
                trend in ("strong", "moderate") and
                macd_line[i] > 0 and
                not np.isnan(sma20[i]) and close[i] > sma20[i]):
                new_position = 1
                quick_reentry = True

        # Consolidation breakout
        if new_position == 0 and position == 0 and consolidation_breakout[i]:
            new_position = 1
            breakout_entry = True

        # V13 NEW: Momentum surge reentry
        if new_position == 0 and position == 0 and momentum_surge[i]:
            if trend in ("strong", "moderate"):
                new_position = 1
                momentum_entry = True

        # Cooldown (V13: reduced from 5 to 3)
        if new_position == 1 and position == 0 and not quick_reentry and not momentum_entry:
            if cooldown_remaining > 0:
                new_position = 0

        # Reentry price filter
        if new_position == 1 and position == 0 and not quick_reentry and not momentum_entry:
            if last_exit_price > 0 and last_exit_reason != "trailing_stop":
                price_diff = abs(close[i] / last_exit_price - 1)
                if price_diff < 0.03:
                    new_position = 0

        # Consecutive prediction (V13: relaxed for moderate trend)
        if new_position == 1 and position == 0 and not quick_reentry and not breakout_entry and not momentum_entry:
            prev_pred = int(y_pred[i - 2]) if i >= 2 else 0
            if bs >= 4 and vs > 1.2:
                pass
            elif trend == "strong" and rs > 0:
                pass
            elif trend == "moderate" and rs > 0 and macd_line[i] > 0:
                pass  # V13: allow moderate trend entry without consecutive pred
            elif prev_pred != 1:
                new_position = 0

        # Below both MAs filter
        if new_position == 1 and position == 0 and not quick_reentry and not momentum_entry:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3 and not breakout_entry:
                        new_position = 0

        # Entry score filter (V13: slightly relaxed)
        if new_position == 1 and position == 0 and not quick_reentry and not momentum_entry:
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
            elif trend == "moderate" and in_uptrend_macro:
                min_score = 1  # V13: relaxed from 2/3
            elif (near_sma_support or near_local_low) and in_uptrend_macro:
                min_score = 2
            elif in_uptrend_macro and rs > 0:
                min_score = 2
            else:
                min_score = 3

            if entry_score < min_score and not breakout_entry:
                new_position = 0
            if wp > 0.9 and rs <= 0 and bs < 2 and trend == "weak" and not breakout_entry:
                new_position = 0
            if bb > 0.85 and bs < 2 and entry_score < 4 and trend == "weak" and not breakout_entry:
                new_position = 0
            if new_position == 1:
                if wp > 0.78 and bb < 0.35 and trend == "weak" and not breakout_entry:
                    new_position = 0
            if new_position == 1 and dp < 0.025:
                if entry_score < 4 and trend == "weak" and not breakout_entry:
                    new_position = 0

        # Overextended filter
        if new_position == 1 and position == 0:
            if ret_5d[i] > 0.06:  # V13: slightly relaxed from 0.05
                new_position = 0

        # Crash filter
        if new_position == 1 and position == 0:
            if drop_from_peak_20[i] <= -0.15 and not stabilized_sideways[i]:
                new_position = 0

        # Volume floor
        if new_position == 1 and position == 0:
            vol_floor = 0.7 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else 0
            if vol_floor > 0 and volume[i] < vol_floor:
                new_position = 0

        # RSI Slope Cap (V13: raised from 12 to 15)
        if new_position == 1 and position == 0 and not quick_reentry and not momentum_entry:
            if rs > 15 and trend != "strong":
                new_position = 0

        # Position sizing
        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            if trend == "strong" and entry_score >= 3:
                position_size = 1.0
            elif dp < 0.025:
                position_size = 0.5
            elif bb > 0.7:
                position_size = 0.7
            elif trend == "strong":
                position_size = 0.9
            else:
                position_size = 1.0

            if close[i] <= opn[i]:
                position_size *= 0.7

            if dist_sma20[i] > 0.05 and position_size > 0.5:
                position_size = 0.5

        # ═══════════════════════════════════════
        # EXIT LOGIC (V13: wider, smarter)
        # ═══════════════════════════════════════
        if position == 1:
            price_ret = (close[i] / entry_close - 1) if entry_close > 0 else 0
            if close[i] > max_price_in_trade:
                max_price_in_trade = close[i]
                last_new_high_bar = i
            max_price_ret = (max_price_in_trade / entry_close - 1) if entry_close > 0 else 0

            cum_ret = price_ret
            max_profit = max_price_ret

            in_uptrend = rs > 0 and hl >= 2
            strong_uptrend = trend == "strong"

            if not np.isnan(atr14[i]) and close[i] > 0:
                atr_stop = ATR_MULT * atr14[i] / close[i]
                atr_stop = max(0.03, min(atr_stop, 0.07))  # V13: wider range
            else:
                atr_stop = 0.05

            # 0) HARD STOP (V13: 10%)
            if cum_ret <= -HARD_STOP:
                new_position = 0
                exit_reason = "hard_stop"

            # 1) ATR stop loss (V13: wider)
            elif cum_ret <= -atr_stop:
                new_position = 0
                exit_reason = "stop_loss"

            # PROFIT FLOOR LOCK (V13: only when lost >70% of peak, peak>25%)
            elif max_profit > 0.25 and cum_ret < max_profit * 0.30:
                new_position = 0
                exit_reason = "profit_floor"

            # Hybrid exit (strong uptrend)
            elif strong_uptrend and cum_ret > 0.05 and max_profit > 0.08:
                macd_bearish = macd_hist[i] < 0 and macd_hist[i - 1] >= 0 if i > 0 else False
                price_below_ma20 = close[i] < sma20[i] if not np.isnan(sma20[i]) else False

                if macd_bearish and price_below_ma20:
                    new_position = 0
                    exit_reason = "hybrid_exit"
                elif price_below_ma20 and cum_ret < max_profit * 0.4:  # V13: was 0.5
                    new_position = 0
                    exit_reason = "hybrid_exit"
                else:
                    new_position = 1

            # Adaptive trailing stop (V13: wider)
            elif max_profit > 0.03 and new_position == 1:
                if max_profit > 0.30:
                    trail_pct = 0.15  # V13: was not this tier
                elif max_profit > 0.25:
                    trail_pct = 0.20  # V13: was 0.18
                elif max_profit > 0.15:
                    trail_pct = 0.28  # V13: was 0.25
                elif max_profit > 0.08:
                    trail_pct = 0.45  # V13: was 0.40
                else:
                    trail_pct = 0.70  # V13: was 0.65

                if strong_uptrend:
                    trail_pct *= STRONG_TREND_TRAIL_MULT  # 0.35 (wider than V12's 0.45)
                elif trend == "moderate":
                    trail_pct *= 0.6  # V13: was 0.7

                # Time-decay — V13: ONLY when RSI<50 or MACD negative
                bars_since_high = i - last_new_high_bar
                rsi_weak = rsi_14[i] < 50
                macd_neg = macd_line[i] < 0
                if bars_since_high > 7 and (rsi_weak or macd_neg):
                    trail_pct *= 0.5
                elif bars_since_high > 4 and (rsi_weak and macd_neg):
                    trail_pct *= 0.7

                giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                if giveback >= trail_pct:
                    new_position = 0
                    exit_reason = "trailing_stop"

            # Profit lock (V13: higher thresholds)
            if new_position == 1 and max_profit >= PROFIT_LOCK_THRESHOLD:
                if cum_ret < PROFIT_LOCK_MIN:
                    if not strong_uptrend:
                        new_position = 0
                        exit_reason = "profit_lock"

            # Zombie exit (V13: ZOMBIE_BARS=11)
            if new_position == 1 and hold_days >= ZOMBIE_BARS and cum_ret < 0.01:
                if not strong_uptrend:
                    new_position = 0
                    exit_reason = "zombie_exit"

            # Min hold (V13: MIN_HOLD=5)
            if new_position == 0 and exit_reason not in ("stop_loss", "hard_stop", "hybrid_exit", "profit_floor") and hold_days < MIN_HOLD:
                if cum_ret > -atr_stop:
                    new_position = 1

            # Exit confirmation
            if new_position == 0 and exit_reason == "signal":
                if raw_signal == 0:
                    consecutive_exit_signals += 1
                else:
                    consecutive_exit_signals = 0
                if consecutive_exit_signals < EXIT_CONFIRM:
                    new_position = 1
                else:
                    consecutive_exit_signals = 0

            # Strong trend override (V13: more generous hold)
            if new_position == 0 and exit_reason == "signal":
                if cum_ret > 0 and trend == "strong":
                    if max_profit < 0.15 or cum_ret > max_profit * 0.45:  # V13: was 0.10/0.50
                        new_position = 1

        else:
            consecutive_exit_signals = 0

        # ═══════════════════════════════════════
        # EXECUTE
        # ═══════════════════════════════════════
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
                entry_close = close[i]
                max_price_in_trade = close[i]
                last_new_high_bar = i
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
                    "momentum_entry": momentum_entry,
                    "entry_ret_5d": round(ret_5d[i] * 100, 2),
                    "entry_drop20d": round(drop_from_peak_20[i] * 100, 2),
                    "entry_dist_sma20": round(dist_sma20[i] * 100, 2),
                }
            else:
                cost = equity[i - 1] * position_size * (commission + tax)
                pnl_pct_now = (close[i] / entry_close - 1) * 100 if entry_close > 0 else 0
                if pnl_pct_now < -5:
                    cooldown_remaining = COOLDOWN_AFTER_BIG_LOSS
                else:
                    cooldown_remaining = 2  # V13: shorter normal cooldown

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
                position_size = 1.0

        if position == 1:
            equity[i] = equity[i - 1] * (1 + ret * position_size) - cost
            hold_days += 1
        else:
            equity[i] = equity[i - 1] - cost
        position = new_position

    if position == 1 and entry_close > 0 and record_trades:
        pnl_pct = (close[-1] / entry_close - 1) * 100
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
    }


def run_comparison(symbols_str="VND"):
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

    results = {"v11": defaultdict(list), "v12": defaultdict(list), "v13": defaultdict(list)}

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

            r11 = backtest_v11(y_pred, rets, sym_test, feature_cols)
            r12 = backtest_v12(y_pred, rets, sym_test, feature_cols)
            r13 = backtest_v13(y_pred, rets, sym_test, feature_cols)

            for t in r11["trades"]:
                t["symbol"] = sym
            for t in r12["trades"]:
                t["symbol"] = sym
            for t in r13["trades"]:
                t["symbol"] = sym

            results["v11"][sym].extend(r11["trades"])
            results["v12"][sym].extend(r12["trades"])
            results["v13"][sym].extend(r13["trades"])

    return results


def calc_metrics(trades):
    if not trades:
        return {"trades": 0, "wr": 0, "avg_pnl": 0, "total_pnl": 0, "pf": 0, "max_loss": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = wins / n * 100
    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    total_pnl = sum(t["pnl_pct"] for t in trades)
    gross_profit = sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0)
    gross_loss = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99
    max_loss = min(t["pnl_pct"] for t in trades)
    return {"trades": n, "wr": round(wr, 1), "avg_pnl": round(avg_pnl, 2),
            "total_pnl": round(total_pnl, 1), "pf": round(pf, 2), "max_loss": round(max_loss, 1)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", type=str, default="VND")
    parser.add_argument("--all-symbols", action="store_true")
    args = parser.parse_args()

    if args.all_symbols:
        symbols_str = "ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC"
    else:
        symbols_str = args.pick

    print("=" * 100)
    print(f"V13 vs V12 vs V11 COMPARISON — Symbols: {symbols_str}")
    print("=" * 100)

    results = run_comparison(symbols_str)

    # Overall comparison
    print(f"\n{'Version':<8} | {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>9} {'PF':>6} {'MaxLoss':>8}")
    print("-" * 60)
    for ver in ["v11", "v12", "v13"]:
        all_trades = []
        for sym_trades in results[ver].values():
            all_trades.extend(sym_trades)
        m = calc_metrics(all_trades)
        print(f"{ver.upper():<8} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+8.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}%")

    # Per-symbol
    print(f"\n{'Symbol':<6} | {'V11 PnL':>8} {'V12 PnL':>8} {'V13 PnL':>8} | {'V11 WR':>7} {'V12 WR':>7} {'V13 WR':>7} | {'V11#':>4} {'V12#':>4} {'V13#':>4}")
    print("-" * 95)
    for sym in sorted(set(list(results["v11"].keys()) + list(results["v12"].keys()) + list(results["v13"].keys()))):
        m11 = calc_metrics(results["v11"].get(sym, []))
        m12 = calc_metrics(results["v12"].get(sym, []))
        m13 = calc_metrics(results["v13"].get(sym, []))
        print(f"{sym:<6} | {m11['total_pnl']:>+8.1f} {m12['total_pnl']:>+8.1f} {m13['total_pnl']:>+8.1f} | "
              f"{m11['wr']:>6.1f}% {m12['wr']:>6.1f}% {m13['wr']:>6.1f}% | "
              f"{m11['trades']:>4} {m12['trades']:>4} {m13['trades']:>4}")

    print("\n" + "=" * 100)
