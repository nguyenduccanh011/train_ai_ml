"""
V10 vs V9 — Fix premature exits in strong uptrends
====================================================
Changes from V9:
  1. ADAPTIVE TRAILING STOP: Trend-aware trailing — wide in strong trends, tight otherwise
  2. QUICK RE-ENTRY: After trailing_stop exit, allow fast re-entry if trend intact
  3. TREND REGIME DETECTION: Track consecutive days above MA20, detect mega-rallies
  4. HYBRID EXIT: In strong uptrends, use rule-based exit (MACD/MA20) instead of trailing
  5. POSITION SIZING: Scale up in strong confirmed trends
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
from run_v10_compare import backtest_v10


def backtest_v11(y_pred, returns, df_test, feature_cols,
                 initial_capital=100_000_000, commission=0.0015, tax=0.001,
                 record_trades=True):
    """V11: V10 exits + stricter entry quality filters + breakout scanner."""
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

    # V10 params (inherited from V9 where unchanged)
    MIN_HOLD = 6
    ZOMBIE_BARS = 14           # V10: slightly longer than V9(12) — give trends more time
    EXIT_CONFIRM = 3
    PROFIT_LOCK_THRESHOLD = 0.12  # V10: higher threshold (V9=0.10) — let winners run more
    PROFIT_LOCK_MIN = 0.06        # V10: lock 6% (V9=5%)
    HARD_STOP = 0.08
    ATR_MULT = 1.8

    # V10 NEW params
    COOLDOWN_AFTER_BIG_LOSS = 5
    QUICK_REENTRY_WINDOW = 3      # bars after trailing_stop exit to allow quick re-entry
    STRONG_TREND_TRAIL_MULT = 0.45  # multiply trail_pct by this in strong trends (wider)

    cooldown_remaining = 0
    last_exit_price = 0
    last_exit_reason = ""
    last_exit_bar = -999
    entry_close = 0

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

    # Moving averages
    sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
    sma50 = pd.Series(close).rolling(50, min_periods=10).mean().values

    # V10: EMA12 and EMA26 for MACD
    ema12 = pd.Series(close).ewm(span=12, min_periods=8).mean().values
    ema26 = pd.Series(close).ewm(span=26, min_periods=15).mean().values
    macd_line = ema12 - ema26
    macd_signal = pd.Series(macd_line).ewm(span=9, min_periods=5).mean().values
    macd_hist = macd_line - macd_signal

    # ATR
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

    # V11: stabilization check after deep drops (5-10 bars with <5% range)
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

    # V11: new consolidation breakout scanner
    consolidation_breakout = np.zeros(n, dtype=bool)
    for i in range(10, n):
        prev_high = np.max(high[i - 10:i])
        prev_low = np.min(low[i - 10:i])
        ref = close[i - 1] if close[i - 1] > 0 else close[i]
        tight_range = ((prev_high - prev_low) / ref) < 0.08 if ref > 0 else False
        vol_ok = volume[i] > 1.2 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else False
        if tight_range and close[i] > prev_high and vol_ok:
            consolidation_breakout[i] = True

    # V10: Trend regime — consecutive days above MA20
    days_above_ma20 = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(sma20[i]) and close[i] > sma20[i]:
            days_above_ma20[i] = days_above_ma20[i - 1] + 1
        else:
            days_above_ma20[i] = 0

    # V10: 52-week (250-day) high distance
    rolling_high_250 = pd.Series(close).rolling(250, min_periods=20).max().values
    dist_from_52w_high = np.where(rolling_high_250 > 0, (close / rolling_high_250), 1.0)

    date_col = "date" if "date" in df_test.columns else ("timestamp" if "timestamp" in df_test.columns else None)
    dates = df_test[date_col].values if date_col else np.arange(n)
    symbols = df_test["symbol"].values if "symbol" in df_test.columns else ["?"] * n

    def gf(name, idx):
        return feat_arrays[name][idx] if idx < n else defaults.get(name, 0)

    def detect_trend_strength(i):
        """
        Returns: 'strong', 'moderate', or 'weak'
        Strong = price > MA20 > MA50, days_above_ma20 >= 10, MACD > 0
        """
        if i < 1:
            return "weak"
        ma20_ok = not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i]
        price_above = close[i] > sma20[i] if not np.isnan(sma20[i]) else False
        macd_pos = macd_line[i] > 0
        days_ab = days_above_ma20[i]
        near_high = dist_from_52w_high[i] > 0.90  # within 10% of 52w high

        score = sum([ma20_ok, price_above, macd_pos, days_ab >= 10, days_ab >= 20, near_high])
        if score >= 4:
            return "strong"
        elif score >= 2:
            return "moderate"
        return "weak"

    entry_features = {}
    n_stop_loss = 0
    n_trailing_stop = 0
    n_zombie_exit = 0
    n_profit_lock = 0
    n_hard_stop = 0
    n_min_hold_saved = 0
    n_quick_reentry = 0
    n_hybrid_exit = 0
    n_breakout_entries = 0
    n_late_entry_blocked = 0
    n_falling_knife_blocked = 0
    n_low_volume_blocked = 0
    n_overextended_capped = 0
    n_bearish_penalized = 0

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

        # ════════════════════════════════════════════════
        # ENTRY LOGIC — V10: QUICK RE-ENTRY + ADAPTIVE
        # ════════════════════════════════════════════════

        # V10 CHANGE #2: Quick re-entry after trailing_stop exit
        quick_reentry = False
        breakout_entry = False
        if new_position == 0 and position == 0 and last_exit_reason == "trailing_stop":
            bars_since_exit = i - last_exit_bar
            if (bars_since_exit <= QUICK_REENTRY_WINDOW and
                trend in ("strong", "moderate") and
                macd_line[i] > 0 and
                not np.isnan(sma20[i]) and close[i] > sma20[i]):
                # Quick re-entry: override ML signal
                new_position = 1
                quick_reentry = True
                n_quick_reentry += 1

        # V11: new scanner-driven breakout entry
        if new_position == 0 and position == 0 and consolidation_breakout[i]:
            new_position = 1
            breakout_entry = True
            n_breakout_entries += 1

        if new_position == 1 and position == 0 and not quick_reentry:
            if cooldown_remaining > 0:
                new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry:
            if last_exit_price > 0 and last_exit_reason != "trailing_stop":
                price_diff = abs(close[i] / last_exit_price - 1)
                if price_diff < 0.03:
                    new_position = 0

        # V9-style: 1-day confirm for strong breakouts, 2-day for others
        if new_position == 1 and position == 0 and not quick_reentry and not breakout_entry:
            prev_pred = int(y_pred[i - 2]) if i >= 2 else 0
            if bs >= 4 and vs > 1.2:
                pass  # accept single-day
            elif trend == "strong" and rs > 0:
                pass  # V10: accept single-day in strong trend
            elif prev_pred != 1:
                new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3 and not breakout_entry:
                        new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            near_sma_support = (not np.isnan(sma20[i]) and
                               close[i] <= sma20[i] * 1.02 and
                               close[i] >= sma20[i] * 0.97)
            near_local_low = (not np.isnan(local_low_20[i]) and
                             close[i] <= local_low_20[i] * 1.05)
            in_uptrend_macro = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and
                               sma20[i] > sma50[i])

            # V10: Even easier entry in strong trend
            if trend == "strong":
                min_score = 1
            elif (near_sma_support or near_local_low) and in_uptrend_macro:
                min_score = 2
            elif in_uptrend_macro and rs > 0:
                min_score = 2
            else:
                min_score = 3

            if entry_score < min_score and not breakout_entry:
                new_position = 0
            if wp > 0.9 and rs <= 0 and bs < 2 and trend != "strong" and not breakout_entry:
                new_position = 0
            if bb > 0.85 and bs < 2 and entry_score < 4 and trend != "strong" and not breakout_entry:
                new_position = 0
            if new_position == 1:
                if wp > 0.78 and bb < 0.35 and trend == "weak" and not breakout_entry:
                    new_position = 0
            if new_position == 1 and dp < 0.025:
                if entry_score < 4 and trend != "strong" and not breakout_entry:
                    new_position = 0

        # V10 CHANGE #5: Position sizing — scale up in strong trends
        if new_position == 1 and position == 0:
            if ret_5d[i] > 0.05:
                new_position = 0
                n_late_entry_blocked += 1

        if new_position == 1 and position == 0:
            if drop_from_peak_20[i] <= -0.15 and not stabilized_sideways[i]:
                new_position = 0
                n_falling_knife_blocked += 1

        if new_position == 1 and position == 0:
            vol_floor = 0.7 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else 0
            if vol_floor > 0 and volume[i] < vol_floor:
                new_position = 0
                n_low_volume_blocked += 1

        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            if trend == "strong" and entry_score >= 3:
                position_size = 1.0  # full size in strong trend
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
                n_bearish_penalized += 1

            if dist_sma20[i] > 0.05 and position_size > 0.5:
                position_size = 0.5
                n_overextended_capped += 1

        # ════════════════════════════════════════════════
        # EXIT LOGIC — V10: ADAPTIVE TRAILING + HYBRID EXIT
        # ════════════════════════════════════════════════
        if position == 1:
            projected = equity[i - 1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0

            in_uptrend = rs > 0 and hl >= 2
            strong_uptrend = trend == "strong"

            # ATR stop
            if not np.isnan(atr14[i]) and close[i] > 0:
                atr_stop = ATR_MULT * atr14[i] / close[i]
                atr_stop = max(0.025, min(atr_stop, 0.06))
            else:
                atr_stop = 0.04

            # 0) HARD STOP
            if cum_ret <= -HARD_STOP:
                new_position = 0
                exit_reason = "hard_stop"
                n_hard_stop += 1

            # 1) ATR stop loss
            elif cum_ret <= -atr_stop:
                new_position = 0
                exit_reason = "stop_loss"
                n_stop_loss += 1

            # V10 CHANGE #4: HYBRID EXIT — In strong uptrend, use rule-based exit
            elif strong_uptrend and cum_ret > 0.05 and max_profit > 0.08:
                # In strong trend with good profit: only exit on MACD cross or MA20 break
                macd_bearish = macd_hist[i] < 0 and macd_hist[i - 1] >= 0 if i > 0 else False
                price_below_ma20 = close[i] < sma20[i] if not np.isnan(sma20[i]) else False

                if macd_bearish and price_below_ma20:
                    new_position = 0
                    exit_reason = "hybrid_exit"
                    n_hybrid_exit += 1
                elif price_below_ma20 and cum_ret < max_profit * 0.5:
                    # Lost half the profit and below MA20
                    new_position = 0
                    exit_reason = "hybrid_exit"
                    n_hybrid_exit += 1
                else:
                    # Hold — override any sell signal from ML
                    new_position = 1

            # V10 CHANGE #1: ADAPTIVE TRAILING STOP
            elif max_profit > 0.03 and new_position == 1:
                if max_profit > 0.25:
                    trail_pct = 0.18
                elif max_profit > 0.15:
                    trail_pct = 0.25
                elif max_profit > 0.08:
                    trail_pct = 0.40
                else:
                    trail_pct = 0.65

                # V10: Adaptive — widen in strong trend
                if strong_uptrend:
                    trail_pct *= STRONG_TREND_TRAIL_MULT  # much wider
                elif trend == "moderate":
                    trail_pct *= 0.7  # somewhat wider

                giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                if giveback >= trail_pct:
                    new_position = 0
                    exit_reason = "trailing_stop"
                    n_trailing_stop += 1

            # 3) Profit lock — V10: higher thresholds
            if new_position == 1 and max_profit >= PROFIT_LOCK_THRESHOLD:
                if cum_ret < PROFIT_LOCK_MIN:
                    # V10: In strong trend, don't profit-lock as aggressively
                    if not strong_uptrend:
                        new_position = 0
                        exit_reason = "profit_lock"
                        n_profit_lock += 1

            # 4) Zombie exit — V10: 14 bars, but exempt strong trends
            if new_position == 1 and hold_days >= ZOMBIE_BARS and cum_ret < 0.01:
                if not strong_uptrend:
                    new_position = 0
                    exit_reason = "zombie_exit"
                    n_zombie_exit += 1

            # 5) Min hold
            if new_position == 0 and exit_reason not in ("stop_loss", "hard_stop", "hybrid_exit") and hold_days < MIN_HOLD:
                if cum_ret > -atr_stop:
                    new_position = 1
                    n_min_hold_saved += 1

            # 6) Exit confirmation (same as V9)
            if new_position == 0 and exit_reason == "signal":
                if raw_signal == 0:
                    consecutive_exit_signals += 1
                else:
                    consecutive_exit_signals = 0
                if consecutive_exit_signals < EXIT_CONFIRM:
                    new_position = 1
                else:
                    consecutive_exit_signals = 0

            # 7) Strong trend override (enhanced V10)
            if new_position == 0 and exit_reason == "signal":
                if cum_ret > 0 and trend == "strong":
                    new_position = 1  # V10: hold in strong trend regardless

        else:
            consecutive_exit_signals = 0

        # ════════════════════════════════════════════════
        # EXECUTE
        # ════════════════════════════════════════════════
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
                position_size = 1.0

        if position == 1:
            equity[i] = equity[i - 1] * (1 + ret * position_size) - cost
            hold_days += 1
        else:
            equity[i] = equity[i - 1] - cost
        position = new_position

    if position == 1 and entry_equity > 0 and record_trades:
        # Keep end-of-data PnL definition consistent with normal exits:
        # use price change from entry_close to last close, not account equity delta.
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
        "n_stop_loss": n_stop_loss, "n_trailing_stop": n_trailing_stop,
        "n_zombie_exit": n_zombie_exit, "n_profit_lock": n_profit_lock,
        "n_hard_stop": n_hard_stop, "n_min_hold_saved": n_min_hold_saved,
        "n_quick_reentry": n_quick_reentry, "n_hybrid_exit": n_hybrid_exit,
        "n_breakout_entries": n_breakout_entries,
        "n_late_entry_blocked": n_late_entry_blocked,
        "n_falling_knife_blocked": n_falling_knife_blocked,
        "n_low_volume_blocked": n_low_volume_blocked,
        "n_overextended_capped": n_overextended_capped,
        "n_bearish_penalized": n_bearish_penalized,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", type=str, default="ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC")
    parser.add_argument("--model", type=str, default="lightgbm")
    parser.add_argument("--target-ma", type=str, default="ema5_20")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")

    if args.target_ma == "ema5_20":
        target_cfg = {"type": "trend_regime", "trend_method": "dual_ma",
                      "short_window": 5, "long_window": 20, "classes": 3}
    else:
        target_cfg = {"type": "trend_regime", "trend_method": "dual_ma",
                      "short_window": 10, "long_window": 40, "classes": 3}

    config = {
        "data": {"data_dir": data_dir},
        "split": {"method": "walk_forward", "train_years": 4, "test_years": 1,
                  "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025},
        "target": target_cfg,
    }

    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    pick = [s.strip() for s in args.pick.split(",")]
    symbols = [s for s in pick if s in loader.symbols]

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    print("=" * 150)
    print(f"V10 vs V11 COMPARISON - Target: {args.target_ma}")
    print("=" * 150)
    print("V11 entry upgrades: late-entry filter, falling-knife stabilization, bullish-candle penalty,")
    print("                    overextended cap, consolidation breakout scanner, volume confirmation")

    v10_all, v11_all = [], []

    for window, train_df, test_df in splitter.split(df):
        model = build_model(args.model)
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        for sym in test_df["symbol"].unique():
            if sym not in symbols:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue

            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values if "return_1d" in sym_test.columns else np.zeros(len(sym_test))

            r10 = backtest_v10(y_pred, rets, sym_test, feature_cols)
            r11 = backtest_v11(y_pred, rets, sym_test, feature_cols)

            for t in r10["trades"]:
                t["symbol"] = sym
                t["window"] = window.label
            for t in r11["trades"]:
                t["symbol"] = sym
                t["window"] = window.label

            v10_all.extend(r10["trades"])
            v11_all.extend(r11["trades"])

    print(f"\n{'-' * 160}")
    print(f"{'Symbol':<8} | {'V10 #':>5} {'V10 WR':>7} {'V10 Avg':>8} {'V10 Tot':>9} {'V10 MaxDD':>10} | "
          f"{'V11 #':>5} {'V11 WR':>7} {'V11 Avg':>8} {'V11 Tot':>9} {'V11 MaxDD':>10} | {'dAvg':>7} {'dTot':>8} {'?':>3}")
    print(f"{'-' * 160}")

    v10_by_sym = defaultdict(list)
    v11_by_sym = defaultdict(list)
    for t in v10_all:
        v10_by_sym[t.get("symbol", t.get("entry_symbol", "?"))].append(t)
    for t in v11_all:
        v11_by_sym[t.get("symbol", t.get("entry_symbol", "?"))].append(t)

    better_count = 0
    all_syms = sorted(set(list(v10_by_sym.keys()) + list(v11_by_sym.keys())))
    for sym in all_syms:
        t10 = v10_by_sym.get(sym, [])
        t11 = v11_by_sym.get(sym, [])
        n10, n11 = len(t10), len(t11)
        wr10 = sum(1 for t in t10 if t["pnl_pct"] > 0) / max(n10, 1) * 100
        wr11 = sum(1 for t in t11 if t["pnl_pct"] > 0) / max(n11, 1) * 100
        pnl10 = sum(t["pnl_pct"] for t in t10)
        pnl11 = sum(t["pnl_pct"] for t in t11)
        avg10 = np.mean([t["pnl_pct"] for t in t10]) if t10 else 0
        avg11 = np.mean([t["pnl_pct"] for t in t11]) if t11 else 0
        maxdd10 = min(t["pnl_pct"] for t in t10) if t10 else 0
        maxdd11 = min(t["pnl_pct"] for t in t11) if t11 else 0
        better = "OK" if avg11 > avg10 else "NO"
        if avg11 > avg10:
            better_count += 1
        print(f"{sym:<8} | {n10:>5} {wr10:>6.1f}% {avg10:>+7.1f}% {pnl10:>+8.1f}% {maxdd10:>+9.1f}% | "
              f"{n11:>5} {wr11:>6.1f}% {avg11:>+7.1f}% {pnl11:>+8.1f}% {maxdd11:>+9.1f}% | "
              f"{avg11-avg10:>+6.1f}% {pnl11-pnl10:>+7.1f}% {better}")

    print(f"\n{'=' * 150}")
    print("AGGREGATE V10:")
    s10 = summarize(v10_all, "V10")
    print("\nAGGREGATE V11:")
    s11 = summarize(v11_all, "V11")

    print(f"\n{'=' * 150}")
    print("IMPROVEMENT V10 -> V11:")
    print(f"{'=' * 150}")
    if s10 and s11:
        print(f"  Trades:     {s10['trades']:>4d} -> {s11['trades']:>4d} (Delta {s11['trades']-s10['trades']:>+4d})")
        print(f"  WR:         {s10['wr']:>5.1f}% -> {s11['wr']:>5.1f}% (Delta {s11['wr']-s10['wr']:>+5.1f}%)")
        print(f"  PF:         {s10['pf']:>5.2f} -> {s11['pf']:>5.2f}")
        print(f"  Avg PnL:    {s10['avg_pnl']:>+6.2f}% -> {s11['avg_pnl']:>+6.2f}%")
        print(f"  Total PnL:  {s10['total_pnl']:>+8.1f}% -> {s11['total_pnl']:>+8.1f}%")
        print(f"  Avg Hold:   {s10['avg_hold']:>5.1f}d -> {s11['avg_hold']:>5.1f}d")
        print(f"  Better:     {better_count}/{len(all_syms)} symbols improved")

    if v11_all:
        exits = defaultdict(int)
        for t in v11_all:
            exits[t.get("exit_reason", "?")] += 1
        print(f"\n  V11 exit breakdown: {dict(exits)}")

        breakout_trades = [t for t in v11_all if t.get("breakout_entry")]
        if breakout_trades:
            b_wr = sum(1 for t in breakout_trades if t["pnl_pct"] > 0) / len(breakout_trades) * 100
            b_avg = np.mean([t["pnl_pct"] for t in breakout_trades])
            print(f"  Breakout entries: {len(breakout_trades)} trades, WR={b_wr:.0f}%, Avg={b_avg:+.1f}%")


if __name__ == "__main__":
    main()
