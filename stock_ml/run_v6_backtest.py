"""
V6 BACKTEST — Anti-Noise + Smart Entry + Improved Exit
=======================================================
Implements 5 improvements over V4:
  A. Signal Smoothing (entry/exit confirmation, min hold 3)
  B. Breakout Quality Filter (reject false breakouts)
  C. Distance-to-Resistance Filter (reduce size near resistance)
  D. Early Trailing (start at 3% instead of 5%)
  E. Zombie Exit (exit flat trades after 5 bars)

Usage:
    python run_v6_backtest.py                    # Quick (5 symbols)
    python run_v6_backtest.py --symbols 20       # 20 symbols
    python run_v6_backtest.py --full             # All symbols
    python run_v6_backtest.py --compare          # V4 vs V6 side-by-side
"""
import argparse, sys, os, numpy as np, pandas as pd
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model, get_available_models
from src.evaluation.metrics import compute_metrics


def backtest_v6(y_pred, returns, df_test, feature_cols,
                initial_capital=100_000_000, commission=0.0015, tax=0.001,
                record_trades=True):
    """
    V6 Backtest with all 5 improvements:
    A. Signal smoothing (entry confirm, exit confirm, min hold 3)
    B. Breakout quality filter
    C. Distance-to-resistance filter
    D. Early trailing (3% threshold)
    E. Zombie exit (5 bars + <1% profit)
    """
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

    # ── V6-A: Signal smoothing state ──
    consecutive_exit_signals = 0  # count consecutive predict=0 bars

    # Extract features
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
    sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
    sma50 = pd.Series(close).rolling(50, min_periods=10).mean().values

    if "high" in df_test.columns and "low" in df_test.columns:
        high, low = df_test["high"].values, df_test["low"].values
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]
        atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().values
    else:
        atr14 = np.full(n, close.mean() * 0.02)

    date_col = "date" if "date" in df_test.columns else ("timestamp" if "timestamp" in df_test.columns else None)
    dates = df_test[date_col].values if date_col else np.arange(n)
    symbols = df_test["symbol"].values if "symbol" in df_test.columns else ["?"] * n

    def gf(name, idx):
        return feat_arrays[name][idx] if idx < n else defaults.get(name, 0)

    entry_features = {}
    n_stop_loss = 0
    n_trailing_stop = 0
    n_zombie_exit = 0
    n_entry_rejected_breakout = 0
    n_entry_rejected_signal = 0
    n_exit_confirmed = 0

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

        # ════════════════════════════════════════════════
        # ENTRY LOGIC (position == 0, want to buy)
        # ════════════════════════════════════════════════
        if new_position == 1 and position == 0:

            # ── V6-A: Entry confirmation — need predict=1 on BOTH current AND previous bar ──
            prev_pred = int(y_pred[i - 2]) if i >= 2 else 0
            if prev_pred != 1:
                new_position = 0
                n_entry_rejected_signal += 1

        # V4 REGIME FILTER (kept from V4)
        if new_position == 1 and position == 0:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3:
                        new_position = 0

        # V4 ENTRY FILTER (kept from V4)
        if new_position == 1 and position == 0:
            entry_score = 0
            if wp < 0.75: entry_score += 1
            if dp > 0.02: entry_score += 1
            if rs > 0: entry_score += 1
            if vs > 1.1: entry_score += 1
            if hl >= 2: entry_score += 1

            if entry_score < 3:
                new_position = 0
            if wp > 0.9 and rs <= 0 and bs < 2:
                new_position = 0
            if bb > 0.85 and bs < 2 and entry_score < 4:
                new_position = 0

            # ── V6-B: Breakout Quality Filter ──
            if new_position == 1:
                if wp > 0.78 and bb < 0.35:
                    new_position = 0  # false breakout territory
                    n_entry_rejected_breakout += 1

            # ── V6-C: Distance-to-Resistance Filter ──
            if new_position == 1 and dp < 0.025:
                if entry_score < 4:
                    new_position = 0  # too close to resistance without strong score

        # POSITION SIZING (V4 + V6-C enhancement)
        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            if dp < 0.025:
                position_size = 0.5  # V6-C: reduce near resistance
            elif bb > 0.7:
                position_size = 0.7  # V4: reduce at high BB
            else:
                position_size = 1.0

        # ════════════════════════════════════════════════
        # EXIT LOGIC (position == 1)
        # ════════════════════════════════════════════════
        if position == 1:
            projected = equity[i - 1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0
            in_uptrend = rs > 0 and hl >= 2

            # ATR-based stop loss (V4)
            if not np.isnan(atr14[i]) and close[i] > 0:
                atr_stop = 2.5 * atr14[i] / close[i]
                atr_stop = max(0.03, min(atr_stop, 0.08))
            else:
                atr_stop = 0.05

            # 1) Stop loss
            if cum_ret <= -atr_stop:
                new_position = 0
                exit_reason = "stop_loss"
                n_stop_loss += 1

            # 2) ── V6-D: Early Trailing — start at 3% (was 5%) ──
            elif max_profit > 0.03 and new_position == 1:
                if max_profit > 0.20:
                    trail_pct = 0.35 if not in_uptrend else 0.50
                elif max_profit > 0.12:
                    trail_pct = 0.45 if not in_uptrend else 0.60
                elif max_profit > 0.05:
                    trail_pct = 0.60 if not in_uptrend else 0.75
                else:
                    # V6-D: new tier for 3-5% profits
                    trail_pct = 0.70 if not in_uptrend else 0.85
                giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                if giveback >= trail_pct:
                    new_position = 0
                    exit_reason = "trailing_stop"
                    n_trailing_stop += 1

            # 3) ── V6-E: Zombie Exit — flat trade after 5 bars ──
            if new_position == 1 and hold_days >= 5 and cum_ret < 0.01:
                new_position = 0
                exit_reason = "zombie_exit"
                n_zombie_exit += 1

            # ── V6-A: Min hold 3 bars (was 2) ──
            if new_position == 0 and exit_reason == "signal" and hold_days < 3:
                # Only override signal exits, not stop/trailing/zombie
                if cum_ret > -0.02:  # not in deep loss
                    new_position = 1

            # ── V6-A: Exit confirmation — need 2 consecutive predict=0 ──
            if new_position == 0 and exit_reason == "signal":
                if raw_signal == 0:
                    consecutive_exit_signals += 1
                else:
                    consecutive_exit_signals = 0

                if consecutive_exit_signals < 2:
                    new_position = 1  # hold — not yet confirmed
                else:
                    n_exit_confirmed += 1
                    consecutive_exit_signals = 0

            # V4 Signal override (keep if strong setup and profitable)
            if new_position == 0 and exit_reason == "signal":
                if cum_ret > 0 and bs >= 3 and hl >= 3 and rs > 0:
                    new_position = 1
        else:
            consecutive_exit_signals = 0

        # ════════════════════════════════════════════════
        # EXECUTE TRADE
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
                }
            else:
                cost = equity[i - 1] * position_size * (commission + tax)
                if record_trades and entry_equity > 0:
                    # Use close prices for accurate PnL (avoids position_size equity mismatch)
                    pnl_pct = (close[i] / entry_close - 1) * 100 if entry_close > 0 else 0
                    max_pnl_pct = (max_equity_in_trade - entry_equity) / entry_equity * 100 if entry_equity > 0 else 0
                    # Future upside
                    future_max = 0
                    if i < n - 1:
                        future_rets = returns[i + 1:min(i + 20, n)]
                        cum_f = np.cumprod(1 + np.nan_to_num(future_rets))
                        if len(cum_f) > 0:
                            future_max = (cum_f.max() - 1) * 100
                    trades.append({
                        "entry_day": current_entry_day, "exit_day": i,
                        "holding_days": i - current_entry_day,
                        "pnl_pct": round(pnl_pct, 2),
                        "max_profit_pct": round(max_pnl_pct, 2),
                        "exit_efficiency": round(pnl_pct / max_pnl_pct * 100, 1) if max_pnl_pct > 0.5 else (100 if pnl_pct > 0 else 0),
                        "future_upside_pct": round(future_max, 2),
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

    # Close open trade
    if position == 1 and entry_equity > 0:
        pnl = equity[-1] - entry_equity
        pnl_pct = (pnl / entry_equity * 100) if entry_equity > 0 else 0
        max_pnl_pct = (max_equity_in_trade - entry_equity) / entry_equity * 100 if entry_equity > 0 else 0
        if record_trades:
            trades.append({
                "entry_day": current_entry_day, "exit_day": n - 1,
                "holding_days": n - 1 - current_entry_day,
                "pnl_pct": round(pnl_pct, 2), "max_profit_pct": round(max_pnl_pct, 2),
                "exit_efficiency": round(pnl_pct / max_pnl_pct * 100, 1) if max_pnl_pct > 0.5 else 100,
                "future_upside_pct": 0, "exit_reason": "end",
                "exit_date": str(dates[-1])[:10], **entry_features,
            })

    # Compute metrics
    total_return = (equity[-1] / initial_capital - 1) * 100
    years = n / 252
    ann_return = ((equity[-1] / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100
    bnh_eq = initial_capital * np.cumprod(1 + np.nan_to_num(returns))
    bnh_return = (bnh_eq[-1] / initial_capital - 1) * 100

    daily_rets = np.diff(equity) / equity[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    sharpe = (np.sqrt(252) * daily_rets.mean() / daily_rets.std()) if len(daily_rets) > 0 and daily_rets.std() > 0 else 0

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100

    # Trade stats
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    gross_wins = sum(t["pnl_pct"] for t in wins)
    gross_losses = abs(sum(t["pnl_pct"] for t in losses))
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    avg_hold = np.mean([t["holding_days"] for t in trades]) if trades else 0

    # Exit reason counts
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t["exit_reason"]] += 1

    return {
        "equity_curve": equity,
        "trades": trades,
        "total_return_pct": round(total_return, 2),
        "ann_return_pct": round(ann_return, 2),
        "bnh_return_pct": round(bnh_return, 2),
        "excess_pct": round(total_return - bnh_return, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "total_trades": n_trades,
        "win_rate_pct": round(win_rate, 1),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "avg_hold_days": round(avg_hold, 1),
        "final_equity": round(equity[-1]),
        "exit_reasons": dict(exit_reasons),
        "n_stop_loss": n_stop_loss,
        "n_trailing_stop": n_trailing_stop,
        "n_zombie_exit": n_zombie_exit,
        "n_entry_rejected_breakout": n_entry_rejected_breakout,
        "n_entry_rejected_signal": n_entry_rejected_signal,
        "n_exit_confirmed": n_exit_confirmed,
    }


def backtest_v4(y_pred, returns, df_test, feature_cols,
                initial_capital=100_000_000, commission=0.0015, tax=0.001):
    """V4 backtest for comparison (same as run_trade_analysis.py)."""
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
    sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
    sma50 = pd.Series(close).rolling(50, min_periods=10).mean().values

    if "high" in df_test.columns and "low" in df_test.columns:
        high, low = df_test["high"].values, df_test["low"].values
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]
        atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().values
    else:
        atr14 = np.full(n, close.mean() * 0.02)

    def gf(name, idx):
        return feat_arrays[name][idx] if idx < n else defaults.get(name, 0)

    for i in range(1, n):
        pred = int(y_pred[i - 1])
        ret = returns[i] if not np.isnan(returns[i]) else 0
        new_position = 1 if pred == 1 else 0
        exit_reason = "signal"

        wp = gf("range_position_20d", i)
        dp = gf("dist_to_resistance", i)
        rs = gf("rsi_slope_5d", i)
        vs = gf("vol_surge_ratio", i)
        bs = gf("breakout_setup_score", i)
        hl = gf("higher_lows_count", i)
        bb = gf("bb_width_percentile", i)

        # V4 REGIME FILTER
        if new_position == 1 and position == 0:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3:
                        new_position = 0

        # V4 ENTRY FILTER
        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            if entry_score < 3:
                new_position = 0
            if wp > 0.9 and rs <= 0 and bs < 2:
                new_position = 0
            if bb > 0.85 and bs < 2 and entry_score < 4:
                new_position = 0

        if new_position == 1 and position == 0:
            position_size = 0.7 if bb > 0.7 else 1.0

        # MIN HOLD 2
        if position == 1 and new_position == 0 and hold_days < 2:
            cum_ret = (equity[i-1] * (1 + ret) - entry_equity) / entry_equity if entry_equity > 0 else 0
            if cum_ret > 0.01:
                new_position = 1

        # ADAPTIVE EXIT
        if position == 1:
            projected = equity[i-1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0
            in_uptrend = rs > 0 and hl >= 2

            atr_stop_val = 2.5 * atr14[i] / close[i] if not np.isnan(atr14[i]) and close[i] > 0 else 0.05
            atr_stop_val = max(0.03, min(atr_stop_val, 0.08))

            if cum_ret <= -atr_stop_val:
                new_position = 0
                exit_reason = "stop_loss"
            elif max_profit > 0.05 and new_position == 1:
                if max_profit > 0.20:
                    trail_pct = 0.35 if not in_uptrend else 0.50
                elif max_profit > 0.12:
                    trail_pct = 0.45 if not in_uptrend else 0.60
                elif max_profit > 0.05:
                    trail_pct = 0.60 if not in_uptrend else 0.75
                else:
                    trail_pct = 0.80
                giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                if giveback >= trail_pct:
                    new_position = 0
                    exit_reason = "trailing_stop"

            if new_position == 0 and exit_reason == "signal":
                if cum_ret > 0 and bs >= 3 and hl >= 3 and rs > 0:
                    new_position = 1

        cost = 0
        if new_position != position:
            if new_position == 1:
                deploy = equity[i-1] * position_size
                cost = deploy * commission
                entry_equity = deploy - cost
                max_equity_in_trade = entry_equity
                current_entry_day = i
                hold_days = 0
            else:
                cost = equity[i-1] * position_size * (commission + tax)
                if entry_equity > 0:
                    exit_eq = equity[i-1] - cost
                    pnl_pct = (exit_eq - entry_equity) / entry_equity * 100
                    max_pnl_pct = (max_equity_in_trade - entry_equity) / entry_equity * 100 if entry_equity > 0 else 0
                    trades.append({
                        "holding_days": i - current_entry_day,
                        "pnl_pct": round(pnl_pct, 2),
                        "max_profit_pct": round(max_pnl_pct, 2),
                        "exit_reason": exit_reason,
                    })
                entry_equity = 0
                max_equity_in_trade = 0
                position_size = 1.0

        if position == 1:
            equity[i] = equity[i-1] * (1 + ret * position_size) - cost
            hold_days += 1
        else:
            equity[i] = equity[i-1] - cost
        position = new_position

    if position == 1 and entry_equity > 0:
        pnl_pct = (equity[-1] - entry_equity) / entry_equity * 100
        max_pnl_pct = (max_equity_in_trade - entry_equity) / entry_equity * 100 if entry_equity > 0 else 0
        trades.append({"holding_days": n-1-current_entry_day, "pnl_pct": round(pnl_pct, 2),
                       "max_profit_pct": round(max_pnl_pct, 2), "exit_reason": "end"})

    total_return = (equity[-1] / initial_capital - 1) * 100
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min() * 100
    daily_rets = np.diff(equity) / equity[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    sharpe = (np.sqrt(252) * daily_rets.mean() / daily_rets.std()) if len(daily_rets) > 0 and daily_rets.std() > 0 else 0

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    gross_w = sum(t["pnl_pct"] for t in wins)
    gross_l = abs(sum(t["pnl_pct"] for t in losses))
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t["exit_reason"]] += 1

    return {
        "equity_curve": equity,
        "trades": trades,
        "total_return_pct": round(total_return, 2),
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "profit_factor": round(gross_w / gross_l if gross_l > 0 else float('inf'), 2),
        "total_trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_win_pct": round(np.mean([t["pnl_pct"] for t in wins]), 2) if wins else 0,
        "avg_loss_pct": round(np.mean([t["pnl_pct"] for t in losses]), 2) if losses else 0,
        "avg_hold_days": round(np.mean([t["holding_days"] for t in trades]), 1) if trades else 0,
        "final_equity": round(equity[-1]),
        "exit_reasons": dict(exit_reasons),
    }


def main():
    parser = argparse.ArgumentParser(description="V6 Backtest — Anti-Noise + Smart Entry")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--symbols", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["random_forest", "xgboost", "lightgbm"])
    parser.add_argument("--compare", action="store_true", help="Compare V4 vs V6")
    parser.add_argument("--capital", type=float, default=100_000_000)
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    config = {
        "data": {"data_dir": data_dir},
        "split": {"method": "walk_forward", "train_years": 4, "test_years": 1,
                  "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025},
        "target": {"type": "trend_regime", "trend_method": "dual_ma",
                   "short_window": 10, "long_window": 40, "classes": 3},
    }

    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    max_symbols = None if args.full else args.symbols
    symbols = loader.symbols[:max_symbols] if max_symbols else None

    print("=" * 120)
    print("🚀 V6 BACKTEST — Anti-Noise + Smart Entry + Improved Exit")
    print(f"   Symbols: {max_symbols or 'all'} | Models: {args.models} | Capital: {args.capital:,.0f}")
    if args.compare:
        print("   📊 Mode: V4 vs V6 COMPARISON")
    print("=" * 120)

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])
    print(f"   Data: {len(df)} rows, {df['symbol'].nunique()} symbols\n")

    all_results = []

    for model_name in args.models:
        print(f"\n{'═' * 100}")
        print(f"🤖 Model: {model_name.upper()}")
        print(f"{'═' * 100}")

        v4_trades_all, v6_trades_all = [], []
        v4_eq_parts, v6_eq_parts = [], []

        for window, train_df, test_df in splitter.split(df):
            try:
                model = build_model(model_name)
                X_train = np.nan_to_num(train_df[feature_cols].values)
                y_train = train_df["target"].values.astype(int)
                X_test = np.nan_to_num(test_df[feature_cols].values)

                offset = 0
                if model_name == "xgboost" and y_train.min() < 0:
                    offset = abs(y_train.min())
                    y_train += offset

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if offset > 0:
                    y_pred -= offset

                rets = test_df["return_1d"].values if "return_1d" in test_df.columns else np.zeros(len(test_df))
                test_reset = test_df.reset_index(drop=True)

                # V6
                r6 = backtest_v6(y_pred, rets, test_reset, feature_cols, args.capital)
                v6_trades_all.extend(r6["trades"])

                line = (f"   {window.label:30s} | V6: Ret={r6['total_return_pct']:>+7.2f}% "
                        f"Trades={r6['total_trades']:>3d} WR={r6['win_rate_pct']:>5.1f}% "
                        f"PF={r6['profit_factor']:>5.2f}")

                if args.compare:
                    r4 = backtest_v4(y_pred, rets, test_reset, feature_cols, args.capital)
                    v4_trades_all.extend(r4["trades"])
                    line += (f" | V4: Ret={r4['total_return_pct']:>+7.2f}% "
                             f"Trades={r4['total_trades']:>3d} WR={r4['win_rate_pct']:>5.1f}%")

                print(line)

            except Exception as e:
                print(f"   {window.label:30s} | ❌ Error: {e}")

        # Aggregate across all windows
        print(f"\n   {'─' * 90}")
        print(f"   📊 AGGREGATE — {model_name}")
        print(f"   {'─' * 90}")

        def summarize(trades, label):
            if not trades:
                print(f"   {label}: No trades")
                return {}
            tdf = pd.DataFrame(trades)
            n = len(tdf)
            wins = tdf[tdf["pnl_pct"] > 0]
            losses = tdf[tdf["pnl_pct"] <= 0]
            wr = len(wins) / n * 100
            gw = wins["pnl_pct"].sum()
            gl = abs(losses["pnl_pct"].sum())
            pf = gw / gl if gl > 0 else float('inf')
            avg_pnl = tdf["pnl_pct"].mean()
            avg_hold = tdf["holding_days"].mean()
            short = tdf[tdf["holding_days"] <= 2]
            marginal = tdf[(tdf["pnl_pct"] > -2) & (tdf["pnl_pct"] < 2)]

            print(f"   {label}:")
            print(f"     Trades: {n:>4d} | Wins: {len(wins):>4d} | Losses: {len(losses):>4d}")
            print(f"     Win Rate: {wr:>5.1f}% | PF: {pf:>5.2f} | Avg PnL: {avg_pnl:>+6.2f}%")
            print(f"     Avg Hold: {avg_hold:>5.1f}d | Short (≤2d): {len(short):>3d} ({len(short)/n*100:.1f}%)")
            print(f"     Marginal (-2~+2%): {len(marginal):>3d} ({len(marginal)/n*100:.1f}%)")

            exit_reasons = tdf["exit_reason"].value_counts()
            er_str = " | ".join([f"{k}:{v}" for k, v in exit_reasons.items()])
            print(f"     Exit reasons: {er_str}")

            return {"trades": n, "wr": wr, "pf": pf, "avg_pnl": avg_pnl,
                    "short": len(short), "marginal": len(marginal), "avg_hold": avg_hold}

        s6 = summarize(v6_trades_all, "V6")

        if args.compare and v4_trades_all:
            print()
            s4 = summarize(v4_trades_all, "V4")

            if s4 and s6:
                print(f"\n   {'─' * 60}")
                print(f"   📈 IMPROVEMENT V4 → V6:")
                print(f"     Trades:   {s4['trades']:>4d} → {s6['trades']:>4d} (Δ{s6['trades']-s4['trades']:>+4d})")
                print(f"     WR:       {s4['wr']:>5.1f}% → {s6['wr']:>5.1f}% (Δ{s6['wr']-s4['wr']:>+5.1f}%)")
                print(f"     PF:       {s4['pf']:>5.2f} → {s6['pf']:>5.2f}")
                print(f"     Avg PnL:  {s4['avg_pnl']:>+6.2f}% → {s6['avg_pnl']:>+6.2f}%")
                print(f"     Short:    {s4['short']:>4d} → {s6['short']:>4d} (Δ{s6['short']-s4['short']:>+4d})")
                print(f"     Marginal: {s4['marginal']:>4d} → {s6['marginal']:>4d} (Δ{s6['marginal']-s4['marginal']:>+4d})")
                print(f"     Avg Hold: {s4['avg_hold']:>5.1f}d → {s6['avg_hold']:>5.1f}d")

        all_results.append({"model": model_name, "v6_trades": v6_trades_all,
                            "v4_trades": v4_trades_all if args.compare else []})

    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save V6 trades
    all_v6 = []
    for r in all_results:
        for t in r["v6_trades"]:
            t["model"] = r["model"]
        all_v6.extend(r["v6_trades"])
    if all_v6:
        v6_df = pd.DataFrame(all_v6)
        v6_path = os.path.join(out_dir, f"v6_trades_{ts}.csv")
        v6_df.to_csv(v6_path, index=False)
        print(f"\n💾 V6 trades saved: results/v6_trades_{ts}.csv ({len(all_v6)} trades)")

    print(f"\n{'=' * 120}")
    print("✅ V6 BACKTEST COMPLETE")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
