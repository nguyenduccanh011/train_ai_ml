"""
V7 vs V6 A/B Comparison
========================
V7 improvements over V6:
  1. Cooldown: After exit, wait 3 bars before re-entry (reduce wasted fees)
  2. Re-entry price filter: Only re-buy if price moved >3% from last exit price
  3. Relax entry to s>=2 when near SMA support or after pullback
  4. Extend zombie exit from 5 to 8 bars
  5. Widen trailing in strong uptrend (sma20 > sma50)
"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v6_backtest import backtest_v6


def backtest_v7(y_pred, returns, df_test, feature_cols,
                initial_capital=100_000_000, commission=0.0015, tax=0.001,
                record_trades=True):
    """V7: V6 + cooldown + re-entry filter + relaxed entry + longer zombie + wider trailing"""
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

    # V7 new state
    cooldown_remaining = 0  # bars to wait after exit
    last_exit_price = 0     # price at last exit

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

    # Local low (20-bar lookback)
    local_low_20 = pd.Series(close).rolling(20, min_periods=5).min().values

    date_col = "date" if "date" in df_test.columns else ("timestamp" if "timestamp" in df_test.columns else None)
    dates = df_test[date_col].values if date_col else np.arange(n)
    symbols = df_test["symbol"].values if "symbol" in df_test.columns else ["?"] * n

    def gf(name, idx):
        return feat_arrays[name][idx] if idx < n else defaults.get(name, 0)

    entry_features = {}
    n_stop_loss = 0
    n_trailing_stop = 0
    n_zombie_exit = 0
    n_cooldown_blocked = 0
    n_reentry_blocked = 0
    n_relaxed_entry = 0

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

        # Decrement cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # ════════════════════════════════════════════════
        # ENTRY LOGIC
        # ════════════════════════════════════════════════
        if new_position == 1 and position == 0:
            # V7-1: Cooldown — block entry if still in cooldown
            if cooldown_remaining > 0:
                new_position = 0
                n_cooldown_blocked += 1

        if new_position == 1 and position == 0:
            # V7-2: Re-entry price filter — block if price near last exit
            if last_exit_price > 0:
                price_diff = abs(close[i] / last_exit_price - 1)
                if price_diff < 0.03:  # within 3% of last exit
                    new_position = 0
                    n_reentry_blocked += 1

        if new_position == 1 and position == 0:
            # V6-A: Entry confirmation
            prev_pred = int(y_pred[i - 2]) if i >= 2 else 0
            if prev_pred != 1:
                new_position = 0

        # Regime filter (same as V6)
        if new_position == 1 and position == 0:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3:
                        new_position = 0

        # V7-3: RELAXED ENTRY FILTER (s>=2 near support)
        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])

            # V7: Allow s>=2 if price is near SMA20 support or near local low
            near_sma_support = (not np.isnan(sma20[i]) and 
                               close[i] <= sma20[i] * 1.02 and 
                               close[i] >= sma20[i] * 0.97)
            near_local_low = (not np.isnan(local_low_20[i]) and 
                             close[i] <= local_low_20[i] * 1.05)
            in_uptrend_macro = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and 
                               sma20[i] > sma50[i])

            if (near_sma_support or near_local_low) and in_uptrend_macro:
                min_score = 2  # V7: relaxed
                if entry_score >= 2:
                    n_relaxed_entry += 1
            else:
                min_score = 3  # V6: strict

            if entry_score < min_score:
                new_position = 0
            if wp > 0.9 and rs <= 0 and bs < 2:
                new_position = 0
            if bb > 0.85 and bs < 2 and entry_score < 4:
                new_position = 0

            # V6-B: Breakout quality
            if new_position == 1:
                if wp > 0.78 and bb < 0.35:
                    new_position = 0

            # V6-C: Distance-to-resistance
            if new_position == 1 and dp < 0.025:
                if entry_score < 4:
                    new_position = 0

        # Position sizing
        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            if dp < 0.025:
                position_size = 0.5
            elif bb > 0.7:
                position_size = 0.7
            else:
                position_size = 1.0

        # ════════════════════════════════════════════════
        # EXIT LOGIC
        # ════════════════════════════════════════════════
        if position == 1:
            projected = equity[i - 1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0

            # V7-5: Stronger uptrend detection
            in_uptrend = rs > 0 and hl >= 2
            strong_uptrend = (in_uptrend and not np.isnan(sma20[i]) and 
                            not np.isnan(sma50[i]) and sma20[i] > sma50[i])

            # ATR stop loss
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

            # 2) V7-5: Wider trailing in strong uptrend
            elif max_profit > 0.03 and new_position == 1:
                if max_profit > 0.20:
                    trail_pct = 0.30 if not strong_uptrend else 0.45
                elif max_profit > 0.12:
                    trail_pct = 0.40 if not strong_uptrend else 0.55
                elif max_profit > 0.05:
                    trail_pct = 0.55 if not strong_uptrend else 0.70
                else:
                    trail_pct = 0.65 if not strong_uptrend else 0.80
                giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                if giveback >= trail_pct:
                    new_position = 0
                    exit_reason = "trailing_stop"
                    n_trailing_stop += 1

            # 3) V7-4: Zombie exit — 8 bars instead of 5
            if new_position == 1 and hold_days >= 8 and cum_ret < 0.01:
                new_position = 0
                exit_reason = "zombie_exit"
                n_zombie_exit += 1

            # Min hold 3 bars
            if new_position == 0 and exit_reason == "signal" and hold_days < 3:
                if cum_ret > -0.02:
                    new_position = 1

            # Exit confirmation (2 consecutive)
            if new_position == 0 and exit_reason == "signal":
                if raw_signal == 0:
                    consecutive_exit_signals += 1
                else:
                    consecutive_exit_signals = 0
                if consecutive_exit_signals < 2:
                    new_position = 1
                else:
                    consecutive_exit_signals = 0

            # Signal override (strong setup)
            if new_position == 0 and exit_reason == "signal":
                if cum_ret > 0 and bs >= 3 and hl >= 3 and rs > 0:
                    new_position = 1
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
                }
            else:
                cost = equity[i - 1] * position_size * (commission + tax)
                # V7: Set cooldown and last exit price
                cooldown_remaining = 3
                last_exit_price = close[i]

                if record_trades and entry_equity > 0:
                    # Use close prices for accurate PnL (avoids position_size equity mismatch)
                    pnl_pct = (close[i] / entry_close - 1) * 100 if entry_close > 0 else 0
                    max_pnl_pct = (max_equity_in_trade - entry_equity) / entry_equity * 100 if entry_equity > 0 else 0
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
                        "exit_reason": exit_reason,
                        "exit_date": str(dates[i])[:10],
                        "future_upside_pct": round(future_max, 2),
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

    # Close open
    if position == 1 and entry_equity > 0 and record_trades:
        pnl = equity[-1] - entry_equity
        pnl_pct = (pnl / entry_equity * 100) if entry_equity > 0 else 0
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
        "n_zombie_exit": n_zombie_exit,
        "n_cooldown_blocked": n_cooldown_blocked,
        "n_reentry_blocked": n_reentry_blocked,
        "n_relaxed_entry": n_relaxed_entry,
    }


def summarize(trades, label):
    if not trades:
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
    marginal = tdf[(tdf["pnl_pct"] > -2) & (tdf["pnl_pct"] < 2)]

    # Wasted fee cycles
    wasted = 0
    for j in range(len(trades) - 1):
        t1, t2 = trades[j], trades[j + 1]
        ed1 = t1.get("exit_date", "")
        ed2 = t2.get("entry_date", "")
        # Can't compute price diff without close data, use pnl as proxy
        if abs(t2.get("pnl_pct", 99)) < 2 and t2.get("holding_days", 99) <= 3:
            wasted += 1

    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t.get("exit_reason", "?")] += 1

    print(f"  {label}:")
    print(f"    Trades: {n:>4d} | Wins: {len(wins):>4d} | Losses: {len(losses):>4d}")
    print(f"    Win Rate: {wr:>5.1f}% | PF: {pf:>5.2f} | Avg PnL: {avg_pnl:>+6.2f}%")
    print(f"    Avg Hold: {avg_hold:>5.1f}d | Marginal (-2~+2%): {len(marginal):>3d} ({len(marginal)/n*100:.1f}%)")
    er_str = " | ".join([f"{k}:{v}" for k, v in sorted(exit_reasons.items())])
    print(f"    Exit: {er_str}")
    return {"trades": n, "wr": wr, "pf": pf, "avg_pnl": avg_pnl, "avg_hold": avg_hold,
            "marginal": len(marginal), "total_pnl": tdf["pnl_pct"].sum()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", type=str, default="ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM")
    parser.add_argument("--model", type=str, default="lightgbm")
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

    pick = [s.strip() for s in args.pick.split(",")]
    symbols = [s for s in pick if s in loader.symbols]

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    print("=" * 120)
    print("🔬 V6 vs V7 A/B COMPARISON")
    print("=" * 120)

    v6_all, v7_all = [], []
    v6_blocked = {"cooldown": 0, "reentry": 0, "relaxed": 0}
    v7_blocked = {"cooldown": 0, "reentry": 0, "relaxed": 0}

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

            r6 = backtest_v6(y_pred, rets, sym_test, feature_cols)
            r7 = backtest_v7(y_pred, rets, sym_test, feature_cols)

            for t in r6["trades"]:
                t["symbol"] = sym
                t["window"] = window.label
            for t in r7["trades"]:
                t["symbol"] = sym
                t["window"] = window.label

            v6_all.extend(r6["trades"])
            v7_all.extend(r7["trades"])
            v7_blocked["cooldown"] += r7.get("n_cooldown_blocked", 0)
            v7_blocked["reentry"] += r7.get("n_reentry_blocked", 0)
            v7_blocked["relaxed"] += r7.get("n_relaxed_entry", 0)

    # Per-symbol comparison
    print(f"\n{'─' * 120}")
    print(f"📊 PER-SYMBOL COMPARISON")
    print(f"{'─' * 120}")
    print(f"  {'Symbol':<8} {'V6 Trades':>10} {'V6 WR':>8} {'V6 PnL':>10} {'V7 Trades':>10} {'V7 WR':>8} {'V7 PnL':>10} {'ΔTrades':>8} {'ΔWR':>8} {'ΔPnL':>10}")

    v6_by_sym = defaultdict(list)
    v7_by_sym = defaultdict(list)
    for t in v6_all:
        v6_by_sym[t.get("symbol", t.get("entry_symbol", "?"))].append(t)
    for t in v7_all:
        v7_by_sym[t.get("symbol", t.get("entry_symbol", "?"))].append(t)

    for sym in sorted(set(list(v6_by_sym.keys()) + list(v7_by_sym.keys()))):
        t6 = v6_by_sym.get(sym, [])
        t7 = v7_by_sym.get(sym, [])
        n6, n7 = len(t6), len(t7)
        wr6 = sum(1 for t in t6 if t["pnl_pct"] > 0) / max(n6, 1) * 100
        wr7 = sum(1 for t in t7 if t["pnl_pct"] > 0) / max(n7, 1) * 100
        pnl6 = sum(t["pnl_pct"] for t in t6)
        pnl7 = sum(t["pnl_pct"] for t in t7)
        print(f"  {sym:<8} {n6:>10} {wr6:>7.1f}% {pnl6:>+9.1f}% {n7:>10} {wr7:>7.1f}% {pnl7:>+9.1f}% {n7-n6:>+8d} {wr7-wr6:>+7.1f}% {pnl7-pnl6:>+9.1f}%")

    # Aggregate
    print(f"\n{'═' * 120}")
    print(f"📊 AGGREGATE V6:")
    s6 = summarize(v6_all, "V6")
    print(f"\n📊 AGGREGATE V7:")
    s7 = summarize(v7_all, "V7")

    print(f"\n{'═' * 120}")
    print(f"📈 IMPROVEMENT V6 → V7:")
    print(f"{'═' * 120}")
    if s6 and s7:
        print(f"  Trades:    {s6['trades']:>4d} → {s7['trades']:>4d} (Δ{s7['trades']-s6['trades']:>+4d})")
        print(f"  WR:        {s6['wr']:>5.1f}% → {s7['wr']:>5.1f}% (Δ{s7['wr']-s6['wr']:>+5.1f}%)")
        print(f"  PF:        {s6['pf']:>5.2f} → {s7['pf']:>5.2f}")
        print(f"  Avg PnL:   {s6['avg_pnl']:>+6.2f}% → {s7['avg_pnl']:>+6.2f}%")
        print(f"  Total PnL: {s6['total_pnl']:>+8.1f}% → {s7['total_pnl']:>+8.1f}%")
        print(f"  Avg Hold:  {s6['avg_hold']:>5.1f}d → {s7['avg_hold']:>5.1f}d")
        print(f"  Marginal:  {s6['marginal']:>4d} → {s7['marginal']:>4d} (Δ{s7['marginal']-s6['marginal']:>+4d})")

    print(f"\n  V7 filter stats:")
    print(f"    Cooldown blocked:  {v7_blocked['cooldown']}")
    print(f"    Re-entry blocked:  {v7_blocked['reentry']}")
    print(f"    Relaxed entries:   {v7_blocked['relaxed']}")


if __name__ == "__main__":
    main()
