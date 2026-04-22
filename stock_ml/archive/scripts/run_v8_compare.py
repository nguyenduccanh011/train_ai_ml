"""
V8 vs V7 A/B Comparison
========================
V8 improvements over V7 (targeting >10% avg PnL per trade):
  1. Min hold 10 days (V7=3) — let winners run
  2. Zombie exit 20 bars (V7=8) — give trades time to develop
  3. Entry score ≥ 4 (V7=3) — fewer but higher conviction
  4. Tighter stop-loss: -4% (V7=ATR ~5-8%)
  5. Better trailing: activate after 5% profit, trail 25% from peak
  6. Profit lock: once >8% profit, never exit below +3%
"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v7_compare import backtest_v7, summarize


def backtest_v8(y_pred, returns, df_test, feature_cols,
                initial_capital=100_000_000, commission=0.0015, tax=0.001,
                record_trades=True):
    """V8: High-conviction swing trading — fewer trades, bigger wins."""
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

    # V8 params
    MIN_HOLD = 10          # Don't sell before 10 days (except stop-loss)
    ZOMBIE_BARS = 20       # Give trades 20 bars to develop
    STOP_LOSS = -0.05      # Fixed 5% stop-loss
    ENTRY_MIN_SCORE = 4    # High conviction only
    TRAIL_ACTIVATE = 0.05  # Activate trailing after 5% profit
    PROFIT_LOCK = 0.03     # Once hit 8%, lock in at least 3%
    PROFIT_LOCK_THRESHOLD = 0.08
    COOLDOWN = 5           # 5 bars cooldown after exit
    REENTRY_FILTER = 0.05  # 5% price change required for re-entry

    cooldown_remaining = 0
    last_exit_price = 0

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
    n_profit_lock = 0
    n_min_hold_saved = 0

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

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # ════════════════════════════════════════════════
        # ENTRY LOGIC — V8: HIGH CONVICTION ONLY
        # ════════════════════════════════════════════════
        if new_position == 1 and position == 0:
            # Cooldown
            if cooldown_remaining > 0:
                new_position = 0

        if new_position == 1 and position == 0:
            # Re-entry price filter — need 5% move
            if last_exit_price > 0:
                price_diff = abs(close[i] / last_exit_price - 1)
                if price_diff < REENTRY_FILTER:
                    new_position = 0

        if new_position == 1 and position == 0:
            # Entry confirmation: 2 consecutive buy signals
            prev_pred = int(y_pred[i - 2]) if i >= 2 else 0
            if prev_pred != 1:
                new_position = 0

        # Regime filter — don't buy in clear downtrend
        if new_position == 1 and position == 0:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    new_position = 0

        # V8: HIGH CONVICTION ENTRY SCORE
        if new_position == 1 and position == 0:
            entry_score = sum([
                wp < 0.70,           # Not overbought
                dp > 0.03,           # Room to resistance
                rs > 0,              # RSI improving
                vs > 1.2,            # Volume surge (stricter)
                hl >= 2,             # Higher lows pattern
                bs >= 2,             # Breakout setup
                od > 0,              # OBV divergence positive
                bb < 0.6,            # Volatility not too wide
            ])

            # V8: Require score ≥ 4 (out of 8)
            if entry_score < ENTRY_MIN_SCORE:
                new_position = 0

            # Overbought filter
            if wp > 0.85 and rs <= 0:
                new_position = 0

            # Uptrend requirement
            in_uptrend = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and
                         sma20[i] > sma50[i])
            if not in_uptrend and entry_score < 5:
                new_position = 0

        # Position sizing
        if new_position == 1 and position == 0:
            position_size = 1.0

        # ════════════════════════════════════════════════
        # EXIT LOGIC — V8: LET WINNERS RUN
        # ════════════════════════════════════════════════
        if position == 1:
            projected = equity[i - 1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0

            in_uptrend = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and
                         sma20[i] > sma50[i])

            # 1) STOP-LOSS: Fixed 5% — always active
            if cum_ret <= STOP_LOSS:
                new_position = 0
                exit_reason = "stop_loss"
                n_stop_loss += 1

            # 2) TRAILING STOP — only after 5% profit
            elif max_profit > TRAIL_ACTIVATE:
                # Dynamic trail based on profit level
                if max_profit > 0.25:
                    trail_pct = 0.20 if not in_uptrend else 0.30
                elif max_profit > 0.15:
                    trail_pct = 0.25 if not in_uptrend else 0.35
                elif max_profit > 0.08:
                    trail_pct = 0.35 if not in_uptrend else 0.45
                else:
                    trail_pct = 0.45 if not in_uptrend else 0.55

                giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                if giveback >= trail_pct:
                    new_position = 0
                    exit_reason = "trailing_stop"
                    n_trailing_stop += 1

            # 3) PROFIT LOCK — once hit 8%, don't let it go below 3%
            if new_position == 1 and max_profit >= PROFIT_LOCK_THRESHOLD:
                if cum_ret < PROFIT_LOCK:
                    new_position = 0
                    exit_reason = "profit_lock"
                    n_profit_lock += 1

            # 4) ZOMBIE EXIT — 20 bars with no progress
            if new_position == 1 and hold_days >= ZOMBIE_BARS and cum_ret < 0.02:
                new_position = 0
                exit_reason = "zombie_exit"
                n_zombie_exit += 1

            # 5) V8 KEY: MIN HOLD 10 DAYS (except stop-loss)
            if new_position == 0 and exit_reason not in ("stop_loss",) and hold_days < MIN_HOLD:
                if cum_ret > STOP_LOSS:  # Not in danger zone
                    new_position = 1
                    n_min_hold_saved += 1

            # 6) Signal exit needs 3 consecutive signals (V7 had 2)
            if new_position == 0 and exit_reason == "signal":
                if raw_signal == 0:
                    consecutive_exit_signals += 1
                else:
                    consecutive_exit_signals = 0
                if consecutive_exit_signals < 3:
                    new_position = 1
                else:
                    consecutive_exit_signals = 0

            # 7) Strong trend override — don't exit if trend is very strong
            if new_position == 0 and exit_reason == "signal":
                if cum_ret > 0.05 and rs > 0 and hl >= 3 and in_uptrend:
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
                entry_features = {
                    "entry_wp": wp, "entry_dp": dp, "entry_rs": rs,
                    "entry_vs": vs, "entry_bs": bs, "entry_hl": hl,
                    "entry_od": od, "entry_bb": bb,
                    "entry_score": sum([wp < 0.70, dp > 0.03, rs > 0, vs > 1.2,
                                       hl >= 2, bs >= 2, od > 0, bb < 0.6]),
                    "entry_date": str(dates[i])[:10],
                    "entry_symbol": str(symbols[i]),
                    "position_size": position_size,
                }
            else:
                cost = equity[i - 1] * position_size * (commission + tax)
                cooldown_remaining = COOLDOWN
                last_exit_price = close[i]

                if record_trades and entry_equity > 0:
                    exit_eq = equity[i - 1] - cost
                    pnl = exit_eq - entry_equity
                    pnl_pct = (pnl / entry_equity * 100) if entry_equity > 0 else 0
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
        "n_zombie_exit": n_zombie_exit, "n_profit_lock": n_profit_lock,
        "n_min_hold_saved": n_min_hold_saved,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", type=str, default="ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC")
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
    print("🔬 V7 vs V8 A/B COMPARISON — Target: >10% avg PnL per trade")
    print("=" * 120)
    print("V8 changes: min_hold=10d, zombie=20d, entry_score≥4/8, stop=-5%, trail@5%+, profit_lock@8%→3%, cooldown=5")

    v7_all, v8_all = [], []

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

            r7 = backtest_v7(y_pred, rets, sym_test, feature_cols)
            r8 = backtest_v8(y_pred, rets, sym_test, feature_cols)

            for t in r7["trades"]:
                t["symbol"] = sym
                t["window"] = window.label
            for t in r8["trades"]:
                t["symbol"] = sym
                t["window"] = window.label

            v7_all.extend(r7["trades"])
            v8_all.extend(r8["trades"])

    # Per-symbol comparison
    print(f"\n{'─' * 130}")
    print(f"{'Symbol':<8} │ {'V7 Trades':>9} {'V7 WR':>7} {'V7 AvgPnL':>10} {'V7 TotPnL':>10} {'V7 Hold':>7} │ "
          f"{'V8 Trades':>9} {'V8 WR':>7} {'V8 AvgPnL':>10} {'V8 TotPnL':>10} {'V8 Hold':>7} │ {'Better':>6}")
    print(f"{'─' * 130}")

    v7_by_sym = defaultdict(list)
    v8_by_sym = defaultdict(list)
    for t in v7_all:
        v7_by_sym[t.get("symbol", t.get("entry_symbol", "?"))].append(t)
    for t in v8_all:
        v8_by_sym[t.get("symbol", t.get("entry_symbol", "?"))].append(t)

    better_count = 0
    for sym in sorted(set(list(v7_by_sym.keys()) + list(v8_by_sym.keys()))):
        t7 = v7_by_sym.get(sym, [])
        t8 = v8_by_sym.get(sym, [])
        n7, n8 = len(t7), len(t8)
        wr7 = sum(1 for t in t7 if t["pnl_pct"] > 0) / max(n7, 1) * 100
        wr8 = sum(1 for t in t8 if t["pnl_pct"] > 0) / max(n8, 1) * 100
        pnl7 = sum(t["pnl_pct"] for t in t7)
        pnl8 = sum(t["pnl_pct"] for t in t8)
        avg7 = np.mean([t["pnl_pct"] for t in t7]) if t7 else 0
        avg8 = np.mean([t["pnl_pct"] for t in t8]) if t8 else 0
        hold7 = np.mean([t["holding_days"] for t in t7]) if t7 else 0
        hold8 = np.mean([t["holding_days"] for t in t8]) if t8 else 0
        better = "✅" if avg8 > avg7 else "❌"
        if avg8 > avg7:
            better_count += 1
        print(f"{sym:<8} │ {n7:>9} {wr7:>6.1f}% {avg7:>+9.2f}% {pnl7:>+9.1f}% {hold7:>6.1f}d │ "
              f"{n8:>9} {wr8:>6.1f}% {avg8:>+9.2f}% {pnl8:>+9.1f}% {hold8:>6.1f}d │ {better:>6}")

    total_syms = len(set(list(v7_by_sym.keys()) + list(v8_by_sym.keys())))

    # Aggregate
    print(f"\n{'═' * 130}")
    print(f"📊 AGGREGATE V7:")
    s7 = summarize(v7_all, "V7")
    print(f"\n📊 AGGREGATE V8:")
    s8 = summarize(v8_all, "V8")

    print(f"\n{'═' * 130}")
    print(f"📈 IMPROVEMENT V7 → V8:")
    print(f"{'═' * 130}")
    if s7 and s8:
        print(f"  Trades:    {s7['trades']:>4d} → {s8['trades']:>4d} (Δ{s8['trades']-s7['trades']:>+4d})")
        print(f"  WR:        {s7['wr']:>5.1f}% → {s8['wr']:>5.1f}% (Δ{s8['wr']-s7['wr']:>+5.1f}%)")
        print(f"  PF:        {s7['pf']:>5.2f} → {s8['pf']:>5.2f}")
        print(f"  Avg PnL:   {s7['avg_pnl']:>+6.2f}% → {s8['avg_pnl']:>+6.2f}% {'✅' if s8['avg_pnl'] > s7['avg_pnl'] else '❌'}")
        print(f"  Total PnL: {s7['total_pnl']:>+8.1f}% → {s8['total_pnl']:>+8.1f}% {'✅' if s8['total_pnl'] > s7['total_pnl'] else '❌'}")
        print(f"  Avg Hold:  {s7['avg_hold']:>5.1f}d → {s8['avg_hold']:>5.1f}d")
        print(f"  Better per-symbol: {better_count}/{total_syms}")

    # V8 exit stats
    if v8_all:
        exits = defaultdict(int)
        for t in v8_all:
            exits[t.get("exit_reason", "?")] += 1
        print(f"\n  V8 exit breakdown: {dict(exits)}")

    # Hold period analysis
    if v8_all:
        holds = [t["holding_days"] for t in v8_all]
        pnls = [t["pnl_pct"] for t in v8_all]
        print(f"\n  V8 Hold Distribution:")
        for lo, hi in [(1, 7), (7, 14), (14, 21), (21, 42), (42, 100)]:
            mask = [(lo <= h < hi) for h in holds]
            cnt = sum(mask)
            if cnt > 0:
                avg_p = np.mean([p for p, m in zip(pnls, mask) if m])
                print(f"    [{lo:2d}-{hi:2d}d): {cnt:4d} trades, avg PnL: {avg_p:+.2f}%")


if __name__ == "__main__":
    main()
