"""
V8b vs V7 — Minimal changes to V7
===================================
V7 is already strong (71.4% WR, PF 35.88). V8 over-engineered and broke it.
V8b: Only change exit logic to let winners run longer.
  1. Min hold 8 days (V7=3) — don't exit on short pullbacks
  2. Zombie exit 15 bars (V7=8) — more time to develop
  3. Require 3 consecutive exit signals (V7=2)
  4. Keep V7's ATR stop-loss (not fixed %) 
  5. Tighter trailing after big profits (>15%)
  6. Profit lock: once >10% profit, don't exit below +4%
  NO changes to entry logic — V7 entry is good!
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


def backtest_v8b(y_pred, returns, df_test, feature_cols,
                 initial_capital=100_000_000, commission=0.0015, tax=0.001,
                 record_trades=True):
    """V8b: V7 entry + improved exit (let winners run)."""
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

    # V8b params — only exit changes
    MIN_HOLD = 8           # V7=3, V8b=8
    ZOMBIE_BARS = 15       # V7=8, V8b=15
    EXIT_CONFIRM = 3       # V7=2, V8b=3 consecutive exit signals
    PROFIT_LOCK_THRESHOLD = 0.10  # Lock profits after 10%
    PROFIT_LOCK_MIN = 0.04        # Keep at least 4%

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
        # ENTRY LOGIC — SAME AS V7 (proven to work)
        # ════════════════════════════════════════════════
        if new_position == 1 and position == 0:
            if cooldown_remaining > 0:
                new_position = 0

        if new_position == 1 and position == 0:
            if last_exit_price > 0:
                price_diff = abs(close[i] / last_exit_price - 1)
                if price_diff < 0.03:
                    new_position = 0

        if new_position == 1 and position == 0:
            prev_pred = int(y_pred[i - 2]) if i >= 2 else 0
            if prev_pred != 1:
                new_position = 0

        if new_position == 1 and position == 0:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3:
                        new_position = 0

        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            near_sma_support = (not np.isnan(sma20[i]) and
                               close[i] <= sma20[i] * 1.02 and
                               close[i] >= sma20[i] * 0.97)
            near_local_low = (not np.isnan(local_low_20[i]) and
                             close[i] <= local_low_20[i] * 1.05)
            in_uptrend_macro = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and
                               sma20[i] > sma50[i])
            if (near_sma_support or near_local_low) and in_uptrend_macro:
                min_score = 2
            else:
                min_score = 3
            if entry_score < min_score:
                new_position = 0
            if wp > 0.9 and rs <= 0 and bs < 2:
                new_position = 0
            if bb > 0.85 and bs < 2 and entry_score < 4:
                new_position = 0
            if new_position == 1:
                if wp > 0.78 and bb < 0.35:
                    new_position = 0
            if new_position == 1 and dp < 0.025:
                if entry_score < 4:
                    new_position = 0

        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            if dp < 0.025:
                position_size = 0.5
            elif bb > 0.7:
                position_size = 0.7
            else:
                position_size = 1.0

        # ════════════════════════════════════════════════
        # EXIT LOGIC — V8b: LET WINNERS RUN
        # ════════════════════════════════════════════════
        if position == 1:
            projected = equity[i - 1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0

            in_uptrend = rs > 0 and hl >= 2
            strong_uptrend = (in_uptrend and not np.isnan(sma20[i]) and
                            not np.isnan(sma50[i]) and sma20[i] > sma50[i])

            # ATR stop loss (same as V7)
            if not np.isnan(atr14[i]) and close[i] > 0:
                atr_stop = 2.5 * atr14[i] / close[i]
                atr_stop = max(0.03, min(atr_stop, 0.08))
            else:
                atr_stop = 0.05

            # 1) Stop loss — same as V7
            if cum_ret <= -atr_stop:
                new_position = 0
                exit_reason = "stop_loss"
                n_stop_loss += 1

            # 2) Trailing stop — V8b: tighter at high profits, wider at low profits
            elif max_profit > 0.03 and new_position == 1:
                if max_profit > 0.25:
                    # Big winner: keep 75-85% of profits
                    trail_pct = 0.20 if not strong_uptrend else 0.15
                elif max_profit > 0.15:
                    trail_pct = 0.30 if not strong_uptrend else 0.25
                elif max_profit > 0.08:
                    trail_pct = 0.45 if not strong_uptrend else 0.35
                else:
                    # Small profit: give more room (V7 was 0.55-0.65)
                    trail_pct = 0.70 if not strong_uptrend else 0.85
                giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                if giveback >= trail_pct:
                    new_position = 0
                    exit_reason = "trailing_stop"
                    n_trailing_stop += 1

            # 3) V8b: Profit lock — once >10%, don't let it go below 4%
            if new_position == 1 and max_profit >= PROFIT_LOCK_THRESHOLD:
                if cum_ret < PROFIT_LOCK_MIN:
                    new_position = 0
                    exit_reason = "profit_lock"
                    n_profit_lock += 1

            # 4) V8b: Zombie exit — 15 bars (V7=8)
            if new_position == 1 and hold_days >= ZOMBIE_BARS and cum_ret < 0.01:
                new_position = 0
                exit_reason = "zombie_exit"
                n_zombie_exit += 1

            # 5) V8b: MIN HOLD 8 DAYS (V7=3) — except stop-loss
            if new_position == 0 and exit_reason not in ("stop_loss",) and hold_days < MIN_HOLD:
                if cum_ret > -atr_stop:  # Not near stop
                    new_position = 1
                    n_min_hold_saved += 1

            # 6) V8b: Exit confirmation — 3 consecutive (V7=2)
            if new_position == 0 and exit_reason == "signal":
                if raw_signal == 0:
                    consecutive_exit_signals += 1
                else:
                    consecutive_exit_signals = 0
                if consecutive_exit_signals < EXIT_CONFIRM:
                    new_position = 1
                else:
                    consecutive_exit_signals = 0

            # 7) Strong trend override (same as V7)
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
                cooldown_remaining = 3
                last_exit_price = close[i]

                if record_trades and entry_equity > 0:
                    # Use close prices for accurate PnL (avoids position_size equity mismatch)
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

    print("=" * 130)
    print("🔬 V7 vs V8b A/B COMPARISON — V8b: Keep V7 entry, improve exit only")
    print("=" * 130)
    print("V8b changes: min_hold=8d(V7=3), zombie=15d(V7=8), exit_confirm=3(V7=2), profit_lock@10%→4%, tighter big-profit trailing")

    v7_all, v8b_all = [], []

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
            r8b = backtest_v8b(y_pred, rets, sym_test, feature_cols)

            for t in r7["trades"]:
                t["symbol"] = sym; t["window"] = window.label
            for t in r8b["trades"]:
                t["symbol"] = sym; t["window"] = window.label

            v7_all.extend(r7["trades"])
            v8b_all.extend(r8b["trades"])

    # Per-symbol comparison
    print(f"\n{'─' * 140}")
    print(f"{'Symbol':<8} │ {'V7 #':>5} {'V7 WR':>7} {'V7 Avg':>8} {'V7 Tot':>9} {'V7 Hld':>6} │ "
          f"{'V8b #':>5} {'V8b WR':>7} {'V8b Avg':>8} {'V8b Tot':>9} {'V8b Hld':>6} │ {'ΔAvg':>7} {'ΔTot':>8} {'?':>3}")
    print(f"{'─' * 140}")

    v7_by_sym = defaultdict(list)
    v8b_by_sym = defaultdict(list)
    for t in v7_all:
        v7_by_sym[t.get("symbol", t.get("entry_symbol", "?"))].append(t)
    for t in v8b_all:
        v8b_by_sym[t.get("symbol", t.get("entry_symbol", "?"))].append(t)

    better_count = 0
    for sym in sorted(set(list(v7_by_sym.keys()) + list(v8b_by_sym.keys()))):
        t7 = v7_by_sym.get(sym, [])
        t8 = v8b_by_sym.get(sym, [])
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
        print(f"{sym:<8} │ {n7:>5} {wr7:>6.1f}% {avg7:>+7.1f}% {pnl7:>+8.1f}% {hold7:>5.1f}d │ "
              f"{n8:>5} {wr8:>6.1f}% {avg8:>+7.1f}% {pnl8:>+8.1f}% {hold8:>5.1f}d │ "
              f"{avg8-avg7:>+6.1f}% {pnl8-pnl7:>+7.1f}% {better}")

    total_syms = len(set(list(v7_by_sym.keys()) + list(v8b_by_sym.keys())))

    print(f"\n{'═' * 140}")
    print(f"📊 AGGREGATE V7:")
    s7 = summarize(v7_all, "V7")
    print(f"\n📊 AGGREGATE V8b:")
    s8 = summarize(v8b_all, "V8b")

    print(f"\n{'═' * 140}")
    print(f"📈 IMPROVEMENT V7 → V8b:")
    print(f"{'═' * 140}")
    if s7 and s8:
        print(f"  Trades:    {s7['trades']:>4d} → {s8['trades']:>4d} (Δ{s8['trades']-s7['trades']:>+4d})")
        print(f"  WR:        {s7['wr']:>5.1f}% → {s8['wr']:>5.1f}% (Δ{s8['wr']-s7['wr']:>+5.1f}%)")
        print(f"  PF:        {s7['pf']:>5.2f} → {s8['pf']:>5.2f}")
        print(f"  Avg PnL:   {s7['avg_pnl']:>+6.2f}% → {s8['avg_pnl']:>+6.2f}% {'✅' if s8['avg_pnl'] > s7['avg_pnl'] else '❌'}")
        print(f"  Total PnL: {s7['total_pnl']:>+8.1f}% → {s8['total_pnl']:>+8.1f}% {'✅' if s8['total_pnl'] > s7['total_pnl'] else '❌'}")
        print(f"  Avg Hold:  {s7['avg_hold']:>5.1f}d → {s8['avg_hold']:>5.1f}d")
        print(f"  Better per-symbol: {better_count}/{total_syms}")

    if v8b_all:
        exits = defaultdict(int)
        for t in v8b_all:
            exits[t.get("exit_reason", "?")] += 1
        print(f"\n  V8b exit breakdown: {dict(exits)}")

        holds = [t["holding_days"] for t in v8b_all]
        pnls = [t["pnl_pct"] for t in v8b_all]
        print(f"\n  V8b Hold Distribution:")
        for lo, hi in [(1, 7), (7, 14), (14, 21), (21, 42), (42, 100)]:
            mask = [(lo <= h < hi) for h in holds]
            cnt = sum(mask)
            if cnt > 0:
                avg_p = np.mean([p for p, m in zip(pnls, mask) if m])
                print(f"    [{lo:2d}-{hi:2d}d): {cnt:4d} trades, avg PnL: {avg_p:+.2f}%")


if __name__ == "__main__":
    main()
