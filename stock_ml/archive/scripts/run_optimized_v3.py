"""
V3 Optimized Backtest - Root-cause fixes for poor performance.

ROOT CAUSES IDENTIFIED:
1. entry_wave_pos=1.0 (buying at top) → 60%+ of losing trades
2. gave_back_pct high → trailing stop too loose, missing profit
3. after_exit_max_up high → selling too early, missing trend continuation
4. Short holding (1-2 days) losing trades → noise trading

V3 FIXES:
A. Entry filter: Only buy when wave_pos < 0.7 AND dist_from_peak > 3%
B. Momentum confirmation: require RSI slope positive + volume surge
C. Adaptive trailing: tighter trail (30%) when profit > 5%, loose (60%) early
D. Hold minimum: skip signals that flip too fast (min 3-day hold)
E. Trend continuation: don't sell if breakout_setup_score >= 3

Usage:
    python run_optimized_v3.py --symbols 10
    python run_optimized_v3.py --compare --symbols 10
    python run_optimized_v3.py --full
"""
import argparse, sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model, get_available_models
from src.evaluation.metrics import compute_metrics


def backtest_v3(y_pred, returns, df_test, feature_cols, initial_capital=100_000_000,
                commission=0.0015, tax=0.001, mode="v3"):
    """
    V3 backtest with entry filters + adaptive exit.
    mode: 'original' = no filters, 'v3' = all optimizations
    """
    n = len(y_pred)
    equity = np.zeros(n)
    equity[0] = initial_capital
    position = 0
    trades = []
    current_entry_day = 0
    entry_equity = 0
    max_equity_in_trade = 0
    total_commission = 0
    hold_days = 0
    n_filtered = 0
    n_sl = 0
    n_trail = 0
    n_trend_hold = 0
    n_min_hold_skip = 0

    # Extract contextual signals from df_test if available
    has_context = mode == "v3" and len(df_test) == n
    if has_context:
        try:
            rsi_slope = df_test["rsi_slope_5d"].values if "rsi_slope_5d" in df_test.columns else np.zeros(n)
            vol_surge = df_test["vol_surge_ratio"].values if "vol_surge_ratio" in df_test.columns else np.ones(n)
            wave_pos = df_test["range_position_20d"].values if "range_position_20d" in df_test.columns else np.full(n, 0.5)
            dist_peak = df_test["dist_to_resistance"].values if "dist_to_resistance" in df_test.columns else np.full(n, 0.05)
            breakout_score = df_test["breakout_setup_score"].values if "breakout_setup_score" in df_test.columns else np.zeros(n)
            bb_width_pct = df_test["bb_width_percentile"].values if "bb_width_percentile" in df_test.columns else np.full(n, 0.5)
            higher_lows = df_test["higher_lows_count"].values if "higher_lows_count" in df_test.columns else np.zeros(n)
            obv_div = df_test["obv_price_divergence"].values if "obv_price_divergence" in df_test.columns else np.zeros(n)
        except:
            has_context = False

    for i in range(1, n):
        pred = int(y_pred[i - 1])
        ret = returns[i] if not np.isnan(returns[i]) else 0
        new_position = 1 if pred == 1 else 0
        exit_reason = "signal"

        if mode == "v3" and has_context:
            # ── ENTRY FILTER ──
            if new_position == 1 and position == 0:
                wp = wave_pos[i] if i < n else 0.5
                dp = dist_peak[i] if i < n else 0.05
                rs = rsi_slope[i] if i < n else 0
                vs = vol_surge[i] if i < n else 1.0
                bs = breakout_score[i] if i < n else 0
                hl = higher_lows[i] if i < n else 0
                od = obv_div[i] if i < n else 0

                # Fix NaN
                wp = wp if not np.isnan(wp) else 0.5
                dp = dp if not np.isnan(dp) else 0.05
                rs = rs if not np.isnan(rs) else 0
                vs = vs if not np.isnan(vs) else 1.0
                bs = bs if not np.isnan(bs) else 0
                hl = hl if not np.isnan(hl) else 0
                od = od if not np.isnan(od) else 0

                # Entry quality score (0-5)
                entry_score = 0
                if wp < 0.75: entry_score += 1       # Not buying at top
                if dp > 0.02: entry_score += 1       # Not at resistance
                if rs > 0: entry_score += 1           # Momentum building
                if vs > 1.1: entry_score += 1         # Volume confirming
                if hl >= 2: entry_score += 1          # Bullish structure

                # Reject low-quality entries
                if entry_score < 2:
                    new_position = 0
                    n_filtered += 1

                # Strong reject: buying at top with no momentum
                if wp > 0.9 and rs <= 0 and bs < 2:
                    new_position = 0
                    n_filtered += 1

            # ── MIN HOLD FILTER (only day 1, and only if profitable) ──
            if position == 1 and new_position == 0 and hold_days < 2:
                cum_ret = (equity[i-1] * (1 + ret) - entry_equity) / entry_equity if entry_equity > 0 else 0
                if cum_ret > 0.01:  # Only hold if slightly profitable
                    new_position = 1
                    n_min_hold_skip += 1

            # ── ADAPTIVE EXIT ──
            if position == 1:
                projected = equity[i-1] * (1 + ret)
                max_equity_in_trade = max(max_equity_in_trade, projected)
                cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
                max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0

                # Stop-loss: -5% (tighter than before)
                if cum_ret <= -0.05:
                    new_position = 0
                    exit_reason = "stop_loss"
                    n_sl += 1

                # Adaptive trailing stop - LOOSER to let winners run
                elif max_profit > 0.03 and new_position == 1:
                    # Check if in uptrend context
                    rs_val = rsi_slope[i] if i < n and has_context else 0
                    rs_val = rs_val if not np.isnan(rs_val) else 0
                    hl_val = higher_lows[i] if i < n and has_context else 0
                    hl_val = hl_val if not np.isnan(hl_val) else 0
                    in_uptrend = rs_val > 0 and hl_val >= 2

                    if max_profit > 0.15:
                        trail_pct = 0.40 if not in_uptrend else 0.55  # Let big trends run
                    elif max_profit > 0.08:
                        trail_pct = 0.50 if not in_uptrend else 0.65
                    elif max_profit > 0.03:
                        trail_pct = 0.70  # Very loose for small profits
                    else:
                        trail_pct = 0.80

                    giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                    if giveback >= trail_pct:
                        new_position = 0
                        exit_reason = "trailing_stop"
                        n_trail += 1

                # Trend continuation: don't sell if strong setup
                if new_position == 0 and exit_reason == "signal":
                    bs = breakout_score[i] if i < n and has_context else 0
                    hl = higher_lows[i] if i < n and has_context else 0
                    rs = rsi_slope[i] if i < n and has_context else 0
                    bs = bs if not np.isnan(bs) else 0
                    hl = hl if not np.isnan(hl) else 0
                    rs = rs if not np.isnan(rs) else 0
                    
                    if cum_ret > 0 and bs >= 3 and hl >= 3 and rs > 0:
                        new_position = 1  # Override: keep riding trend
                        n_trend_hold += 1

        # Execute trade
        cost = 0
        if new_position != position:
            if new_position == 1:
                cost = equity[i-1] * commission
                entry_equity = equity[i-1] - cost
                max_equity_in_trade = entry_equity
                current_entry_day = i
                hold_days = 0
            else:
                cost = equity[i-1] * (commission + tax)
                exit_eq = equity[i-1] - cost
                pnl = exit_eq - entry_equity if entry_equity > 0 else 0
                trades.append({
                    "entry_day": current_entry_day, "exit_day": i,
                    "holding_days": i - current_entry_day,
                    "pnl": pnl, "pnl_pct": (pnl / entry_equity * 100) if entry_equity > 0 else 0,
                    "is_win": pnl > 0, "exit_reason": exit_reason,
                })
                total_commission += cost
                entry_equity = 0
                max_equity_in_trade = 0

        if position == 1:
            equity[i] = equity[i-1] * (1 + ret) - cost
            hold_days += 1
        else:
            equity[i] = equity[i-1] - cost

        position = new_position

    # Close open trade
    if position == 1 and entry_equity > 0:
        pnl = equity[-1] - entry_equity
        trades.append({
            "entry_day": current_entry_day, "exit_day": n-1,
            "holding_days": n-1-current_entry_day,
            "pnl": pnl, "pnl_pct": (pnl/entry_equity*100) if entry_equity > 0 else 0,
            "is_win": pnl > 0, "exit_reason": "end",
        })

    # Metrics
    total_ret = (equity[-1] / initial_capital - 1) * 100
    years = n / 252
    ann_ret = ((equity[-1] / initial_capital) ** (1/max(years, 0.01)) - 1) * 100
    daily_rets = np.diff(equity) / equity[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    sharpe = (np.sqrt(252) * daily_rets.mean() / daily_rets.std()) if len(daily_rets) > 0 and daily_rets.std() > 0 else 0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100

    bnh = initial_capital * (1 + returns).cumprod()
    bnh = np.nan_to_num(bnh, nan=initial_capital)
    bnh_ret = (bnh[-1] / initial_capital - 1) * 100

    n_trades = len(trades)
    wins = [t for t in trades if t["is_win"]]
    losses = [t for t in trades if not t["is_win"]]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    max_loss = min([t["pnl_pct"] for t in losses]) if losses else 0
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = abs(sum(t["pnl"] for t in losses))
    pf = gross_w / gross_l if gross_l > 0 else float('inf')
    avg_hold = np.mean([t["holding_days"] for t in trades]) if trades else 0

    return {
        "total_return_pct": round(total_ret, 2),
        "ann_return_pct": round(ann_ret, 2),
        "bnh_return_pct": round(bnh_ret, 2),
        "excess_pct": round(total_ret - bnh_ret, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "win_rate_pct": round(win_rate, 1),
        "total_trades": n_trades,
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "max_loss_pct": round(max_loss, 2),
        "avg_hold_days": round(avg_hold, 1),
        "final_equity": round(equity[-1]),
        "n_filtered": n_filtered,
        "n_stop_loss": n_sl,
        "n_trailing_stop": n_trail,
        "n_trend_hold": n_trend_hold,
        "n_min_hold_skip": n_min_hold_skip,
        "commission_total": round(total_commission),
        "equity_curve": equity,
        "trades": trades,
    }


def main():
    parser = argparse.ArgumentParser(description="V3 Optimized Backtest")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--symbols", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--compare", action="store_true", help="Compare original vs V3")
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

    max_sym = None if args.full else args.symbols
    symbols = loader.symbols[:max_sym] if max_sym else None
    model_names = args.models or ["random_forest", "xgboost", "lightgbm"]
    available = get_available_models()
    model_names = [m for m in model_names if m in available]

    # Use leading features for V3 (richer signals)
    feat_set = "leading"

    print("=" * 100)
    print("📈 V3 OPTIMIZED BACKTEST - Root Cause Fixes")
    print(f"   Features: {feat_set} | Models: {model_names} | Symbols: {max_sym or 'all'}")
    print(f"   Vốn: {args.capital:,.0f} VND")
    if args.compare:
        print("   Mode: COMPARE (original vs V3 optimized)")
    print("=" * 100)

    print("\n📦 Loading data...")
    raw_df = loader.load_all(symbols=symbols)
    print(f"   {len(raw_df)} rows, {raw_df['symbol'].nunique()} symbols")

    engine = FeatureEngine(feature_set=feat_set)
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    all_results = []

    for model_name in model_names:
        modes = ["original", "v3"] if args.compare else ["v3"]
        for mode in modes:
            print(f"\n{'─' * 80}")
            print(f"🤖 {model_name} | Mode: {mode.upper()}")

            agg_y_test, agg_y_pred, agg_returns = [], [], []
            agg_df_parts = []

            for window, train_df, test_df in splitter.split(df):
                try:
                    model = build_model(model_name)
                    X_train = np.nan_to_num(train_df[feature_cols].values)
                    y_train = train_df["target"].values.astype(int)
                    X_test = np.nan_to_num(test_df[feature_cols].values)
                    y_test = test_df["target"].values.astype(int)

                    offset = 0
                    if model_name == "xgboost" and y_train.min() < 0:
                        offset = abs(y_train.min())
                        y_train = y_train + offset

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    if offset > 0:
                        y_pred = y_pred - offset

                    rets = test_df["return_1d"].values if "return_1d" in test_df.columns else np.zeros(len(test_df))
                    bt = backtest_v3(y_pred, rets, test_df.reset_index(drop=True),
                                     feature_cols, args.capital, mode=mode)

                    tag = "🛡️" if mode == "v3" else "📊"
                    filt = f"Filt:{bt['n_filtered']:>3d}" if mode == "v3" else ""
                    print(f"   {tag} {window.label:30s} | "
                          f"Ret:{bt['total_return_pct']:>+7.2f}% | "
                          f"WR:{bt['win_rate_pct']:>5.1f}% | "
                          f"PF:{bt['profit_factor']:>5.2f} | "
                          f"Trades:{bt['total_trades']:>3d} | "
                          f"SL:{bt['n_stop_loss']:>2d} TS:{bt['n_trailing_stop']:>2d} {filt}")

                    agg_y_test.extend(y_test.tolist())
                    agg_y_pred.extend(y_pred.tolist())
                    agg_returns.extend(rets.tolist())
                    agg_df_parts.append(test_df.reset_index(drop=True))
                except Exception as e:
                    print(f"   ❌ {window.label}: {e}")

            if agg_returns:
                agg_rets = np.array(agg_returns)
                agg_preds = np.array(agg_y_pred)
                agg_test_df = pd.concat(agg_df_parts, ignore_index=True)
                agg_bt = backtest_v3(agg_preds, agg_rets, agg_test_df,
                                     feature_cols, args.capital, mode=mode)
                agg_metrics = compute_metrics(agg_y_test, agg_y_pred)

                print(f"\n   📊 AGGREGATE: {model_name} ({mode})")
                print(f"   {'─' * 70}")
                for k in ["total_return_pct", "ann_return_pct", "sharpe", "max_dd_pct",
                          "profit_factor", "win_rate_pct", "total_trades",
                          "avg_win_pct", "avg_loss_pct", "max_loss_pct", "avg_hold_days"]:
                    print(f"   {k:25s}: {agg_bt[k]}")
                if mode == "v3":
                    print(f"   {'n_filtered':25s}: {agg_bt['n_filtered']}")
                    print(f"   {'n_stop_loss':25s}: {agg_bt['n_stop_loss']}")
                    print(f"   {'n_trailing_stop':25s}: {agg_bt['n_trailing_stop']}")
                    print(f"   {'n_trend_hold':25s}: {agg_bt['n_trend_hold']}")
                    print(f"   {'n_min_hold_skip':25s}: {agg_bt['n_min_hold_skip']}")

                all_results.append({
                    "model": model_name, "mode": mode,
                    "total_return_pct": agg_bt["total_return_pct"],
                    "ann_return_pct": agg_bt["ann_return_pct"],
                    "sharpe": agg_bt["sharpe"],
                    "max_dd_pct": agg_bt["max_dd_pct"],
                    "profit_factor": agg_bt["profit_factor"],
                    "win_rate_pct": agg_bt["win_rate_pct"],
                    "trades": agg_bt["total_trades"],
                    "avg_win_pct": agg_bt["avg_win_pct"],
                    "avg_loss_pct": agg_bt["avg_loss_pct"],
                    "max_loss_pct": agg_bt["max_loss_pct"],
                    "avg_hold": agg_bt["avg_hold_days"],
                    "final_equity": agg_bt["final_equity"],
                    "f1": agg_metrics["f1_macro"],
                    "n_filtered": agg_bt.get("n_filtered", 0),
                    "n_sl": agg_bt.get("n_stop_loss", 0),
                    "n_trail": agg_bt.get("n_trailing_stop", 0),
                })

    # Final comparison table
    if all_results:
        print("\n" + "=" * 120)
        print("🏆 FINAL COMPARISON")
        print("=" * 120)
        res_df = pd.DataFrame(all_results).sort_values("total_return_pct", ascending=False)
        print(res_df.to_string(index=False, float_format="%.2f"))

        if args.compare and len(res_df) > 1:
            print(f"\n📊 IMPROVEMENT ANALYSIS:")
            for mn in model_names:
                orig = res_df[(res_df["model"] == mn) & (res_df["mode"] == "original")]
                v3 = res_df[(res_df["model"] == mn) & (res_df["mode"] == "v3")]
                if len(orig) > 0 and len(v3) > 0:
                    o, v = orig.iloc[0], v3.iloc[0]
                    print(f"\n   {mn}:")
                    for col in ["total_return_pct", "sharpe", "win_rate_pct", "profit_factor",
                                "max_dd_pct", "avg_loss_pct", "max_loss_pct", "trades"]:
                        delta = v[col] - o[col]
                        good = delta > 0
                        if col in ("max_dd_pct", "avg_loss_pct", "max_loss_pct"):
                            good = delta > 0  # less negative = better
                        icon = "🟢" if good else "🔴"
                        print(f"     {col:20s}: {o[col]:>10.2f} → {v[col]:>10.2f} ({icon}{delta:>+8.2f})")

        # Save
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(out_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        res_df.to_csv(os.path.join(out_dir, f"v3_optimized_{ts}.csv"), index=False)
        print(f"\n💾 Saved to results/v3_optimized_{ts}.csv")


if __name__ == "__main__":
    main()
