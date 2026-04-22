"""
DEEP TRADE ANALYSIS — V4 Trade Quality Diagnostic
===================================================
Phân tích chi tiết từng giao dịch:
1. So sánh điểm bán vs lợi nhuận tốt nhất (exit efficiency)
2. Xếp hạng giao dịch lỗ lớn và giao dịch kém
3. Tìm pattern chung của giao dịch xấu
4. Đề xuất filter mới
"""
import sys, os, numpy as np, pandas as pd
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model, get_available_models
from src.evaluation.metrics import compute_metrics


def backtest_with_trade_details(y_pred, returns, df_test, feature_cols,
                                 initial_capital=100_000_000, commission=0.0015, tax=0.001):
    """V4 backtest that records detailed per-trade info including max possible profit."""
    n = len(y_pred)
    equity = np.zeros(n)
    equity[0] = initial_capital
    position = 0
    trades = []
    current_entry_day = 0
    entry_equity = 0
    entry_price = 0
    max_equity_in_trade = 0
    hold_days = 0
    position_size = 1.0

    # Extract features
    feat_arrays = {}
    feat_names = ["rsi_slope_5d", "vol_surge_ratio", "range_position_20d",
                   "dist_to_resistance", "breakout_setup_score", "bb_width_percentile",
                   "higher_lows_count", "obv_price_divergence"]
    defaults = {"rsi_slope_5d": 0, "vol_surge_ratio": 1.0, "range_position_20d": 0.5,
                 "dist_to_resistance": 0.05, "breakout_setup_score": 0, "bb_width_percentile": 0.5,
                 "higher_lows_count": 0, "obv_price_divergence": 0}
    for fn in feat_names:
        if fn in df_test.columns:
            arr = df_test[fn].values.copy()
            arr = np.where(np.isnan(arr), defaults[fn], arr)
            feat_arrays[fn] = arr
        else:
            feat_arrays[fn] = np.full(n, defaults[fn])

    close = df_test["close"].values if "close" in df_test.columns else np.zeros(n)
    sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
    sma50 = pd.Series(close).rolling(50, min_periods=10).mean().values

    if "high" in df_test.columns and "low" in df_test.columns:
        high = df_test["high"].values
        low = df_test["low"].values
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]
        atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().values
    else:
        atr14 = np.full(n, close.mean() * 0.02)

    # Date info
    dates = df_test["date"].values if "date" in df_test.columns else np.arange(n)
    symbols = df_test["symbol"].values if "symbol" in df_test.columns else ["?" ] * n

    def get_feat(name, idx):
        return feat_arrays[name][idx] if name in feat_arrays and idx < n else 0

    # Track entry features for each trade
    entry_features = {}

    for i in range(1, n):
        pred = int(y_pred[i - 1])
        ret = returns[i] if not np.isnan(returns[i]) else 0
        new_position = 1 if pred == 1 else 0
        exit_reason = "signal"

        wp = get_feat("range_position_20d", i)
        dp = get_feat("dist_to_resistance", i)
        rs = get_feat("rsi_slope_5d", i)
        vs = get_feat("vol_surge_ratio", i)
        bs = get_feat("breakout_setup_score", i)
        hl = get_feat("higher_lows_count", i)
        od = get_feat("obv_price_divergence", i)
        bb = get_feat("bb_width_percentile", i)

        # V4 REGIME FILTER
        if new_position == 1 and position == 0:
            if close is not None and sma50 is not None:
                price_below_sma50 = close[i] < sma50[i] if not np.isnan(sma50[i]) else False
                price_below_sma20 = close[i] < sma20[i] if not np.isnan(sma20[i]) else False
                if price_below_sma50 and price_below_sma20 and rs <= 0:
                    if bs < 3:
                        new_position = 0

        # V4 ENTRY FILTER
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

        # V4 POSITION SIZING
        if new_position == 1 and position == 0:
            position_size = 0.7 if bb > 0.7 else 1.0

        # MIN HOLD
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

            if atr14 is not None and not np.isnan(atr14[i]):
                atr_stop = 2.5 * atr14[i] / close[i]
                atr_stop = max(0.03, min(atr_stop, 0.08))
            else:
                atr_stop = 0.05

            if cum_ret <= -atr_stop:
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

        # EXECUTE
        cost = 0
        if new_position != position:
            if new_position == 1:
                deploy = equity[i-1] * position_size
                cost = deploy * commission
                entry_equity = deploy - cost
                max_equity_in_trade = entry_equity
                current_entry_day = i
                hold_days = 0
                entry_price = close[i] if close is not None else 0
                entry_features = {
                    "entry_wp": wp, "entry_dp": dp, "entry_rs": rs,
                    "entry_vs": vs, "entry_bs": bs, "entry_hl": hl,
                    "entry_od": od, "entry_bb": bb,
                    "entry_score": sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2]),
                    "entry_sma50_dist": (close[i] - sma50[i]) / sma50[i] if not np.isnan(sma50[i]) and sma50[i] > 0 else 0,
                    "entry_atr_pct": atr14[i] / close[i] if close[i] > 0 and not np.isnan(atr14[i]) else 0,
                    "entry_date": str(dates[i])[:10] if hasattr(dates[i], '__str__') else str(dates[i]),
                    "entry_symbol": str(symbols[i]),
                }
            else:
                cost = equity[i-1] * position_size * (commission + tax)
                exit_eq = equity[i-1] - cost
                pnl = exit_eq - entry_equity if entry_equity > 0 else 0
                pnl_pct = (pnl / entry_equity * 100) if entry_equity > 0 else 0

                # Calculate max possible profit (what we could have had)
                max_possible_pnl_pct = (max_equity_in_trade - entry_equity) / entry_equity * 100 if entry_equity > 0 else 0

                # Forward-looking: what was the best exit in next 5-20 days?
                future_max = 0
                if i < n - 1:
                    future_end = min(i + 20, n)
                    future_rets = returns[i+1:future_end]
                    cum_future = np.cumprod(1 + np.nan_to_num(future_rets))
                    if len(cum_future) > 0:
                        future_max = (cum_future.max() - 1) * 100

                trade_info = {
                    "entry_day": current_entry_day, "exit_day": i,
                    "holding_days": i - current_entry_day,
                    "pnl_pct": round(pnl_pct, 2),
                    "max_profit_pct": round(max_possible_pnl_pct, 2),
                    "exit_efficiency": round(pnl_pct / max_possible_pnl_pct * 100, 1) if max_possible_pnl_pct > 0.5 else (100 if pnl_pct > 0 else 0),
                    "left_on_table_pct": round(max_possible_pnl_pct - pnl_pct, 2),
                    "future_upside_pct": round(future_max, 2),
                    "exit_reason": exit_reason,
                    "position_size": position_size,
                    "exit_date": str(dates[i])[:10] if hasattr(dates[i], '__str__') else str(dates[i]),
                    **entry_features,
                }
                trades.append(trade_info)
                entry_equity = 0
                max_equity_in_trade = 0
                position_size = 1.0

        if position == 1:
            equity[i] = equity[i-1] * (1 + ret * position_size) - cost
            hold_days += 1
        else:
            equity[i] = equity[i-1] - cost

        position = new_position

    # Close open
    if position == 1 and entry_equity > 0:
        pnl = equity[-1] - entry_equity
        pnl_pct = (pnl / entry_equity * 100) if entry_equity > 0 else 0
        max_possible_pnl_pct = (max_equity_in_trade - entry_equity) / entry_equity * 100 if entry_equity > 0 else 0
        trades.append({
            "entry_day": current_entry_day, "exit_day": n-1,
            "holding_days": n-1-current_entry_day,
            "pnl_pct": round(pnl_pct, 2),
            "max_profit_pct": round(max_possible_pnl_pct, 2),
            "exit_efficiency": round(pnl_pct / max_possible_pnl_pct * 100, 1) if max_possible_pnl_pct > 0.5 else 100,
            "left_on_table_pct": round(max_possible_pnl_pct - pnl_pct, 2),
            "future_upside_pct": 0,
            "exit_reason": "end",
            "position_size": position_size,
            "exit_date": str(dates[-1])[:10],
            **entry_features,
        })

    return trades, equity


def main():
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

    symbols = loader.symbols[:5]
    model_names = ["random_forest", "xgboost", "lightgbm"]
    feat_set = "leading"

    print("=" * 120)
    print("🔍 DEEP TRADE ANALYSIS — V4 Trade Quality Diagnostic")
    print("=" * 120)

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set=feat_set)
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    all_trades = []

    for model_name in model_names:
        for window, train_df, test_df in splitter.split(df):
            try:
                model = build_model(model_name)
                X_train = np.nan_to_num(train_df[feature_cols].values)
                y_train = train_df["target"].values.astype(int)
                X_test = np.nan_to_num(test_df[feature_cols].values)

                offset = 0
                if model_name == "xgboost" and y_train.min() < 0:
                    offset = abs(y_train.min())
                    y_train = y_train + offset

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if offset > 0:
                    y_pred = y_pred - offset

                rets = test_df["return_1d"].values if "return_1d" in test_df.columns else np.zeros(len(test_df))
                trades, eq = backtest_with_trade_details(y_pred, rets, test_df.reset_index(drop=True), feature_cols)

                for t in trades:
                    t["model"] = model_name
                    t["window"] = window.label
                all_trades.extend(trades)
            except Exception as e:
                print(f"   ❌ {model_name} {window.label}: {e}")

    tdf = pd.DataFrame(all_trades)

    # ══════════════════════════════════════════════════════════════
    # 1. EXIT EFFICIENCY ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("📊 1. EXIT EFFICIENCY — Bán ở đâu so với lợi nhuận tốt nhất?")
    print("=" * 120)

    for mn in model_names:
        mt = tdf[tdf["model"] == mn]
        wins = mt[mt["pnl_pct"] > 0]
        losses = mt[mt["pnl_pct"] <= 0]

        print(f"\n  🤖 {mn.upper()} ({len(mt)} trades)")
        print(f"  {'─' * 90}")

        if len(wins) > 0:
            print(f"  WINNERS ({len(wins)}):")
            print(f"    Avg PnL:           {wins['pnl_pct'].mean():>+7.2f}%")
            print(f"    Avg Max Profit:    {wins['max_profit_pct'].mean():>+7.2f}%")
            print(f"    Avg Exit Eff:      {wins['exit_efficiency'].mean():>6.1f}%")
            print(f"    Avg Left on Table: {wins['left_on_table_pct'].mean():>+7.2f}%")
            print(f"    Avg Future Upside: {wins['future_upside_pct'].mean():>+7.2f}%")

        if len(losses) > 0:
            print(f"  LOSERS ({len(losses)}):")
            print(f"    Avg PnL:           {losses['pnl_pct'].mean():>+7.2f}%")
            print(f"    Avg Max Profit:    {losses['max_profit_pct'].mean():>+7.2f}%")
            print(f"    Avg Future Upside: {losses['future_upside_pct'].mean():>+7.2f}%")
            print(f"    Had profit >5%:    {len(losses[losses['max_profit_pct'] > 5])} trades")
            print(f"    Had profit >10%:   {len(losses[losses['max_profit_pct'] > 10])} trades")

    # ══════════════════════════════════════════════════════════════
    # 2. WORST TRADES RANKING
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("📉 2. TOP 30 WORST TRADES — Giao dịch lỗ lớn nhất")
    print("=" * 120)

    worst = tdf.nsmallest(30, "pnl_pct")
    print(f"\n  {'#':>3s} {'Model':>15s} {'Window':>30s} {'PnL%':>8s} {'MaxProf%':>9s} {'Eff%':>6s} "
          f"{'Hold':>5s} {'Exit':>12s} {'Symbol':>8s} {'EntryDate':>12s} "
          f"{'WP':>5s} {'RS':>6s} {'BS':>4s} {'HL':>4s} {'BB':>5s} {'Score':>6s} {'SMA50d':>7s}")
    print(f"  {'─' * 170}")
    for idx, (_, t) in enumerate(worst.iterrows()):
        print(f"  {idx+1:>3d} {t['model']:>15s} {t['window']:>30s} {t['pnl_pct']:>+8.2f} {t['max_profit_pct']:>+9.2f} "
              f"{t['exit_efficiency']:>6.1f} {t['holding_days']:>5.0f} {t['exit_reason']:>12s} "
              f"{t.get('entry_symbol','?'):>8s} {str(t.get('entry_date','?'))[:10]:>12s} "
              f"{t.get('entry_wp',0):>5.2f} {t.get('entry_rs',0):>6.3f} {t.get('entry_bs',0):>4.0f} "
              f"{t.get('entry_hl',0):>4.0f} {t.get('entry_bb',0):>5.2f} {t.get('entry_score',0):>6.0f} "
              f"{t.get('entry_sma50_dist',0):>7.3f}")

    # ══════════════════════════════════════════════════════════════
    # 3. FLAT/MARGINAL TRADES (< 2% profit)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("😐 3. MARGINAL TRADES — Giao dịch gần hòa vốn (-2% < PnL < 2%)")
    print("=" * 120)

    marginal = tdf[(tdf["pnl_pct"] > -2) & (tdf["pnl_pct"] < 2)]
    print(f"\n  Total marginal trades: {len(marginal)} / {len(tdf)} ({len(marginal)/len(tdf)*100:.1f}%)")
    if len(marginal) > 0:
        print(f"  Avg max_profit they had: {marginal['max_profit_pct'].mean():>+7.2f}%")
        print(f"  Avg future upside:       {marginal['future_upside_pct'].mean():>+7.2f}%")
        print(f"  Avg hold days:           {marginal['holding_days'].mean():>5.1f}")

    # ══════════════════════════════════════════════════════════════
    # 4. PATTERN ANALYSIS — What features predict bad trades?
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("🔬 4. FEATURE PATTERNS — Đặc điểm giao dịch xấu vs tốt")
    print("=" * 120)

    feature_cols_trade = ["entry_wp", "entry_dp", "entry_rs", "entry_vs",
                          "entry_bs", "entry_hl", "entry_bb", "entry_score",
                          "entry_sma50_dist", "entry_atr_pct"]

    # Categorize trades
    big_wins = tdf[tdf["pnl_pct"] > 10]
    small_wins = tdf[(tdf["pnl_pct"] > 0) & (tdf["pnl_pct"] <= 10)]
    small_losses = tdf[(tdf["pnl_pct"] <= 0) & (tdf["pnl_pct"] > -3)]
    big_losses = tdf[tdf["pnl_pct"] <= -3]

    categories = {
        "Big Wins (>10%)": big_wins,
        "Small Wins (0-10%)": small_wins,
        "Small Losses (0 to -3%)": small_losses,
        "Big Losses (<-3%)": big_losses,
    }

    print(f"\n  {'Feature':>20s}", end="")
    for cat in categories:
        print(f" | {cat:>22s}", end="")
    print()
    print(f"  {'─' * 120}")

    for feat in feature_cols_trade:
        print(f"  {feat:>20s}", end="")
        for cat, cat_df in categories.items():
            if len(cat_df) > 0 and feat in cat_df.columns:
                val = cat_df[feat].mean()
                print(f" | {val:>22.3f}", end="")
            else:
                print(f" | {'N/A':>22s}", end="")
        print()

    print(f"\n  {'Count':>20s}", end="")
    for cat, cat_df in categories.items():
        print(f" | {len(cat_df):>22d}", end="")
    print()

    # ══════════════════════════════════════════════════════════════
    # 5. EXIT REASON BREAKDOWN
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("🚪 5. EXIT REASON ANALYSIS")
    print("=" * 120)

    for reason in tdf["exit_reason"].unique():
        rt = tdf[tdf["exit_reason"] == reason]
        print(f"\n  {reason:>15s}: {len(rt):>4d} trades | "
              f"Avg PnL: {rt['pnl_pct'].mean():>+7.2f}% | "
              f"Avg MaxProf: {rt['max_profit_pct'].mean():>+7.2f}% | "
              f"Avg Eff: {rt['exit_efficiency'].mean():>5.1f}% | "
              f"WR: {(rt['pnl_pct']>0).mean()*100:>5.1f}%")

    # ══════════════════════════════════════════════════════════════
    # 6. LOSERS THAT HAD PROFIT — Missed opportunities
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("💔 6. LOSERS THAT HAD PROFIT — Giao dịch có lãi nhưng kết thúc lỗ")
    print("=" * 120)

    losers_had_profit = tdf[(tdf["pnl_pct"] <= 0) & (tdf["max_profit_pct"] > 3)]
    print(f"\n  Total: {len(losers_had_profit)} trades had >3% profit but ended in loss")
    if len(losers_had_profit) > 0:
        print(f"  Avg max profit they had: {losers_had_profit['max_profit_pct'].mean():>+7.2f}%")
        print(f"  Avg final PnL:           {losers_had_profit['pnl_pct'].mean():>+7.2f}%")
        print(f"  Avg hold days:           {losers_had_profit['holding_days'].mean():>5.1f}")
        print(f"\n  Top 15 worst 'could have won' trades:")
        worst_missed = losers_had_profit.nlargest(15, "max_profit_pct")
        for _, t in worst_missed.iterrows():
            print(f"    {t['model']:>15s} | {t['window']:>30s} | "
                  f"Had +{t['max_profit_pct']:.1f}% → ended {t['pnl_pct']:+.1f}% | "
                  f"Hold:{t['holding_days']:.0f}d | Exit:{t['exit_reason']} | "
                  f"WP:{t.get('entry_wp',0):.2f} RS:{t.get('entry_rs',0):.3f} BS:{t.get('entry_bs',0):.0f}")

    # ══════════════════════════════════════════════════════════════
    # 7. HOLDING DAYS vs OUTCOME
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("⏱️  7. HOLDING PERIOD ANALYSIS")
    print("=" * 120)

    for days_range, label in [((0, 2), "1-2 days"), ((2, 5), "3-5 days"),
                               ((5, 10), "6-10 days"), ((10, 20), "11-20 days"),
                               ((20, 999), "20+ days")]:
        ht = tdf[(tdf["holding_days"] >= days_range[0]) & (tdf["holding_days"] < days_range[1])]
        if len(ht) > 0:
            print(f"  {label:>12s}: {len(ht):>4d} trades | "
                  f"Avg PnL: {ht['pnl_pct'].mean():>+7.2f}% | "
                  f"WR: {(ht['pnl_pct']>0).mean()*100:>5.1f}% | "
                  f"Avg MaxProf: {ht['max_profit_pct'].mean():>+7.2f}%")

    # ══════════════════════════════════════════════════════════════
    # 8. RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("💡 8. ROOT CAUSE ANALYSIS & RECOMMENDATIONS")
    print("=" * 120)

    # Analyze what differentiates big losses
    if len(big_losses) > 0 and len(big_wins) > 0:
        print("\n  📌 KEY DIFFERENCES (Big Wins vs Big Losses):")
        for feat in feature_cols_trade:
            if feat in big_wins.columns and feat in big_losses.columns:
                w_mean = big_wins[feat].mean()
                l_mean = big_losses[feat].mean()
                diff = w_mean - l_mean
                if abs(diff) > 0.01:
                    direction = "↑ Winners higher" if diff > 0 else "↓ Winners lower"
                    print(f"    {feat:>20s}: Win={w_mean:>7.3f}  Loss={l_mean:>7.3f}  Δ={diff:>+7.3f} ({direction})")

    # Check if signal exits are premature
    signal_exits = tdf[tdf["exit_reason"] == "signal"]
    if len(signal_exits) > 0:
        premature = signal_exits[signal_exits["future_upside_pct"] > 5]
        print(f"\n  📌 PREMATURE SIGNAL EXITS: {len(premature)}/{len(signal_exits)} "
              f"({len(premature)/len(signal_exits)*100:.1f}%) had >5% upside after exit")

    # Save detailed trades
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tdf.to_csv(os.path.join(out_dir, f"trade_analysis_{ts}.csv"), index=False)
    print(f"\n  💾 Saved {len(tdf)} trades to results/trade_analysis_{ts}.csv")


if __name__ == "__main__":
    main()
