"""
Deep Analysis of V10 Trades — Identify patterns in losing/marginal trades
Goal: Find filters to cut bad trades while keeping winners
"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v10_compare import backtest_v10


def analyze_entry_context(df_sym, entry_idx, lookback=20):
    """Analyze price context around entry point."""
    close = df_sym["close"].values
    volume = df_sym["volume"].values if "volume" in df_sym.columns else np.ones(len(close))
    high = df_sym["high"].values if "high" in df_sym.columns else close
    low = df_sym["low"].values if "low" in df_sym.columns else close
    opn = df_sym["open"].values if "open" in df_sym.columns else close
    n = len(close)
    i = entry_idx

    if i < lookback or i >= n:
        return {}

    # Price context
    recent_close = close[max(0, i-lookback):i+1]
    recent_vol = volume[max(0, i-lookback):i+1]

    # 1. Consolidation detection: narrow range in last N bars
    range_5d = (max(high[i-5:i+1]) - min(low[i-5:i+1])) / close[i] * 100 if i >= 5 else 99
    range_10d = (max(high[i-10:i+1]) - min(low[i-10:i+1])) / close[i] * 100 if i >= 10 else 99
    range_20d = (max(high[i-20:i+1]) - min(low[i-20:i+1])) / close[i] * 100 if i >= 20 else 99

    # 2. Volume analysis
    avg_vol_20 = np.mean(volume[max(0,i-20):i]) if i >= 1 else volume[i]
    vol_ratio_today = volume[i] / avg_vol_20 if avg_vol_20 > 0 else 1
    avg_vol_5 = np.mean(volume[max(0,i-5):i]) if i >= 1 else volume[i]
    vol_ratio_5d = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1

    # 3. Distance from recent low/high
    low_20d = min(low[max(0,i-20):i+1])
    high_20d = max(high[max(0,i-20):i+1])
    dist_from_low = (close[i] - low_20d) / low_20d * 100 if low_20d > 0 else 0
    dist_from_high = (high_20d - close[i]) / close[i] * 100 if close[i] > 0 else 0

    # 4. Recent trend (5-day return)
    ret_5d = (close[i] / close[i-5] - 1) * 100 if i >= 5 else 0
    ret_10d = (close[i] / close[i-10] - 1) * 100 if i >= 10 else 0
    ret_3d = (close[i] / close[i-3] - 1) * 100 if i >= 3 else 0

    # 5. Candle pattern at entry
    body_pct = (close[i] - opn[i]) / opn[i] * 100 if opn[i] > 0 else 0
    upper_shadow = (high[i] - max(close[i], opn[i])) / close[i] * 100 if close[i] > 0 else 0
    lower_shadow = (min(close[i], opn[i]) - low[i]) / close[i] * 100 if close[i] > 0 else 0

    # 6. SMA context
    sma20 = np.mean(close[max(0,i-19):i+1])
    sma50 = np.mean(close[max(0,i-49):i+1]) if i >= 49 else np.mean(close[:i+1])
    dist_sma20 = (close[i] / sma20 - 1) * 100 if sma20 > 0 else 0
    dist_sma50 = (close[i] / sma50 - 1) * 100 if sma50 > 0 else 0
    sma20_above_sma50 = 1 if sma20 > sma50 else 0

    # 7. Consolidation breakout detection
    # Check if price was in tight range (< 5%) for at least 5 bars then broke out
    is_breakout_from_consolidation = 0
    if i >= 10:
        pre_range = (max(high[i-10:i-1]) - min(low[i-10:i-1])) / close[i-5] * 100 if close[i-5] > 0 else 99
        if pre_range < 8 and close[i] > max(high[i-10:i-1]) and vol_ratio_today > 1.2:
            is_breakout_from_consolidation = 1

    # 8. After big drop detection
    max_drop_20d = 0
    if i >= 20:
        for j in range(i-20, i):
            drop = (close[j] / max(close[max(0,j-10):j+1]) - 1) * 100 if j >= 1 else 0
            max_drop_20d = min(max_drop_20d, drop)

    # 9. Consecutive up/down days before entry
    consec_up = 0
    for j in range(i, max(0, i-10), -1):
        if close[j] > close[j-1] if j > 0 else False:
            consec_up += 1
        else:
            break

    consec_down = 0
    for j in range(i-1, max(0, i-10), -1):
        if close[j] < close[j-1] if j > 0 else False:
            consec_down += 1
        else:
            break

    return {
        "range_5d": round(range_5d, 2),
        "range_10d": round(range_10d, 2),
        "range_20d_pct": round(range_20d, 2),
        "vol_ratio_today": round(vol_ratio_today, 2),
        "vol_ratio_5d": round(vol_ratio_5d, 2),
        "dist_from_low_20d": round(dist_from_low, 2),
        "dist_from_high_20d": round(dist_from_high, 2),
        "ret_3d": round(ret_3d, 2),
        "ret_5d": round(ret_5d, 2),
        "ret_10d": round(ret_10d, 2),
        "body_pct": round(body_pct, 2),
        "upper_shadow": round(upper_shadow, 2),
        "lower_shadow": round(lower_shadow, 2),
        "dist_sma20": round(dist_sma20, 2),
        "dist_sma50": round(dist_sma50, 2),
        "sma20_above_sma50": sma20_above_sma50,
        "is_consolidation_breakout": is_breakout_from_consolidation,
        "max_drop_20d": round(max_drop_20d, 2),
        "consec_up_at_entry": consec_up,
        "consec_down_before": consec_down,
    }


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")

    config = {
        "data": {"data_dir": data_dir},
        "split": {"method": "walk_forward", "train_years": 4, "test_years": 1,
                  "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025},
        "target": {"type": "trend_regime", "trend_method": "dual_ma",
                   "short_window": 5, "long_window": 20, "classes": 3},
    }

    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    test_symbols = ["ACB", "FPT", "HPG", "SSI", "VND", "MBB", "TCB", "VNM", "DGC", "AAS", "AAV", "REE", "BID", "VIC"]
    symbols = [s for s in test_symbols if s in loader.symbols]

    raw_df = loader.load_all(symbols=symbols)
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
            if sym not in symbols:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values if "return_1d" in sym_test.columns else np.zeros(len(sym_test))

            r10 = backtest_v10(y_pred, rets, sym_test, feature_cols)

            for t in r10["trades"]:
                t["symbol"] = sym
                t["window"] = window.label
                # Add entry context
                entry_day = t.get("entry_day", 0)
                ctx = analyze_entry_context(sym_test, entry_day)
                t.update(ctx)

            all_trades.extend(r10["trades"])

    tdf = pd.DataFrame(all_trades)
    if tdf.empty:
        print("No trades found!")
        return

    # Categorize trades
    tdf["category"] = "winner"
    tdf.loc[tdf["pnl_pct"] <= -5, "category"] = "big_loss"
    tdf.loc[(tdf["pnl_pct"] > -5) & (tdf["pnl_pct"] <= 0), "category"] = "small_loss"
    tdf.loc[(tdf["pnl_pct"] > 0) & (tdf["pnl_pct"] <= 2), "category"] = "marginal"
    tdf.loc[(tdf["pnl_pct"] > 2) & (tdf["pnl_pct"] <= 10), "category"] = "small_win"
    tdf.loc[tdf["pnl_pct"] > 10, "category"] = "big_win"

    print("=" * 140)
    print("🔬 V10 TRADE DEEP ANALYSIS — Entry Pattern Profiling")
    print("=" * 140)

    # ═══════════════════════════════════════════════════════
    # 1. OVERVIEW BY CATEGORY
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 100}")
    print("📊 1. TRADE DISTRIBUTION BY CATEGORY")
    print(f"{'═' * 100}")
    for cat in ["big_loss", "small_loss", "marginal", "small_win", "big_win"]:
        ct = tdf[tdf["category"] == cat]
        n = len(ct)
        pct = n / len(tdf) * 100
        avg = ct["pnl_pct"].mean() if n > 0 else 0
        total = ct["pnl_pct"].sum() if n > 0 else 0
        print(f"  {cat:<12}: {n:>4} trades ({pct:>5.1f}%)  Avg: {avg:>+6.2f}%  Total: {total:>+8.1f}%")

    # ═══════════════════════════════════════════════════════
    # 2. LOSING TRADE ANALYSIS
    # ═══════════════════════════════════════════════════════
    losers = tdf[tdf["pnl_pct"] <= 0].copy()
    winners = tdf[tdf["pnl_pct"] > 2].copy()

    context_features = ["range_5d", "range_10d", "vol_ratio_today", "vol_ratio_5d",
                        "dist_from_low_20d", "dist_from_high_20d", "ret_3d", "ret_5d", "ret_10d",
                        "body_pct", "dist_sma20", "dist_sma50", "sma20_above_sma50",
                        "is_consolidation_breakout", "max_drop_20d", "consec_up_at_entry"]

    print(f"\n{'═' * 100}")
    print("📉 2. LOSING vs WINNING ENTRY CONTEXT COMPARISON")
    print(f"{'═' * 100}")
    print(f"{'Feature':<28} │ {'Losers(≤0%)':>12} │ {'Winners(>2%)':>12} │ {'Δ':>8} │ Insight")
    print(f"{'─' * 100}")

    insights = []
    for feat in context_features:
        if feat not in tdf.columns:
            continue
        lv = losers[feat].mean() if len(losers) > 0 else 0
        wv = winners[feat].mean() if len(winners) > 0 else 0
        delta = wv - lv

        # Generate insight
        insight = ""
        if feat == "vol_ratio_today" and lv < wv:
            insight = "Winners have higher volume at entry"
        elif feat == "dist_sma20" and abs(delta) > 1:
            insight = f"Winners {'closer to' if abs(wv) < abs(lv) else 'farther from'} SMA20"
        elif feat == "sma20_above_sma50" and delta > 0.1:
            insight = "Winners more often in uptrend"
        elif feat == "ret_5d" and delta > 1:
            insight = "Winners enter after stronger short-term momentum"
        elif feat == "range_5d" and abs(delta) > 1:
            insight = "Different volatility context"
        elif feat == "dist_from_low_20d" and abs(delta) > 1:
            insight = f"Winners enter {'higher' if wv > lv else 'lower'} from 20d low"
        elif feat == "is_consolidation_breakout" and wv > lv:
            insight = "⭐ Consolidation breakouts have better outcomes"
        elif feat == "consec_up_at_entry" and abs(delta) > 0.3:
            insight = "Different momentum pattern"

        print(f"  {feat:<26} │ {lv:>11.2f} │ {wv:>11.2f} │ {delta:>+7.2f} │ {insight}")

        if abs(delta) > 0.5:
            insights.append((feat, lv, wv, delta, insight))

    # ═══════════════════════════════════════════════════════
    # 3. EXIT REASON ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 100}")
    print("🚪 3. EXIT REASON vs PnL ANALYSIS")
    print(f"{'═' * 100}")
    print(f"{'Exit Reason':<18} │ {'Count':>5} │ {'Avg PnL':>8} │ {'WR':>6} │ {'Total':>9} │ {'Avg Hold':>8}")
    print(f"{'─' * 70}")
    for reason in tdf["exit_reason"].unique():
        rt = tdf[tdf["exit_reason"] == reason]
        n = len(rt)
        avg = rt["pnl_pct"].mean()
        wr = (rt["pnl_pct"] > 0).sum() / n * 100
        total = rt["pnl_pct"].sum()
        ah = rt["holding_days"].mean()
        flag = " ⚠️" if avg < 0 else ""
        print(f"  {reason:<16} │ {n:>5} │ {avg:>+7.2f}% │ {wr:>5.1f}% │ {total:>+8.1f}% │ {ah:>7.1f}d{flag}")

    # ═══════════════════════════════════════════════════════
    # 4. ENTRY TREND vs PnL
    # ═══════════════════════════════════════════════════════
    if "entry_trend" in tdf.columns:
        print(f"\n{'═' * 100}")
        print("📈 4. ENTRY TREND REGIME vs PnL")
        print(f"{'═' * 100}")
        for trend in ["strong", "moderate", "weak"]:
            tt = tdf[tdf["entry_trend"] == trend]
            if len(tt) == 0:
                continue
            n = len(tt)
            avg = tt["pnl_pct"].mean()
            wr = (tt["pnl_pct"] > 0).sum() / n * 100
            total = tt["pnl_pct"].sum()
            print(f"  {trend:<10}: {n:>4} trades  WR={wr:>5.1f}%  Avg={avg:>+6.2f}%  Total={total:>+8.1f}%")

    # ═══════════════════════════════════════════════════════
    # 5. ENTRY SCORE vs PnL
    # ═══════════════════════════════════════════════════════
    if "entry_score" in tdf.columns:
        print(f"\n{'═' * 100}")
        print("🎯 5. ENTRY SCORE vs PnL")
        print(f"{'═' * 100}")
        for score in sorted(tdf["entry_score"].unique()):
            st = tdf[tdf["entry_score"] == score]
            n = len(st)
            avg = st["pnl_pct"].mean()
            wr = (st["pnl_pct"] > 0).sum() / n * 100
            total = st["pnl_pct"].sum()
            flag = " ⚠️ FILTER CANDIDATE" if avg < 1 and n > 5 else ""
            print(f"  Score={int(score)}: {n:>4} trades  WR={wr:>5.1f}%  Avg={avg:>+6.2f}%  Total={total:>+8.1f}%{flag}")

    # ═══════════════════════════════════════════════════════
    # 6. SPECIFIC LOSING PATTERNS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 100}")
    print("🔍 6. SPECIFIC LOSING PATTERNS (FILTER CANDIDATES)")
    print(f"{'═' * 100}")

    filters = []

    # Pattern A: Entry after consecutive up days (chasing)
    if "consec_up_at_entry" in tdf.columns:
        chasing = tdf[tdf["consec_up_at_entry"] >= 3]
        not_chasing = tdf[tdf["consec_up_at_entry"] < 3]
        if len(chasing) > 5:
            avg_c = chasing["pnl_pct"].mean()
            avg_nc = not_chasing["pnl_pct"].mean()
            print(f"\n  A. CHASING (≥3 consec up days at entry):")
            print(f"     Chasing:     {len(chasing):>4} trades, Avg={avg_c:>+6.2f}%, WR={(chasing['pnl_pct']>0).sum()/len(chasing)*100:.1f}%")
            print(f"     Not chasing: {len(not_chasing):>4} trades, Avg={avg_nc:>+6.2f}%, WR={(not_chasing['pnl_pct']>0).sum()/len(not_chasing)*100:.1f}%")
            if avg_c < avg_nc:
                filters.append(("Avoid entry after ≥3 consecutive up days", avg_nc - avg_c, len(chasing)))
                print(f"     💡 RECOMMENDATION: Filter out chasing entries → saves {len(chasing)} bad trades, improves avg by {avg_nc-avg_c:+.2f}%")

    # Pattern B: Entry when far above SMA20 (overextended)
    if "dist_sma20" in tdf.columns:
        overext = tdf[tdf["dist_sma20"] > 5]
        normal = tdf[tdf["dist_sma20"] <= 5]
        if len(overext) > 5:
            avg_o = overext["pnl_pct"].mean()
            avg_n = normal["pnl_pct"].mean()
            print(f"\n  B. OVEREXTENDED (>5% above SMA20 at entry):")
            print(f"     Overextended: {len(overext):>4} trades, Avg={avg_o:>+6.2f}%")
            print(f"     Normal:       {len(normal):>4} trades, Avg={avg_n:>+6.2f}%")
            if avg_o < avg_n:
                filters.append(("Avoid entry >5% above SMA20", avg_n - avg_o, len(overext)))

    # Pattern C: Entry on declining volume
    if "vol_ratio_today" in tdf.columns:
        low_vol = tdf[tdf["vol_ratio_today"] < 0.7]
        high_vol = tdf[tdf["vol_ratio_today"] >= 0.7]
        if len(low_vol) > 5:
            avg_lv = low_vol["pnl_pct"].mean()
            avg_hv = high_vol["pnl_pct"].mean()
            print(f"\n  C. LOW VOLUME ENTRY (vol < 70% of 20d avg):")
            print(f"     Low volume:  {len(low_vol):>4} trades, Avg={avg_lv:>+6.2f}%")
            print(f"     Normal vol:  {len(high_vol):>4} trades, Avg={avg_hv:>+6.2f}%")
            if avg_lv < avg_hv:
                filters.append(("Require volume > 70% of 20d avg", avg_hv - avg_lv, len(low_vol)))

    # Pattern D: Entry below both SMA20 and SMA50 (downtrend)
    if "sma20_above_sma50" in tdf.columns:
        downtrend = tdf[(tdf["sma20_above_sma50"] == 0) & (tdf["dist_sma20"] < 0)]
        uptrend = tdf[tdf["sma20_above_sma50"] == 1]
        if len(downtrend) > 5:
            avg_d = downtrend["pnl_pct"].mean()
            avg_u = uptrend["pnl_pct"].mean()
            print(f"\n  D. DOWNTREND ENTRY (SMA20 < SMA50 AND price < SMA20):")
            print(f"     Downtrend: {len(downtrend):>4} trades, Avg={avg_d:>+6.2f}%, WR={(downtrend['pnl_pct']>0).sum()/len(downtrend)*100:.1f}%")
            print(f"     Uptrend:   {len(uptrend):>4} trades, Avg={avg_u:>+6.2f}%, WR={(uptrend['pnl_pct']>0).sum()/len(uptrend)*100:.1f}%")
            if avg_d < avg_u:
                filters.append(("Avoid downtrend entries (SMA20<SMA50 + price<SMA20)", avg_u - avg_d, len(downtrend)))

    # Pattern E: Entry after big recent drop (catching falling knife)
    if "max_drop_20d" in tdf.columns:
        falling = tdf[tdf["max_drop_20d"] < -15]
        stable = tdf[tdf["max_drop_20d"] >= -15]
        if len(falling) > 5:
            avg_f = falling["pnl_pct"].mean()
            avg_s = stable["pnl_pct"].mean()
            print(f"\n  E. FALLING KNIFE (>15% drop in last 20d):")
            print(f"     After big drop: {len(falling):>4} trades, Avg={avg_f:>+6.2f}%")
            print(f"     Stable:         {len(stable):>4} trades, Avg={avg_s:>+6.2f}%")
            if avg_f < avg_s:
                filters.append(("Avoid entry after >15% drop in 20d", avg_s - avg_f, len(falling)))

    # Pattern F: Short hold losing trades (quick stop-outs)
    short_losers = tdf[(tdf["holding_days"] <= 3) & (tdf["pnl_pct"] < 0)]
    if len(short_losers) > 5:
        avg_sl = short_losers["pnl_pct"].mean()
        print(f"\n  F. QUICK STOP-OUTS (hold ≤ 3 days, loss):")
        print(f"     Count: {len(short_losers)}, Avg loss: {avg_sl:+.2f}%, Total: {short_losers['pnl_pct'].sum():+.1f}%")
        # Analyze what these have in common
        if "entry_trend" in short_losers.columns:
            for trend in ["strong", "moderate", "weak"]:
                tt = short_losers[short_losers["entry_trend"] == trend]
                if len(tt) > 0:
                    print(f"       - {trend}: {len(tt)} trades ({len(tt)/len(short_losers)*100:.0f}%)")

    # Pattern G: Missed consolidation breakout opportunities
    if "is_consolidation_breakout" in tdf.columns:
        cb = tdf[tdf["is_consolidation_breakout"] == 1]
        ncb = tdf[tdf["is_consolidation_breakout"] == 0]
        print(f"\n  G. CONSOLIDATION BREAKOUT ENTRIES:")
        print(f"     Breakout entries: {len(cb):>4} trades, Avg={cb['pnl_pct'].mean():>+6.2f}%" if len(cb) > 0 else "     No breakout entries detected")
        print(f"     Non-breakout:     {len(ncb):>4} trades, Avg={ncb['pnl_pct'].mean():>+6.2f}%" if len(ncb) > 0 else "")

    # Pattern H: Bearish candle at entry
    if "body_pct" in tdf.columns:
        bearish_entry = tdf[tdf["body_pct"] < -0.5]
        bullish_entry = tdf[tdf["body_pct"] > 0.5]
        if len(bearish_entry) > 5:
            print(f"\n  H. CANDLE PATTERN AT ENTRY:")
            print(f"     Bearish candle: {len(bearish_entry):>4} trades, Avg={bearish_entry['pnl_pct'].mean():>+6.2f}%")
            print(f"     Bullish candle: {len(bullish_entry):>4} trades, Avg={bullish_entry['pnl_pct'].mean():>+6.2f}%")
            if bearish_entry["pnl_pct"].mean() < bullish_entry["pnl_pct"].mean():
                filters.append(("Prefer bullish candle at entry", bullish_entry['pnl_pct'].mean() - bearish_entry['pnl_pct'].mean(), len(bearish_entry)))

    # Pattern I: Range position + recent trend combo
    if "dist_from_low_20d" in tdf.columns and "ret_5d" in tdf.columns:
        # Near bottom with negative momentum = potential bounce but risky
        bottom_neg = tdf[(tdf["dist_from_low_20d"] < 3) & (tdf["ret_5d"] < -3)]
        if len(bottom_neg) > 3:
            print(f"\n  I. BOTTOM FISHING (near 20d low + negative 5d momentum):")
            print(f"     Count: {len(bottom_neg)}, Avg={bottom_neg['pnl_pct'].mean():>+6.2f}%, WR={(bottom_neg['pnl_pct']>0).sum()/len(bottom_neg)*100:.1f}%")

    # ═══════════════════════════════════════════════════════
    # 7. V10 TIMING ANALYSIS (buy/sell timing vs Rule)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 100}")
    print("⏰ 7. TIMING ANALYSIS — LATE ENTRY/EXIT PATTERNS")
    print(f"{'═' * 100}")

    # Analyze trades that entered late (after price already moved up)
    late_entries = tdf[tdf["ret_5d"] > 5]  # price already up 5% in 5 days
    early_entries = tdf[(tdf["ret_5d"] >= -2) & (tdf["ret_5d"] <= 2)]  # flat entry
    if len(late_entries) > 3 and len(early_entries) > 3:
        print(f"  Late entry (5d return > +5%):  {len(late_entries):>4} trades, Avg={late_entries['pnl_pct'].mean():>+6.2f}%")
        print(f"  Flat entry (5d return ~0%):    {len(early_entries):>4} trades, Avg={early_entries['pnl_pct'].mean():>+6.2f}%")
        if late_entries["pnl_pct"].mean() < early_entries["pnl_pct"].mean():
            filters.append(("Avoid entry when 5d return > +5% (late entry)", early_entries['pnl_pct'].mean() - late_entries['pnl_pct'].mean(), len(late_entries)))

    # ═══════════════════════════════════════════════════════
    # 8. RECOMMENDATIONS SUMMARY
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═' * 100}")
    print("💡 8. V11 IMPROVEMENT RECOMMENDATIONS (ranked by impact)")
    print(f"{'═' * 100}")

    filters.sort(key=lambda x: x[1] * x[2], reverse=True)  # sort by total impact
    for i, (desc, avg_improvement, trades_affected) in enumerate(filters):
        impact = avg_improvement * trades_affected / len(tdf)
        print(f"\n  {i+1}. {desc}")
        print(f"     Avg improvement: {avg_improvement:+.2f}%/trade  |  Trades affected: {trades_affected}  |  Portfolio impact: {impact:+.2f}%")

    # Additional: consolidation breakout opportunity
    print(f"\n  NEW ENTRY SIGNAL PROPOSALS:")
    print(f"  ─────────────────────────────")
    print(f"  A. CONSOLIDATION BREAKOUT SCANNER:")
    print(f"     - Detect 5-10 bar tight range (< 5-8% width)")
    print(f"     - Entry when price breaks above range high with volume > 1.2x avg")
    print(f"     - This pattern was missed by all 3 strategies")
    print(f"  B. POST-DUMP ACCUMULATION:")
    print(f"     - After >15% drop, wait for 5-10 bars of sideways (range < 5%)")
    print(f"     - Entry on first bullish candle with rising volume")
    print(f"     - V10 currently enters too early (during the drop) or too late")
    print(f"  C. VOLUME-CONFIRMED REVERSAL:")
    print(f"     - At 20d low + volume spike (>1.5x avg) + bullish candle")
    print(f"     - Currently all 3 models miss this because ML model hasn't learned this specific pattern")

    # Top 10 worst trades detail
    print(f"\n{'═' * 100}")
    print("📋 TOP 15 WORST V10 TRADES (for manual inspection)")
    print(f"{'═' * 100}")
    worst = tdf.nsmallest(15, "pnl_pct")
    for _, t in worst.iterrows():
        print(f"  {t.get('symbol','?'):<6} {t.get('entry_date','')[:10]} → {t.get('exit_date','')[:10]}  "
              f"PnL={t['pnl_pct']:>+6.1f}%  Hold={t['holding_days']:>2d}d  "
              f"Exit={t.get('exit_reason',''):<14} Trend={t.get('entry_trend','?'):<8} "
              f"Score={t.get('entry_score',0):.0f}  "
              f"Ret5d={t.get('ret_5d',0):>+5.1f}%  VolR={t.get('vol_ratio_today',0):.1f}  "
              f"dSMA20={t.get('dist_sma20',0):>+5.1f}%")


if __name__ == "__main__":
    main()
