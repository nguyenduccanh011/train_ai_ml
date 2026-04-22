"""
Deep analysis of V6 trade signals:
1. Entry timing vs wave bottom
2. Post-exit price movement (sold too early?)
3. Consecutive buy-sell-buy at similar prices (wasted fees)
4. Export s=1, s=2 filtered signals to chart
"""
import sys, os, json, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v6_backtest import backtest_v6


def analyze_trades(symbols_pick, model_name="lightgbm"):
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

    symbols = [s for s in symbols_pick if s in loader.symbols]
    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    date_col = "date" if "date" in raw_df.columns else "timestamp"

    # Collect ALL signals (including filtered ones) per symbol
    all_trades_by_sym = defaultdict(list)
    all_filtered_signals = defaultdict(list)  # s=1,2 signals that were rejected

    for window, train_df, test_df in splitter.split(df):
        model = build_model(model_name)
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        for sym in test_df["symbol"].unique():
            if sym not in symbols:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue

            X_test_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred_sym = model.predict(X_test_sym)

            # Extract features for filtered signal analysis
            feat_names = ["rsi_slope_5d", "vol_surge_ratio", "range_position_20d",
                          "dist_to_resistance", "breakout_setup_score", "bb_width_percentile",
                          "higher_lows_count", "obv_price_divergence"]
            defaults = {"rsi_slope_5d": 0, "vol_surge_ratio": 1.0, "range_position_20d": 0.5,
                        "dist_to_resistance": 0.05, "breakout_setup_score": 0, "bb_width_percentile": 0.5,
                        "higher_lows_count": 0, "obv_price_divergence": 0}

            close = sym_test["close"].values
            dates = sym_test[date_col].values if date_col in sym_test.columns else np.arange(len(sym_test))
            sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
            sma50 = pd.Series(close).rolling(50, min_periods=10).mean().values

            # Find ALL predict=1 signals and compute their entry_score
            for i in range(1, len(y_pred_sym)):
                pred = int(y_pred_sym[i])
                if pred != 1:
                    continue

                # Compute entry score
                feat_vals = {}
                for fn in feat_names:
                    if fn in sym_test.columns:
                        v = sym_test[fn].iloc[i]
                        feat_vals[fn] = v if not np.isnan(v) else defaults[fn]
                    else:
                        feat_vals[fn] = defaults[fn]

                wp = feat_vals["range_position_20d"]
                dp = feat_vals["dist_to_resistance"]
                rs = feat_vals["rsi_slope_5d"]
                vs = feat_vals["vol_surge_ratio"]
                hl = feat_vals["higher_lows_count"]
                bs = feat_vals["breakout_setup_score"]

                entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])

                # Check regime filter
                regime_reject = False
                if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                    if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                        if bs < 3:
                            regime_reject = True

                # Check entry confirmation (V6-A)
                prev_pred = int(y_pred_sym[i - 1]) if i >= 1 else 0
                confirm_reject = (prev_pred != 1)

                d = str(dates[i])[:10]
                # Future 5, 10, 20 day returns
                future_5 = (close[min(i+5, len(close)-1)] / close[i] - 1) * 100
                future_10 = (close[min(i+10, len(close)-1)] / close[i] - 1) * 100
                future_20 = (close[min(i+20, len(close)-1)] / close[i] - 1) * 100

                if entry_score < 3 or regime_reject or confirm_reject:
                    all_filtered_signals[sym].append({
                        "date": d, "close": round(float(close[i]), 2),
                        "entry_score": entry_score,
                        "regime_reject": regime_reject,
                        "confirm_reject": confirm_reject,
                        "future_5d": round(future_5, 1),
                        "future_10d": round(future_10, 1),
                        "future_20d": round(future_20, 1),
                    })

            # Run actual backtest for trades
            rets = sym_test["return_1d"].values if "return_1d" in sym_test.columns else np.zeros(len(sym_test))
            r = backtest_v6(y_pred_sym, rets, sym_test, feature_cols)
            for t in r["trades"]:
                t["window"] = window.label
                t["entry_symbol"] = sym
            all_trades_by_sym[sym].extend(r["trades"])

    # ═══ ANALYSIS ═══
    print("=" * 120)
    print("🔍 DEEP TRADE ANALYSIS")
    print("=" * 120)

    # Build date->close lookup per symbol
    date_close = {}
    for sym in symbols:
        sym_raw = raw_df[raw_df["symbol"] == sym].sort_values(date_col)
        date_close[sym] = {str(r[date_col])[:10]: float(r["close"]) for _, r in sym_raw.iterrows()}
        # Also store sorted dates for wave analysis
        date_close[sym + "_dates"] = sorted(date_close[sym].keys())
        date_close[sym + "_prices"] = [date_close[sym][d] for d in date_close[sym + "_dates"]]

    total_wasted = 0
    total_early_exit = 0
    total_late_entry = 0
    total_trades = 0

    for sym in sorted(all_trades_by_sym.keys()):
        trades = all_trades_by_sym[sym]
        if not trades:
            continue

        print(f"\n{'═' * 100}")
        print(f"📊 {sym} — {len(trades)} trades")
        print(f"{'═' * 100}")

        dates_list = date_close.get(sym + "_dates", [])
        prices_list = date_close.get(sym + "_prices", [])
        dc = date_close.get(sym, {})

        # ── 1. Entry timing analysis ──
        print(f"\n  📍 ENTRY TIMING (vs local wave bottom):")
        late_entries = 0
        for t in trades:
            ed = t.get("entry_date", "")
            entry_price = dc.get(ed, 0)
            if entry_price == 0 or ed not in dates_list:
                continue

            # Find local bottom in 20 bars before entry
            idx = dates_list.index(ed)
            lookback = max(0, idx - 20)
            local_prices = prices_list[lookback:idx + 1]
            if not local_prices:
                continue
            local_min = min(local_prices)
            pct_from_bottom = (entry_price / local_min - 1) * 100

            if pct_from_bottom > 10:
                late_entries += 1
                total_late_entry += 1
                print(f"    ⚠️  {ed} Buy @{entry_price} | Bottom={local_min:.1f} ({pct_from_bottom:+.1f}% from bottom) — LATE ENTRY")

        # ── 2. Post-exit analysis ──
        print(f"\n  📍 POST-EXIT ANALYSIS (did price keep rising?):")
        early_exits = 0
        for t in trades:
            xd = t.get("exit_date", "")
            exit_price = dc.get(xd, 0)
            if exit_price == 0 or xd not in dates_list:
                continue

            idx = dates_list.index(xd)
            # Future 5, 10 bars
            f5 = prices_list[min(idx + 5, len(prices_list) - 1)]
            f10 = prices_list[min(idx + 10, len(prices_list) - 1)]
            f5_pct = (f5 / exit_price - 1) * 100
            f10_pct = (f10 / exit_price - 1) * 100

            reason = t.get("exit_reason", "")
            pnl = t.get("chart_pnl_pct", t.get("pnl_pct", 0))

            if f10_pct > 5:
                early_exits += 1
                total_early_exit += 1
                print(f"    ⚠️  {xd} Sell @{exit_price} ({reason}, PnL={pnl:+.1f}%) | +5d: {f5_pct:+.1f}%, +10d: {f10_pct:+.1f}% — SOLD TOO EARLY")

        # ── 3. Consecutive buy-sell-buy at similar prices ──
        print(f"\n  📍 CONSECUTIVE TRADES (sell then re-buy at similar price):")
        wasted_trades = 0
        for j in range(len(trades) - 1):
            t1 = trades[j]
            t2 = trades[j + 1]
            sell_price = dc.get(t1.get("exit_date", ""), 0)
            rebuy_price = dc.get(t2.get("entry_date", ""), 0)
            if sell_price == 0 or rebuy_price == 0:
                continue

            price_diff_pct = abs(rebuy_price / sell_price - 1) * 100
            if price_diff_pct < 2:  # re-buy within 2% of sell price
                wasted_trades += 1
                total_wasted += 1
                gap_days = 0
                if t1.get("exit_date") in dates_list and t2.get("entry_date") in dates_list:
                    gap_days = dates_list.index(t2["entry_date"]) - dates_list.index(t1["exit_date"])
                print(f"    ⚠️  Sell {t1.get('exit_date')} @{sell_price} → Buy {t2.get('entry_date')} @{rebuy_price} "
                      f"(diff={price_diff_pct:.1f}%, gap={gap_days}d) — WASTED FEES")

        total_trades += len(trades)

        # ── 4. Filtered signals analysis (s=1, s=2) ──
        filtered = all_filtered_signals.get(sym, [])
        if filtered:
            # Only show s=1, s=2 that would have been profitable
            good_filtered = [f for f in filtered if f["future_10d"] > 5 and f["entry_score"] <= 2]
            if good_filtered:
                print(f"\n  📍 FILTERED SIGNALS (s≤2) that were PROFITABLE:")
                for f in good_filtered[:10]:
                    print(f"    💡 {f['date']} @{f['close']} s={f['entry_score']} "
                          f"| +5d:{f['future_5d']:+.1f}% +10d:{f['future_10d']:+.1f}% +20d:{f['future_20d']:+.1f}% "
                          f"{'[regime]' if f['regime_reject'] else ''}"
                          f"{'[no-confirm]' if f['confirm_reject'] else ''}")

    # ═══ SUMMARY ═══
    print(f"\n{'═' * 120}")
    print(f"📊 SUMMARY ACROSS ALL SYMBOLS")
    print(f"{'═' * 120}")
    print(f"  Total trades:        {total_trades}")
    print(f"  Late entries (>10%): {total_late_entry} ({total_late_entry/max(total_trades,1)*100:.1f}%)")
    print(f"  Sold too early:      {total_early_exit} ({total_early_exit/max(total_trades,1)*100:.1f}%)")
    print(f"  Wasted fee cycles:   {total_wasted} ({total_wasted/max(total_trades,1)*100:.1f}%)")

    # ═══ Export filtered signals to chart ═══
    print(f"\n📊 Updating chart data with s=1,2 markers...")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization", "data")
    for sym in symbols:
        json_path = os.path.join(out_dir, f"{sym}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)

        # Add filtered signal markers (small gray triangles)
        filtered = all_filtered_signals.get(sym, [])
        for sig in filtered:
            if sig["entry_score"] <= 2:
                color = "#888888" if sig["future_10d"] <= 0 else "#00bcd4"
                data["markers"].append({
                    "time": sig["date"],
                    "position": "belowBar",
                    "color": color,
                    "shape": "circle",
                    "text": f"s={sig['entry_score']}",
                    "size": 1,
                })

        with open(json_path, "w") as f:
            json.dump(data, f)
        print(f"  ✅ {sym}: added {len([s for s in filtered if s['entry_score'] <= 2])} filtered markers")

    # ═══ IMPROVEMENT PROPOSALS ═══
    print(f"\n{'═' * 120}")
    print(f"💡 IMPROVEMENT PROPOSALS")
    print(f"{'═' * 120}")
    print("""
  1. WASTED FEE CYCLES (sell → re-buy at same price):
     Root cause: Model flips signal briefly then re-enters.
     Fix options:
       a) "Cooldown period" — after exit, wait N bars before re-entry
       b) "Re-entry price filter" — only re-buy if price > exit_price * 1.02 OR price < exit_price * 0.95
       c) "Hold extension" — if exit signal is weak and no stop triggered, extend hold

  2. LATE ENTRIES (buying >10% from wave bottom):
     Root cause: Entry confirmation (need 2 consecutive predict=1) + high entry_score filter delays entry.
     Fix options:
       a) Relax s≥3 to s≥2 for entries near SMA support or after strong pullback
       b) "First signal" mode — if rsi_slope just turned positive, allow s=2 entry
       c) Add a "distance from local low" feature — prefer entries closer to bottom

  3. SELLING TOO EARLY (price rises >5% after exit):
     Root cause: Trailing stop too tight, zombie exit too aggressive, or exit confirmation too fast.
     Fix options:
       a) Widen trailing in strong uptrend (increase trail_pct thresholds)
       b) Extend zombie threshold from 5 bars to 7-8 bars
       c) "Trend strength" override — if sma20 > sma50 and rising, hold longer
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", type=str, default="ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM")
    args = parser.parse_args()
    symbols = [s.strip() for s in args.pick.split(",")]
    analyze_trades(symbols)
