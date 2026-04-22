"""
Deep analysis of V11 trades to find improvement opportunities for V13.
Analyzes: early entry potential, rebuy-at-higher-price, top-buying, bottom-selling.
"""
import sys, os, json, glob, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v11_compare import backtest_v11

def run_deep_analysis():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    config = {
        "data": {"data_dir": data_dir},
        "split": {"method": "walk_forward", "train_years": 4, "test_years": 1,
                  "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025},
        "target": {"type": "trend_regime", "trend_method": "dual_ma",
                   "short_window": 5, "long_window": 20, "classes": 3},
    }

    symbols_str = "ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC"
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

    # Collect all V11 trades with price data context
    all_analyses = []

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
            
            close = sym_test["close"].values
            high = sym_test["high"].values
            low = sym_test["low"].values
            dates = sym_test["date"].values if "date" in sym_test.columns else np.arange(len(sym_test))
            
            sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
            sma50 = pd.Series(close).rolling(50, min_periods=10).mean().values

            trades = r11["trades"]
            for t in trades:
                t["symbol"] = sym
            
            for idx, t in enumerate(trades):
                entry_day = t.get("entry_day", 0)
                exit_day = t.get("exit_day", 0)
                entry_price = close[entry_day] if entry_day < len(close) else 0
                exit_price = close[exit_day] if exit_day < len(close) else 0
                
                # Look back 5 bars before entry - could we have entered earlier/cheaper?
                lookback = 5
                pre_entry_start = max(0, entry_day - lookback)
                pre_entry_lows = low[pre_entry_start:entry_day+1]
                pre_entry_closes = close[pre_entry_start:entry_day+1]
                min_low_before = np.min(pre_entry_lows) if len(pre_entry_lows) > 0 else entry_price
                early_entry_savings_pct = (entry_price / min_low_before - 1) * 100 if min_low_before > 0 else 0
                
                # Low during the trade (could have entered at low of trade)
                trade_lows = low[entry_day:exit_day+1] if exit_day < len(low) else low[entry_day:]
                min_in_trade = np.min(trade_lows) if len(trade_lows) > 0 else entry_price
                
                # High during the trade
                trade_highs = high[entry_day:exit_day+1] if exit_day < len(high) else high[entry_day:]
                max_in_trade = np.max(trade_highs) if len(trade_highs) > 0 else exit_price
                max_possible_pnl = (max_in_trade / entry_price - 1) * 100 if entry_price > 0 else 0
                
                # Was this a top-buy? (entry near highest in next 10 bars)
                post_entry_end = min(entry_day + 10, len(close))
                post_entry_prices = close[entry_day:post_entry_end]
                if len(post_entry_prices) > 1:
                    entry_rank = (entry_price - np.min(post_entry_prices)) / (np.max(post_entry_prices) - np.min(post_entry_prices) + 0.001)
                else:
                    entry_rank = 0.5
                is_top_buy = entry_rank > 0.8
                
                # Was this a bottom-sell? (exit near lowest in surrounding 10 bars)
                pre_exit_start = max(0, exit_day - 5)
                post_exit_end = min(exit_day + 5, len(close))
                surrounding_exit = close[pre_exit_start:post_exit_end]
                if len(surrounding_exit) > 1:
                    exit_rank = (exit_price - np.min(surrounding_exit)) / (np.max(surrounding_exit) - np.min(surrounding_exit) + 0.001)
                else:
                    exit_rank = 0.5
                is_bottom_sell = exit_rank < 0.2
                
                # Rebuy analysis: if there's a next trade, what happened between?
                rebuy_gap_pct = None
                gap_min_price = None
                gap_days = None
                if idx + 1 < len(trades):
                    next_t = trades[idx + 1]
                    next_entry_day = next_t.get("entry_day", 0)
                    next_entry_price = close[next_entry_day] if next_entry_day < len(close) else 0
                    if exit_price > 0 and next_entry_price > 0:
                        rebuy_gap_pct = (next_entry_price / exit_price - 1) * 100
                        gap_days = next_entry_day - exit_day
                        if exit_day < next_entry_day and next_entry_day < len(low):
                            gap_prices = low[exit_day:next_entry_day+1]
                            gap_min_price = np.min(gap_prices)
                
                # Entry vs SMA position
                entry_vs_sma20 = (entry_price / sma20[entry_day] - 1) * 100 if entry_day < len(sma20) and not np.isnan(sma20[entry_day]) and sma20[entry_day] > 0 else 0
                
                analysis = {
                    **t,
                    "entry_price": round(entry_price, 0),
                    "exit_price": round(exit_price, 0),
                    "early_entry_savings_pct": round(early_entry_savings_pct, 2),
                    "max_possible_pnl": round(max_possible_pnl, 2),
                    "actual_capture_ratio": round(t["pnl_pct"] / max_possible_pnl * 100, 1) if max_possible_pnl > 0 else 0,
                    "is_top_buy": is_top_buy,
                    "is_bottom_sell": is_bottom_sell,
                    "entry_rank_10d": round(entry_rank, 2),
                    "exit_rank_10d": round(exit_rank, 2),
                    "rebuy_gap_pct": round(rebuy_gap_pct, 2) if rebuy_gap_pct is not None else None,
                    "gap_days": gap_days,
                    "entry_vs_sma20": round(entry_vs_sma20, 2),
                }
                all_analyses.append(analysis)

    return all_analyses


if __name__ == "__main__":
    print("Running deep V11 trade analysis...")
    analyses = run_deep_analysis()
    
    print(f"\n{'='*100}")
    print(f"DEEP V11 TRADE ANALYSIS — {len(analyses)} trades")
    print(f"{'='*100}")
    
    # 1. EARLY ENTRY ANALYSIS
    print(f"\n{'='*80}")
    print("1. COULD WE HAVE ENTERED EARLIER/CHEAPER? (5 bars lookback)")
    print(f"{'='*80}")
    savings = [a["early_entry_savings_pct"] for a in analyses if a["early_entry_savings_pct"] > 0.5]
    print(f"  Trades where entry was >0.5% above recent low: {len(savings)}/{len(analyses)}")
    print(f"  Average potential savings: {np.mean(savings):.2f}%")
    print(f"  Median: {np.median(savings):.2f}%, Max: {np.max(savings):.2f}%")
    print(f"  Total potential savings: {sum(savings):.1f}%")
    
    # Entry vs SMA20 distribution
    above_sma = [a for a in analyses if a["entry_vs_sma20"] > 3]
    at_sma = [a for a in analyses if -1 <= a["entry_vs_sma20"] <= 3]
    below_sma = [a for a in analyses if a["entry_vs_sma20"] < -1]
    print(f"\n  Entry position vs SMA20:")
    print(f"    >3% above SMA20: {len(above_sma)} trades, avg PnL: {np.mean([a['pnl_pct'] for a in above_sma]):.2f}%")
    print(f"    Near SMA20 (±3%): {len(at_sma)} trades, avg PnL: {np.mean([a['pnl_pct'] for a in at_sma]):.2f}%")
    print(f"    <1% below SMA20: {len(below_sma)} trades, avg PnL: {np.mean([a['pnl_pct'] for a in below_sma]) if below_sma else 0:.2f}%")
    
    # 2. TOP-BUYING ANALYSIS
    print(f"\n{'='*80}")
    print("2. TOP-BUY ANALYSIS (entry rank > 0.8 in next 10 days)")
    print(f"{'='*80}")
    top_buys = [a for a in analyses if a["is_top_buy"]]
    non_top = [a for a in analyses if not a["is_top_buy"]]
    print(f"  Top-buys: {len(top_buys)}/{len(analyses)} ({len(top_buys)/len(analyses)*100:.1f}%)")
    print(f"  Top-buy avg PnL: {np.mean([a['pnl_pct'] for a in top_buys]):.2f}% (WR: {sum(1 for a in top_buys if a['pnl_pct']>0)/len(top_buys)*100:.1f}%)")
    print(f"  Non-top avg PnL: {np.mean([a['pnl_pct'] for a in non_top]):.2f}% (WR: {sum(1 for a in non_top if a['pnl_pct']>0)/len(non_top)*100:.1f}%)")
    print(f"\n  Top-buy characteristics:")
    top_buy_trends = defaultdict(int)
    for a in top_buys:
        top_buy_trends[a.get("entry_trend", "?")] += 1
    for t, c in sorted(top_buy_trends.items(), key=lambda x: -x[1]):
        pnl = np.mean([a["pnl_pct"] for a in top_buys if a.get("entry_trend") == t])
        print(f"    Trend={t}: {c} trades, avg PnL: {pnl:+.2f}%")
    
    top_buy_reasons = defaultdict(int)
    for a in top_buys:
        top_buy_reasons[a.get("exit_reason", "?")] += 1
    for r, c in sorted(top_buy_reasons.items(), key=lambda x: -x[1]):
        print(f"    Exit={r}: {c} trades")
    
    print(f"\n  Worst top-buys:")
    for a in sorted(top_buys, key=lambda x: x["pnl_pct"])[:15]:
        print(f"    {a['symbol']:5s} {a.get('entry_date',''):10s} PnL:{a['pnl_pct']:+.1f}% "
              f"Hold:{a['holding_days']:3d}d Trend:{a.get('entry_trend','')} "
              f"vs_SMA20:{a['entry_vs_sma20']:+.1f}% EntryRank:{a['entry_rank_10d']:.2f}")
    
    # 3. BOTTOM-SELLING ANALYSIS
    print(f"\n{'='*80}")
    print("3. BOTTOM-SELL ANALYSIS (exit rank < 0.2 in surrounding 10 days)")
    print(f"{'='*80}")
    bottom_sells = [a for a in analyses if a["is_bottom_sell"]]
    non_bottom = [a for a in analyses if not a["is_bottom_sell"]]
    print(f"  Bottom-sells: {len(bottom_sells)}/{len(analyses)} ({len(bottom_sells)/len(analyses)*100:.1f}%)")
    if bottom_sells:
        print(f"  Bottom-sell avg PnL: {np.mean([a['pnl_pct'] for a in bottom_sells]):.2f}%")
        print(f"  Non-bottom avg PnL: {np.mean([a['pnl_pct'] for a in non_bottom]):.2f}%")
        
        # What happens AFTER bottom-sells?
        bs_with_rebuy = [a for a in bottom_sells if a["rebuy_gap_pct"] is not None]
        if bs_with_rebuy:
            rebuy_higher = [a for a in bs_with_rebuy if a["rebuy_gap_pct"] > 0]
            print(f"\n  After bottom-sell, rebuy higher: {len(rebuy_higher)}/{len(bs_with_rebuy)}")
            if rebuy_higher:
                print(f"  Avg rebuy premium: +{np.mean([a['rebuy_gap_pct'] for a in rebuy_higher]):.2f}%")
        
        print(f"\n  Bottom-sell exit reasons:")
        bs_reasons = defaultdict(int)
        for a in bottom_sells:
            bs_reasons[a.get("exit_reason", "?")] += 1
        for r, c in sorted(bs_reasons.items(), key=lambda x: -x[1]):
            print(f"    {r}: {c} trades")
        
        print(f"\n  Worst bottom-sells (sold low, price recovered):")
        for a in sorted(bottom_sells, key=lambda x: x["pnl_pct"])[:15]:
            rb = f"rebuy+{a['rebuy_gap_pct']:.1f}%" if a["rebuy_gap_pct"] is not None else "no rebuy"
            print(f"    {a['symbol']:5s} {a.get('entry_date',''):10s} PnL:{a['pnl_pct']:+.1f}% "
                  f"Hold:{a['holding_days']:3d}d Reason:{a.get('exit_reason','')} "
                  f"ExitRank:{a['exit_rank_10d']:.2f} {rb}")
    
    # 4. REBUY-AT-HIGHER-PRICE ANALYSIS
    print(f"\n{'='*80}")
    print("4. REBUY-AT-HIGHER-PRICE ANALYSIS")
    print(f"{'='*80}")
    rebuys = [a for a in analyses if a["rebuy_gap_pct"] is not None]
    rebuy_higher = [a for a in rebuys if a["rebuy_gap_pct"] > 2]
    rebuy_lower = [a for a in rebuys if a["rebuy_gap_pct"] < -2]
    rebuy_similar = [a for a in rebuys if -2 <= a["rebuy_gap_pct"] <= 2]
    
    print(f"  Total sell→rebuy pairs: {len(rebuys)}")
    print(f"  Rebuy >2% higher: {len(rebuy_higher)} ({len(rebuy_higher)/max(len(rebuys),1)*100:.1f}%)")
    print(f"  Rebuy ±2% similar: {len(rebuy_similar)} ({len(rebuy_similar)/max(len(rebuys),1)*100:.1f}%)")
    print(f"  Rebuy >2% lower: {len(rebuy_lower)} ({len(rebuy_lower)/max(len(rebuys),1)*100:.1f}%)")
    
    if rebuy_higher:
        total_premium = sum(a["rebuy_gap_pct"] for a in rebuy_higher)
        print(f"\n  Total cost of rebuying higher: +{total_premium:.1f}%")
        print(f"  Avg gap days: {np.mean([a['gap_days'] for a in rebuy_higher]):.1f}")
        print(f"  Avg premium: +{np.mean([a['rebuy_gap_pct'] for a in rebuy_higher]):.2f}%")
        
        print(f"\n  Worst rebuy-higher cases (sold then rebought much higher):")
        for a in sorted(rebuy_higher, key=lambda x: -x["rebuy_gap_pct"])[:20]:
            print(f"    {a['symbol']:5s} Sold:{a.get('exit_date',''):10s} PnL:{a['pnl_pct']:+.1f}% "
                  f"Reason:{a.get('exit_reason','')} → Rebuy+{a['rebuy_gap_pct']:.1f}% "
                  f"after {a['gap_days']}d Trend:{a.get('entry_trend','')}")
    
    # 5. PROFIT CAPTURE EFFICIENCY
    print(f"\n{'='*80}")
    print("5. PROFIT CAPTURE EFFICIENCY")
    print(f"{'='*80}")
    winning = [a for a in analyses if a["pnl_pct"] > 0 and a["max_possible_pnl"] > 0]
    if winning:
        captures = [a["actual_capture_ratio"] for a in winning]
        print(f"  Winning trades: {len(winning)}")
        print(f"  Avg capture ratio: {np.mean(captures):.1f}% of max possible profit")
        print(f"  Median capture: {np.median(captures):.1f}%")
        
        low_capture = [a for a in winning if a["actual_capture_ratio"] < 30]
        print(f"\n  Low capture (<30% of max): {len(low_capture)} trades")
        if low_capture:
            print(f"  These trades captured avg {np.mean([a['actual_capture_ratio'] for a in low_capture]):.1f}% of max")
            total_left = sum(a["max_possible_pnl"] - a["pnl_pct"] for a in low_capture)
            print(f"  Total profit left on table: {total_left:.1f}%")
    
    # 6. EXIT REASON PATTERNS
    print(f"\n{'='*80}")
    print("6. EXIT REASON → WHAT HAPPENS NEXT")
    print(f"{'='*80}")
    for reason in ["signal", "trailing_stop", "stop_loss", "hard_stop"]:
        reason_trades = [a for a in rebuys if a.get("exit_reason") == reason]
        if not reason_trades:
            continue
        rebuy_h = [a for a in reason_trades if a["rebuy_gap_pct"] > 2]
        avg_gap = np.mean([a["rebuy_gap_pct"] for a in reason_trades])
        avg_gap_days = np.mean([a["gap_days"] for a in reason_trades])
        print(f"\n  Exit={reason}: {len(reason_trades)} sell→rebuy pairs")
        print(f"    Avg rebuy gap: {avg_gap:+.2f}%, Avg gap days: {avg_gap_days:.1f}")
        print(f"    Rebuy >2% higher: {len(rebuy_h)} ({len(rebuy_h)/len(reason_trades)*100:.1f}%)")
        
        # For signal exits: could we have held longer?
        if reason == "signal":
            held_then_rebuy_higher = [a for a in reason_trades if a["rebuy_gap_pct"] > 3 and a["gap_days"] <= 5]
            if held_then_rebuy_higher:
                print(f"    *** {len(held_then_rebuy_higher)} times sold on signal, price went UP >3% within 5 days ***")
                for a in held_then_rebuy_higher[:10]:
                    print(f"      {a['symbol']:5s} {a.get('exit_date',''):10s} PnL:{a['pnl_pct']:+.1f}% "
                          f"→ +{a['rebuy_gap_pct']:.1f}% in {a['gap_days']}d")
    
    # 7. KEY FINDINGS SUMMARY
    print(f"\n{'='*100}")
    print("7. ROOT CAUSE SUMMARY & V13 RECOMMENDATIONS")
    print(f"{'='*100}")
    
    top_buy_count = len([a for a in analyses if a["is_top_buy"]])
    bottom_sell_count = len([a for a in analyses if a["is_bottom_sell"]])
    signal_rebuy_higher = len([a for a in rebuys if a.get("exit_reason") == "signal" and a["rebuy_gap_pct"] > 2])
    
    print(f"""
ROOT CAUSES:
A. TOP-BUYING ({top_buy_count}/{len(analyses)} trades = {top_buy_count/len(analyses)*100:.1f}%):
   → Entry khi giá đã quá cao so với recent range
   → Fix: Chờ pullback về SMA20 hoặc entry khi giá gần support

B. BOTTOM-SELLING ({bottom_sell_count}/{len(analyses)} trades = {bottom_sell_count/len(analyses)*100:.1f}%):
   → Bán ngay đáy, giá hồi phục ngay sau đó
   → Fix: Thêm reversal confirmation trước khi exit

C. SIGNAL EXIT → REBUY HIGHER ({signal_rebuy_higher} cases):
   → Model predict sell nhưng giá tiếp tục tăng
   → Fix: Trong uptrend, require EXIT_CONFIRM cao hơn hoặc chỉ exit khi price < SMA20

D. ENTRY TOO HIGH vs SMA20:
   → Avg entry {np.mean([a['entry_vs_sma20'] for a in analyses]):.1f}% above SMA20
   → Fix: Limit entry chỉ khi price <= SMA20 * 1.03 (trừ breakout)
""")

    # Save analysis for V13 development
    import json
    output_path = os.path.join(os.path.dirname(__file__), "v11_deep_analysis.json")
    
    # Save key stats
    stats = {
        "total_trades": len(analyses),
        "top_buy_count": top_buy_count,
        "bottom_sell_count": bottom_sell_count,
        "avg_early_entry_savings": round(np.mean([a["early_entry_savings_pct"] for a in analyses]), 2),
        "rebuy_higher_count": len(rebuy_higher),
        "rebuy_higher_total_premium": round(sum(a["rebuy_gap_pct"] for a in rebuy_higher), 1) if rebuy_higher else 0,
        "avg_entry_vs_sma20": round(np.mean([a["entry_vs_sma20"] for a in analyses]), 2),
        "avg_capture_ratio": round(np.mean([a["actual_capture_ratio"] for a in winning]), 1) if winning else 0,
    }
    with open(output_path, "w") as f:
        json.dump({"stats": stats, "trades": analyses[:50]}, f, indent=2, default=str)
    print(f"\nSaved analysis to {output_path}")

run_deep_analysis()
