"""
V2 Pipeline: Forward Risk-Reward Target + Leading Features + Fixed Hold Backtest
Giải quyết gốc rễ: target leading, features leading, buy-point prediction
"""
import sys, os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model


def run_v2():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    
    config = {
        "split": {"method": "walk_forward", "train_years": 4, "test_years": 1, "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025},
        "target": {
            "type": "forward_risk_reward",
            "forward_window": 10,
            "gain_threshold": 0.05,   # 5% upside needed
            "loss_threshold": 0.03,   # max 3% downside
            "rr_threshold": 2.0,      # reward/risk >= 2
        },
    }
    
    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)
    engine = FeatureEngine(feature_set="leading")  # NEW leading features
    
    symbols = loader.symbols[:10]
    hold_period = 10  # Fixed hold after buy signal
    prob_threshold = 0.6  # Only buy when probability > 60%
    
    print("=" * 100)
    print("🚀 V2 PIPELINE: Forward Risk-Reward + Leading Features")
    print(f"   Target: gain≥5%, loss<3%, RR≥2 in {config['target']['forward_window']}d")
    print(f"   Hold: {hold_period} days fixed | Prob threshold: {prob_threshold}")
    print("=" * 100)
    
    raw_df = loader.load_all(symbols=symbols)
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])
    
    # Check target distribution
    buy_pct = (df["target"] == 1).mean() * 100
    print(f"\n📊 Target distribution: BUY={buy_pct:.1f}%, AVOID={100-buy_pct:.1f}%")
    
    all_trades = []
    all_metrics = []
    
    for symbol in symbols:
        sym_df = df[df["symbol"] == symbol].copy().reset_index(drop=True)
        if len(sym_df) < 100:
            continue
        
        prices = sym_df["close"].values
        
        for window, train_df, test_df in splitter.split(sym_df):
            try:
                X_train = np.nan_to_num(train_df[feature_cols].values)
                y_train = train_df["target"].values.astype(int)
                X_test = np.nan_to_num(test_df[feature_cols].values)
                y_test = test_df["target"].values.astype(int)
                
                if len(X_train) < 50 or len(X_test) < 10:
                    continue
                
                model = build_model("random_forest")
                model.fit(X_train, y_train)
                
                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test)
                    if probs.shape[1] >= 2:
                        buy_probs = probs[:, 1]
                    else:
                        buy_probs = model.predict(X_test).astype(float)
                else:
                    buy_probs = model.predict(X_test).astype(float)
                
                test_indices = test_df.index.tolist()
                
                # Simulate fixed-hold trades
                i = 0
                while i < len(test_indices):
                    idx = test_indices[i]
                    prob = buy_probs[i]
                    
                    if prob >= prob_threshold and idx + hold_period < len(prices):
                        entry_price = prices[idx]
                        # Hold for fixed period
                        exit_idx = min(idx + hold_period, len(prices) - 1)
                        exit_price = prices[exit_idx]
                        
                        # Track within-trade stats
                        trade_prices = prices[idx:exit_idx+1]
                        max_price = np.max(trade_prices)
                        min_price = np.min(trade_prices)
                        
                        trade_return = (exit_price - entry_price) / entry_price * 100
                        max_dd = (min_price - entry_price) / entry_price * 100
                        max_profit = (max_price - entry_price) / entry_price * 100
                        
                        # Wave position at entry
                        lookback = max(0, idx - 20)
                        past = prices[lookback:idx+1]
                        wave_pos = (prices[idx] - np.min(past)) / (np.max(past) - np.min(past)) if np.max(past) > np.min(past) else 0.5
                        
                        # After exit
                        after_end = min(idx + hold_period + 10, len(prices))
                        after_prices = prices[exit_idx+1:after_end] if exit_idx+1 < len(prices) else []
                        after_5d_ret = (prices[min(exit_idx+5, len(prices)-1)] - exit_price) / exit_price * 100 if exit_idx+5 < len(prices) else 0
                        
                        all_trades.append({
                            "symbol": symbol,
                            "window": window,
                            "entry_idx": idx,
                            "exit_idx": exit_idx,
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(exit_price, 2),
                            "hold_days": hold_period,
                            "buy_prob": round(prob, 3),
                            "trade_return_pct": round(trade_return, 2),
                            "max_dd_pct": round(max_dd, 2),
                            "max_profit_pct": round(max_profit, 2),
                            "gave_back_pct": round(max_profit - trade_return, 2),
                            "entry_wave_pos": round(wave_pos, 2),
                            "actual_target": int(y_test[i]),
                            "after_exit_5d_pct": round(after_5d_ret, 2),
                            "is_win": trade_return > 0,
                        })
                        
                        # Skip ahead past hold period (no overlapping trades)
                        skip = 0
                        while i < len(test_indices) and test_indices[i] <= exit_idx:
                            i += 1
                        continue
                    i += 1
                
                # Classification metrics
                y_pred = (buy_probs >= prob_threshold).astype(int)
                tp = ((y_pred == 1) & (y_test == 1)).sum()
                fp = ((y_pred == 1) & (y_test == 0)).sum()
                fn = ((y_pred == 0) & (y_test == 1)).sum()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                all_metrics.append({
                    "symbol": symbol, "window": window,
                    "precision": precision, "recall": recall,
                    "buy_signals": int(y_pred.sum()), "actual_buys": int(y_test.sum()),
                })
                
            except Exception as e:
                print(f"   Error {symbol} w{window}: {e}")
                continue
    
    # ═══════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════
    if not all_trades:
        print("No trades generated!")
        return
    
    trades_df = pd.DataFrame(all_trades)
    metrics_df = pd.DataFrame(all_metrics)
    wins = trades_df[trades_df["is_win"]]
    losses = trades_df[~trades_df["is_win"]]
    
    print(f"\n{'═'*80}")
    print(f"📊 V2 RESULTS: {len(trades_df)} trades")
    print(f"{'═'*80}")
    print(f"   Win Rate: {len(wins)}/{len(trades_df)} = {len(wins)/len(trades_df)*100:.1f}%")
    print(f"   Avg Return/Trade: {trades_df['trade_return_pct'].mean():+.2f}%")
    print(f"   Median Return: {trades_df['trade_return_pct'].median():+.2f}%")
    print(f"   Total Return: {trades_df['trade_return_pct'].sum():+.2f}%")
    print(f"   Avg Max DD: {trades_df['max_dd_pct'].mean():.2f}%")
    print(f"   Avg Max Profit: {trades_df['max_profit_pct'].mean():+.2f}%")
    print(f"   Avg Gave Back: {trades_df['gave_back_pct'].mean():.2f}%")
    
    print(f"\n📍 ENTRY WAVE POSITION:")
    print(f"   All trades: {trades_df['entry_wave_pos'].mean():.2f}")
    print(f"   Wins: {wins['entry_wave_pos'].mean():.2f}")
    print(f"   Losses: {losses['entry_wave_pos'].mean():.2f}")
    
    print(f"\n📈 AFTER EXIT:")
    after_up = (trades_df["after_exit_5d_pct"] > 0).sum()
    print(f"   Giá tăng sau bán: {after_up}/{len(trades_df)} ({after_up/len(trades_df)*100:.1f}%)")
    print(f"   Avg 5d after exit: {trades_df['after_exit_5d_pct'].mean():+.2f}%")
    
    print(f"\n🎯 PRECISION/RECALL (is this a buy point?):")
    print(f"   Avg Precision: {metrics_df['precision'].mean():.3f}")
    print(f"   Avg Recall: {metrics_df['recall'].mean():.3f}")
    
    # By probability bucket
    print(f"\n📊 BY BUY PROBABILITY:")
    prob_bins = [(0.5, 0.6, "50-60%"), (0.6, 0.7, "60-70%"), (0.7, 0.8, "70-80%"), (0.8, 0.9, "80-90%"), (0.9, 1.01, "90-100%")]
    print(f"   {'Prob':<10} {'Trades':>8} {'Win%':>8} {'AvgRet%':>10} {'AvgDD%':>10}")
    for lo, hi, label in prob_bins:
        mask = (trades_df["buy_prob"] >= lo) & (trades_df["buy_prob"] < hi)
        subset = trades_df[mask]
        if len(subset) > 0:
            wr = subset["is_win"].mean() * 100
            avg_ret = subset["trade_return_pct"].mean()
            avg_dd = subset["max_dd_pct"].mean()
            print(f"   {label:<10} {len(subset):>8} {wr:>7.1f}% {avg_ret:>+9.2f}% {avg_dd:>9.2f}%")
    
    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_df.to_csv(os.path.join(out_dir, f"v2_trades_{ts}.csv"), index=False)
    print(f"\n💾 Saved to results/v2_trades_{ts}.csv")


if __name__ == "__main__":
    run_v2()
