"""
V2 Tuned: Relaxed thresholds + class_weight balanced + multiple configs comparison
"""
import sys, os, warnings
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model


def run_experiment(symbols_data, feature_cols, splitter, target_cfg, model_name, hold_period, prob_threshold, label=""):
    target_gen = TargetGenerator.from_config({"target": target_cfg})
    
    all_trades = []
    
    for symbol, sym_raw in symbols_data.items():
        sym_df = target_gen.generate(sym_raw.copy())
        sym_df = sym_df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
        if len(sym_df) < 100:
            continue
        
        prices = sym_df["close"].values
        
        for window, train_df, test_df in splitter.split(sym_df):
            try:
                X_train = np.nan_to_num(train_df[feature_cols].values)
                y_train = train_df["target"].values.astype(int)
                X_test = np.nan_to_num(test_df[feature_cols].values)
                
                if len(X_train) < 50 or len(X_test) < 10:
                    continue
                if y_train.sum() < 5:  # need some positive examples
                    continue
                
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                
                if model_name == "rf_balanced":
                    mdl = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
                elif model_name == "gbm":
                    # Calculate scale_pos_weight for imbalance
                    neg = (y_train == 0).sum()
                    pos = (y_train == 1).sum()
                    spw = neg / pos if pos > 0 else 1
                    mdl = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
                else:
                    mdl = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
                
                mdl.fit(X_train, y_train)
                
                probs = mdl.predict_proba(X_test)
                buy_probs = probs[:, 1] if probs.shape[1] >= 2 else mdl.predict(X_test).astype(float)
                
                test_indices = test_df.index.tolist()
                
                i = 0
                while i < len(test_indices):
                    idx = test_indices[i]
                    prob = buy_probs[i]
                    
                    if prob >= prob_threshold and idx + hold_period < len(prices):
                        entry_price = prices[idx]
                        exit_idx = min(idx + hold_period, len(prices) - 1)
                        exit_price = prices[exit_idx]
                        
                        trade_prices = prices[idx:exit_idx+1]
                        trade_return = (exit_price - entry_price) / entry_price * 100
                        max_dd = (np.min(trade_prices) - entry_price) / entry_price * 100
                        max_profit = (np.max(trade_prices) - entry_price) / entry_price * 100
                        
                        lookback = max(0, idx - 20)
                        past = prices[lookback:idx+1]
                        wave_pos = (prices[idx] - np.min(past)) / (np.max(past) - np.min(past)) if np.max(past) > np.min(past) else 0.5
                        
                        all_trades.append({
                            "symbol": symbol, "trade_return_pct": round(trade_return, 2),
                            "max_dd_pct": round(max_dd, 2), "max_profit_pct": round(max_profit, 2),
                            "gave_back_pct": round(max_profit - trade_return, 2),
                            "entry_wave_pos": round(wave_pos, 2), "buy_prob": round(prob, 3),
                            "is_win": trade_return > 0,
                        })
                        
                        while i < len(test_indices) and test_indices[i] <= exit_idx:
                            i += 1
                        continue
                    i += 1
            except:
                continue
    
    if not all_trades:
        return None
    
    tdf = pd.DataFrame(all_trades)
    wins = tdf[tdf["is_win"]]
    
    return {
        "label": label,
        "trades": len(tdf),
        "win_rate": len(wins)/len(tdf)*100,
        "avg_return": tdf["trade_return_pct"].mean(),
        "total_return": tdf["trade_return_pct"].sum(),
        "avg_dd": tdf["max_dd_pct"].mean(),
        "avg_gave_back": tdf["gave_back_pct"].mean(),
        "entry_wave": tdf["entry_wave_pos"].mean(),
        "median_return": tdf["trade_return_pct"].median(),
        "profit_factor": abs(wins["trade_return_pct"].sum() / tdf[~tdf["is_win"]]["trade_return_pct"].sum()) if len(tdf[~tdf["is_win"]]) > 0 and tdf[~tdf["is_win"]]["trade_return_pct"].sum() != 0 else 0,
    }


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    
    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config({
        "split": {"method": "walk_forward", "train_years": 4, "test_years": 1, "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025}
    })
    engine = FeatureEngine(feature_set="leading")
    
    symbols = loader.symbols[:10]
    
    print("=" * 100)
    print("🧪 V2 TUNED: Comparing multiple configurations")
    print("=" * 100)
    
    raw_df = loader.load_all(symbols=symbols)
    df = engine.compute_for_all_symbols(raw_df)
    feature_cols = engine.get_feature_columns(df)
    
    # Prepare per-symbol data
    symbols_data = {}
    for sym in symbols:
        sym_df = df[df["symbol"] == sym].copy().reset_index(drop=True)
        if len(sym_df) >= 100:
            symbols_data[sym] = sym_df
    
    # Define experiments
    experiments = [
        # (target_cfg, model, hold, prob_threshold, label)
        ({"type": "forward_risk_reward", "forward_window": 10, "gain_threshold": 0.05, "loss_threshold": 0.03, "rr_threshold": 2.0},
         "rf_balanced", 10, 0.5, "A: strict target, RF balanced, prob>50%"),
        
        ({"type": "forward_risk_reward", "forward_window": 10, "gain_threshold": 0.03, "loss_threshold": 0.05, "rr_threshold": 1.5},
         "rf_balanced", 10, 0.5, "B: relaxed target(3%/5%/1.5), RF balanced"),
        
        ({"type": "forward_risk_reward", "forward_window": 15, "gain_threshold": 0.05, "loss_threshold": 0.05, "rr_threshold": 1.5},
         "rf_balanced", 15, 0.5, "C: 15d window, gain5%/loss5%/rr1.5"),
        
        ({"type": "forward_risk_reward", "forward_window": 10, "gain_threshold": 0.03, "loss_threshold": 0.05, "rr_threshold": 1.5},
         "gbm", 10, 0.5, "D: relaxed + GBM"),
        
        ({"type": "forward_risk_reward", "forward_window": 10, "gain_threshold": 0.03, "loss_threshold": 0.05, "rr_threshold": 1.5},
         "rf_balanced", 10, 0.4, "E: relaxed + prob>40%"),
         
        ({"type": "forward_risk_reward", "forward_window": 10, "gain_threshold": 0.03, "loss_threshold": 0.05, "rr_threshold": 1.5},
         "rf_balanced", 10, 0.6, "F: relaxed + prob>60%"),
         
        # Old approach for comparison
        ({"type": "trend_regime", "trend_method": "dual_ma", "short_window": 10, "long_window": 40, "classes": 3},
         "rf", 10, 0.5, "OLD: trend_regime + RF (baseline)"),
    ]
    
    results = []
    for target_cfg, model_name, hold, prob_thresh, label in experiments:
        print(f"\n🔄 Running: {label}...")
        
        # For old approach, need special handling (predict=1 means buy)
        if target_cfg["type"] == "trend_regime":
            # Use prob_threshold differently - predict class 1
            r = run_experiment(symbols_data, feature_cols, splitter, target_cfg, "rf", hold, 0.5, label)
        else:
            r = run_experiment(symbols_data, feature_cols, splitter, target_cfg, model_name, hold, prob_thresh, label)
        
        if r:
            results.append(r)
            print(f"   ✅ {r['trades']} trades, Win={r['win_rate']:.1f}%, AvgRet={r['avg_return']:+.2f}%, WavePos={r['entry_wave']:.2f}")
        else:
            print(f"   ❌ No trades")
    
    # Summary table
    print(f"\n{'═'*120}")
    print(f"📊 COMPARISON TABLE")
    print(f"{'═'*120}")
    print(f"{'Config':<45} {'Trades':>7} {'Win%':>7} {'AvgRet':>8} {'MedRet':>8} {'TotalRet':>10} {'AvgDD':>8} {'GaveBack':>9} {'Wave':>6} {'PF':>6}")
    print(f"{'─'*120}")
    
    for r in results:
        print(f"{r['label']:<45} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['avg_return']:>+7.2f}% {r['median_return']:>+7.2f}% {r['total_return']:>+9.1f}% {r['avg_dd']:>7.2f}% {r['avg_gave_back']:>8.2f}% {r['entry_wave']:>5.2f} {r['profit_factor']:>5.2f}")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['avg_return'])
        print(f"\n🏆 BEST: {best['label']}")
        print(f"   Win Rate: {best['win_rate']:.1f}%, Avg Return: {best['avg_return']:+.2f}%, Profit Factor: {best['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
