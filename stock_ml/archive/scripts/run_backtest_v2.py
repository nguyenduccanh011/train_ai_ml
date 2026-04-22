"""
Backtest V2 Runner - Per-symbol backtest with Smart Exit strategy.
Compares: Original (v1) vs Smart Exit (v2) strategies.

Usage:
    python run_backtest_v2.py                   # Quick (10 symbols)
    python run_backtest_v2.py --symbols 20      # 20 symbols
    python run_backtest_v2.py --full             # All symbols
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model, get_available_models
from src.evaluation.backtest import backtest_predictions
from src.evaluation.backtest_v2 import backtest_smart_exit, backtest_per_symbol, format_smart_report
from src.evaluation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="VN Stock ML Backtest V2 - Smart Exit")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--symbols", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--features", nargs="+", default=["technical"])
    parser.add_argument("--capital", type=float, default=100_000_000)
    args = parser.parse_args()

    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "portable_data", "vn_stock_ai_dataset_cleaned"
    )

    config = {
        "data": {"data_dir": data_dir},
        "split": {
            "method": "walk_forward",
            "train_years": 4, "test_years": 1, "gap_days": 0,
            "first_test_year": 2020, "last_test_year": 2025,
        },
        "target": {
            "type": "trend_regime", "trend_method": "dual_ma",
            "short_window": 10, "long_window": 40, "classes": 3,
        },
    }

    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    max_symbols = None if args.full else args.symbols
    symbols = loader.symbols[:max_symbols] if max_symbols else loader.symbols

    model_names = args.models or ["random_forest", "xgboost", "lightgbm", "logistic_regression"]
    available = get_available_models()
    model_names = [m for m in model_names if m in available]

    print("=" * 90)
    print("📈 VN STOCK ML BACKTEST V2 - SMART EXIT")
    print(f"   Vốn: {args.capital:,.0f} VND")
    print(f"   Models: {model_names}")
    print(f"   Features: {args.features}")
    print(f"   Symbols ({len(symbols)}): {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
    print("=" * 90)

    # Load & prepare data
    print("\n📦 Loading data...")
    raw_df = loader.load_all(symbols=symbols)
    print(f"   {len(raw_df)} rows, {raw_df['symbol'].nunique()} symbols")

    all_comparisons = []

    for feat_set in args.features:
        print(f"\n{'━' * 90}")
        print(f"📊 Feature set: {feat_set}")

        engine = FeatureEngine(feature_set=feat_set)
        df = engine.compute_for_all_symbols(raw_df)
        df = target_gen.generate_for_all_symbols(df)
        feature_cols = engine.get_feature_columns(df)
        df = df.dropna(subset=feature_cols + ["target"])

        for model_name in model_names:
            print(f"\n   🤖 Model: {model_name}")
            print(f"   {'─' * 80}")

            # ══════════════════════════════════════════════════════
            # PER-SYMBOL backtest (fixes the concatenation bug)
            # ══════════════════════════════════════════════════════
            
            symbol_v1_results = {}  # Original strategy per symbol
            symbol_v2_results = {}  # Smart exit per symbol
            symbol_metrics = {}     # Classification metrics per symbol
            
            for symbol in symbols:
                sym_df = df[df["symbol"] == symbol].copy()
                if len(sym_df) < 100:
                    continue
                
                # Collect predictions across all walk-forward windows for this symbol
                sym_y_test_all = []
                sym_y_pred_all = []
                sym_returns_all = []
                sym_prices_all = []
                sym_atr_all = []
                sym_vol_ratio_all = []
                sym_rsi_all = []
                sym_trend_short_all = []
                sym_trend_medium_all = []
                
                for window, train_df, test_df in splitter.split(sym_df):
                    try:
                        # Filter for this symbol in train/test
                        X_train = np.nan_to_num(train_df[feature_cols].values)
                        y_train = train_df["target"].values.astype(int)
                        X_test = np.nan_to_num(test_df[feature_cols].values)
                        y_test = test_df["target"].values.astype(int)
                        
                        if len(X_train) < 50 or len(X_test) < 10:
                            continue
                        
                        model = build_model(model_name)
                        
                        # XGBoost offset
                        offset = 0
                        if model_name == "xgboost" and y_train.min() < 0:
                            offset = abs(y_train.min())
                            y_train_fit = y_train + offset
                        else:
                            y_train_fit = y_train
                        
                        model.fit(X_train, y_train_fit)
                        y_pred = model.predict(X_test)
                        
                        if offset > 0:
                            y_pred = y_pred - offset
                        
                        # Collect data
                        sym_y_test_all.extend(y_test.tolist())
                        sym_y_pred_all.extend(y_pred.tolist())
                        
                        # Get signal data for smart exit
                        returns = test_df["return_1d"].values if "return_1d" in test_df.columns else np.zeros(len(test_df))
                        prices = test_df["close"].values if "close" in test_df.columns else np.ones(len(test_df))
                        atr_vals = test_df["atr_14"].values if "atr_14" in test_df.columns else np.full(len(test_df), np.nan)
                        vol_ratio = test_df["volume_ratio_20d"].values if "volume_ratio_20d" in test_df.columns else np.ones(len(test_df))
                        rsi_vals = test_df["rsi_14"].values if "rsi_14" in test_df.columns else np.full(len(test_df), 50.0)
                        
                        # Trend signals
                        if "sma_5" in test_df.columns and "sma_10" in test_df.columns:
                            t_short = (test_df["sma_5"].values > test_df["sma_10"].values).astype(float)
                        else:
                            t_short = np.ones(len(test_df))
                        
                        if "sma_20" in test_df.columns and "sma_50" in test_df.columns:
                            t_medium = (test_df["sma_20"].values > test_df["sma_50"].values).astype(float)
                        else:
                            t_medium = np.ones(len(test_df))
                        
                        sym_returns_all.extend(returns.tolist())
                        sym_prices_all.extend(prices.tolist())
                        sym_atr_all.extend(atr_vals.tolist())
                        sym_vol_ratio_all.extend(vol_ratio.tolist())
                        sym_rsi_all.extend(rsi_vals.tolist())
                        sym_trend_short_all.extend(t_short.tolist())
                        sym_trend_medium_all.extend(t_medium.tolist())
                        
                    except Exception as e:
                        pass  # Skip failed windows silently
                
                if not sym_returns_all or len(sym_returns_all) < 20:
                    continue
                
                # Convert to arrays
                pred_arr = np.array(sym_y_pred_all)
                ret_arr = np.array(sym_returns_all)
                price_arr = np.array(sym_prices_all)
                atr_arr = np.array(sym_atr_all)
                vol_arr = np.array(sym_vol_ratio_all)
                rsi_arr = np.array(sym_rsi_all)
                ts_arr = np.array(sym_trend_short_all)
                tm_arr = np.array(sym_trend_medium_all)
                
                cap_per_sym = args.capital / len(symbols)
                
                # V1: Original backtest
                v1 = backtest_predictions(pred_arr, ret_arr, cap_per_sym)
                symbol_v1_results[symbol] = v1
                
                # V2: Smart Exit backtest
                v2 = backtest_smart_exit(
                    y_pred=pred_arr,
                    returns=ret_arr,
                    prices=price_arr,
                    atr=atr_arr,
                    volume_ratio=vol_arr,
                    rsi=rsi_arr,
                    trend_short=ts_arr,
                    trend_medium=tm_arr,
                    initial_capital=cap_per_sym,
                )
                symbol_v2_results[symbol] = v2
                
                # Classification metrics
                if sym_y_test_all:
                    sym_metrics = compute_metrics(sym_y_test_all, sym_y_pred_all)
                    symbol_metrics[symbol] = sym_metrics
            
            # ══════════════════════════════════════════════════════
            # AGGREGATE RESULTS
            # ══════════════════════════════════════════════════════
            
            if not symbol_v1_results:
                print(f"      ❌ No valid results")
                continue
            
            n_valid = len(symbol_v1_results)
            
            # Portfolio V1
            v1_total_equity = sum(r["final_equity"] for r in symbol_v1_results.values())
            v1_return = (v1_total_equity / args.capital - 1) * 100
            v1_avg_sharpe = np.mean([r["sharpe_ratio"] for r in symbol_v1_results.values()])
            v1_avg_dd = np.mean([r["max_drawdown_pct"] for r in symbol_v1_results.values()])
            v1_worst_dd = min(r["max_drawdown_pct"] for r in symbol_v1_results.values())
            v1_total_trades = sum(r["total_trades"] for r in symbol_v1_results.values())
            v1_total_wins = sum(r["winning_trades"] for r in symbol_v1_results.values())
            v1_win_rate = (v1_total_wins / v1_total_trades * 100) if v1_total_trades > 0 else 0
            v1_avg_bnh = np.mean([r["buy_hold_return_pct"] for r in symbol_v1_results.values()])
            
            # Portfolio V2
            v2_total_equity = sum(r["final_equity"] for r in symbol_v2_results.values())
            v2_return = (v2_total_equity / args.capital - 1) * 100
            v2_avg_sharpe = np.mean([r["sharpe_ratio"] for r in symbol_v2_results.values()])
            v2_avg_dd = np.mean([r["max_drawdown_pct"] for r in symbol_v2_results.values()])
            v2_worst_dd = min(r["max_drawdown_pct"] for r in symbol_v2_results.values())
            v2_total_trades = sum(r["total_trades"] for r in symbol_v2_results.values())
            v2_total_wins = sum(r["winning_trades"] for r in symbol_v2_results.values())
            v2_win_rate = (v2_total_wins / v2_total_trades * 100) if v2_total_trades > 0 else 0
            
            # Classification metrics aggregate
            avg_f1 = np.mean([m["f1_macro"] for m in symbol_metrics.values()]) if symbol_metrics else 0
            avg_acc = np.mean([m["accuracy"] for m in symbol_metrics.values()]) if symbol_metrics else 0
            
            # Print per-symbol details
            print(f"\n      📊 Per-symbol results ({n_valid} symbols):")
            print(f"      {'Symbol':<8} {'V1 Ret%':>8} {'V2 Ret%':>8} {'V1 DD%':>8} {'V2 DD%':>8} {'V1 WR%':>8} {'V2 WR%':>8} {'B&H%':>8} {'F1':>6}")
            print(f"      {'─'*74}")
            
            for sym in sorted(symbol_v1_results.keys()):
                v1r = symbol_v1_results[sym]
                v2r = symbol_v2_results[sym]
                f1 = symbol_metrics.get(sym, {}).get("f1_macro", 0)
                print(f"      {sym:<8} {v1r['total_return_pct']:>+8.2f} {v2r['total_return_pct']:>+8.2f} "
                      f"{v1r['max_drawdown_pct']:>8.2f} {v2r['max_drawdown_pct']:>8.2f} "
                      f"{v1r['win_rate_pct']:>8.1f} {v2r['win_rate_pct']:>8.1f} "
                      f"{v1r['buy_hold_return_pct']:>+8.2f} {f1:>6.3f}")
            
            # Portfolio summary
            print(f"\n      {'═'*74}")
            print(f"      🏦 PORTFOLIO COMPARISON: {model_name} + {feat_set}")
            print(f"      {'─'*74}")
            print(f"      {'Metric':<25} {'V1 (Original)':>20} {'V2 (Smart Exit)':>20}")
            print(f"      {'─'*74}")
            print(f"      {'Total Return':.<25} {v1_return:>+19.2f}% {v2_return:>+19.2f}%")
            print(f"      {'Buy & Hold (avg)':.<25} {v1_avg_bnh:>+19.2f}% {v1_avg_bnh:>+19.2f}%")
            print(f"      {'Excess Return':.<25} {v1_return-v1_avg_bnh:>+19.2f}% {v2_return-v1_avg_bnh:>+19.2f}%")
            print(f"      {'Avg Sharpe':.<25} {v1_avg_sharpe:>20.3f} {v2_avg_sharpe:>20.3f}")
            print(f"      {'Avg Max DD':.<25} {v1_avg_dd:>19.2f}% {v2_avg_dd:>19.2f}%")
            print(f"      {'Worst DD':.<25} {v1_worst_dd:>19.2f}% {v2_worst_dd:>19.2f}%")
            print(f"      {'Total Trades':.<25} {v1_total_trades:>20d} {v2_total_trades:>20d}")
            print(f"      {'Win Rate':.<25} {v1_win_rate:>19.1f}% {v2_win_rate:>19.1f}%")
            print(f"      {'Avg F1-Macro':.<25} {avg_f1:>20.3f} {avg_f1:>20.3f}")
            print(f"      {'Final Equity':.<25} {v1_total_equity:>17,.0f} VND {v2_total_equity:>17,.0f} VND")
            profit_v1 = v1_total_equity - args.capital
            profit_v2 = v2_total_equity - args.capital
            print(f"      {'Profit':.<25} {profit_v1:>+17,.0f} VND {profit_v2:>+17,.0f} VND")
            print(f"      {'═'*74}")
            
            # V2 Exit reason breakdown
            all_exit_reasons = defaultdict(int)
            for sym_r in symbol_v2_results.values():
                for reason, count in sym_r.get("exit_reasons", {}).items():
                    all_exit_reasons[reason] += count
            if all_exit_reasons:
                print(f"\n      🚪 V2 Exit Reasons:")
                for reason, count in sorted(all_exit_reasons.items(), key=lambda x: -x[1]):
                    print(f"         {reason}: {count}")
            
            # Store for final comparison
            all_comparisons.append({
                "model": model_name,
                "feature_set": feat_set,
                "n_symbols": n_valid,
                "v1_return_pct": round(v1_return, 2),
                "v2_return_pct": round(v2_return, 2),
                "v1_sharpe": round(v1_avg_sharpe, 3),
                "v2_sharpe": round(v2_avg_sharpe, 3),
                "v1_avg_dd_pct": round(v1_avg_dd, 2),
                "v2_avg_dd_pct": round(v2_avg_dd, 2),
                "v1_worst_dd_pct": round(v1_worst_dd, 2),
                "v2_worst_dd_pct": round(v2_worst_dd, 2),
                "v1_win_rate": round(v1_win_rate, 1),
                "v2_win_rate": round(v2_win_rate, 1),
                "v1_trades": v1_total_trades,
                "v2_trades": v2_total_trades,
                "buy_hold_pct": round(v1_avg_bnh, 2),
                "f1_macro": round(avg_f1, 3),
                "accuracy": round(avg_acc, 3),
                "v1_profit_M": round(profit_v1 / 1e6, 1),
                "v2_profit_M": round(profit_v2 / 1e6, 1),
            })

    # ══════════════════════════════════════════════════════
    # FINAL COMPARISON TABLE
    # ══════════════════════════════════════════════════════
    if all_comparisons:
        print("\n" + "═" * 100)
        print("🏆 FINAL COMPARISON: V1 (Original) vs V2 (Smart Exit)")
        print("═" * 100)
        
        comp_df = pd.DataFrame(all_comparisons)
        print(comp_df.to_string(index=False, float_format="%.2f"))
        
        print(f"\n💰 Vốn ban đầu: {args.capital:,.0f} VND")
        
        # Best model by V2 return
        best_v2 = comp_df.loc[comp_df["v2_return_pct"].idxmax()]
        print(f"\n🥇 Best V2 Model: {best_v2['model']} ({best_v2['feature_set']})")
        print(f"   V2 Return: {best_v2['v2_return_pct']:+.2f}% | V1 Return: {best_v2['v1_return_pct']:+.2f}%")
        print(f"   V2 Sharpe: {best_v2['v2_sharpe']:.3f} | V1 Sharpe: {best_v2['v1_sharpe']:.3f}")
        print(f"   V2 Avg DD: {best_v2['v2_avg_dd_pct']:.2f}% | V1 Avg DD: {best_v2['v1_avg_dd_pct']:.2f}%")
        print(f"   V2 Win Rate: {best_v2['v2_win_rate']:.1f}% | V1 Win Rate: {best_v2['v1_win_rate']:.1f}%")
        
        # Improvement summary
        print(f"\n📊 Improvement Summary (V2 vs V1):")
        for _, row in comp_df.iterrows():
            ret_diff = row["v2_return_pct"] - row["v1_return_pct"]
            dd_diff = row["v2_avg_dd_pct"] - row["v1_avg_dd_pct"]
            wr_diff = row["v2_win_rate"] - row["v1_win_rate"]
            sharpe_diff = row["v2_sharpe"] - row["v1_sharpe"]
            print(f"   {row['model']:<20} | Return: {ret_diff:+.2f}% | DD: {dd_diff:+.2f}% | WinRate: {wr_diff:+.1f}% | Sharpe: {sharpe_diff:+.3f}")
        
        # Save results
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        comp_df.to_csv(os.path.join(out_dir, f"backtest_v2_{ts}.csv"), index=False)
        print(f"\n💾 Saved to results/backtest_v2_{ts}.csv")
        
        print("═" * 100)


if __name__ == "__main__":
    main()
