"""
Backtest runner - trains models and simulates trading profits.
Now with Smart Exit strategies (stop-loss + trailing stop).

Usage:
    python run_backtest.py                          # Quick test (10 symbols, no smart exit)
    python run_backtest.py --smart-exit              # With smart exit (SL=-7%, Trail=50%)
    python run_backtest.py --smart-exit --compare    # Compare original vs smart exit
    python run_backtest.py --symbols 20 --smart-exit # 20 symbols with smart exit
    python run_backtest.py --full --smart-exit       # All symbols with smart exit
    python run_backtest.py --sl -0.05 --trail 0.3    # Custom stop-loss & trailing
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model, get_available_models
from src.evaluation.backtest import backtest_predictions, format_backtest_report, SmartExitConfig
from src.evaluation.metrics import compute_metrics


def run_single_backtest(splitter, engine, target_gen, raw_df, feature_cols, df,
                        model_name, capital, smart_exit=None, label=""):
    """Run backtest for one model with optional smart exit. Returns aggregated results."""
    all_y_test = []
    all_y_pred = []
    all_returns = []

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

            returns = test_df["return_1d"].values if "return_1d" in test_df.columns else np.zeros(len(test_df))

            bt = backtest_predictions(y_pred, returns, capital, smart_exit=smart_exit)
            print(f"      {label}{window.label:30s} | "
                  f"Return: {bt['total_return_pct']:>+7.2f}% | "
                  f"Trades: {bt['total_trades']:>3d} | "
                  f"WR: {bt['win_rate_pct']:>5.1f}% | "
                  f"PF: {bt['profit_factor']:>5.2f} | "
                  f"SL:{bt.get('n_stop_loss',0):>2d} TS:{bt.get('n_trailing_stop',0):>2d}")

            all_y_test.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_returns.extend(returns.tolist())

        except Exception as e:
            print(f"      {label}{window.label:30s} | ❌ Error: {e}")

    if not all_returns:
        return None

    all_returns_arr = np.array(all_returns)
    all_pred_arr = np.array(all_y_pred)
    agg_bt = backtest_predictions(all_pred_arr, all_returns_arr, capital, smart_exit=smart_exit)
    agg_metrics = compute_metrics(all_y_test, all_y_pred)

    return {
        "bt": agg_bt,
        "metrics": agg_metrics,
        "model": model_name,
    }


def main():
    parser = argparse.ArgumentParser(description="VN Stock ML Backtest with Smart Exit")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--symbols", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--features", nargs="+", default=["technical"])
    parser.add_argument("--capital", type=float, default=100_000_000)
    # Smart exit options
    parser.add_argument("--smart-exit", action="store_true", help="Enable smart exit (SL + trailing)")
    parser.add_argument("--compare", action="store_true", help="Compare original vs smart exit")
    parser.add_argument("--sl", type=float, default=-0.07, help="Stop-loss threshold (default: -0.07)")
    parser.add_argument("--trail", type=float, default=0.50, help="Trailing stop giveback ratio (default: 0.50)")
    parser.add_argument("--no-sl", action="store_true", help="Disable stop-loss (trailing only)")
    parser.add_argument("--no-trail", action="store_true", help="Disable trailing stop (SL only)")
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

    # Setup smart exit config
    smart_exit_config = None
    if args.smart_exit or args.compare:
        smart_exit_config = SmartExitConfig(
            stop_loss_pct=args.sl,
            trailing_stop_pct=args.trail,
            enable_stop_loss=not args.no_sl,
            enable_trailing_stop=not args.no_trail,
        )

    # Setup
    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    max_symbols = None if args.full else args.symbols
    symbols = loader.symbols[:max_symbols] if max_symbols else None

    model_names = args.models or ["random_forest", "xgboost", "lightgbm", "logistic_regression"]
    available = get_available_models()
    model_names = [m for m in model_names if m in available]

    print("=" * 90)
    print("📈 VN STOCK ML BACKTEST" + (" + 🛡️ SMART EXIT" if (args.smart_exit or args.compare) else ""))
    print(f"   Vốn: {args.capital:,.0f} VND")
    print(f"   Models: {model_names}")
    print(f"   Features: {args.features}")
    print(f"   Symbols: {max_symbols or 'all'}")
    if smart_exit_config:
        print(f"   🛡️ Stop-loss: {smart_exit_config.stop_loss_pct:.1%} ({'ON' if smart_exit_config.enable_stop_loss else 'OFF'})")
        print(f"   🛡️ Trailing stop: {smart_exit_config.trailing_stop_pct:.0%} giveback ({'ON' if smart_exit_config.enable_trailing_stop else 'OFF'})")
    if args.compare:
        print(f"   📊 Mode: COMPARE (original vs smart exit)")
    print("=" * 90)

    # Load data
    print("\n📦 Loading data...")
    raw_df = loader.load_all(symbols=symbols)
    print(f"   {len(raw_df)} rows, {raw_df['symbol'].nunique()} symbols")

    all_backtests = []

    for feat_set in args.features:
        print(f"\n{'─' * 70}")
        print(f"📊 Feature set: {feat_set}")

        engine = FeatureEngine(feature_set=feat_set)
        df = engine.compute_for_all_symbols(raw_df)
        df = target_gen.generate_for_all_symbols(df)
        feature_cols = engine.get_feature_columns(df)
        df = df.dropna(subset=feature_cols + ["target"])

        for model_name in model_names:
            print(f"\n   🤖 Model: {model_name}")

            if args.compare:
                # ── Run ORIGINAL (no smart exit) ──
                print(f"\n      --- ORIGINAL (no smart exit) ---")
                res_orig = run_single_backtest(
                    splitter, engine, target_gen, raw_df, feature_cols, df,
                    model_name, args.capital, smart_exit=None, label="[ORIG] "
                )

                # ── Run SMART EXIT ──
                print(f"\n      --- SMART EXIT (SL={args.sl:.0%}, Trail={args.trail:.0%}) ---")
                res_smart = run_single_backtest(
                    splitter, engine, target_gen, raw_df, feature_cols, df,
                    model_name, args.capital, smart_exit=smart_exit_config, label="[SMART] "
                )

                # Print comparison
                if res_orig and res_smart:
                    o = res_orig["bt"]
                    s = res_smart["bt"]
                    print(f"\n      {'─' * 60}")
                    print(f"      📊 SO SÁNH: {model_name} ({feat_set})")
                    print(f"      {'─' * 60}")
                    print(f"      {'Metric':<25s} {'Original':>12s} {'Smart Exit':>12s} {'Δ':>10s}")
                    print(f"      {'─' * 60}")
                    
                    comparisons = [
                        ("Total Return %", o["total_return_pct"], s["total_return_pct"]),
                        ("Ann. Return %", o["annualized_return_pct"], s["annualized_return_pct"]),
                        ("Sharpe Ratio", o["sharpe_ratio"], s["sharpe_ratio"]),
                        ("Max Drawdown %", o["max_drawdown_pct"], s["max_drawdown_pct"]),
                        ("Profit Factor", o["profit_factor"], s["profit_factor"]),
                        ("Win Rate %", o["win_rate_pct"], s["win_rate_pct"]),
                        ("Total Trades", o["total_trades"], s["total_trades"]),
                        ("Avg Loss %", o["avg_loss_pct"], s["avg_loss_pct"]),
                        ("Max Loss %", o["max_loss_pct"], s["max_loss_pct"]),
                    ]
                    for name, orig_v, smart_v in comparisons:
                        delta = smart_v - orig_v
                        arrow = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
                        if name in ("Max Drawdown %", "Avg Loss %", "Max Loss %"):
                            arrow = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")  # less negative = better
                        print(f"      {name:<25s} {orig_v:>12.2f} {smart_v:>12.2f} {arrow}{delta:>+8.2f}")
                    
                    print(f"\n      🛡️ Smart Exit: {s.get('n_stop_loss',0)} stop-loss, {s.get('n_trailing_stop',0)} trailing stop")
                    print(f"\n      ORIGINAL:")
                    print(format_backtest_report(o))
                    print(f"\n      SMART EXIT:")
                    print(format_backtest_report(s))

                    all_backtests.append({
                        "model": model_name, "feature_set": feat_set, "mode": "original",
                        "total_return_pct": o["total_return_pct"],
                        "ann_return_pct": o["annualized_return_pct"],
                        "sharpe": o["sharpe_ratio"], "max_dd_pct": o["max_drawdown_pct"],
                        "profit_factor": o["profit_factor"], "win_rate_pct": o["win_rate_pct"],
                        "trades": o["total_trades"], "final_equity": o["final_equity"],
                        "avg_loss_pct": o["avg_loss_pct"], "max_loss_pct": o["max_loss_pct"],
                    })
                    all_backtests.append({
                        "model": model_name, "feature_set": feat_set, "mode": "smart_exit",
                        "total_return_pct": s["total_return_pct"],
                        "ann_return_pct": s["annualized_return_pct"],
                        "sharpe": s["sharpe_ratio"], "max_dd_pct": s["max_drawdown_pct"],
                        "profit_factor": s["profit_factor"], "win_rate_pct": s["win_rate_pct"],
                        "trades": s["total_trades"], "final_equity": s["final_equity"],
                        "avg_loss_pct": s["avg_loss_pct"], "max_loss_pct": s["max_loss_pct"],
                        "n_stop_loss": s.get("n_stop_loss", 0),
                        "n_trailing_stop": s.get("n_trailing_stop", 0),
                    })
            else:
                # Single mode (original or smart exit)
                se = smart_exit_config if args.smart_exit else None
                label = "[SMART] " if args.smart_exit else ""
                res = run_single_backtest(
                    splitter, engine, target_gen, raw_df, feature_cols, df,
                    model_name, args.capital, smart_exit=se, label=label
                )
                if res:
                    bt = res["bt"]
                    metrics = res["metrics"]
                    print(f"\n      {'─' * 50}")
                    print(format_backtest_report(bt))

                    all_backtests.append({
                        "model": model_name, "feature_set": feat_set,
                        "mode": "smart_exit" if args.smart_exit else "original",
                        "total_return_pct": bt["total_return_pct"],
                        "ann_return_pct": bt["annualized_return_pct"],
                        "buy_hold_pct": bt["buy_hold_return_pct"],
                        "excess_pct": bt["excess_return_pct"],
                        "sharpe": bt["sharpe_ratio"],
                        "max_dd_pct": bt["max_drawdown_pct"],
                        "profit_factor": bt["profit_factor"],
                        "win_rate_pct": bt["win_rate_pct"],
                        "f1_macro": metrics["f1_macro"],
                        "accuracy": metrics["accuracy"],
                        "trades": bt["total_trades"],
                        "final_equity": bt["final_equity"],
                        "profit_vnd": bt["final_equity"] - args.capital,
                        "n_stop_loss": bt.get("n_stop_loss", 0),
                        "n_trailing_stop": bt.get("n_trailing_stop", 0),
                    })

    # Final comparison
    if all_backtests:
        print("\n" + "=" * 110)
        print("🏆 BACKTEST COMPARISON")
        print("=" * 110)
        bt_df = pd.DataFrame(all_backtests).sort_values("total_return_pct", ascending=False)
        print(bt_df.to_string(index=False, float_format="%.2f"))
        print(f"\n💰 Vốn ban đầu: {args.capital:,.0f} VND")
        best = bt_df.iloc[0]
        print(f"🥇 Tốt nhất: {best['model']} ({best.get('mode','')}) | Return: {best['total_return_pct']:+.2f}%")
        print("=" * 110)

        # Save
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(out_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"backtest_smart_{ts}.csv" if (args.smart_exit or args.compare) else f"backtest_{ts}.csv"
        bt_df.to_csv(os.path.join(out_dir, fname), index=False)
        print(f"💾 Saved to results/{fname}")


if __name__ == "__main__":
    main()
