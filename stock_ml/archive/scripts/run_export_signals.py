"""
Export OHLCV + V6 trade signals as JSON for chart visualization.
Runs V6 backtest on all symbols, exports per-symbol JSON files.

Usage:
    python run_export_signals.py                # 5 symbols
    python run_export_signals.py --symbols 20   # 20 symbols
    python run_export_signals.py --full         # All symbols
    python run_export_signals.py --model lightgbm
"""
import argparse, sys, os, json, numpy as np, pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v6_backtest import backtest_v6


def main():
    parser = argparse.ArgumentParser(description="Export OHLCV + trade signals for visualization")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--symbols", type=int, default=5)
    parser.add_argument("--pick", type=str, default="", help="Comma-separated symbol list, e.g. VNM,FPT,HPG")
    parser.add_argument("--model", type=str, default="lightgbm")
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

    if args.pick:
        symbols = [s.strip() for s in args.pick.split(",") if s.strip() in loader.symbols]
    elif args.full:
        symbols = loader.symbols
    else:
        symbols = loader.symbols[:args.symbols]

    print(f"📊 Exporting signals for {len(symbols)} symbols, model={args.model}")

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization", "data")
    os.makedirs(out_dir, exist_ok=True)

    # Collect all trades — run backtest PER SYMBOL to get correct per-stock PnL
    all_trades = []

    for window, train_df, test_df in splitter.split(df):
        try:
            model = build_model(args.model)
            X_train = np.nan_to_num(train_df[feature_cols].values)
            y_train = train_df["target"].values.astype(int)

            offset = 0
            if args.model == "xgboost" and y_train.min() < 0:
                offset = abs(y_train.min())
                y_train += offset

            model.fit(X_train, y_train)

            # Run backtest per symbol for accurate per-stock PnL
            window_trades = 0
            for sym in test_df["symbol"].unique():
                sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
                if len(sym_test) < 10:
                    continue

                X_test_sym = np.nan_to_num(sym_test[feature_cols].values)
                y_pred_sym = model.predict(X_test_sym)
                if offset > 0:
                    y_pred_sym -= offset

                rets_sym = sym_test["return_1d"].values if "return_1d" in sym_test.columns else np.zeros(len(sym_test))

                r = backtest_v6(y_pred_sym, rets_sym, sym_test, feature_cols, args.capital)
                for t in r["trades"]:
                    t["window"] = window.label
                    t["entry_symbol"] = sym  # ensure symbol is set
                all_trades.extend(r["trades"])
                window_trades += len(r["trades"])

            print(f"   {window.label}: {window_trades} trades")

        except Exception as e:
            print(f"   {window.label}: ❌ {e}")

    # Group trades by symbol
    trades_by_symbol = {}
    for t in all_trades:
        sym = t.get("entry_symbol", "?")
        if sym not in trades_by_symbol:
            trades_by_symbol[sym] = []
        trades_by_symbol[sym].append(t)

    # Export per-symbol OHLCV + markers
    symbol_list = []
    for sym in sorted(trades_by_symbol.keys()):
        if sym == "?":
            continue
        # Column might be "date" or "timestamp"
        date_col = "date" if "date" in raw_df.columns else "timestamp"
        sym_df = raw_df[raw_df["symbol"] == sym].sort_values(date_col).copy()
        if len(sym_df) == 0:
            continue

        # OHLCV data
        ohlcv = []
        for _, row in sym_df.iterrows():
            d = str(row[date_col])[:10]
            ohlcv.append({
                "time": d,
                "open": round(float(row["open"]), 2) if "open" in row else 0,
                "high": round(float(row["high"]), 2) if "high" in row else 0,
                "low": round(float(row["low"]), 2) if "low" in row else 0,
                "close": round(float(row["close"]), 2) if "close" in row else 0,
                "volume": int(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else 0,
            })

        # Build date->close lookup from raw OHLCV
        date_to_close = {}
        for _, row in sym_df.iterrows():
            d = str(row[date_col])[:10]
            date_to_close[d] = float(row["close"])

        # Trade markers — recalculate PnL from actual close prices
        markers = []
        trades = trades_by_symbol[sym]
        for t in trades:
            entry_date = t.get("entry_date", "")
            exit_date = t.get("exit_date", "")
            exit_reason = t.get("exit_reason", "signal")

            # Recalculate PnL from actual chart prices
            entry_close = date_to_close.get(entry_date, 0)
            exit_close = date_to_close.get(exit_date, 0)
            if entry_close > 0 and exit_close > 0:
                pnl = (exit_close - entry_close) / entry_close * 100
            else:
                pnl = t.get("pnl_pct", 0)
            t["chart_pnl_pct"] = round(pnl, 1)  # store corrected PnL

            # Buy marker
            if entry_date:
                score = t.get("entry_score", 0)
                markers.append({
                    "time": entry_date,
                    "position": "belowBar",
                    "color": "#2196F3",
                    "shape": "arrowUp",
                    "text": f"Buy (s={score})",
                    "size": 2,
                })

            # Sell marker with exit reason color
            if exit_date and exit_reason != "end":
                if exit_reason == "stop_loss":
                    color, shape, label = "#ff5252", "circle", f"SL {pnl:+.1f}%"
                elif exit_reason == "trailing_stop":
                    color, shape, label = "#ff9800", "arrowDown", f"Trail {pnl:+.1f}%"
                elif exit_reason == "zombie_exit":
                    color, shape, label = "#9e9e9e", "square", f"Zombie {pnl:+.1f}%"
                else:
                    color = "#4caf50" if pnl > 0 else "#e91e63"
                    shape = "arrowDown"
                    label = f"Exit {pnl:+.1f}%"

                markers.append({
                    "time": exit_date,
                    "position": "aboveBar",
                    "color": color,
                    "shape": shape,
                    "text": label,
                    "size": 2,
                })

        n_trades = len(trades)
        wins = sum(1 for t in trades if t.get("chart_pnl_pct", 0) > 0)
        total_pnl = sum(t.get("chart_pnl_pct", 0) for t in trades)

        symbol_data = {
            "symbol": sym,
            "ohlcv": ohlcv,
            "markers": markers,
            "stats": {
                "total_trades": n_trades,
                "wins": wins,
                "losses": n_trades - wins,
                "win_rate": round(wins / n_trades * 100, 1) if n_trades > 0 else 0,
                "total_pnl_pct": round(total_pnl, 2),
                "avg_pnl_pct": round(total_pnl / n_trades, 2) if n_trades > 0 else 0,
            },
        }

        fname = f"{sym}.json"
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(symbol_data, f)

        symbol_list.append({
            "symbol": sym,
            "file": f"data/{fname}",
            "trades": n_trades,
            "win_rate": symbol_data["stats"]["win_rate"],
            "total_pnl": symbol_data["stats"]["total_pnl_pct"],
        })
        print(f"   ✅ {sym}: {n_trades} trades, WR={symbol_data['stats']['win_rate']:.1f}%, PnL={total_pnl:+.1f}%")

    # Save index
    index = {
        "model": args.model,
        "generated": datetime.now().isoformat(),
        "symbols": symbol_list,
    }
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n💾 Exported {len(symbol_list)} symbols to visualization/data/")
    print(f"   Open visualization/index.html in browser to view charts")


if __name__ == "__main__":
    main()
