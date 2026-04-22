"""Generate visualization data with V7 backtest trades for all symbols."""
import sys, os, json, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v7_compare import backtest_v7


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

    # Load all symbols that have existing viz data
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization", "data")
    existing = [f.replace(".json", "") for f in os.listdir(viz_dir) if f.endswith(".json") and f != "index.json"]
    symbols = [s for s in existing if s in loader.symbols]
    print(f"Processing {len(symbols)} symbols: {symbols}")

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    # Collect trades per symbol across all windows
    all_trades = {s: [] for s in symbols}

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

            result = backtest_v7(y_pred, rets, sym_test, feature_cols)
            all_trades[sym].extend(result["trades"])

    # Now update each symbol's JSON
    index_data = []
    for sym in symbols:
        json_path = os.path.join(viz_dir, f"{sym}.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        # Build date->close lookup from ohlcv
        close_map = {c["time"]: c["close"] for c in data["ohlcv"]}

        trades = all_trades.get(sym, [])
        markers = []
        wins, losses = 0, 0
        total_pnl = 0

        for t in trades:
            entry_date = str(t.get("entry_date", ""))[:10]
            exit_date = str(t.get("exit_date", ""))[:10]
            score = t.get("entry_score", 0)
            exit_reason = t.get("exit_reason", "signal")
            hold = t.get("holding_days", 0)

            if not entry_date or not exit_date:
                continue

            # Calculate REAL PnL from actual close prices
            entry_close = close_map.get(entry_date, 0)
            exit_close = close_map.get(exit_date, 0)
            if entry_close > 0 and exit_close > 0:
                pnl = (exit_close / entry_close - 1) * 100
            else:
                pnl = t.get("pnl_pct", 0)  # fallback

            # Buy marker
            markers.append({
                "time": entry_date,
                "position": "belowBar",
                "color": "#2196F3",
                "shape": "arrowUp",
                "text": f"Buy @{entry_close:.2f} (s={score})",
                "size": 2
            })

            # Sell marker
            if pnl > 0:
                color = "#4caf50"  # green
                wins += 1
            else:
                color = "#f44336"  # red
                losses += 1
            total_pnl += pnl

            reason_tag = ""
            if exit_reason == "stop_loss":
                reason_tag = " SL"
            elif exit_reason == "trailing_stop":
                reason_tag = " TS"
            elif exit_reason == "zombie_exit":
                reason_tag = " ZE"

            markers.append({
                "time": exit_date,
                "position": "aboveBar",
                "color": color,
                "shape": "arrowDown",
                "text": f"Exit @{exit_close:.2f} {pnl:+.1f}%{reason_tag} ({hold}d)",
                "size": 2
            })

        n_trades = wins + losses
        data["markers"] = markers
        data["stats"] = {
            "total_trades": n_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(n_trades, 1) * 100, 1),
            "total_pnl_pct": round(total_pnl, 1),
            "avg_pnl_pct": round(total_pnl / max(n_trades, 1), 2),
            "version": "V7"
        }

        with open(json_path, "w") as f:
            json.dump(data, f)

        index_data.append({
            "symbol": sym,
            "trades": n_trades,
            "win_rate": data["stats"]["win_rate"],
            "total_pnl": data["stats"]["total_pnl_pct"],
        })
        print(f"  {sym}: {n_trades} trades, WR {data['stats']['win_rate']}%, PnL {data['stats']['total_pnl_pct']:+.1f}%")

    # Update index (with file field for HTML)
    for d in index_data:
        d["file"] = f"./data/{d['symbol']}.json"
    with open(os.path.join(viz_dir, "index.json"), "w") as f:
        json.dump({"symbols": index_data, "stats": index_data, "version": "V7"}, f)

    print(f"\n✅ Updated {len(symbols)} symbol charts with V7 trades")


if __name__ == "__main__":
    main()
