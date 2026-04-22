"""
Export V8b backtest trades + OHLCV data as JSON for frontend visualization.
Ranks symbols by performance and exports top performers.
"""
import sys, os, json, numpy as np, pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v7_compare import backtest_v7, summarize
from run_v8b_compare import backtest_v8b
from run_v9_compare import backtest_v9


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", type=str, default="ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC")
    parser.add_argument("--model", type=str, default="lightgbm")
    parser.add_argument("--top", type=int, default=10, help="Number of top symbols to export")
    args = parser.parse_args()

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

    pick = [s.strip() for s in args.pick.split(",")]
    symbols = [s for s in pick if s in loader.symbols]

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    # Collect V8b trades per symbol
    trades_by_sym = defaultdict(list)

    for window, train_df, test_df in splitter.split(df):
        model = build_model(args.model)
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

            r9 = backtest_v9(y_pred, rets, sym_test, feature_cols)

            for t in r9["trades"]:
                t["symbol"] = sym
                t["window"] = window.label
            trades_by_sym[sym].extend(r9["trades"])

    # Rank symbols by total PnL
    sym_stats = []
    for sym, trades in trades_by_sym.items():
        total_pnl = sum(t["pnl_pct"] for t in trades)
        n_trades = len(trades)
        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        wr = wins / max(n_trades, 1) * 100
        avg_pnl = np.mean([t["pnl_pct"] for t in trades]) if trades else 0
        sym_stats.append({
            "symbol": sym,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "win_rate": round(wr, 1),
            "trades": n_trades,
        })

    sym_stats.sort(key=lambda x: x["total_pnl"], reverse=True)
    top_symbols = [s["symbol"] for s in sym_stats[:args.top]]

    print(f"\n📊 Symbol Rankings (V8b):")
    print(f"{'Rank':<5} {'Symbol':<8} {'Trades':>6} {'WR':>6} {'Avg PnL':>8} {'Total PnL':>10}")
    print("-" * 50)
    for i, s in enumerate(sym_stats):
        marker = " ⭐" if s["symbol"] in top_symbols else ""
        print(f"{i+1:<5} {s['symbol']:<8} {s['trades']:>6} {s['win_rate']:>5.1f}% {s['avg_pnl']:>+7.2f}% {s['total_pnl']:>+9.2f}%{marker}")

    # Load OHLCV for top symbols and build JSON
    output = {
        "generated_at": datetime.now().isoformat(),
        "model": "V9 (LightGBM, EMA 5/20)",
        "rankings": sym_stats,
        "symbols": {}
    }

    for sym in top_symbols:
        # Load raw OHLCV
        sym_raw = loader.load_symbol(sym)
        date_col = "timestamp" if "timestamp" in sym_raw.columns else "date"
        sym_raw = sym_raw.sort_values(date_col).reset_index(drop=True)

        ohlcv = []
        for _, row in sym_raw.iterrows():
            d = str(row[date_col])[:10]
            ohlcv.append({
                "time": d,
                "open": round(float(row["open"]), 2) if pd.notna(row.get("open")) else 0,
                "high": round(float(row["high"]), 2) if pd.notna(row.get("high")) else 0,
                "low": round(float(row["low"]), 2) if pd.notna(row.get("low")) else 0,
                "close": round(float(row["close"]), 2) if pd.notna(row.get("close")) else 0,
                "volume": int(row["volume"]) if pd.notna(row.get("volume")) else 0,
            })

        # Format trades as markers
        sym_trades = trades_by_sym[sym]
        markers = []
        for t in sym_trades:
            entry_date = t.get("entry_date", "")
            exit_date = t.get("exit_date", "")
            pnl = t.get("pnl_pct", 0)
            reason = t.get("exit_reason", "")
            hold = t.get("holding_days", 0)

            if entry_date:
                markers.append({
                    "type": "buy",
                    "time": entry_date,
                    "text": f"BUY",
                    "tooltip": f"Entry | Hold: {hold}d | PnL: {pnl:+.1f}%"
                })
            if exit_date:
                marker_type = "sell"
                color = "#4caf50" if pnl > 0 else "#ff5252"
                markers.append({
                    "type": "sell",
                    "time": exit_date,
                    "text": f"{pnl:+.1f}%",
                    "color": color,
                    "tooltip": f"Exit ({reason}) | {pnl:+.1f}% | {hold}d"
                })

        output["symbols"][sym] = {
            "ohlcv": ohlcv,
            "trades": [{
                "entry_date": t.get("entry_date", ""),
                "exit_date": t.get("exit_date", ""),
                "pnl_pct": t.get("pnl_pct", 0),
                "holding_days": t.get("holding_days", 0),
                "exit_reason": t.get("exit_reason", ""),
            } for t in sym_trades],
            "markers": markers,
            "stats": next((s for s in sym_stats if s["symbol"] == sym), {}),
        }

    # Save JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "New folder (7)", "stock_signals.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Exported {len(top_symbols)} symbols to: {out_path}")
    print(f"   Top symbols: {', '.join(top_symbols)}")


if __name__ == "__main__":
    main()
