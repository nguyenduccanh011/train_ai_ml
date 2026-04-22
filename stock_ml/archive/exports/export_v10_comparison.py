"""
Export V10 ML vs Rule-Based comparison data for visualization.
Generates JSON files for each symbol with OHLCV, markers, trades, and stats.
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
from run_v10_compare import backtest_v10
from run_v9_compare import backtest_v9
from compare_rule_vs_model import backtest_rule, compute_macd


def make_markers(trades, method="v10"):
    """Create chart markers from trade list."""
    markers = []
    if method == "v10":
        buy_color, win_color, loss_color = "#9C27B0", "#4caf50", "#f44336"
        buy_shape, sell_shape = "arrowUp", "arrowDown"
        prefix = "V10"
    elif method == "v9":
        buy_color, win_color, loss_color = "#2196F3", "#4caf50", "#f44336"
        buy_shape, sell_shape = "arrowUp", "arrowDown"
        prefix = "V9"
    else:
        buy_color, win_color, loss_color = "#FF9800", "#8BC34A", "#E91E63"
        buy_shape, sell_shape = "circle", "circle"
        prefix = "Rule"

    for t in trades:
        entry_date = t.get("entry_date", "")
        exit_date = t.get("exit_date", "")
        pnl = t.get("pnl_pct", 0)
        reason = t.get("exit_reason", "")

        if entry_date:
            markers.append({
                "time": entry_date,
                "position": "belowBar",
                "color": buy_color,
                "shape": buy_shape,
                "text": f"{prefix} Buy",
                "size": 1,
                "method": method,
            })
        if exit_date:
            reason_tag = ""
            if reason == "stop_loss": reason_tag = " SL"
            elif reason == "trailing_stop": reason_tag = " TS"
            elif reason == "hard_stop": reason_tag = " HS"
            elif reason == "zombie_exit": reason_tag = " ZE"
            elif reason == "hybrid_exit": reason_tag = " HX"
            elif reason == "profit_lock": reason_tag = " PL"

            markers.append({
                "time": exit_date,
                "position": "aboveBar",
                "color": win_color if pnl >= 0 else loss_color,
                "shape": sell_shape,
                "text": f"{prefix} {pnl:+.1f}%{reason_tag}",
                "size": 1,
                "method": method,
            })
    return markers


def compute_stats(trades, label=""):
    if not trades:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl_pct": 0, "avg_pnl_pct": 0, "avg_hold": 0, "pf": 0, "version": label}
    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holds = [t.get("holding_days", 0) for t in trades]
    gl = abs(sum(losses))
    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_pnl_pct": round(sum(pnls), 1),
        "avg_pnl_pct": round(np.mean(pnls), 2),
        "avg_hold": round(np.mean(holds), 1) if holds else 0,
        "pf": round(sum(wins) / gl, 2) if gl > 0 else 99,
        "version": label,
    }


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization", "data")
    os.makedirs(viz_dir, exist_ok=True)

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

    # Collect V10 and V9 trades
    v10_by_sym = defaultdict(list)
    v9_by_sym = defaultdict(list)
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
            r9 = backtest_v9(y_pred, rets, sym_test, feature_cols)
            for t in r10["trades"]:
                t["symbol"] = sym
            for t in r9["trades"]:
                t["symbol"] = sym
            v10_by_sym[sym].extend(r10["trades"])
            v9_by_sym[sym].extend(r9["trades"])

    # Collect Rule-based trades
    rule_by_sym = defaultdict(list)
    for sym in symbols:
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])
        sym_test = sym_data[sym_data[date_col] >= "2020-01-01"].reset_index(drop=True)
        if len(sym_test) < 50:
            continue
        rule_by_sym[sym] = backtest_rule(sym_test)

    # Generate per-symbol JSON
    index_entries = []
    for sym in symbols:
        print(f"  Exporting {sym}...")
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])

        # OHLCV
        ohlcv = []
        for _, row in sym_data.iterrows():
            ohlcv.append({
                "time": str(row[date_col])[:10],
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]) if "volume" in row else 0,
            })

        v10_trades = v10_by_sym.get(sym, [])
        v9_trades = v9_by_sym.get(sym, [])
        rule_trades = rule_by_sym.get(sym, [])

        v10_markers = make_markers(v10_trades, "v10")
        v9_markers = make_markers(v9_trades, "v9")
        rule_markers = make_markers(rule_trades, "rule")

        v10_stats = compute_stats(v10_trades, "V10 ML")
        v9_stats = compute_stats(v9_trades, "V9 ML")
        rule_stats = compute_stats(rule_trades, "Rule-Based")

        sym_json = {
            "symbol": sym,
            "ohlcv": ohlcv,
            "v10_markers": v10_markers,
            "v9_markers": v9_markers,
            "rule_markers": rule_markers,
            "markers": v10_markers + rule_markers,  # default combined
            "v10_trades": [{k: v for k, v in t.items() if k in
                           ("entry_date", "exit_date", "pnl_pct", "holding_days", "exit_reason", "entry_trend", "quick_reentry")}
                          for t in v10_trades],
            "v9_trades": [{k: v for k, v in t.items() if k in
                          ("entry_date", "exit_date", "pnl_pct", "holding_days", "exit_reason")}
                         for t in v9_trades],
            "rule_trades": [{k: v for k, v in t.items() if k in
                            ("entry_date", "exit_date", "pnl_pct", "holding_days", "exit_reason")}
                           for t in rule_trades],
            "v10_stats": v10_stats,
            "v9_stats": v9_stats,
            "rule_stats": rule_stats,
            "stats": v10_stats,  # backward compat
        }

        out_path = os.path.join(viz_dir, f"{sym}.json")
        with open(out_path, "w") as f:
            json.dump(sym_json, f)

        index_entries.append({
            "symbol": sym,
            "v10_trades": v10_stats["total_trades"],
            "v10_wr": v10_stats["win_rate"],
            "v10_pnl": v10_stats["total_pnl_pct"],
            "v9_trades": v9_stats["total_trades"],
            "v9_wr": v9_stats["win_rate"],
            "v9_pnl": v9_stats["total_pnl_pct"],
            "rule_trades": rule_stats["total_trades"],
            "rule_pnl": rule_stats["total_pnl_pct"],
            "file": f"./data/{sym}.json",
        })

    # Sort by V10 total PnL descending
    index_entries.sort(key=lambda x: x["v10_pnl"], reverse=True)

    index_json = {
        "symbols": index_entries,
        "version": "V10_vs_V9_vs_Rule",
        "generated_at": datetime.now().isoformat(),
    }
    with open(os.path.join(viz_dir, "index.json"), "w") as f:
        json.dump(index_json, f, indent=2)

    print(f"\n✅ Exported {len(index_entries)} symbols to {viz_dir}")
    print(f"   Open visualization/index.html to view")


if __name__ == "__main__":
    main()
