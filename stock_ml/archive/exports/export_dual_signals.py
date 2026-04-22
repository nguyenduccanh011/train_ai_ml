"""
Export both Rule-based and V9 ML trades + OHLCV for frontend comparison.
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
from run_v9_compare import backtest_v9
from compare_rule_vs_model import backtest_rule, compute_macd


def main():
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

    test_symbols = ["VND", "AAS", "BID", "SSI", "FPT", "ACB", "MBB", "DGC", "HPG", "TCB"]
    symbols = [s for s in test_symbols if s in loader.symbols]

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    # V9 trades
    v9_by_sym = defaultdict(list)
    for window, train_df, test_df in splitter.split(df):
        model = build_model("lightgbm")
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)
        for sym in test_df["symbol"].unique():
            if sym not in symbols: continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10: continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values if "return_1d" in sym_test.columns else np.zeros(len(sym_test))
            r9 = backtest_v9(y_pred, rets, sym_test, feature_cols)
            for t in r9["trades"]: t["symbol"] = sym
            v9_by_sym[sym].extend(r9["trades"])

    # Rule-based trades
    rule_by_sym = defaultdict(list)
    for sym in symbols:
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])
        sym_test = sym_data[sym_data[date_col] >= "2020-01-01"].reset_index(drop=True)
        if len(sym_test) < 50: continue
        rule_by_sym[sym] = backtest_rule(sym_test)

    # Build output
    def calc_stats(trades, label):
        if not trades: return {"label": label, "trades": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}
        n = len(trades)
        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        return {
            "label": label, "trades": n,
            "win_rate": round(wins / n * 100, 1),
            "avg_pnl": round(np.mean([t["pnl_pct"] for t in trades]), 2),
            "total_pnl": round(sum(t["pnl_pct"] for t in trades), 2),
        }

    rankings = []
    for sym in symbols:
        v9s = calc_stats(v9_by_sym[sym], "V9")
        rs = calc_stats(rule_by_sym[sym], "Rule")
        rankings.append({
            "symbol": sym,
            "v9_total_pnl": v9s["total_pnl"], "v9_avg_pnl": v9s["avg_pnl"],
            "v9_wr": v9s["win_rate"], "v9_trades": v9s["trades"],
            "rule_total_pnl": rs["total_pnl"], "rule_avg_pnl": rs["avg_pnl"],
            "rule_wr": rs["win_rate"], "rule_trades": rs["trades"],
            "total_pnl": v9s["total_pnl"],  # sort by V9
        })
    rankings.sort(key=lambda x: x["total_pnl"], reverse=True)

    output = {
        "generated_at": datetime.now().isoformat(),
        "model": "V9 ML vs Rule-Based",
        "rankings": rankings,
        "symbols": {},
    }

    for sym in symbols:
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

        def make_markers(trades, method):
            markers = []
            for t in trades:
                entry_date = t.get("entry_date", "")
                exit_date = t.get("exit_date", "")
                pnl = t.get("pnl_pct", 0)
                reason = t.get("exit_reason", "")
                hold = t.get("holding_days", 0)
                if entry_date:
                    markers.append({"type": "buy", "method": method, "time": entry_date,
                                    "text": f"{method} BUY",
                                    "tooltip": f"{method} Entry | Hold:{hold}d | PnL:{pnl:+.1f}%"})
                if exit_date:
                    markers.append({"type": "sell", "method": method, "time": exit_date,
                                    "text": f"{pnl:+.1f}%",
                                    "color": "#4caf50" if pnl > 0 else "#ff5252",
                                    "tooltip": f"{method} Exit ({reason}) | {pnl:+.1f}% | {hold}d"})
            return markers

        def format_trades(trades):
            return [{"entry_date": t.get("entry_date",""), "exit_date": t.get("exit_date",""),
                     "pnl_pct": t.get("pnl_pct",0), "holding_days": t.get("holding_days",0),
                     "exit_reason": t.get("exit_reason","")} for t in trades]

        output["symbols"][sym] = {
            "ohlcv": ohlcv,
            "v9_trades": format_trades(v9_by_sym[sym]),
            "v9_markers": make_markers(v9_by_sym[sym], "V9"),
            "rule_trades": format_trades(rule_by_sym[sym]),
            "rule_markers": make_markers(rule_by_sym[sym], "Rule"),
            "v9_stats": calc_stats(v9_by_sym[sym], "V9"),
            "rule_stats": calc_stats(rule_by_sym[sym], "Rule"),
        }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "New folder (7)", "stock_signals.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Exported {len(symbols)} symbols with dual signals to: {out_path}")


if __name__ == "__main__":
    main()
