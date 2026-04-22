"""
Export V9 ML + Rule-Based signals to visualization/data/ for comparison chart.
Rule: BUY when MACD_hist > 0 AND Close > MA20 AND Close > Open
      SELL when MACD_hist < 0 AND Close < MA20 AND Close < Open
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

    test_symbols = ["VND", "AAS", "BID", "SSI", "FPT", "ACB", "MBB", "DGC", "HPG", "TCB", "REE", "VNM"]
    symbols = [s for s in test_symbols if s in loader.symbols]

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    print(f"📊 Training V9 model on {len(symbols)} symbols...")

    # V9 trades
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
            r9 = backtest_v9(y_pred, rets, sym_test, feature_cols)
            for t in r9["trades"]:
                t["symbol"] = sym
            v9_by_sym[sym].extend(r9["trades"])

    # Rule-based trades
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

    # Stats helper
    def calc_stats(trades):
        if not trades:
            return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                    "total_pnl_pct": 0, "avg_pnl_pct": 0, "avg_hold": 0}
        n = len(trades)
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]
        pnls = [t["pnl_pct"] for t in trades]
        holds = [t.get("holding_days", 0) for t in trades]
        return {
            "total_trades": n, "wins": len(wins), "losses": len(losses),
            "win_rate": round(len(wins) / n * 100, 1),
            "total_pnl_pct": round(sum(pnls), 1),
            "avg_pnl_pct": round(np.mean(pnls), 2),
            "avg_hold": round(np.mean(holds), 1),
            "avg_win": round(np.mean([t["pnl_pct"] for t in wins]), 2) if wins else 0,
            "avg_loss": round(np.mean([t["pnl_pct"] for t in losses]), 2) if losses else 0,
        }

    # Export per-symbol JSON
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization", "data")
    os.makedirs(viz_dir, exist_ok=True)

    index_symbols = []

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

        # V9 markers
        v9_markers = []
        for t in v9_by_sym[sym]:
            ed = t.get("entry_date", "")
            xd = t.get("exit_date", "")
            pnl = t.get("pnl_pct", 0)
            reason = t.get("exit_reason", "")
            hold = t.get("holding_days", 0)
            if ed:
                v9_markers.append({
                    "time": ed, "position": "belowBar", "color": "#2196F3",
                    "shape": "arrowUp", "text": f"V9 Buy", "size": 2,
                    "method": "V9"
                })
            if xd:
                c = "#4caf50" if pnl > 0 else "#f44336"
                v9_markers.append({
                    "time": xd, "position": "aboveBar", "color": c,
                    "shape": "arrowDown", "text": f"V9 {pnl:+.1f}%",
                    "size": 2, "method": "V9"
                })

        # Rule markers
        rule_markers = []
        for t in rule_by_sym[sym]:
            ed = t.get("entry_date", "")
            xd = t.get("exit_date", "")
            pnl = t.get("pnl_pct", 0)
            hold = t.get("holding_days", 0)
            if ed:
                rule_markers.append({
                    "time": ed, "position": "belowBar", "color": "#FF9800",
                    "shape": "circle", "text": f"Rule Buy", "size": 1,
                    "method": "Rule"
                })
            if xd:
                c = "#8BC34A" if pnl > 0 else "#E91E63"
                rule_markers.append({
                    "time": xd, "position": "aboveBar", "color": c,
                    "shape": "circle", "text": f"Rule {pnl:+.1f}%",
                    "size": 1, "method": "Rule"
                })

        v9_stats = calc_stats(v9_by_sym[sym])
        rule_stats = calc_stats(rule_by_sym[sym])

        # Format trades for table
        def fmt_trades(trades):
            return [{"entry_date": t.get("entry_date", ""), "exit_date": t.get("exit_date", ""),
                     "pnl_pct": t.get("pnl_pct", 0), "holding_days": t.get("holding_days", 0),
                     "exit_reason": t.get("exit_reason", "")} for t in trades]

        output = {
            "symbol": sym,
            "ohlcv": ohlcv,
            "markers": v9_markers,  # backward compat
            "v9_markers": v9_markers,
            "rule_markers": rule_markers,
            "v9_trades": fmt_trades(v9_by_sym[sym]),
            "rule_trades": fmt_trades(rule_by_sym[sym]),
            "stats": v9_stats,
            "v9_stats": v9_stats,
            "rule_stats": rule_stats,
        }

        out_path = os.path.join(viz_dir, f"{sym}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False)
        
        v9_pnl = v9_stats["total_pnl_pct"]
        rule_pnl = rule_stats["total_pnl_pct"]
        winner = "V9" if v9_pnl > rule_pnl else "Rule"
        print(f"  {sym}: V9={v9_stats['total_trades']}trades/{v9_pnl:+.1f}% | "
              f"Rule={rule_stats['total_trades']}trades/{rule_pnl:+.1f}% → {winner} wins")

        index_symbols.append({
            "symbol": sym,
            "trades": v9_stats["total_trades"],
            "win_rate": v9_stats["win_rate"],
            "total_pnl": v9_pnl,
            "rule_trades": rule_stats["total_trades"],
            "rule_pnl": rule_pnl,
            "file": f"./data/{sym}.json",
        })

    # Write index
    index = {
        "symbols": index_symbols,
        "stats": index_symbols,
        "version": "V9_vs_Rule",
        "generated_at": datetime.now().isoformat(),
    }
    idx_path = os.path.join(viz_dir, "index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Exported {len(symbols)} symbols to {viz_dir}")

    # Print aggregate comparison
    all_v9 = []
    all_rule = []
    for sym in symbols:
        all_v9.extend(v9_by_sym[sym])
        all_rule.extend(rule_by_sym[sym])

    v9_agg = calc_stats(all_v9)
    rule_agg = calc_stats(all_rule)

    print(f"\n{'='*80}")
    print(f"📊 AGGREGATE COMPARISON — {len(symbols)} symbols, 2020-2025")
    print(f"{'='*80}")
    print(f"  {'Metric':<20} {'V9 ML Model':>15} {'Rule-Based':>15} {'Winner':>10}")
    print(f"  {'─'*60}")
    print(f"  {'Trades':<20} {v9_agg['total_trades']:>15} {rule_agg['total_trades']:>15}")
    wr_win = "V9 ✅" if v9_agg['win_rate'] > rule_agg['win_rate'] else "Rule ✅"
    print(f"  {'Win Rate':<20} {v9_agg['win_rate']:>14.1f}% {rule_agg['win_rate']:>14.1f}% {wr_win:>10}")
    avg_win = "V9 ✅" if v9_agg['avg_pnl_pct'] > rule_agg['avg_pnl_pct'] else "Rule ✅"
    print(f"  {'Avg PnL':<20} {v9_agg['avg_pnl_pct']:>+14.2f}% {rule_agg['avg_pnl_pct']:>+14.2f}% {avg_win:>10}")
    tot_win = "V9 ✅" if v9_agg['total_pnl_pct'] > rule_agg['total_pnl_pct'] else "Rule ✅"
    print(f"  {'Total PnL':<20} {v9_agg['total_pnl_pct']:>+14.1f}% {rule_agg['total_pnl_pct']:>+14.1f}% {tot_win:>10}")
    print(f"  {'Avg Hold':<20} {v9_agg['avg_hold']:>14.1f}d {rule_agg['avg_hold']:>14.1f}d")
    print(f"  {'Avg Win':<20} {v9_agg['avg_win']:>+14.2f}% {rule_agg['avg_win']:>+14.2f}%")
    print(f"  {'Avg Loss':<20} {v9_agg['avg_loss']:>+14.2f}% {rule_agg['avg_loss']:>+14.2f}%")


if __name__ == "__main__":
    main()
