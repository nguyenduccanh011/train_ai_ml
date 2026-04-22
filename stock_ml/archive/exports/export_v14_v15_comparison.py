"""
Export V15 ML vs V14 ML vs V11 ML vs Rule-Based comparison data for visualization.
Replaces V12/V10 with V14/V15. Generates JSON files for each symbol with OHLCV, markers, trades, and stats.
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
from run_v11_compare import backtest_v11
from run_v14_compare import backtest_v14
from run_v15_compare import backtest_v15
from compare_rule_vs_model import backtest_rule


def make_markers(trades, method="v15"):
    """Create chart markers from trade list."""
    markers = []
    colors = {
        "v15": ("#FF5722", "#4caf50", "#f44336", "arrowUp", "arrowDown", "V15"),
        "v14": ("#00BCD4", "#4caf50", "#f44336", "arrowUp", "arrowDown", "V14"),
        "v11": ("#9C27B0", "#4caf50", "#f44336", "arrowUp", "arrowDown", "V11"),
        "rule": ("#FF9800", "#8BC34A", "#E91E63", "circle", "circle", "Rule"),
    }
    buy_color, win_color, loss_color, buy_shape, sell_shape, prefix = colors.get(method, colors["rule"])

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
            reason_tags = {
                "stop_loss": " SL", "trailing_stop": " TS", "hard_stop": " HS",
                "zombie_exit": " ZE", "hybrid_exit": " HX", "profit_lock": " PL",
                "profit_floor": " PF", "time_decay": " TD",
                "profit_peak_sma10": " PP", "profit_peak_ema8": " PE",
            }
            reason_tag = reason_tags.get(reason, "")
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
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl_pct": 0, "avg_pnl_pct": 0, "avg_hold": 0, "pf": 0,
            "version": label,
        }
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
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "portable_data", "vn_stock_ai_dataset_cleaned"
    )
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization", "data")
    os.makedirs(viz_dir, exist_ok=True)

    config = {
        "data": {"data_dir": data_dir},
        "split": {
            "method": "walk_forward",
            "train_years": 4, "test_years": 1,
            "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025,
        },
        "target": {
            "type": "trend_regime", "trend_method": "dual_ma",
            "short_window": 5, "long_window": 20, "classes": 3,
        },
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

    v15_by_sym = defaultdict(list)
    v14_by_sym = defaultdict(list)
    v11_by_sym = defaultdict(list)

    print("=" * 70)
    print("RUNNING V15 vs V14 vs V11 vs Rule-Based COMPARISON")
    print("=" * 70)
    print("\nRunning walk-forward backtests...")
    
    for fold_idx, (_, train_df, test_df) in enumerate(splitter.split(df)):
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

            r15 = backtest_v15(y_pred, rets, sym_test, feature_cols)
            r14 = backtest_v14(y_pred, rets, sym_test, feature_cols)
            r11 = backtest_v11(y_pred, rets, sym_test, feature_cols)
            for t in r15["trades"]:
                t["symbol"] = sym
            for t in r14["trades"]:
                t["symbol"] = sym
            for t in r11["trades"]:
                t["symbol"] = sym
            v15_by_sym[sym].extend(r15["trades"])
            v14_by_sym[sym].extend(r14["trades"])
            v11_by_sym[sym].extend(r11["trades"])

        print(f"  Fold {fold_idx+1} done")

    # Rule-based
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

    # Print summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON: V15 vs V14 vs V11 (Baseline)")
    print("=" * 70)
    
    total_v15_pnl = 0
    total_v14_pnl = 0
    total_v11_pnl = 0
    total_rule_pnl = 0
    
    print(f"\n{'Symbol':<8} {'V15 PnL':>10} {'V14 PnL':>10} {'V11 PnL':>10} {'Rule PnL':>10} {'V15 WR':>8} {'V14 WR':>8} {'V11 WR':>8} {'V15 #':>6} {'V14 #':>6} {'V11 #':>6}")
    print("-" * 100)
    
    for sym in symbols:
        s15 = compute_stats(v15_by_sym.get(sym, []), "V15")
        s14 = compute_stats(v14_by_sym.get(sym, []), "V14")
        s11 = compute_stats(v11_by_sym.get(sym, []), "V11")
        sr = compute_stats(rule_by_sym.get(sym, []), "Rule")
        
        total_v15_pnl += s15["total_pnl_pct"]
        total_v14_pnl += s14["total_pnl_pct"]
        total_v11_pnl += s11["total_pnl_pct"]
        total_rule_pnl += sr["total_pnl_pct"]
        
        best = max(s15["total_pnl_pct"], s14["total_pnl_pct"], s11["total_pnl_pct"])
        v15_mark = " *" if s15["total_pnl_pct"] == best else ""
        v14_mark = " *" if s14["total_pnl_pct"] == best else ""
        v11_mark = " *" if s11["total_pnl_pct"] == best else ""
        
        print(f"{sym:<8} {s15['total_pnl_pct']:>+9.1f}%{v15_mark} {s14['total_pnl_pct']:>+9.1f}%{v14_mark} {s11['total_pnl_pct']:>+9.1f}%{v11_mark} {sr['total_pnl_pct']:>+9.1f}% {s15['win_rate']:>7.1f}% {s14['win_rate']:>7.1f}% {s11['win_rate']:>7.1f}% {s15['total_trades']:>5} {s14['total_trades']:>5} {s11['total_trades']:>5}")
    
    print("-" * 100)
    print(f"{'TOTAL':<8} {total_v15_pnl:>+9.1f}%  {total_v14_pnl:>+9.1f}%  {total_v11_pnl:>+9.1f}%  {total_rule_pnl:>+9.1f}%")
    print(f"\nV15 vs V11: {total_v15_pnl - total_v11_pnl:>+.1f}% difference")
    print(f"V14 vs V11: {total_v14_pnl - total_v11_pnl:>+.1f}% difference")
    print(f"V15 vs V14: {total_v15_pnl - total_v14_pnl:>+.1f}% difference")

    # Export JSON files
    print("\nExporting visualization data...")
    index_entries = []
    for sym in symbols:
        print(f"  Exporting {sym}...")
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])

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

        v15_trades = v15_by_sym.get(sym, [])
        v14_trades = v14_by_sym.get(sym, [])
        v11_trades = v11_by_sym.get(sym, [])
        rule_trades = rule_by_sym.get(sym, [])

        v15_markers = make_markers(v15_trades, "v15")
        v14_markers = make_markers(v14_trades, "v14")
        v11_markers = make_markers(v11_trades, "v11")
        rule_markers = make_markers(rule_trades, "rule")

        v15_stats = compute_stats(v15_trades, "V15 ML")
        v14_stats = compute_stats(v14_trades, "V14 ML")
        v11_stats = compute_stats(v11_trades, "V11 ML")
        rule_stats = compute_stats(rule_trades, "Rule-Based")

        trade_keys = ("entry_date", "exit_date", "pnl_pct", "holding_days", "exit_reason",
                      "entry_trend", "quick_reentry", "breakout_entry", "vshape_entry", "secondary_breakout")

        sym_json = {
            "symbol": sym,
            "ohlcv": ohlcv,
            "v15_markers": v15_markers,
            "v14_markers": v14_markers,
            "v11_markers": v11_markers,
            "rule_markers": rule_markers,
            "markers": v15_markers + v14_markers + v11_markers + rule_markers,
            "v15_trades": [{k: v for k, v in t.items() if k in trade_keys} for t in v15_trades],
            "v14_trades": [{k: v for k, v in t.items() if k in trade_keys} for t in v14_trades],
            "v11_trades": [{k: v for k, v in t.items() if k in trade_keys} for t in v11_trades],
            "rule_trades": [{k: v for k, v in t.items() if k in ("entry_date", "exit_date", "pnl_pct", "holding_days", "exit_reason")} for t in rule_trades],
            "v15_stats": v15_stats,
            "v14_stats": v14_stats,
            "v11_stats": v11_stats,
            "rule_stats": rule_stats,
            "stats": v15_stats,
        }

        out_path = os.path.join(viz_dir, f"{sym}.json")
        with open(out_path, "w") as f:
            json.dump(sym_json, f)

        index_entries.append({
            "symbol": sym,
            "v15_trades": v15_stats["total_trades"],
            "v15_wr": v15_stats["win_rate"],
            "v15_pnl": v15_stats["total_pnl_pct"],
            "v14_trades": v14_stats["total_trades"],
            "v14_wr": v14_stats["win_rate"],
            "v14_pnl": v14_stats["total_pnl_pct"],
            "v11_trades": v11_stats["total_trades"],
            "v11_wr": v11_stats["win_rate"],
            "v11_pnl": v11_stats["total_pnl_pct"],
            "rule_trades": rule_stats["total_trades"],
            "rule_pnl": rule_stats["total_pnl_pct"],
            "file": f"./data/{sym}.json",
        })

    index_entries.sort(key=lambda x: x["v15_pnl"], reverse=True)

    index_json = {
        "symbols": index_entries,
        "version": "V15_vs_V14_vs_V11_vs_Rule",
        "generated_at": datetime.now().isoformat(),
    }
    with open(os.path.join(viz_dir, "index.json"), "w") as f:
        json.dump(index_json, f, indent=2)

    print(f"\nExported {len(index_entries)} symbols to {viz_dir}")
    print("Open visualization/index.html to view")


if __name__ == "__main__":
    main()
