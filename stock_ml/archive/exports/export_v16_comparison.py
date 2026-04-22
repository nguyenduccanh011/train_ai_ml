"""
Export V11 + V14 (A+B) + V15 (A+B+E) + V16 (A+B+E+F+G) trades for visualization.
Generates JSON files per symbol with OHLCV, markers, trades, stats.
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
from run_v16_compare import backtest_v16
from compare_rule_vs_model import backtest_rule


COLORS = {
    "v16": "#FFEB3B",  # yellow — newest/best
    "v15": "#FF5722",  # orange-red
    "v14": "#00BCD4",  # cyan
    "v11": "#9C27B0",  # purple
    "rule": "#FF9800",
}


def make_markers(trades, method):
    markers = []
    buy_color = COLORS[method]
    win_color, loss_color = "#4caf50", "#f44336"
    if method == "rule":
        win_color, loss_color = "#8BC34A", "#E91E63"
        buy_shape, sell_shape = "circle", "circle"
    else:
        buy_shape, sell_shape = "arrowUp", "arrowDown"
    prefix = method.upper()

    short = {
        "stop_loss": "SL", "trailing_stop": "TS", "hard_stop": "HS",
        "zombie_exit": "ZE", "hybrid_exit": "HX", "profit_lock": "PL",
        "profit_floor": "PF", "peak_protect_dist": "PP", "peak_protect_ema": "PE",
        "fast_loss_cut": "FC", "signal": "", "end": "END",
    }

    for t in trades:
        ed, xd = t.get("entry_date", ""), t.get("exit_date", "")
        pnl = t.get("pnl_pct", 0)
        reason = t.get("exit_reason", "")
        tag = short.get(reason, reason[:2].upper() if reason else "")
        if ed:
            markers.append({"time": ed, "position": "belowBar", "color": buy_color,
                           "shape": buy_shape, "text": f"{prefix} Buy", "size": 1, "method": method})
        if xd:
            markers.append({"time": xd, "position": "aboveBar",
                           "color": win_color if pnl >= 0 else loss_color,
                           "shape": sell_shape, "text": f"{prefix} {pnl:+.1f}%{(' '+tag) if tag else ''}",
                           "size": 1, "method": method})
    return markers


def compute_stats(trades, label):
    if not trades:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl_pct": 0, "avg_pnl_pct": 0, "avg_hold": 0, "pf": 0, "version": label}
    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holds = [t.get("holding_days", 0) for t in trades]
    gl = abs(sum(losses))
    return {
        "total_trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_pnl_pct": round(sum(pnls), 1),
        "avg_pnl_pct": round(np.mean(pnls), 2),
        "avg_hold": round(np.mean(holds), 1) if holds else 0,
        "pf": round(sum(wins) / gl, 2) if gl > 0 else 99,
        "version": label,
    }


def enrich_trades_with_prices(trades, close_by_date):
    enriched = []
    for t in trades:
        row = dict(t)
        entry_date = row.get("entry_date")
        exit_date = row.get("exit_date")
        entry_price = close_by_date.get(entry_date) if entry_date else None
        exit_price = close_by_date.get(exit_date) if exit_date else None
        row["entry_price"] = round(float(entry_price), 2) if entry_price is not None else None
        row["exit_price"] = round(float(exit_price), 2) if exit_price is not None else None
        if entry_price and exit_price:
            pnl_check = (exit_price / entry_price - 1) * 100
            row["pnl_check_pct"] = round(float(pnl_check), 2)
            row["pnl_gap_pct"] = round(float(row.get("pnl_pct", 0) - pnl_check), 2)
        else:
            row["pnl_check_pct"] = None
            row["pnl_gap_pct"] = None
        enriched.append(row)
    return enriched


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

    test_symbols = ["ACB", "FPT", "HPG", "SSI", "VND", "MBB", "TCB", "VNM",
                    "DGC", "AAS", "AAV", "REE", "BID", "VIC"]
    symbols = [s for s in test_symbols if s in loader.symbols]

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    v16_by, v15_by, v14_by, v11_by = (defaultdict(list) for _ in range(4))

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
            rets = sym_test["return_1d"].values

            r16 = backtest_v16(y_pred, rets, sym_test, feature_cols,
                               mod_a=True, mod_b=True, mod_c=False, mod_d=False,
                               mod_e=True, mod_f=True, mod_g=True)
            r15 = backtest_v15(y_pred, rets, sym_test, feature_cols,
                               mod_a=True, mod_b=True, mod_c=False, mod_d=False, mod_e=True)
            r14 = backtest_v14(y_pred, rets, sym_test, feature_cols, mod_a=True, mod_b=True)
            r11 = backtest_v11(y_pred, rets, sym_test, feature_cols)

            for t in r16["trades"]: t["symbol"] = sym
            for t in r15["trades"]: t["symbol"] = sym
            for t in r14["trades"]: t["symbol"] = sym
            for t in r11["trades"]: t["symbol"] = sym
            v16_by[sym].extend(r16["trades"])
            v15_by[sym].extend(r15["trades"])
            v14_by[sym].extend(r14["trades"])
            v11_by[sym].extend(r11["trades"])

    rule_by = defaultdict(list)
    for sym in symbols:
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])
        sym_test = sym_data[sym_data[date_col] >= "2020-01-01"].reset_index(drop=True)
        if len(sym_test) < 50:
            continue
        rule_by[sym] = backtest_rule(sym_test)

    fields = ("entry_date", "exit_date", "pnl_pct", "holding_days", "exit_reason",
              "entry_trend", "quick_reentry", "breakout_entry", "vshape_entry", "secondary_breakout")

    print("\n" + "=" * 116)
    print("DETAILED COMPARISON: V11 vs V14(A+B) vs V15(A+B+E) vs V16(A+B+E+F+G)")
    print("=" * 116)
    print(f"{'Symbol':<8} {'V11 PnL':>10} {'V14 PnL':>10} {'V15 PnL':>10} {'V16 PnL':>10} "
          f"{'V11 WR':>8} {'V14 WR':>8} {'V15 WR':>8} {'V16 WR':>8} {'Best':>8}")
    print("-" * 116)

    totals = {"v11": 0.0, "v14": 0.0, "v15": 0.0, "v16": 0.0, "rule": 0.0}
    sanity_issues = []

    index_entries = []
    for sym in symbols:
        print(f"  Exporting {sym}...")
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])

        ohlcv = []
        for _, row in sym_data.iterrows():
            ohlcv.append({"time": str(row[date_col])[:10],
                          "open": round(float(row["open"]), 2),
                          "high": round(float(row["high"]), 2),
                          "low": round(float(row["low"]), 2),
                          "close": round(float(row["close"]), 2),
                          "volume": int(row["volume"]) if "volume" in row else 0})
        close_by_date = {bar["time"]: bar["close"] for bar in ohlcv}

        v16t_raw, v15t_raw = v16_by.get(sym, []), v15_by.get(sym, [])
        v14t_raw, v11t_raw = v14_by.get(sym, []), v11_by.get(sym, [])
        rt_raw = rule_by.get(sym, [])

        v16t = enrich_trades_with_prices(v16t_raw, close_by_date)
        v15t = enrich_trades_with_prices(v15t_raw, close_by_date)
        v14t = enrich_trades_with_prices(v14t_raw, close_by_date)
        v11t = enrich_trades_with_prices(v11t_raw, close_by_date)
        rt = enrich_trades_with_prices(rt_raw, close_by_date)

        s16 = compute_stats(v16t, "V16 ML")
        s15 = compute_stats(v15t, "V15 ML")
        s14 = compute_stats(v14t, "V14 ML")
        s11 = compute_stats(v11t, "V11 ML")
        srule = compute_stats(rt, "Rule")

        totals["v11"] += s11["total_pnl_pct"]
        totals["v14"] += s14["total_pnl_pct"]
        totals["v15"] += s15["total_pnl_pct"]
        totals["v16"] += s16["total_pnl_pct"]
        totals["rule"] += srule["total_pnl_pct"]

        pnl_map = {
            "V11": s11["total_pnl_pct"],
            "V14": s14["total_pnl_pct"],
            "V15": s15["total_pnl_pct"],
            "V16": s16["total_pnl_pct"],
        }
        best_name = max(pnl_map, key=pnl_map.get)
        print(f"{sym:<8} {s11['total_pnl_pct']:>+9.1f}% {s14['total_pnl_pct']:>+9.1f}% "
              f"{s15['total_pnl_pct']:>+9.1f}% {s16['total_pnl_pct']:>+9.1f}% "
              f"{s11['win_rate']:>7.1f}% {s14['win_rate']:>7.1f}% {s15['win_rate']:>7.1f}% "
              f"{s16['win_rate']:>7.1f}% {best_name:>8}")

        for tag, trades in (("V11", v11t), ("V14", v14t), ("V15", v15t), ("V16", v16t), ("RULE", rt)):
            for tr in trades:
                gap = tr.get("pnl_gap_pct")
                if gap is not None and abs(gap) > 1.0:
                    sanity_issues.append(
                        f"{sym} {tag} {tr.get('entry_date')}->{tr.get('exit_date')} "
                        f"pnl={tr.get('pnl_pct')} check={tr.get('pnl_check_pct')} gap={gap}"
                    )

        sym_json = {
            "symbol": sym, "ohlcv": ohlcv,
            "v16_markers": make_markers(v16t, "v16"),
            "v15_markers": make_markers(v15t, "v15"),
            "v14_markers": make_markers(v14t, "v14"),
            "v11_markers": make_markers(v11t, "v11"),
            "rule_markers": make_markers(rt, "rule"),
            "v16_trades": [{k: v for k, v in t.items() if k in fields} for t in v16t],
            "v15_trades": [{k: v for k, v in t.items() if k in fields} for t in v15t],
            "v14_trades": [{k: v for k, v in t.items() if k in fields} for t in v14t],
            "v11_trades": [{k: v for k, v in t.items() if k in fields} for t in v11t],
            "rule_trades": [{k: v for k, v in t.items() if k in ("entry_date", "exit_date", "pnl_pct", "holding_days", "exit_reason")} for t in rt],
            "v16_stats": s16, "v15_stats": s15, "v14_stats": s14, "v11_stats": s11, "rule_stats": srule,
            "stats": s16,
        }

        with open(os.path.join(viz_dir, f"{sym}.json"), "w") as f:
            json.dump(sym_json, f)

        index_entries.append({
            "symbol": sym,
            "v16_trades": s16["total_trades"], "v16_pnl": s16["total_pnl_pct"], "v16_wr": s16["win_rate"],
            "v15_trades": s15["total_trades"], "v15_pnl": s15["total_pnl_pct"], "v15_wr": s15["win_rate"],
            "v14_trades": s14["total_trades"], "v14_pnl": s14["total_pnl_pct"], "v14_wr": s14["win_rate"],
            "v11_trades": s11["total_trades"], "v11_pnl": s11["total_pnl_pct"], "v11_wr": s11["win_rate"],
            "rule_trades": srule["total_trades"], "rule_pnl": srule["total_pnl_pct"],
            "file": f"./data/{sym}.json",
        })

    print("-" * 116)
    print(f"{'TOTAL':<8} {totals['v11']:>+9.1f}% {totals['v14']:>+9.1f}% "
          f"{totals['v15']:>+9.1f}% {totals['v16']:>+9.1f}%")
    print(f"Delta V14-V11: {totals['v14'] - totals['v11']:+.1f}%")
    print(f"Delta V15-V14: {totals['v15'] - totals['v14']:+.1f}%")
    print(f"Delta V16-V15: {totals['v16'] - totals['v15']:+.1f}%")
    print(f"Delta V16-V11: {totals['v16'] - totals['v11']:+.1f}%")
    if sanity_issues:
        print("\nSANITY CHECK: Found pnl mismatches > 1.0%")
        for line in sanity_issues[:20]:
            print("  " + line)
        if len(sanity_issues) > 20:
            print(f"  ... and {len(sanity_issues) - 20} more")
    else:
        print("\nSANITY CHECK: No pnl mismatches > 1.0%")

    index_entries.sort(key=lambda x: x["v16_pnl"], reverse=True)
    with open(os.path.join(viz_dir, "index.json"), "w") as f:
        json.dump({"symbols": index_entries, "version": "V16_vs_V15_vs_V14_vs_V11_vs_Rule",
                   "generated_at": datetime.now().isoformat()}, f, indent=2)

    print(f"\nExported {len(index_entries)} symbols to {viz_dir}")
    print("Open visualization/index.html (V16 included)")


if __name__ == "__main__":
    main()
