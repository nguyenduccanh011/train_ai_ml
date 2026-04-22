"""
Export V23 Optimal trades for web visualization (data_v23/{SYM}.json).
V23 = V22 base + graduated fast_exit + restored peak_protect + trend-specific caps.
Best config: peak_protect_strong_threshold=0.12.
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v23_optimal import backtest_v23


COLORS = {"v23": "#66BB6A"}


def make_markers(trades, method="v23"):
    markers = []
    buy_color = COLORS[method]
    win_color, loss_color = "#4caf50", "#f44336"
    prefix = "V23"

    short = {
        "stop_loss": "SL", "trailing_stop": "TS", "hard_stop": "HS",
        "zombie_exit": "ZE", "hybrid_exit": "HX", "profit_lock": "PL",
        "profit_floor": "PF", "peak_protect_dist": "PP", "peak_protect_ema": "PE",
        "fast_loss_cut": "FC", "fast_exit_loss": "FE", "signal_hard_cap": "HC",
        "signal": "", "end": "END",
    }

    for t in trades:
        ed, xd = t.get("entry_date", ""), t.get("exit_date", "")
        pnl = t.get("pnl_pct", 0)
        reason = t.get("exit_reason", "")
        tag = short.get(reason, reason[:2].upper() if reason else "")
        if ed:
            markers.append({
                "time": ed, "position": "belowBar", "color": buy_color,
                "shape": "arrowUp", "text": f"{prefix} Buy", "size": 1, "method": method,
            })
        if xd:
            markers.append({
                "time": xd, "position": "aboveBar",
                "color": win_color if pnl >= 0 else loss_color,
                "shape": "arrowDown",
                "text": f"{prefix} {pnl:+.1f}%{(' ' + tag) if tag else ''}",
                "size": 1, "method": method,
            })
    return markers


def compute_stats(trades, label="V23 Optimal"):
    if not trades:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl_pct": 0, "avg_pnl_pct": 0, "avg_hold": 0, "pf": 0,
                "version": label}
    pnls = [float(t["pnl_pct"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holds = [t.get("holding_days", 0) for t in trades]
    gl = abs(sum(losses))
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0
    return {
        "total_trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_pnl_pct": round(sum(pnls), 1),
        "avg_pnl_pct": round(np.mean(pnls), 2),
        "median_pnl_pct": round(float(np.median(pnls)), 2),
        "std_pnl_pct": round(float(np.std(pnls)), 2),
        "avg_win_pct": round(avg_win, 2), "avg_loss_pct": round(avg_loss, 2),
        "payoff_ratio": round(payoff, 2),
        "max_win_pct": round(float(max(pnls)), 2),
        "max_loss_pct": round(float(min(pnls)), 2),
        "avg_hold": round(np.mean(holds), 1) if holds else 0,
        "pf": round(sum(wins) / gl, 2) if gl > 0 else 99,
        "version": label,
    }


def enrich_trades_with_prices(trades, close_by_date):
    out = []
    for t in trades:
        row = dict(t)
        ed, xd = row.get("entry_date"), row.get("exit_date")
        ep = close_by_date.get(ed) if ed else None
        xp = close_by_date.get(xd) if xd else None
        row["entry_price"] = round(float(ep), 2) if ep is not None else None
        row["exit_price"] = round(float(xp), 2) if xp is not None else None
        out.append(row)
    return out


def select_fields(trades, fields):
    out = []
    for t in trades:
        row = {}
        for k in fields:
            if k not in t:
                continue
            v = t[k]
            if isinstance(v, np.generic):
                v = v.item()
            row[k] = v
        out.append(row)
    return out


MIN_ROWS = 2000


def get_all_viable_symbols(data_dir):
    base = os.path.join(data_dir, "all_symbols")
    viable = []
    for d in os.listdir(base):
        if not d.startswith("symbol="):
            continue
        sym = d.replace("symbol=", "")
        f = os.path.join(base, d, "timeframe=1D", "data.csv")
        if not os.path.exists(f):
            continue
        try:
            with open(f) as fh:
                n = sum(1 for _ in fh) - 1
            if n >= MIN_ROWS:
                viable.append(sym)
        except Exception:
            continue
    return sorted(viable)


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "visualization", "data_v23")
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

    all_viable = get_all_viable_symbols(data_dir)
    all_symbols = [s for s in all_viable if s in loader.symbols]
    print(f"Training & testing on {len(all_symbols)} viable symbols (>= {MIN_ROWS} rows)")

    print("Loading all symbols...", end=" ", flush=True)
    t0 = time.time()
    raw_df = loader.load_all(symbols=all_symbols, show_progress=False)
    print(f"done in {time.time()-t0:.0f}s.")

    print("Computing features...", end=" ", flush=True)
    t0 = time.time()
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])
    print(f"done in {time.time()-t0:.0f}s. Total rows: {len(df):,}")

    v23_by = defaultdict(list)

    print("Running walk-forward V23 backtest...")
    bt_start = time.time()
    for window, train_df, test_df in splitter.split(df):
        model = build_model("lightgbm")
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        test_syms = test_df["symbol"].unique()
        print(f"  Window {window}: train {len(train_df):,} rows, test {len(test_syms)} symbols...",
              end=" ", flush=True)
        w_start = time.time()

        for sym in test_syms:
            if sym not in all_symbols:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values

            r23 = backtest_v23(y_pred, rets, sym_test, feature_cols,
                               peak_protect_strong_threshold=0.12,
                               mod_a=True, mod_b=True, mod_c=False, mod_d=False,
                               mod_e=True, mod_f=True, mod_g=True, mod_h=True,
                               mod_i=True, mod_j=True)
            for t in r23["trades"]:
                t["symbol"] = sym
            v23_by[sym].extend(r23["trades"])

        print(f"{time.time()-w_start:.0f}s")

    print(f"Walk-forward V23 complete in {time.time()-bt_start:.0f}s.")

    fields = (
        "entry_date", "exit_date", "pnl_pct", "holding_days", "exit_reason",
        "entry_trend", "quick_reentry", "breakout_entry", "vshape_entry",
        "entry_profile", "entry_choppy_regime", "position_size",
        "entry_price", "exit_price",
    )

    print("Exporting JSON files...", end=" ", flush=True)
    t0 = time.time()
    index_entries = []
    for sym in all_symbols:
        if sym not in v23_by:
            continue

        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])

        close_by_date = {str(row[date_col])[:10]: float(row["close"]) for _, row in sym_data.iterrows()}

        v23t = enrich_trades_with_prices(v23_by.get(sym, []), close_by_date)
        s23 = compute_stats(v23t, "V23 Optimal")

        sym_json = {
            "symbol": sym,
            "v23_markers": make_markers(v23t, "v23"),
            "v23_trades": select_fields(v23t, fields),
            "v23_stats": s23,
        }

        with open(os.path.join(viz_dir, f"{sym}.json"), "w", encoding="utf-8") as f:
            json.dump(sym_json, f)

        index_entries.append({
            "symbol": sym,
            "v23_trades": s23["total_trades"], "v23_pnl": s23["total_pnl_pct"], "v23_wr": s23["win_rate"],
        })

    index_entries.sort(key=lambda x: x["v23_pnl"], reverse=True)
    with open(os.path.join(viz_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({
            "symbols": index_entries,
            "version": "V23_Optimal",
            "generated_at": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"done in {time.time()-t0:.0f}s.")
    total = sum(e["v23_pnl"] for e in index_entries)
    pos = sum(1 for e in index_entries if e["v23_pnl"] > 0)
    print(f"\nExported {len(index_entries)} symbols to {viz_dir}")
    print(f"V23 Aggregate PnL: {total:+.1f}% | {pos}/{len(index_entries)} profitable")


if __name__ == "__main__":
    main()
