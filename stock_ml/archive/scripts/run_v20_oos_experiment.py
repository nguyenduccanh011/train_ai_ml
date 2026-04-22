"""
Out-of-Sample Experiment: Test A/B/C models on 20 NEW symbols.
===============================================================
Scenario A trains on 14 original symbols → pure OOS for 20 new symbols.
Scenario B trains on top 54 → most of the 20 are in training set.
Scenario C trains on all 370 → all 20 are in training set.

If B/C >> A → larger training helps generalization.
If A ~ B ~ C → 14 symbols already sufficient.
"""
import sys, os, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v20_compare import backtest_v20

TRAIN_14 = ["ACB", "FPT", "HPG", "SSI", "VND", "MBB", "TCB", "VNM",
             "DGC", "AAS", "AAV", "REE", "BID", "VIC"]

OOS_TEST_SYMBOLS = ["STB", "VHM", "VPB", "SHB", "MWG", "GEX", "DIG", "NVL",
                     "MSN", "DXG", "SHS", "KBC", "HSG", "CTG", "PVS", "PDR",
                     "VRE", "CEO", "CII", "PVD"]

MIN_ROWS = 2000


def get_symbol_liquidity(data_dir):
    base = os.path.join(data_dir, "all_symbols")
    results = []
    for d in os.listdir(base):
        if not d.startswith("symbol="):
            continue
        sym = d.replace("symbol=", "")
        f = os.path.join(base, d, "timeframe=1D", "data.csv")
        if not os.path.exists(f):
            continue
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"])
            if len(df) < MIN_ROWS:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            recent = df[df["timestamp"] >= "2020-01-01"]
            if len(recent) < 100:
                continue
            avg_tv = recent["traded_value"].mean() if "traded_value" in recent.columns else 0
            results.append((sym, avg_tv, len(df)))
        except Exception:
            continue
    results.sort(key=lambda x: -x[1])
    return results


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


def calc_metrics(trades):
    if not trades:
        return {"trades": 0, "wr": 0, "avg_pnl": 0, "total_pnl": 0, "pf": 0,
                "max_loss": 0, "avg_hold": 0, "median_pnl": 0}
    pnls = [t["pnl_pct"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    return {
        "trades": len(pnls),
        "wr": round(wins / len(pnls) * 100, 1),
        "avg_pnl": round(np.mean(pnls), 2),
        "total_pnl": round(sum(pnls), 1),
        "pf": round(gp / gl, 2) if gl > 0 else 99,
        "max_loss": round(min(pnls), 1),
        "avg_hold": round(np.mean([t.get("holding_days", 0) for t in trades]), 1),
        "median_pnl": round(float(np.median(pnls)), 2),
    }


def calc_per_symbol(trades):
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t.get("symbol", "?")].append(t)
    return {sym: calc_metrics(ts) for sym, ts in sorted(by_sym.items())}


def run_scenario(scenario_name, train_symbols, test_symbols, data_dir):
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

    all_symbols = sorted(set(train_symbols) | set(test_symbols))
    available = [s for s in all_symbols if s in loader.symbols]

    print(f"  Loading {len(available)} symbols...", end=" ", flush=True)
    t0 = time.time()
    raw_df = loader.load_all(symbols=available, show_progress=False)
    print(f"loaded in {time.time()-t0:.0f}s.", end=" ", flush=True)

    print("Computing features...", end=" ", flush=True)
    t0 = time.time()
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])
    print(f"done in {time.time()-t0:.0f}s. Total rows: {len(df):,}")

    all_trades = []
    for window, train_df, test_df in splitter.split(df):
        train_mask = train_df["symbol"].isin(train_symbols)
        train_data = train_df[train_mask]

        if len(train_data) < 100:
            continue

        model = build_model("lightgbm")
        X_train = np.nan_to_num(train_data[feature_cols].values)
        y_train = train_data["target"].values.astype(int)
        model.fit(X_train, y_train)

        for sym in test_symbols:
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values

            r = backtest_v20(y_pred, rets, sym_test, feature_cols,
                             mod_a=True, mod_b=True, mod_c=False, mod_d=False,
                             mod_e=True, mod_f=True, mod_g=True, mod_h=True,
                             mod_i=True, mod_j=True)
            for t in r["trades"]:
                t["symbol"] = sym
            all_trades.extend(r["trades"])

    return all_trades


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")

    print("=" * 140)
    print("OUT-OF-SAMPLE EXPERIMENT: Test A/B/C models on 20 NEW symbols")
    print(f"OOS test symbols: {', '.join(OOS_TEST_SYMBOLS)}")
    print(f"Original train symbols (A): {', '.join(TRAIN_14)}")
    print("=" * 140)

    print("\nRanking symbols by liquidity...")
    liquidity = get_symbol_liquidity(data_dir)
    top50_symbols = [sym for sym, _, _ in liquidity[:50]]
    all_viable = get_all_viable_symbols(data_dir)

    train_14 = list(TRAIN_14)
    train_50 = sorted(set(top50_symbols) | set(TRAIN_14))
    train_all = sorted(set(all_viable) | set(TRAIN_14))

    oos_in_b = sum(1 for s in OOS_TEST_SYMBOLS if s in set(train_50))
    oos_in_c = sum(1 for s in OOS_TEST_SYMBOLS if s in set(train_all))
    print(f"  Scenario A: Train {len(train_14)} syms | OOS symbols in train: 0/20 (pure OOS)")
    print(f"  Scenario B: Train {len(train_50)} syms | OOS symbols in train: {oos_in_b}/20")
    print(f"  Scenario C: Train {len(train_all)} syms | OOS symbols in train: {oos_in_c}/20")

    results = {}

    for key, label, train_syms in [
        ("A", f"Train 14 (pure OOS)", train_14),
        ("B", f"Train {len(train_50)} (top 50)", train_50),
        ("C", f"Train {len(train_all)} (all viable)", train_all),
    ]:
        print(f"\n{'='*80}")
        print(f"SCENARIO {key}: {label}")
        print(f"{'='*80}")
        t0 = time.time()
        trades = run_scenario(key, train_syms, OOS_TEST_SYMBOLS, data_dir)
        elapsed = time.time() - t0
        results[key] = {
            "trades": trades, "metrics": calc_metrics(trades),
            "per_sym": calc_per_symbol(trades), "n_train": len(train_syms),
            "label": label,
        }
        print(f"\n  => {elapsed:.0f}s | PnL: {results[key]['metrics']['total_pnl']:+.1f}%")

    # COMPARISON
    base_pnl = results["A"]["metrics"]["total_pnl"]

    print("\n" + "=" * 140)
    print("AGGREGATE COMPARISON (tested on 20 OOS symbols)")
    print("=" * 140)
    print(f"{'Scenario':<50} | {'#Train':>6} {'#Tr':>5} {'WR%':>6} {'AvgPnL':>8} {'MedPnL':>8} "
          f"{'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgHold':>8} | {'vs A':>8}")
    print("-" * 140)

    for key in ["A", "B", "C"]:
        m = results[key]["metrics"]
        delta = m["total_pnl"] - base_pnl
        marker = " ***" if delta > 0 and key != "A" else ""
        print(f"{key}: {results[key]['label']:<48} | {results[key]['n_train']:>6} {m['trades']:>5} {m['wr']:>5.1f}% "
              f"{m['avg_pnl']:>+7.2f}% {m['median_pnl']:>+7.2f}% {m['total_pnl']:>+9.1f}% "
              f"{m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d | {delta:>+7.1f}%{marker}")

    # PER-SYMBOL
    print("\n" + "=" * 140)
    print("PER-SYMBOL TOTAL PNL (%) — 20 OOS symbols")
    print("=" * 140)
    print(f"{'Sym':<6} | {'A (14)':>10} | {'B (50)':>10} | {'C (all)':>10} | {'B vs A':>8} {'C vs A':>8} | Best")
    print("-" * 140)

    b_wins = c_wins = 0
    for sym in OOS_TEST_SYMBOLS:
        pa = results["A"]["per_sym"].get(sym, {}).get("total_pnl", 0)
        pb = results["B"]["per_sym"].get(sym, {}).get("total_pnl", 0)
        pc = results["C"]["per_sym"].get(sym, {}).get("total_pnl", 0)
        db = pb - pa
        dc = pc - pa
        best = "A"
        best_val = pa
        if pb > best_val:
            best = "B"
            best_val = pb
        if pc > best_val:
            best = "C"
        if db > 0:
            b_wins += 1
        if dc > 0:
            c_wins += 1
        print(f"{sym:<6} | {pa:>+9.1f}% | {pb:>+9.1f}% | {pc:>+9.1f}% | {db:>+7.1f}% {dc:>+7.1f}% | {best}")

    print("-" * 140)
    ta = sum(results["A"]["per_sym"].get(s, {}).get("total_pnl", 0) for s in OOS_TEST_SYMBOLS)
    tb = sum(results["B"]["per_sym"].get(s, {}).get("total_pnl", 0) for s in OOS_TEST_SYMBOLS)
    tc = sum(results["C"]["per_sym"].get(s, {}).get("total_pnl", 0) for s in OOS_TEST_SYMBOLS)
    print(f"{'TOTAL':<6} | {ta:>+9.1f}% | {tb:>+9.1f}% | {tc:>+9.1f}% | {tb-ta:>+7.1f}% {tc-ta:>+7.1f}% |")

    # VERDICT
    print("\n" + "=" * 140)
    print("VERDICT")
    print("=" * 140)
    for key, label in [("B", "Top 50"), ("C", "All viable")]:
        m = results[key]["metrics"]
        ma = results["A"]["metrics"]
        wins = b_wins if key == "B" else c_wins
        print(f"  {label} vs Baseline(14):  PnL {m['total_pnl'] - ma['total_pnl']:>+7.1f}%  "
              f"WR {m['wr'] - ma['wr']:>+5.1f}%  PF {m['pf'] - ma['pf']:>+5.2f}  "
              f"Improved {wins}/20 OOS symbols")

    best_scenario = max(["A", "B", "C"], key=lambda k: results[k]["metrics"]["total_pnl"])
    print(f"\n  >>> BEST OOS SCENARIO: {best_scenario}: {results[best_scenario]['label']}")
    if best_scenario != "A":
        print(f"  >>> Larger training set HELPS generalization on unseen symbols!")
    else:
        print(f"  >>> Even on unseen symbols, 14-symbol training holds up!")


if __name__ == "__main__":
    main()
