"""
Test P2+P3 combo vs P2-only vs P3-only vs Baseline.
If combo wins, it becomes V20.
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
from run_v19_3_compare import backtest_v19_3
from run_experiment_proposals import (
    backtest_v19_3_patched, calc_metrics, calc_per_symbol, run_rule_baseline
)
from compare_rule_vs_model import backtest_rule


def backtest_v19_3_p2(y_pred, returns, df_test, feature_cols, **kw):
    return backtest_v19_3_patched(y_pred, returns, df_test, feature_cols, patch="confirm_bars", **kw)

def backtest_v19_3_p3(y_pred, returns, df_test, feature_cols, **kw):
    return backtest_v19_3_patched(y_pred, returns, df_test, feature_cols, patch="peak_protect", **kw)

def backtest_v19_3_p2p3(y_pred, returns, df_test, feature_cols, **kw):
    return backtest_v19_3_patched(y_pred, returns, df_test, feature_cols, patch="p2p3", **kw)


def backtest_v19_3_combo(y_pred, returns, df_test, feature_cols,
                          initial_capital=100_000_000, commission=0.0015, tax=0.001,
                          record_trades=True,
                          mod_a=True, mod_b=True, mod_c=False, mod_d=False,
                          mod_e=True, mod_f=True, mod_g=True, mod_h=True,
                          mod_i=True, mod_j=True):
    """V19.3 with BOTH P2 (reduced confirm_bars) AND P3 (adaptive peak_protect)."""
    return backtest_v19_3_patched(y_pred, returns, df_test, feature_cols,
                                   initial_capital=initial_capital, commission=commission, tax=tax,
                                   record_trades=record_trades, patch="p2p3",
                                   mod_a=mod_a, mod_b=mod_b, mod_c=mod_c, mod_d=mod_d,
                                   mod_e=mod_e, mod_f=mod_f, mod_g=mod_g, mod_h=mod_h,
                                   mod_i=mod_i, mod_j=mod_j)


def run_backtest(pick, backtest_fn=backtest_v19_3):
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

    raw_df = loader.load_all(symbols=pick)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    all_trades = []
    for window, train_df, test_df in splitter.split(df):
        model = build_model("lightgbm")
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        for sym in test_df["symbol"].unique():
            if sym not in pick:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values

            r = backtest_fn(y_pred, rets, sym_test, feature_cols,
                            mod_a=True, mod_b=True, mod_c=False, mod_d=False,
                            mod_e=True, mod_f=True, mod_g=True, mod_h=True,
                            mod_i=True, mod_j=True)
            for t in r["trades"]:
                t["symbol"] = sym
            all_trades.extend(r["trades"])

    return all_trades


def main():
    pick = ["ACB", "FPT", "HPG", "SSI", "VND", "MBB", "TCB", "VNM", "DGC", "AAS", "AAV", "REE", "BID", "VIC"]

    print("=" * 140)
    print("P2+P3 COMBO TEST: Does combining both proposals beat each individually?")
    print(f"Symbols: {', '.join(pick)}")
    print("=" * 140)

    configs = [
        ("BASELINE (V19.3)",        backtest_v19_3),
        ("P2: Reduced confirm_bars", backtest_v19_3_p2),
        ("P3: Adaptive peak_protect", backtest_v19_3_p3),
        ("P2+P3 COMBO (V20 candidate)", backtest_v19_3_combo),
    ]

    results = {}
    for label, fn in configs:
        print(f"\nRunning {label}...")
        t0 = time.time()
        trades = run_backtest(pick, backtest_fn=fn)
        elapsed = time.time() - t0
        m = calc_metrics(trades)
        ps = calc_per_symbol(trades)
        results[label] = {"metrics": m, "per_sym": ps, "trades": trades}
        print(f"  Done in {elapsed:.0f}s - PnL: {m['total_pnl']:+.1f}%  WR: {m['wr']:.1f}%  PF: {m['pf']:.2f}")

    # Rule baseline
    print("\nRunning RULE baseline...")
    rule_trades = run_rule_baseline(pick)
    rule_m = calc_metrics(rule_trades)
    rule_ps = calc_per_symbol(rule_trades)
    results["RULE"] = {"metrics": rule_m, "per_sym": rule_ps}

    # ═══════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════
    base_pnl = results["BASELINE (V19.3)"]["metrics"]["total_pnl"]

    print("\n" + "=" * 140)
    print("AGGREGATE COMPARISON")
    print("=" * 140)
    print(f"{'Version':<35} | {'#Tr':>4} {'WR%':>6} {'AvgPnL':>8} {'MedPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgHold':>8} | {'vs Base':>8}")
    print("-" * 140)

    for label in ["BASELINE (V19.3)", "P2: Reduced confirm_bars", "P3: Adaptive peak_protect",
                   "P2+P3 COMBO (V20 candidate)", "RULE"]:
        m = results[label]["metrics"]
        delta = m["total_pnl"] - base_pnl
        marker = " ***" if delta > 0 and label != "BASELINE (V19.3)" and label != "RULE" else ""
        print(f"{label:<35} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['median_pnl']:>+7.2f}% {m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} "
              f"{m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d | {delta:>+7.1f}%{marker}")

    # ═══════════════════════════════════════
    # PER-SYMBOL
    # ═══════════════════════════════════════
    print("\n" + "=" * 140)
    print("PER-SYMBOL TOTAL PNL (%)")
    print("=" * 140)
    labels_order = ["BASELINE (V19.3)", "P2: Reduced confirm_bars", "P3: Adaptive peak_protect",
                    "P2+P3 COMBO (V20 candidate)", "RULE"]
    short = ["BASE", "P2", "P3", "P2+P3", "RULE"]
    header = f"{'Sym':<6}"
    for s in short:
        header += f" | {s:>10}"
    header += " |  P2+P3 vs Base  P2+P3 vs P2  P2+P3 vs P3"
    print(header)
    print("-" * 140)

    combo_wins_vs_p2 = 0
    combo_wins_vs_p3 = 0
    combo_wins_vs_base = 0

    for sym in pick:
        row = f"{sym:<6}"
        pnls = {}
        for label, s in zip(labels_order, short):
            pnl = results[label]["per_sym"].get(sym, {}).get("total_pnl", 0)
            pnls[s] = pnl
            row += f" | {pnl:>+9.1f}%"

        d_base = pnls["P2+P3"] - pnls["BASE"]
        d_p2 = pnls["P2+P3"] - pnls["P2"]
        d_p3 = pnls["P2+P3"] - pnls["P3"]
        row += f" |     {d_base:>+7.1f}%     {d_p2:>+7.1f}%     {d_p3:>+7.1f}%"
        print(row)

        if d_base > 0: combo_wins_vs_base += 1
        if d_p2 > 0: combo_wins_vs_p2 += 1
        if d_p3 > 0: combo_wins_vs_p3 += 1

    print("-" * 140)

    # Totals
    row = f"{'TOTAL':<6}"
    for label in labels_order:
        tot = sum(results[label]["per_sym"].get(sym, {}).get("total_pnl", 0) for sym in pick)
        row += f" | {tot:>+9.1f}%"
    combo_tot = sum(results["P2+P3 COMBO (V20 candidate)"]["per_sym"].get(sym, {}).get("total_pnl", 0) for sym in pick)
    base_tot = sum(results["BASELINE (V19.3)"]["per_sym"].get(sym, {}).get("total_pnl", 0) for sym in pick)
    p2_tot = sum(results["P2: Reduced confirm_bars"]["per_sym"].get(sym, {}).get("total_pnl", 0) for sym in pick)
    p3_tot = sum(results["P3: Adaptive peak_protect"]["per_sym"].get(sym, {}).get("total_pnl", 0) for sym in pick)
    row += f" |     {combo_tot - base_tot:>+7.1f}%     {combo_tot - p2_tot:>+7.1f}%     {combo_tot - p3_tot:>+7.1f}%"
    print(row)

    # ═══════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════
    combo_m = results["P2+P3 COMBO (V20 candidate)"]["metrics"]
    p2_m = results["P2: Reduced confirm_bars"]["metrics"]
    base_m = results["BASELINE (V19.3)"]["metrics"]

    print("\n" + "=" * 140)
    print("VERDICT")
    print("=" * 140)
    print(f"  P2+P3 COMBO vs BASELINE:  PnL {combo_m['total_pnl'] - base_m['total_pnl']:>+7.1f}%  "
          f"WR {combo_m['wr'] - base_m['wr']:>+5.1f}%  PF {combo_m['pf'] - base_m['pf']:>+5.2f}  "
          f"Wins {combo_wins_vs_base}/14 symbols")
    print(f"  P2+P3 COMBO vs P2-only:   PnL {combo_m['total_pnl'] - p2_m['total_pnl']:>+7.1f}%  "
          f"WR {combo_m['wr'] - p2_m['wr']:>+5.1f}%  PF {combo_m['pf'] - p2_m['pf']:>+5.2f}  "
          f"Wins {combo_wins_vs_p2}/14 symbols")

    if combo_m['total_pnl'] > p2_m['total_pnl'] and combo_m['pf'] >= p2_m['pf'] - 0.05:
        print("\n  >>> RECOMMENDATION: P2+P3 COMBO is BETTER than P2 alone => Use as V20")
    elif combo_m['total_pnl'] > base_m['total_pnl'] and combo_m['total_pnl'] <= p2_m['total_pnl']:
        print("\n  >>> RECOMMENDATION: P2+P3 COMBO beats baseline but NOT P2 alone => Use P2-only as V20")
    else:
        print("\n  >>> RECOMMENDATION: P2+P3 COMBO does NOT beat baseline => Keep V19.3")


if __name__ == "__main__":
    main()
