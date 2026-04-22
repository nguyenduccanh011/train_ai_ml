"""
Feature Ablation Experiment — Test each feature group's impact on V27 model.

Runs 8 experiments:
  0. Baseline (leading features only)
  1. Baseline + Group A (Market Structure)
  2. Baseline + Group B (Exhaustion & Failure Signals)
  3. Baseline + Group C (Volatility Regime)
  4. Baseline + Group D (Multi-timeframe)
  5. Baseline + Group E (Relative Strength)
  6. Baseline + Group F (Liquidity)
  7. Baseline + Best Groups (auto-selected from 1-6)

Usage:
  python run_feature_ablation.py              # Run all experiments
  python run_feature_ablation.py --group A    # Run only Group A
  python run_feature_ablation.py --group A,B  # Run Group A and B
"""
import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model, detect_device
from src.config_loader import get_training_device
from run_v27 import backtest_v27_default
from compare_rule_vs_model import backtest_rule


def calc_metrics(trades):
    if not trades:
        return {"trades": 0, "wr": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0,
                "pf": 0.0, "max_loss": 0.0, "avg_hold": 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = wins / n * 100
    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    total_pnl = sum(t["pnl_pct"] for t in trades)
    gp = sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0)
    gl = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))
    pf = gp / gl if gl > 0 else 99
    max_loss = min(t["pnl_pct"] for t in trades)
    avg_hold = np.mean([t.get("holding_days", 0) for t in trades])
    return {"trades": n, "wr": wr, "avg_pnl": avg_pnl, "total_pnl": total_pnl,
            "pf": pf, "max_loss": max_loss, "avg_hold": avg_hold}


def comp_score(m):
    """Composite score (higher is better)."""
    if m["trades"] == 0:
        return 0
    return (m["wr"] * 0.3 + m["pf"] * 20 + m["avg_pnl"] * 50 +
            m["total_pnl"] * 0.05 - abs(m["max_loss"]) * 2)


def resolve_symbols(min_rows=2000):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    loader = DataLoader(data_dir)
    symbols = []
    for sym in loader.symbols:
        try:
            df = loader.load_symbol(sym)
            if len(df) >= min_rows:
                symbols.append(sym)
        except FileNotFoundError:
            continue
    return symbols


def run_experiment(symbols, extra_groups=None, device="cpu"):
    """Run a single ablation experiment with given extra feature groups."""
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

    raw_df = loader.load_all(symbols=symbols)
    engine = FeatureEngine(feature_set="leading", extra_groups=extra_groups)
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    all_trades = []
    feature_importances = defaultdict(list)

    for window, train_df, test_df in splitter.split(df):
        model = build_model("lightgbm", device=device)
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        lgb_model = model.named_steps["model"]
        if hasattr(lgb_model, "feature_importances_"):
            for fname, imp in zip(feature_cols, lgb_model.feature_importances_):
                feature_importances[fname].append(imp)

        for sym in test_df["symbol"].unique():
            if sym not in symbols:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values

            r = backtest_v27_default(y_pred, rets, sym_test, feature_cols,
                                     mod_a=True, mod_b=True, mod_c=False, mod_d=False,
                                     mod_e=True, mod_f=True, mod_g=True, mod_h=True,
                                     mod_i=True, mod_j=True)
            for t in r["trades"]:
                t["symbol"] = sym
            all_trades.extend(r["trades"])

    avg_importance = {k: np.mean(v) for k, v in feature_importances.items()}
    avg_importance = dict(sorted(avg_importance.items(), key=lambda x: -x[1]))

    return all_trades, avg_importance, len(feature_cols)


GROUP_NAMES = {
    None: "Baseline (leading)",
    "A": "Market Structure",
    "B": "Exhaustion & Failure",
    "C": "Volatility Regime",
    "D": "Multi-timeframe",
    "E": "Relative Strength",
    "F": "Liquidity",
}


def main():
    parser = argparse.ArgumentParser(description="Feature Ablation Experiment")
    parser.add_argument("--group", type=str, default="",
                        help="Comma-separated groups to test (A,B,C,D,E,F). Empty=all")
    parser.add_argument("--min-rows", type=int, default=2000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    device = args.device or get_training_device()
    resolved = detect_device(device)
    print(f"Training device: {resolved.upper()}")

    print("\nResolving symbols...")
    symbols = resolve_symbols(min_rows=args.min_rows)
    print(f"Using {len(symbols)} symbols")

    if args.group:
        groups_to_test = [g.strip().upper() for g in args.group.split(",")]
    else:
        groups_to_test = ["A", "B", "C", "D", "E", "F"]

    experiments = [(None, "Baseline")]
    for g in groups_to_test:
        experiments.append(([g], f"Baseline + Group {g} ({GROUP_NAMES.get(g, g)})"))

    results = {}
    all_importances = {}

    print(f"\n{'=' * 120}")
    print(f"FEATURE ABLATION EXPERIMENT — {len(experiments)} experiments x {len(symbols)} symbols")
    print(f"{'=' * 120}")

    total_start = time.time()

    for extra_groups, label in experiments:
        t0 = time.time()
        group_key = ",".join(extra_groups) if extra_groups else "baseline"
        print(f"\n{'─' * 100}")
        print(f"  Experiment: {label}")
        print(f"  Extra groups: {extra_groups or 'none'}")
        print(f"{'─' * 100}")

        trades, importance, n_features = run_experiment(
            symbols, extra_groups=extra_groups, device=device
        )
        dt = time.time() - t0
        metrics = calc_metrics(trades)
        score = comp_score(metrics)

        results[group_key] = {**metrics, "score": score, "n_features": n_features,
                              "time": round(dt, 1), "label": label}
        all_importances[group_key] = importance

        print(f"\n  Results ({label}):")
        print(f"    Features: {n_features}")
        print(f"    Trades={metrics['trades']}, WR={metrics['wr']:.1f}%, "
              f"AvgPnL={metrics['avg_pnl']:+.2f}%, TotalPnL={metrics['total_pnl']:+.1f}%, "
              f"PF={metrics['pf']:.2f}, MaxLoss={metrics['max_loss']:.1f}%, "
              f"AvgHold={metrics['avg_hold']:.1f}d")
        print(f"    Composite Score: {score:.0f}")
        print(f"    Time: {dt:.1f}s")

        if importance:
            print(f"\n    Top-15 Feature Importance:")
            for i, (fname, imp) in enumerate(list(importance.items())[:15]):
                marker = " *NEW*" if extra_groups and any(
                    fname.startswith(p) for p in _get_group_prefixes(extra_groups)
                ) else ""
                print(f"      {i + 1:>2}. {fname:<35s} {imp:>8.0f}{marker}")

    # ── Summary Comparison ──
    print(f"\n\n{'=' * 120}")
    print(f"SUMMARY COMPARISON")
    print(f"{'=' * 120}")

    baseline = results.get("baseline", {})
    bl_pnl = baseline.get("total_pnl", 0)
    bl_wr = baseline.get("wr", 0)
    bl_pf = baseline.get("pf", 0)
    bl_score = baseline.get("score", 0)

    header = (f"  {'Experiment':<40s} | {'#Feat':>5} {'#Trd':>5} {'WR':>6} "
              f"{'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} "
              f"{'Score':>7} | {'dPnL':>8} {'dWR':>6} {'dPF':>6} {'dScore':>7} | {'Time':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for key, r in results.items():
        d_pnl = r["total_pnl"] - bl_pnl
        d_wr = r["wr"] - bl_wr
        d_pf = r["pf"] - bl_pf
        d_score = r["score"] - bl_score
        print(f"  {r['label']:<40s} | {r['n_features']:>5} {r['trades']:>5} "
              f"{r['wr']:>5.1f}% {r['avg_pnl']:>+7.2f}% {r['total_pnl']:>+9.1f}% "
              f"{r['pf']:>5.2f} {r['max_loss']:>+7.1f}% {r['score']:>6.0f} | "
              f"{d_pnl:>+7.1f}% {d_wr:>+5.1f}% {d_pf:>+5.2f} {d_score:>+6.0f} | "
              f"{r['time']:>5.1f}s")

    # ── Identify best groups ──
    improved_groups = []
    for key, r in results.items():
        if key == "baseline":
            continue
        d_score = r["score"] - bl_score
        d_pnl = r["total_pnl"] - bl_pnl
        d_pf = r["pf"] - bl_pf
        if d_score > 0 and d_pnl > 0:
            improved_groups.append((key, d_score, d_pnl, d_pf))

    improved_groups.sort(key=lambda x: -x[1])

    print(f"\n{'─' * 80}")
    if improved_groups:
        print(f"  GROUPS THAT IMPROVED (score + TotalPnL > baseline):")
        for g, ds, dp, dpf in improved_groups:
            print(f"    Group {g}: Score +{ds:.0f}, TotalPnL +{dp:.1f}%, PF +{dpf:.2f}")

        best_combo = [g[0] for g in improved_groups]
        print(f"\n  RECOMMENDED BEST COMBO: {best_combo}")

        # ── Run best combo ──
        if len(best_combo) > 1 and not args.group:
            print(f"\n{'─' * 100}")
            print(f"  Running BEST COMBO: Baseline + Groups {best_combo}")
            print(f"{'─' * 100}")

            t0 = time.time()
            trades, importance, n_features = run_experiment(
                symbols, extra_groups=best_combo, device=device
            )
            dt = time.time() - t0
            metrics = calc_metrics(trades)
            score = comp_score(metrics)

            print(f"\n  BEST COMBO Results:")
            print(f"    Features: {n_features}")
            print(f"    Trades={metrics['trades']}, WR={metrics['wr']:.1f}%, "
                  f"AvgPnL={metrics['avg_pnl']:+.2f}%, TotalPnL={metrics['total_pnl']:+.1f}%, "
                  f"PF={metrics['pf']:.2f}, MaxLoss={metrics['max_loss']:.1f}%")
            print(f"    Composite Score: {score:.0f} (Baseline: {bl_score:.0f}, "
                  f"Delta: {score - bl_score:+.0f})")
            print(f"    Time: {dt:.1f}s")

            d_pnl = metrics["total_pnl"] - bl_pnl
            d_wr = metrics["wr"] - bl_wr
            d_pf = metrics["pf"] - bl_pf

            results["best_combo"] = {
                **metrics, "score": score, "n_features": n_features,
                "time": round(dt, 1), "label": f"BEST COMBO {best_combo}",
                "groups": best_combo,
            }

            print(f"\n    vs Baseline: TotalPnL {d_pnl:+.1f}%, WR {d_wr:+.1f}%, PF {d_pf:+.2f}")

            if importance:
                print(f"\n    Top-20 Feature Importance (Best Combo):")
                for i, (fname, imp) in enumerate(list(importance.items())[:20]):
                    marker = " *NEW*" if any(
                        fname.startswith(p) for p in _get_group_prefixes(best_combo)
                    ) else ""
                    print(f"      {i + 1:>2}. {fname:<35s} {imp:>8.0f}{marker}")
    else:
        print(f"  NO GROUP IMPROVED over baseline. Current features are already well-tuned.")

    # ── Save results ──
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for key, r in results.items():
        rows.append({
            "experiment": key, "label": r["label"],
            "n_features": r["n_features"], "trades": r["trades"],
            "win_rate": round(r["wr"], 2), "avg_pnl": round(r["avg_pnl"], 3),
            "total_pnl": round(r["total_pnl"], 2), "profit_factor": round(r["pf"], 3),
            "max_loss": round(r["max_loss"], 2), "avg_hold": round(r["avg_hold"], 1),
            "composite_score": round(r["score"], 1), "time_sec": r["time"],
        })
    df_results = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "feature_ablation_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")

    imp_rows = []
    for exp_key, imp_dict in all_importances.items():
        for fname, imp_val in imp_dict.items():
            imp_rows.append({"experiment": exp_key, "feature": fname, "importance": round(imp_val, 1)})
    if imp_rows:
        df_imp = pd.DataFrame(imp_rows)
        imp_path = os.path.join(results_dir, "feature_importance_ablation.csv")
        df_imp.to_csv(imp_path, index=False)
        print(f"  Feature importance saved to {imp_path}")

    total_time = time.time() - total_start
    print(f"\n{'=' * 120}")
    print(f"  TOTAL TIME: {total_time:.0f}s ({total_time / 60:.1f} minutes)")
    print(f"{'=' * 120}")


def _get_group_prefixes(groups):
    """Return feature name prefixes for each group to identify new features."""
    prefix_map = {
        "A": ["pivot_", "dist_to_last_swing", "bos_", "choch", "hh_hl_regime"],
        "B": ["upthrust", "spring", "climax_", "gap_up", "gap_down", "gap_filled", "reversal_"],
        "C": ["atr_percentile", "true_range_percentile", "overnight_gap", "compression_duration", "post_expansion"],
        "D": ["weekly_", "price_vs_wma"],
        "E": ["rs_vs_", "rs_rank", "rs_divergence"],
        "F": ["amihud_", "turnover_percentile", "avg_spread_proxy", "volume_consistency"],
    }
    result = []
    for g in groups:
        result.extend(prefix_map.get(g, []))
    return result


if __name__ == "__main__":
    main()
