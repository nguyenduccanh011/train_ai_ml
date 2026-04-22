"""
FULL MODEL COMPARISON: V11, V15, V16, V17, V18, V19, V19.1 + Rule-based
========================================================================
Runs all models on the same data, computes metrics, ranks objectively.
"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v11_compare import backtest_v11
from run_v15_compare import backtest_v15
from run_v16_compare import backtest_v16
from run_v17_compare import backtest_v17
from run_v18_compare import backtest_v18
from run_v19_compare import backtest_v19
from run_v19_1_compare import backtest_v19_1
from compare_rule_vs_model import backtest_rule


def calc_metrics(trades):
    if not trades:
        return {"trades": 0, "wr": 0, "avg_pnl": 0, "total_pnl": 0, "pf": 0,
                "max_loss": 0, "avg_hold": 0, "avg_win": 0, "avg_loss": 0,
                "max_win": 0, "max_dd_trade": 0, "win_trades": 0, "lose_trades": 0}
    n = len(trades)
    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0
    return {
        "trades": n,
        "wr": len(wins) / n * 100,
        "avg_pnl": np.mean(pnls),
        "total_pnl": sum(pnls),
        "pf": gp / gl if gl > 0 else 99,
        "max_loss": min(pnls),
        "max_win": max(pnls),
        "avg_hold": np.mean([t.get("holding_days", 0) for t in trades]),
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "win_trades": len(wins),
        "lose_trades": len(losses),
    }


def calc_drawdown(trades):
    """Max consecutive losing streak and max cumulative drawdown from trades."""
    if not trades:
        return 0, 0, 0
    cum = 0
    peak = 0
    max_dd = 0
    streak = 0
    max_streak = 0
    for t in trades:
        cum += t["pnl_pct"]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
        if t["pnl_pct"] <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_dd, max_streak, cum


def run_all_models(symbols_str):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    config = {
        "data": {"data_dir": data_dir},
        "split": {"method": "walk_forward", "train_years": 4, "test_years": 1,
                  "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025},
        "target": {"type": "trend_regime", "trend_method": "dual_ma",
                   "short_window": 5, "long_window": 20, "classes": 3},
    }

    pick = [s.strip() for s in symbols_str.split(",")]
    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    raw_df = loader.load_all(symbols=pick)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    # Model configs: (name, backtest_fn, mod_a..mod_j)
    # V11 baseline uses backtest_v11 directly (no mods)
    # V15-V19.1 use their respective functions with full mods enabled
    model_configs = {
        "V11":    (backtest_v11, {}),
        "V15":    (backtest_v15, {"mod_a": True, "mod_b": True, "mod_c": False, "mod_d": False, "mod_e": True}),
        "V16":    (backtest_v16, {"mod_a": True, "mod_b": True, "mod_c": False, "mod_d": False, "mod_e": True,
                                   "mod_f": True, "mod_g": True}),
        "V17":    (backtest_v17, {"mod_a": True, "mod_b": True, "mod_c": False, "mod_d": False, "mod_e": True,
                                   "mod_f": True, "mod_g": True, "mod_h": True, "mod_i": True, "mod_j": True}),
        "V18":    (backtest_v18, {"mod_a": True, "mod_b": True, "mod_c": False, "mod_d": False, "mod_e": True,
                                   "mod_f": True, "mod_g": True, "mod_h": True, "mod_i": True, "mod_j": True}),
        "V19":    (backtest_v19, {"mod_a": True, "mod_b": True, "mod_c": False, "mod_d": False, "mod_e": True,
                                   "mod_f": True, "mod_g": True, "mod_h": True, "mod_i": True, "mod_j": True}),
        "V19.1":  (backtest_v19_1, {"mod_a": True, "mod_b": True, "mod_c": False, "mod_d": False, "mod_e": True,
                                     "mod_f": True, "mod_g": True, "mod_h": True, "mod_i": True, "mod_j": True}),
    }

    all_results = {}  # model_name -> {sym -> trades}

    for model_name in model_configs:
        all_results[model_name] = defaultdict(list)

    # Run ML models
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

            for model_name, (bt_fn, mods) in model_configs.items():
                r = bt_fn(y_pred, rets, sym_test, feature_cols, **mods)
                for t in r["trades"]:
                    t["symbol"] = sym
                all_results[model_name][sym].extend(r["trades"])

    # Run rule-based
    all_results["Rule"] = defaultdict(list)
    symbols = [s for s in pick if s in loader.symbols]
    for sym in symbols:
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])
        sym_test = sym_data[sym_data[date_col] >= "2020-01-01"].reset_index(drop=True)
        if len(sym_test) < 50:
            continue
        trades = backtest_rule(sym_test)
        for t in trades:
            t["symbol"] = sym
        all_results["Rule"][sym].extend(trades)

    return all_results, pick


def print_report(all_results, pick):
    model_names = ["V11", "V15", "V16", "V17", "V18", "V19", "V19.1", "Rule"]

    # ===== OVERALL METRICS =====
    print("=" * 160)
    print("SECTION 1: OVERALL METRICS (all symbols combined)")
    print("=" * 160)

    overall = {}
    for mn in model_names:
        all_trades = []
        for sym_trades in all_results[mn].values():
            all_trades.extend(sym_trades)
        m = calc_metrics(all_trades)
        max_dd, max_streak, cum_pnl = calc_drawdown(all_trades)
        m["max_dd"] = max_dd
        m["max_streak"] = max_streak
        overall[mn] = m

    print(f"{'Model':<8} | {'#Trades':>7} {'WR%':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} "
          f"{'AvgWin':>8} {'AvgLoss':>8} {'MaxWin':>8} {'MaxLoss':>8} {'MaxDD':>7} {'Streak':>6} {'AvgHold':>7}")
    print("-" * 160)
    for mn in model_names:
        m = overall[mn]
        print(f"{mn:<8} | {m['trades']:>7} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['avg_win']:>+7.2f}% {m['avg_loss']:>+7.2f}% "
              f"{m['max_win']:>+7.1f}% {m['max_loss']:>+7.1f}% {m['max_dd']:>6.1f}% {m['max_streak']:>6} {m['avg_hold']:>6.1f}d")

    # ===== PER-SYMBOL BREAKDOWN =====
    print("\n" + "=" * 160)
    print("SECTION 2: PER-SYMBOL TOTAL PnL%")
    print("=" * 160)

    header = f"{'Sym':<6}"
    for mn in model_names:
        header += f" | {mn+' Tot':>9} {mn+' PF':>6}"
    print(header)
    print("-" * 160)

    sym_data = {}
    totals = {mn: 0 for mn in model_names}
    all_syms = sorted(set().union(*[all_results[mn].keys() for mn in model_names]))

    for sym in all_syms:
        row = f"{sym:<6}"
        sym_data[sym] = {}
        for mn in model_names:
            m = calc_metrics(all_results[mn].get(sym, []))
            sym_data[sym][mn] = m
            totals[mn] += m["total_pnl"]
            row += f" | {m['total_pnl']:>+8.1f}% {m['pf']:>5.2f}"
        print(row)

    print("-" * 160)
    row = f"{'TOTAL':<6}"
    for mn in model_names:
        row += f" | {totals[mn]:>+8.1f}% {'':>5}"
    print(row)

    # ===== PER-SYMBOL WIN RATE =====
    print("\n" + "=" * 160)
    print("SECTION 3: PER-SYMBOL WIN RATE%")
    print("=" * 160)
    header = f"{'Sym':<6}"
    for mn in model_names:
        header += f" | {mn:>8}"
    print(header)
    print("-" * 160)
    for sym in all_syms:
        row = f"{sym:<6}"
        for mn in model_names:
            m = sym_data[sym].get(mn, {"wr": 0})
            row += f" | {m['wr']:>7.1f}%"
        print(row)

    # ===== RANKING =====
    print("\n" + "=" * 160)
    print("SECTION 4: OBJECTIVE RANKING (lower = better)")
    print("=" * 160)

    # Rank on multiple criteria
    criteria = {
        "Total PnL":   {mn: overall[mn]["total_pnl"] for mn in model_names},
        "Win Rate":     {mn: overall[mn]["wr"] for mn in model_names},
        "Profit Factor":{mn: overall[mn]["pf"] for mn in model_names},
        "Avg PnL":      {mn: overall[mn]["avg_pnl"] for mn in model_names},
        "Max Drawdown":  {mn: -overall[mn]["max_dd"] for mn in model_names},  # lower DD = better, so negate
        "Max Loss":     {mn: overall[mn]["max_loss"] for mn in model_names},  # less negative = better
        "Avg Hold Days": {mn: -overall[mn]["avg_hold"] for mn in model_names},  # shorter = more capital efficient (negate)
    }

    ranks = {mn: [] for mn in model_names}
    print(f"{'Criterion':<16}", end="")
    for mn in model_names:
        print(f" | {mn:>8}", end="")
    print(" | Best")
    print("-" * 160)

    for crit_name, values in criteria.items():
        # Sort descending (higher = better for most metrics)
        sorted_models = sorted(values.keys(), key=lambda x: values[x], reverse=True)
        rank_map = {mn: i+1 for i, mn in enumerate(sorted_models)}
        for mn in model_names:
            ranks[mn].append(rank_map[mn])
        row = f"{crit_name:<16}"
        for mn in model_names:
            row += f" |     #{rank_map[mn]:>1}  "
        row += f" | {sorted_models[0]}"
        print(row)

    # Average rank
    print("-" * 160)
    avg_ranks = {mn: np.mean(ranks[mn]) for mn in model_names}
    row = f"{'AVG RANK':<16}"
    for mn in model_names:
        row += f" | {avg_ranks[mn]:>7.2f} "
    print(row)

    final_ranking = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])
    print(f"\n{'FINAL RANKING':>16}: ", end="")
    for i, mn in enumerate(final_ranking):
        print(f"#{i+1} {mn} ({avg_ranks[mn]:.2f})", end="  ")
    print()

    # ===== SYMBOL-LEVEL WINS =====
    print("\n" + "=" * 160)
    print("SECTION 5: BEST MODEL PER SYMBOL (by Total PnL)")
    print("=" * 160)

    sym_wins = defaultdict(int)
    for sym in all_syms:
        best_mn = max(model_names, key=lambda mn: sym_data[sym].get(mn, {"total_pnl": -999})["total_pnl"])
        best_pnl = sym_data[sym][best_mn]["total_pnl"]
        worst_mn = min(model_names, key=lambda mn: sym_data[sym].get(mn, {"total_pnl": 999})["total_pnl"])
        worst_pnl = sym_data[sym][worst_mn]["total_pnl"]
        sym_wins[best_mn] += 1
        print(f"  {sym:<6}: BEST = {best_mn:<8} ({best_pnl:>+7.1f}%)   WORST = {worst_mn:<8} ({worst_pnl:>+7.1f}%)")

    print(f"\n  Symbol wins count:")
    for mn in sorted(sym_wins.keys(), key=lambda x: sym_wins[x], reverse=True):
        print(f"    {mn:<8}: {sym_wins[mn]} symbols")

    # ===== STRENGTHS & WEAKNESSES =====
    print("\n" + "=" * 160)
    print("SECTION 6: STRENGTHS & WEAKNESSES ANALYSIS")
    print("=" * 160)

    for mn in model_names:
        m = overall[mn]
        strengths = []
        weaknesses = []

        # Check if best in any criterion
        for crit_name, values in criteria.items():
            sorted_models = sorted(values.keys(), key=lambda x: values[x], reverse=True)
            if sorted_models[0] == mn:
                strengths.append(f"BEST {crit_name}")
            if sorted_models[-1] == mn:
                weaknesses.append(f"WORST {crit_name}")
            if sorted_models[-2] == mn:
                weaknesses.append(f"2nd worst {crit_name}")

        # Profitable symbols
        prof_syms = [s for s in all_syms if sym_data[s].get(mn, {"total_pnl": 0})["total_pnl"] > 0]
        loss_syms = [s for s in all_syms if sym_data[s].get(mn, {"total_pnl": 0})["total_pnl"] < 0]

        print(f"\n  {mn}:")
        print(f"    Rank: #{final_ranking.index(mn)+1}/{len(model_names)} (avg rank {avg_ranks[mn]:.2f})")
        print(f"    TotalPnL: {m['total_pnl']:>+.1f}%  WR: {m['wr']:.1f}%  PF: {m['pf']:.2f}  MaxDD: {m['max_dd']:.1f}%")
        print(f"    Profitable: {len(prof_syms)}/{len(all_syms)} symbols ({', '.join(prof_syms)})")
        if loss_syms:
            print(f"    Losing:     {len(loss_syms)}/{len(all_syms)} symbols ({', '.join(loss_syms)})")
        if strengths:
            print(f"    STRENGTHS:  {', '.join(strengths)}")
        if weaknesses:
            print(f"    WEAKNESSES: {', '.join(weaknesses)}")

        # Verdict
        rank_pos = final_ranking.index(mn) + 1
        if rank_pos <= 2:
            print(f"    VERDICT: TOP PERFORMER - keep and refine")
        elif rank_pos <= 4:
            print(f"    VERDICT: COMPETITIVE - has specific strengths worth preserving")
        elif rank_pos <= 6:
            print(f"    VERDICT: MEDIOCRE - no standout advantage, consider deprecating")
        else:
            print(f"    VERDICT: BOTTOM TIER - candidate for removal or major overhaul")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", type=str,
                        default="ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC")
    args = parser.parse_args()

    print(f"\nRunning FULL MODEL COMPARISON on: {args.pick}")
    print(f"Models: V11, V15, V16, V17, V18, V19, V19.1, Rule-based\n")

    all_results, pick = run_all_models(args.pick)
    print_report(all_results, pick)
