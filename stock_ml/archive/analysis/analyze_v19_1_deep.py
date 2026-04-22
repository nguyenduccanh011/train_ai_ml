"""
V19.1 Deep Analysis: Trade-level comparison with Rule and other models.
Focus on VND and problematic symbols.
"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_v19_1_compare import run_test, run_rule_test, calc_metrics, backtest_v19_1
from run_v11_compare import backtest_v11
from run_v17_compare import backtest_v17
from run_v18_compare import backtest_v18
from run_v19_compare import backtest_v19


def run_model_test(symbols_str, backtest_fn, label):
    """Run a specific model backtest and return trades."""
    from src.data.loader import DataLoader
    from src.data.splitter import WalkForwardSplitter
    from src.data.target import TargetGenerator
    from src.features.engine import FeatureEngine
    from src.models.registry import build_model

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
                t["model"] = label
            all_trades.extend(r["trades"])
    return all_trades


def run_v11_test(symbols_str):
    """V11 baseline (all mods off)."""
    from src.data.loader import DataLoader
    from src.data.splitter import WalkForwardSplitter
    from src.data.target import TargetGenerator
    from src.features.engine import FeatureEngine
    from src.models.registry import build_model

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
            r = backtest_v17(y_pred, rets, sym_test, feature_cols,
                             mod_a=False, mod_b=False, mod_c=False, mod_d=False,
                             mod_e=False, mod_f=False, mod_g=False, mod_h=False,
                             mod_i=False, mod_j=False)
            for t in r["trades"]:
                t["symbol"] = sym
                t["model"] = "V11"
            all_trades.extend(r["trades"])
    return all_trades


def analyze_trades_detail(trades, label):
    """Detailed trade analysis."""
    if not trades:
        return
    df = pd.DataFrame(trades)
    print(f"\n{'='*80}")
    print(f"  {label}: {len(trades)} trades")
    print(f"{'='*80}")

    # Win/Loss breakdown
    wins = df[df["pnl_pct"] > 0]
    losses = df[df["pnl_pct"] <= 0]
    print(f"  Wins: {len(wins)} ({len(wins)/len(df)*100:.1f}%)  Avg: {wins['pnl_pct'].mean():.2f}%" if len(wins) else "  Wins: 0")
    print(f"  Losses: {len(losses)} ({len(losses)/len(df)*100:.1f}%)  Avg: {losses['pnl_pct'].mean():.2f}%" if len(losses) else "  Losses: 0")
    print(f"  Total PnL: {df['pnl_pct'].sum():.1f}%  Avg: {df['pnl_pct'].mean():.2f}%")

    # Exit reason breakdown
    if "exit_reason" in df.columns:
        print(f"\n  Exit Reasons:")
        for reason, grp in df.groupby("exit_reason"):
            avg_pnl = grp["pnl_pct"].mean()
            total = grp["pnl_pct"].sum()
            print(f"    {reason:<25}: {len(grp):>3} trades, avg={avg_pnl:>+6.2f}%, total={total:>+8.1f}%")

    # Worst trades
    print(f"\n  Top 5 WORST trades:")
    worst = df.nsmallest(5, "pnl_pct")
    for _, t in worst.iterrows():
        sym = t.get("symbol", "?")
        entry = t.get("entry_date", "?")
        exit_d = t.get("exit_date", "?")
        reason = t.get("exit_reason", "?")
        print(f"    {sym} {entry}→{exit_d}: {t['pnl_pct']:>+6.2f}% (exit: {reason}, hold: {t.get('holding_days',0)}d)")

    # Best trades
    print(f"\n  Top 5 BEST trades:")
    best = df.nlargest(5, "pnl_pct")
    for _, t in best.iterrows():
        sym = t.get("symbol", "?")
        entry = t.get("entry_date", "?")
        exit_d = t.get("exit_date", "?")
        reason = t.get("exit_reason", "?")
        print(f"    {sym} {entry}→{exit_d}: {t['pnl_pct']:>+6.2f}% (exit: {reason}, hold: {t.get('holding_days',0)}d)")

    return df


def find_rule_profits_missed_by_model(rule_trades, model_trades, sym):
    """Find profitable rule trades that v19.1 missed or lost money on."""
    rule_sym = [t for t in rule_trades if t.get("symbol") == sym and t["pnl_pct"] > 3]
    model_sym = [t for t in model_trades if t.get("symbol") == sym]

    if not rule_sym:
        return []

    missed = []
    for rt in rule_sym:
        r_entry = rt.get("entry_date", "")
        r_exit = rt.get("exit_date", "")
        # Check if model had overlapping trade
        overlapping = []
        for mt in model_sym:
            m_entry = mt.get("entry_date", "")
            m_exit = mt.get("exit_date", "")
            # Simple overlap check
            if m_entry <= r_exit and m_exit >= r_entry:
                overlapping.append(mt)

        if not overlapping:
            missed.append({"type": "MISSED", "rule_trade": rt, "model_trades": []})
        else:
            # Check if model lost money during rule's profitable period
            model_total = sum(mt["pnl_pct"] for mt in overlapping)
            if model_total < rt["pnl_pct"] * 0.3:  # Model captured < 30% of rule profit
                missed.append({"type": "UNDERPERFORM", "rule_trade": rt,
                              "model_trades": overlapping, "model_total": model_total})
    return missed


def find_noisy_trades(trades, sym):
    """Find clusters of small losing trades (noise/chop)."""
    sym_trades = [t for t in trades if t.get("symbol") == sym]
    if len(sym_trades) < 3:
        return []

    noisy_clusters = []
    i = 0
    while i < len(sym_trades) - 2:
        # Look for 3+ consecutive small trades (hold < 10d, |pnl| < 3%)
        cluster = []
        j = i
        while j < len(sym_trades):
            t = sym_trades[j]
            if t.get("holding_days", 0) < 10 and abs(t["pnl_pct"]) < 3:
                cluster.append(t)
                j += 1
            else:
                break
        if len(cluster) >= 3:
            total_pnl = sum(t["pnl_pct"] for t in cluster)
            noisy_clusters.append({"trades": cluster, "total_pnl": total_pnl,
                                   "count": len(cluster)})
            i = j
        else:
            i += 1
    return noisy_clusters


def compare_model_entries(all_models_trades, sym):
    """Compare entry/exit quality across models for a symbol."""
    print(f"\n  === {sym}: Cross-model comparison ===")
    for label, trades in all_models_trades.items():
        sym_trades = [t for t in trades if t.get("symbol") == sym]
        m = calc_metrics(sym_trades)
        print(f"    {label:<12}: {m['trades']:>3} trades, WR={m['wr']:>5.1f}%, "
              f"AvgPnL={m['avg_pnl']:>+6.2f}%, TotPnL={m['total_pnl']:>+8.1f}%, PF={m['pf']:>5.2f}")

    # Find best trades unique to each model
    for label, trades in all_models_trades.items():
        sym_trades = [t for t in trades if t.get("symbol") == sym and t["pnl_pct"] > 5]
        if sym_trades:
            best = max(sym_trades, key=lambda x: x["pnl_pct"])
            # Check if other models captured this
            for other_label, other_trades in all_models_trades.items():
                if other_label == label:
                    continue
                other_sym = [t for t in other_trades if t.get("symbol") == sym]
                overlapping = [t for t in other_sym
                              if t.get("entry_date","") <= best.get("exit_date","")
                              and t.get("exit_date","") >= best.get("entry_date","")]
                if not overlapping:
                    print(f"    ★ {label} captured {best['pnl_pct']:+.1f}% trade "
                          f"({best.get('entry_date','')}→{best.get('exit_date','')}) "
                          f"that {other_label} MISSED")


if __name__ == "__main__":
    SYMBOLS = "ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC"

    print("=" * 100)
    print("V19.1 DEEP TRADE ANALYSIS")
    print("=" * 100)
    print("Loading data and running backtests... (this may take a few minutes)")

    # Run all models
    print("\n[1/6] Running V11 baseline...")
    t_v11 = run_v11_test(SYMBOLS)
    print(f"  → {len(t_v11)} trades")

    print("[2/6] Running V15/V16 (V17 variant)...")
    t_v15 = run_model_test(SYMBOLS, backtest_v17, "V15")  # V17 with same mods = ~V15/V16
    print(f"  → {len(t_v15)} trades")

    print("[3/6] Running V17...")
    t_v17 = run_model_test(SYMBOLS, backtest_v17, "V17")
    print(f"  → {len(t_v17)} trades")

    print("[4/6] Running V19...")
    t_v19 = run_model_test(SYMBOLS, backtest_v19, "V19")
    print(f"  → {len(t_v19)} trades")

    print("[5/6] Running V19.1...")
    t_v191 = run_model_test(SYMBOLS, backtest_v19_1, "V19.1")
    print(f"  → {len(t_v191)} trades")

    print("[6/6] Running Rule-based...")
    t_rule = run_rule_test(SYMBOLS)
    for t in t_rule:
        t["model"] = "Rule"
    print(f"  → {len(t_rule)} trades")

    all_models = {
        "V11": t_v11,
        "V17": t_v17,
        "V19": t_v19,
        "V19.1": t_v191,
        "Rule": t_rule,
    }

    # ══════════════════════════════════════════════════════════════
    # SECTION 1: Overall metrics comparison
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 1: OVERALL METRICS COMPARISON")
    print("=" * 100)
    print(f"{'Model':<12} | {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgHold':>8}")
    print("-" * 80)
    for label, trades in all_models.items():
        m = calc_metrics(trades)
        print(f"{label:<12} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d")

    # ══════════════════════════════════════════════════════════════
    # SECTION 2: Per-symbol breakdown
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 2: PER-SYMBOL BREAKDOWN")
    print("=" * 100)

    symbols_list = sorted(set(t.get("symbol","?") for t in t_v191) | set(t.get("symbol","?") for t in t_rule))
    print(f"{'Sym':<6}| {'V11':>9} | {'V17':>9} | {'V19':>9} | {'V19.1':>9} | {'Rule':>9} | {'V19.1-Rule':>11}")
    print("-" * 80)
    for sym in symbols_list:
        metrics = {}
        for label, trades in all_models.items():
            sym_t = [t for t in trades if t.get("symbol") == sym]
            metrics[label] = calc_metrics(sym_t)
        v191_tot = metrics["V19.1"]["total_pnl"]
        rule_tot = metrics["Rule"]["total_pnl"]
        print(f"{sym:<6}| {metrics['V11']['total_pnl']:>+8.1f}% | {metrics['V17']['total_pnl']:>+8.1f}% | "
              f"{metrics['V19']['total_pnl']:>+8.1f}% | {v191_tot:>+8.1f}% | {rule_tot:>+8.1f}% | {v191_tot-rule_tot:>+10.1f}%")

    # ══════════════════════════════════════════════════════════════
    # SECTION 3: DEEP DIVE - VND
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 3: DEEP DIVE - VND (high_beta profile)")
    print("=" * 100)

    for label in ["V19.1", "Rule"]:
        sym_trades = [t for t in all_models[label] if t.get("symbol") == "VND"]
        analyze_trades_detail(sym_trades, f"VND - {label}")

    # Rule profits missed by V19.1
    print("\n  --- VND: Rule profits MISSED or UNDERPERFORMED by V19.1 ---")
    missed = find_rule_profits_missed_by_model(t_rule, t_v191, "VND")
    for item in missed:
        rt = item["rule_trade"]
        print(f"  [{item['type']}] Rule: {rt.get('entry_date','')}→{rt.get('exit_date','')}: "
              f"{rt['pnl_pct']:+.1f}% ({rt.get('holding_days',0)}d)")
        if item["model_trades"]:
            for mt in item["model_trades"]:
                print(f"    V19.1: {mt.get('entry_date','')}→{mt.get('exit_date','')}: "
                      f"{mt['pnl_pct']:+.1f}% ({mt.get('exit_reason','')})")
            print(f"    Model total: {item.get('model_total',0):+.1f}% vs Rule: {rt['pnl_pct']:+.1f}%")

    # Noisy trades
    print("\n  --- VND: V19.1 Noisy trade clusters ---")
    noisy = find_noisy_trades(t_v191, "VND")
    for cluster in noisy:
        print(f"  Cluster of {cluster['count']} small trades, total PnL: {cluster['total_pnl']:+.1f}%")
        for t in cluster["trades"]:
            print(f"    {t.get('entry_date','')}→{t.get('exit_date','')}: {t['pnl_pct']:+.2f}% "
                  f"({t.get('exit_reason','')}, {t.get('holding_days',0)}d)")

    # Cross-model comparison for VND
    compare_model_entries(all_models, "VND")

    # ══════════════════════════════════════════════════════════════
    # SECTION 4: PROBLEMATIC SYMBOLS ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 4: PROBLEMATIC SYMBOLS (V19.1 underperforms Rule)")
    print("=" * 100)

    problem_symbols = []
    for sym in symbols_list:
        v191_t = [t for t in t_v191 if t.get("symbol") == sym]
        rule_t = [t for t in t_rule if t.get("symbol") == sym]
        m191 = calc_metrics(v191_t)
        mr = calc_metrics(rule_t)
        gap = m191["total_pnl"] - mr["total_pnl"]
        if gap < -5:  # V19.1 underperforms rule by >5%
            problem_symbols.append((sym, gap, m191, mr))

    for sym, gap, m191, mr in sorted(problem_symbols, key=lambda x: x[1]):
        print(f"\n  ▼ {sym}: V19.1={m191['total_pnl']:+.1f}% vs Rule={mr['total_pnl']:+.1f}% (gap={gap:+.1f}%)")

        # Missed profitable rule trades
        missed = find_rule_profits_missed_by_model(t_rule, t_v191, sym)
        if missed:
            print(f"    Missed/underperformed rule trades:")
            for item in missed[:5]:
                rt = item["rule_trade"]
                print(f"      [{item['type']}] Rule: {rt.get('entry_date','')}→{rt.get('exit_date','')}: "
                      f"{rt['pnl_pct']:+.1f}%")

        # Noisy clusters
        noisy = find_noisy_trades(t_v191, sym)
        if noisy:
            total_noise_loss = sum(c["total_pnl"] for c in noisy)
            print(f"    Noisy clusters: {len(noisy)}, total noise PnL: {total_noise_loss:+.1f}%")

        # V19.1 worst trades for this symbol
        sym_trades = [t for t in t_v191 if t.get("symbol") == sym]
        worst = sorted(sym_trades, key=lambda x: x["pnl_pct"])[:3]
        if worst:
            print(f"    Worst V19.1 trades:")
            for t in worst:
                print(f"      {t.get('entry_date','')}→{t.get('exit_date','')}: {t['pnl_pct']:+.2f}% "
                      f"(exit: {t.get('exit_reason','')}, trend: {t.get('entry_trend','')}, "
                      f"score: {t.get('entry_score','')}, size: {t.get('position_size','')}, "
                      f"choppy: {t.get('entry_choppy_regime','')})")

    # ══════════════════════════════════════════════════════════════
    # SECTION 5: WHERE OTHER MODELS DO BETTER
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 5: CROSS-MODEL COMPARISON - WHERE OTHERS EXCEL")
    print("=" * 100)

    for sym in symbols_list:
        compare_model_entries(all_models, sym)

    # ══════════════════════════════════════════════════════════════
    # SECTION 6: V19.1 EXIT REASON ANALYSIS (all symbols)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 6: V19.1 EXIT REASON ANALYSIS (all symbols)")
    print("=" * 100)
    df_v191 = pd.DataFrame(t_v191)
    if len(df_v191) > 0 and "exit_reason" in df_v191.columns:
        for reason, grp in df_v191.groupby("exit_reason"):
            wins = len(grp[grp["pnl_pct"] > 0])
            avg = grp["pnl_pct"].mean()
            total = grp["pnl_pct"].sum()
            print(f"  {reason:<25}: {len(grp):>4} trades, WR={wins/len(grp)*100:>5.1f}%, "
                  f"avg={avg:>+6.2f}%, total={total:>+8.1f}%")

    # ══════════════════════════════════════════════════════════════
    # SECTION 7: V19.1 ENTRY TYPE ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 7: V19.1 ENTRY TYPE ANALYSIS")
    print("=" * 100)
    if len(df_v191) > 0:
        for entry_type in ["quick_reentry", "breakout_entry", "vshape_entry"]:
            if entry_type in df_v191.columns:
                grp = df_v191[df_v191[entry_type] == True]
                if len(grp) > 0:
                    m = calc_metrics(grp.to_dict("records"))
                    print(f"  {entry_type:<20}: {m['trades']:>3} trades, WR={m['wr']:>5.1f}%, "
                          f"avg={m['avg_pnl']:>+6.2f}%, total={m['total_pnl']:>+8.1f}%, PF={m['pf']:>5.2f}")

        # Normal ML entries
        normal = df_v191[
            (df_v191.get("quick_reentry", False) != True) &
            (df_v191.get("breakout_entry", False) != True) &
            (df_v191.get("vshape_entry", False) != True)
        ] if all(c in df_v191.columns for c in ["quick_reentry","breakout_entry","vshape_entry"]) else df_v191
        m = calc_metrics(normal.to_dict("records"))
        print(f"  {'normal_ml':<20}: {m['trades']:>3} trades, WR={m['wr']:>5.1f}%, "
              f"avg={m['avg_pnl']:>+6.2f}%, total={m['total_pnl']:>+8.1f}%, PF={m['pf']:>5.2f}")

    # ══════════════════════════════════════════════════════════════
    # SECTION 8: TREND ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 8: V19.1 BY ENTRY TREND")
    print("=" * 100)
    if len(df_v191) > 0 and "entry_trend" in df_v191.columns:
        for trend, grp in df_v191.groupby("entry_trend"):
            m = calc_metrics(grp.to_dict("records"))
            print(f"  {trend:<12}: {m['trades']:>4} trades, WR={m['wr']:>5.1f}%, "
                  f"avg={m['avg_pnl']:>+6.2f}%, total={m['total_pnl']:>+8.1f}%, PF={m['pf']:>5.2f}")

    # ══════════════════════════════════════════════════════════════
    # SECTION 9: POSITION SIZE ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 9: V19.1 BY POSITION SIZE")
    print("=" * 100)
    if len(df_v191) > 0 and "position_size" in df_v191.columns:
        df_v191["size_bucket"] = pd.cut(df_v191["position_size"], bins=[0, 0.4, 0.6, 0.8, 1.01],
                                         labels=["0-40%", "40-60%", "60-80%", "80-100%"])
        for bucket, grp in df_v191.groupby("size_bucket"):
            if len(grp) == 0:
                continue
            m = calc_metrics(grp.to_dict("records"))
            print(f"  {str(bucket):<12}: {m['trades']:>4} trades, WR={m['wr']:>5.1f}%, "
                  f"avg={m['avg_pnl']:>+6.2f}%, total={m['total_pnl']:>+8.1f}%, PF={m['pf']:>5.2f}")

    # ══════════════════════════════════════════════════════════════
    # SECTION 10: ROOT CAUSE ANALYSIS & RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 10: ROOT CAUSE ANALYSIS & IMPROVEMENT RECOMMENDATIONS")
    print("=" * 100)

    # Compute key findings
    v191_metrics = calc_metrics(t_v191)
    rule_metrics = calc_metrics(t_rule)

    # Count losing exit reasons
    if len(df_v191) > 0:
        loss_by_reason = df_v191[df_v191["pnl_pct"] < 0].groupby("exit_reason")["pnl_pct"].agg(["sum","count"])
        worst_reason = loss_by_reason["sum"].idxmin() if len(loss_by_reason) > 0 else "unknown"
        worst_reason_loss = loss_by_reason["sum"].min() if len(loss_by_reason) > 0 else 0

    # Total noise
    total_noise_trades = 0
    total_noise_loss = 0
    for sym in symbols_list:
        noisy = find_noisy_trades(t_v191, sym)
        for c in noisy:
            total_noise_trades += c["count"]
            total_noise_loss += c["total_pnl"]

    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║                    ROOT CAUSE ANALYSIS                          ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║                                                                  ║
  ║  1. V19.1 Total PnL: {v191_metrics['total_pnl']:>+8.1f}%  vs Rule: {rule_metrics['total_pnl']:>+8.1f}%         ║
  ║     Gap: {v191_metrics['total_pnl']-rule_metrics['total_pnl']:>+8.1f}%                                         ║
  ║                                                                  ║
  ║  2. V19.1 Win Rate: {v191_metrics['wr']:>5.1f}%  vs Rule: {rule_metrics['wr']:>5.1f}%            ║
  ║     V19.1 trades: {v191_metrics['trades']:>4}  vs Rule: {rule_metrics['trades']:>4}              ║
  ║                                                                  ║
  ║  3. Noise trades: {total_noise_trades:>3} trades, total loss: {total_noise_loss:>+7.1f}%     ║
  ║     → Many short-hold losing trades in choppy regimes            ║
  ║                                                                  ║
  ║  4. Worst exit reason: {str(worst_reason):<20} ({worst_reason_loss:>+7.1f}%)     ║
  ║                                                                  ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║                  IMPROVEMENT RECOMMENDATIONS                     ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║                                                                  ║
  ║  A. ENTRY QUALITY IMPROVEMENTS:                                  ║
  ║     1. Stricter anti-chop filter: require min 2 of:              ║
  ║        - MACD histogram > 0                                      ║
  ║        - Close > SMA20                                           ║
  ║        - RSI slope positive                                      ║
  ║        - Volume above 1.2x avg                                   ║
  ║     2. Add regime-adaptive entry score minimum:                   ║
  ║        weak trend → min_score=4 (currently 3)                    ║
  ║     3. For high_beta (VND, SSI): require breakout_setup >= 2     ║
  ║        before allowing non-breakout entries                       ║
  ║                                                                  ║
  ║  B. EXIT QUALITY IMPROVEMENTS:                                   ║
  ║     1. Tighter profit protection: start trailing at 10%          ║
  ║        (currently 15-20%)                                        ║
  ║     2. For signal exits: use MACD crossover as primary           ║
  ║        rather than multi-score system                             ║
  ║     3. Add time-decay: if hold > 20d and < 5% profit,           ║
  ║        reduce exit threshold to accelerate rotation              ║
  ║                                                                  ║
  ║  C. POSITION SIZING:                                             ║
  ║     1. Reduce size more aggressively in weak trends              ║
  ║     2. Scale down further for symbols with high ATR ratio        ║
  ║     3. Consider 0% allocation in confirmed bear regimes          ║
  ║                                                                  ║
  ║  D. RULE INTEGRATION:                                            ║
  ║     1. Use rule-based entry timing as confirmation signal         ║
  ║     2. Add rule's SMA crossover as hard entry requirement        ║
  ║     3. Blend ML probability with rule confidence score           ║
  ║                                                                  ║
  ║  E. NEW FEATURES TO EXPLORE:                                     ║
  ║     1. Sector momentum (relative strength vs VN-Index)           ║
  ║     2. Institutional flow proxy (large-block volume)             ║
  ║     3. Volatility regime classifier (low/normal/high/crisis)     ║
  ║     4. Price-volume trend divergence on longer timeframe         ║
  ║                                                                  ║
  ╚══════════════════════════════════════════════════════════════════╝
""")

    # ══════════════════════════════════════════════════════════════
    # SECTION 11: SPECIFIC MODEL ADVANTAGES
    # ══════════════════════════════════════════════════════════════
    print("=" * 100)
    print("SECTION 11: WHERE EACH MODEL HAS UNIQUE ADVANTAGES")
    print("=" * 100)

    for sym in symbols_list:
        best_model = None
        best_pnl = -999
        model_pnls = {}
        for label, trades in all_models.items():
            sym_t = [t for t in trades if t.get("symbol") == sym]
            m = calc_metrics(sym_t)
            model_pnls[label] = m["total_pnl"]
            if m["total_pnl"] > best_pnl:
                best_pnl = m["total_pnl"]
                best_model = label

        v191_pnl = model_pnls.get("V19.1", 0)
        if best_model != "V19.1" and best_pnl - v191_pnl > 5:
            print(f"\n  {sym}: Best={best_model} ({best_pnl:+.1f}%) vs V19.1 ({v191_pnl:+.1f}%)")
            # Show best model's best trades
            best_trades = [t for t in all_models[best_model] if t.get("symbol") == sym]
            best_trade = max(best_trades, key=lambda x: x["pnl_pct"]) if best_trades else None
            if best_trade:
                print(f"    Best trade by {best_model}: {best_trade.get('entry_date','')}→"
                      f"{best_trade.get('exit_date','')}: {best_trade['pnl_pct']:+.1f}%")

            # Root cause analysis
            v191_sym_trades = [t for t in t_v191 if t.get("symbol") == sym]
            if v191_sym_trades:
                losses = [t for t in v191_sym_trades if t["pnl_pct"] < -3]
                if losses:
                    print(f"    V19.1 has {len(losses)} trades losing >3%:")
                    for t in sorted(losses, key=lambda x: x["pnl_pct"])[:3]:
                        print(f"      {t.get('entry_date','')}→{t.get('exit_date','')}: "
                              f"{t['pnl_pct']:+.2f}% (exit: {t.get('exit_reason','')}, "
                              f"trend: {t.get('entry_trend','')}, choppy: {t.get('entry_choppy_regime','')})")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
