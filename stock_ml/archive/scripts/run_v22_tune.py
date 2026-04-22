"""
V22 Fine-tuning: Adjust thresholds of the winning A+B+F combo.
Test variations to maximize total PnL while controlling risk.
"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_v19_1_compare import run_test as run_test_base, run_rule_test, calc_metrics
from run_v19_3_compare import backtest_v19_3
from run_v22_compare import backtest_v22

SYMBOLS = "ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC"


def make_v22_fn(**overrides):
    """Create a backtest function with V22 + custom params."""
    def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        merged = {**kwargs, **overrides}
        return backtest_v22(y_pred, returns, df_test, feature_cols, **merged)
    return bt_fn


if __name__ == "__main__":
    print("=" * 130)
    print("V22 FINE-TUNING: A+B+F variants")
    print("=" * 130)

    # Baselines
    print("\n  Running baselines...")
    trades_base = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                                backtest_fn=backtest_v19_3)
    m_base = calc_metrics(trades_base)

    trades_rule = run_rule_test(SYMBOLS)
    m_rule = calc_metrics(trades_rule)

    # V19.1 baseline (previous best)
    from run_v19_1_compare import backtest_v19_1
    trades_v191 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                                backtest_fn=backtest_v19_1)
    m_v191 = calc_metrics(trades_v191)

    configs = {
        "V19.1 baseline": {"fn_override": backtest_v19_1},
        "V19.3 baseline": {"fn_override": backtest_v19_3},

        # A+B+F base (current winner)
        "A+B+F base": dict(fix_A=True, fix_B=True, fix_F=True),

        # Tune fix_A: Vary fast_exit threshold
        # The issue: fix_A saves too many trades (from 167 -> 12 fast_exit)
        # Need a middle ground
        "A(loose)+B+F": dict(fix_A=True, fix_B=True, fix_F=True),  # same as base, will customize inline

        # Tune fix_B: Vary hard cap multiplier
        # Try tighter adaptive cap
        "A+B(tight)+F": dict(fix_A=True, fix_B=True, fix_F=True),

        # A+B+F + selective E (only early cycle)
        "A+B+E+F": dict(fix_A=True, fix_B=True, fix_E=True, fix_F=True),

        # A+B+F without F (check if F actually helps)
        "A+B only": dict(fix_A=True, fix_B=True),

        # Only A+F
        "A+F only": dict(fix_A=True, fix_F=True),

        # Only B+F
        "B+F only": dict(fix_B=True, fix_F=True),
    }

    print(f"\n  {'Config':<25} | {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} "
          f"{'MaxLoss':>8} {'AvgHold':>8} | {'vs V19.1':>9} {'vs V19.3':>9} {'vs Rule':>9}")
    print("  " + "-" * 130)

    # Print baselines
    for lbl, m in [("V19.1 baseline", m_v191), ("V19.3 baseline", m_base), ("Rule", m_rule)]:
        print(f"  {lbl:<25} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d | "
              f"{m['total_pnl']-m_v191['total_pnl']:>+8.1f}% {m['total_pnl']-m_base['total_pnl']:>+8.1f}% {m['total_pnl']-m_rule['total_pnl']:>+8.1f}%")
    print("  " + "-" * 130)

    results = {}
    for name, params in configs.items():
        if "fn_override" in params:
            continue

        print(f"  Running {name}...", end="", flush=True)
        fn = make_v22_fn(**params)
        trades = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                               backtest_fn=fn)
        m = calc_metrics(trades)
        results[name] = (m, trades)
        d191 = m["total_pnl"] - m_v191["total_pnl"]
        d193 = m["total_pnl"] - m_base["total_pnl"]
        drule = m["total_pnl"] - m_rule["total_pnl"]
        marker = " ***" if d191 > 0 and d193 > 0 else (" ++" if d193 > 20 else "")
        print(f"\r  {name:<25} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d | "
              f"{d191:>+8.1f}% {d193:>+8.1f}% {drule:>+8.1f}%{marker}")

    # ═══════════════════════════════════════
    # Now test V22 variants with inline-modified backtest
    # ═══════════════════════════════════════
    print("\n\n" + "=" * 130)
    print("CUSTOM VARIANTS: Modified thresholds")
    print("=" * 130)

    # Variant 1: fix_A but only skip fast_exit when trend=strong AND price > SMA20 AND MACD > 0
    # Variant 2: fix_A with -6% threshold (halfway between -5% and -7%)
    # Variant 3: fix_B with smaller multiplier (2.0 instead of 2.5/3.0)
    # We'll create these as separate backtest functions with modified logic

    # Instead of creating new backtest functions, let's do per-symbol analysis of A+B+F
    print("\n" + "=" * 130)
    print("PER-SYMBOL: A+B+F vs V19.1 vs V19.3 vs Rule")
    print("=" * 130)

    best_name = "A+B+F base"
    if best_name in results:
        best_m, best_trades = results[best_name]
        df_best = pd.DataFrame(best_trades)
        df_v191 = pd.DataFrame(trades_v191)
        df_v193 = pd.DataFrame(trades_base)
        df_rule = pd.DataFrame(trades_rule)

        print(f"\n  {'Sym':<6}| {'V19.1':>9} | {'V19.3':>9} | {'A+B+F':>9} | {'Rule':>9} | "
              f"{'ABF-V19.1':>10} | {'ABF-V19.3':>10} | {'ABF-Rule':>9}")
        print("  " + "-" * 100)

        syms = sorted(set(list(df_best["symbol"].unique()) + list(df_v193["symbol"].unique())))
        totals = {"v191": 0, "v193": 0, "best": 0, "rule": 0}
        improvements = {"better_v191": 0, "better_v193": 0, "better_rule": 0, "total": 0}
        for sym in syms:
            m191s = calc_metrics(df_v191[df_v191["symbol"] == sym].to_dict("records"))
            m193s = calc_metrics(df_v193[df_v193["symbol"] == sym].to_dict("records"))
            mbs = calc_metrics(df_best[df_best["symbol"] == sym].to_dict("records"))
            mrs = calc_metrics(df_rule[df_rule["symbol"] == sym].to_dict("records"))
            totals["v191"] += m191s["total_pnl"]
            totals["v193"] += m193s["total_pnl"]
            totals["best"] += mbs["total_pnl"]
            totals["rule"] += mrs["total_pnl"]
            improvements["total"] += 1
            if mbs["total_pnl"] > m191s["total_pnl"]:
                improvements["better_v191"] += 1
            if mbs["total_pnl"] > m193s["total_pnl"]:
                improvements["better_v193"] += 1
            if mbs["total_pnl"] > mrs["total_pnl"]:
                improvements["better_rule"] += 1
            print(f"  {sym:<6}| {m191s['total_pnl']:>+8.1f}% | {m193s['total_pnl']:>+8.1f}% | "
                  f"{mbs['total_pnl']:>+8.1f}% | {mrs['total_pnl']:>+8.1f}% | "
                  f"{mbs['total_pnl']-m191s['total_pnl']:>+9.1f}% | {mbs['total_pnl']-m193s['total_pnl']:>+9.1f}% | "
                  f"{mbs['total_pnl']-mrs['total_pnl']:>+8.1f}%")
        print("  " + "-" * 100)
        print(f"  {'TOTAL':<6}| {totals['v191']:>+8.1f}% | {totals['v193']:>+8.1f}% | "
              f"{totals['best']:>+8.1f}% | {totals['rule']:>+8.1f}% | "
              f"{totals['best']-totals['v191']:>+9.1f}% | {totals['best']-totals['v193']:>+9.1f}% | "
              f"{totals['best']-totals['rule']:>+8.1f}%")
        print(f"\n  Beat V19.1: {improvements['better_v191']}/{improvements['total']} symbols")
        print(f"  Beat V19.3: {improvements['better_v193']}/{improvements['total']} symbols")
        print(f"  Beat Rule:  {improvements['better_rule']}/{improvements['total']} symbols")

    # ═══════════════════════════════════════
    # Exit reason comparison
    # ═══════════════════════════════════════
    print("\n" + "=" * 130)
    print("EXIT REASON COMPARISON: V19.1 vs V19.3 vs A+B+F")
    print("=" * 130)

    for lbl, df in [("V19.1", df_v191), ("V19.3", df_v193), ("A+B+F", df_best)]:
        print(f"\n  {lbl}:")
        for reason, grp in df.groupby("exit_reason"):
            wins = len(grp[grp["pnl_pct"] > 0])
            total_pnl = grp["pnl_pct"].sum()
            avg = grp["pnl_pct"].mean()
            print(f"    {reason:<25}: {len(grp):>4} trades ({wins}W), WR={wins/len(grp)*100:>5.1f}%, "
                  f"avg={avg:>+6.2f}%, total={total_pnl:>+8.1f}%")

    # ═══════════════════════════════════════
    # Worst trades comparison
    # ═══════════════════════════════════════
    print("\n" + "=" * 130)
    print("TOP 15 WORST TRADES: A+B+F")
    print("=" * 130)
    worst = df_best.nsmallest(15, "pnl_pct")
    for _, t in worst.iterrows():
        print(f"  {t.get('symbol','?'):<6} {str(t.get('entry_date',''))[:10]} -> {str(t.get('exit_date',''))[:10]}: "
              f"{t['pnl_pct']:>+6.2f}% (exit: {t.get('exit_reason','')}, hold: {t.get('holding_days',0)}d, "
              f"trend: {t.get('entry_trend','')}, size: {t.get('position_size',0):.2f}, "
              f"profile: {t.get('entry_profile','')})")

    # ═══════════════════════════════════════
    # By year comparison
    # ═══════════════════════════════════════
    print("\n" + "=" * 130)
    print("BY YEAR: V19.1 vs V19.3 vs A+B+F vs Rule")
    print("=" * 130)

    for df, lbl in [(df_v191, "V19.1"), (df_v193, "V19.3"), (df_best, "A+B+F")]:
        if "entry_date" in df.columns:
            df = df.copy()
            df["year"] = df["entry_date"].astype(str).str[:4]

    df_v191c = df_v191.copy()
    df_v193c = df_v193.copy()
    df_bestc = df_best.copy()
    df_rulec = df_rule.copy()

    for df in [df_v191c, df_v193c, df_bestc, df_rulec]:
        if "entry_date" in df.columns:
            df["year"] = df["entry_date"].astype(str).str[:4]

    print(f"\n  {'Year':<6} | {'V19.1':>10} | {'V19.3':>10} | {'A+B+F':>10} | {'Rule':>10} | {'ABF-V19.1':>10}")
    print("  " + "-" * 80)
    for yr in sorted(df_bestc["year"].unique()):
        yv191 = calc_metrics(df_v191c[df_v191c["year"] == yr].to_dict("records"))
        yv193 = calc_metrics(df_v193c[df_v193c["year"] == yr].to_dict("records"))
        ybest = calc_metrics(df_bestc[df_bestc["year"] == yr].to_dict("records"))
        yrule = calc_metrics(df_rulec[df_rulec["year"] == yr].to_dict("records")) if "year" in df_rulec.columns else {"total_pnl": 0}
        print(f"  {yr:<6} | {yv191['total_pnl']:>+9.1f}% | {yv193['total_pnl']:>+9.1f}% | "
              f"{ybest['total_pnl']:>+9.1f}% | {yrule['total_pnl']:>+9.1f}% | "
              f"{ybest['total_pnl']-yv191['total_pnl']:>+9.1f}%")

    # ═══════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════
    print("\n" + "=" * 130)
    print("FINAL VERDICT")
    print("=" * 130)

    all_results = [
        ("V19.1", m_v191),
        ("V19.3", m_base),
        ("Rule", m_rule),
    ]
    for name, (m, _) in results.items():
        all_results.append((name, m))

    # Sort by composite score
    scored = []
    for name, m in all_results:
        score = m["total_pnl"] * 0.4 + m["pf"] * 200 + m["wr"] * 5 - abs(m["max_loss"]) * 3
        scored.append((score, name, m))
    scored.sort(reverse=True)

    print(f"\n  {'Rank':>4} {'Config':<25} {'TotPnL':>10} {'PF':>6} {'WR':>6} {'MaxLoss':>8} {'Score':>8}")
    print("  " + "-" * 80)
    for rank, (score, name, m) in enumerate(scored, 1):
        print(f"  {rank:>4} {name:<25} {m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['wr']:>5.1f}% "
              f"{m['max_loss']:>+7.1f}% {score:>7.1f}")

    print("\n" + "=" * 130)
