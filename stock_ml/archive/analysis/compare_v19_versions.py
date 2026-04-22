"""
Compare V19.1 vs V19.2 vs V19.3 vs V19.4 + Rule baseline.
Run full backtest on all 14 symbols and produce detailed comparison.
"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_v19_1_compare import run_test as run_test_v191, run_rule_test, calc_metrics, backtest_v19_1
from run_v19_2_compare import backtest_v19_2
from run_v19_3_compare import backtest_v19_3
from run_v19_4_compare import backtest_v19_4


SYMBOLS = "ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC"


def run_version(backtest_fn, label):
    """Run a backtest version."""
    trades = run_test_v191(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                           backtest_fn=backtest_fn)
    for t in trades:
        t["model"] = label
    return trades


def print_section(title):
    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")


if __name__ == "__main__":
    print("=" * 120)
    print("V19.x VERSION COMPARISON - Full Backtest")
    print("=" * 120)

    # Run all versions
    versions = {}
    for label, fn in [
        ("V19.1", backtest_v19_1),
        ("V19.2", backtest_v19_2),
        ("V19.3", backtest_v19_3),
        ("V19.4", backtest_v19_4),
    ]:
        print(f"\n  Running {label}...")
        trades = run_version(fn, label)
        versions[label] = trades
        m = calc_metrics(trades)
        print(f"    -> {m['trades']} trades, WR={m['wr']:.1f}%, TotPnL={m['total_pnl']:+.1f}%, PF={m['pf']:.2f}")

    print(f"\n  Running Rule...")
    t_rule = run_rule_test(SYMBOLS)
    for t in t_rule:
        t["model"] = "Rule"
    versions["Rule"] = t_rule
    m = calc_metrics(t_rule)
    print(f"    -> {m['trades']} trades, WR={m['wr']:.1f}%, TotPnL={m['total_pnl']:+.1f}%, PF={m['pf']:.2f}")

    # ═══════════════════════════════════════
    # SECTION 1: Overall comparison
    # ═══════════════════════════════════════
    print_section("1. OVERALL METRICS COMPARISON")
    print(f"  {'Model':<10} | {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgHold':>8}")
    print("  " + "-" * 80)
    for label, trades in versions.items():
        m = calc_metrics(trades)
        print(f"  {label:<10} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d")

    # Delta table
    print(f"\n  {'Metric':<12} | {'V19.2-V19.1':>12} | {'V19.3-V19.1':>12} | {'V19.4-V19.1':>12} | {'V19.4-Rule':>12}")
    print("  " + "-" * 65)
    m191 = calc_metrics(versions["V19.1"])
    mr = calc_metrics(versions["Rule"])
    # Delta print
    for metric_name, key in [("TotalPnL", "total_pnl"), ("WinRate", "wr"), ("PF", "pf"), ("MaxLoss", "max_loss"), ("AvgPnL", "avg_pnl")]:
        vals = []
        for label in ["V19.2", "V19.3", "V19.4"]:
            m = calc_metrics(versions[label])
            delta_191 = m[key] - m191[key]
            vals.append(delta_191)
        delta_rule = calc_metrics(versions["V19.4"])[key] - mr[key]
        sfx = "%" if key != "pf" else ""
        if key == "pf":
            print(f"  {metric_name:<12} | {vals[0]:>+11.2f}{sfx} | {vals[1]:>+11.2f}{sfx} | {vals[2]:>+11.2f}{sfx} | {delta_rule:>+11.2f}{sfx}")
        else:
            print(f"  {metric_name:<12} | {vals[0]:>+11.1f}{sfx} | {vals[1]:>+11.1f}{sfx} | {vals[2]:>+11.1f}{sfx} | {delta_rule:>+11.1f}{sfx}")

    # ═══════════════════════════════════════
    # SECTION 2: Per-symbol comparison
    # ═══════════════════════════════════════
    print_section("2. PER-SYMBOL COMPARISON")

    symbols_list = sorted(set(t.get("symbol", "?") for t in versions["V19.1"]))
    print(f"  {'Sym':<6}| {'V19.1':>9} | {'V19.2':>9} | {'V19.3':>9} | {'V19.4':>9} | {'Rule':>9} | {'V19.4-V19.1':>12} | {'V19.4-Rule':>11}")
    print("  " + "-" * 100)

    total = {k: 0 for k in versions}
    for sym in symbols_list:
        row = {}
        for label, trades in versions.items():
            sym_t = [t for t in trades if t.get("symbol") == sym]
            row[label] = calc_metrics(sym_t)["total_pnl"]
            total[label] += row[label]
        d191 = row["V19.4"] - row["V19.1"]
        dr = row["V19.4"] - row["Rule"]
        print(f"  {sym:<6}| {row['V19.1']:>+8.1f}% | {row['V19.2']:>+8.1f}% | {row['V19.3']:>+8.1f}% | "
              f"{row['V19.4']:>+8.1f}% | {row['Rule']:>+8.1f}% | {d191:>+11.1f}% | {dr:>+10.1f}%")

    print("  " + "-" * 100)
    d191 = total["V19.4"] - total["V19.1"]
    dr = total["V19.4"] - total["Rule"]
    print(f"  {'TOTAL':<6}| {total['V19.1']:>+8.1f}% | {total['V19.2']:>+8.1f}% | {total['V19.3']:>+8.1f}% | "
          f"{total['V19.4']:>+8.1f}% | {total['Rule']:>+8.1f}% | {d191:>+11.1f}% | {dr:>+10.1f}%")

    # ═══════════════════════════════════════
    # SECTION 3: Exit reason comparison
    # ═══════════════════════════════════════
    print_section("3. EXIT REASON ANALYSIS (V19.1 vs V19.4)")

    for label in ["V19.1", "V19.2", "V19.3", "V19.4"]:
        trades = versions[label]
        df = pd.DataFrame(trades)
        if len(df) == 0 or "exit_reason" not in df.columns:
            continue
        print(f"\n  {label}:")
        for reason, grp in df.groupby("exit_reason"):
            wins = len(grp[grp["pnl_pct"] > 0])
            avg = grp["pnl_pct"].mean()
            total_pnl = grp["pnl_pct"].sum()
            print(f"    {reason:<25}: {len(grp):>4} trades, WR={wins/len(grp)*100:>5.1f}%, "
                  f"avg={avg:>+6.2f}%, total={total_pnl:>+8.1f}%")

    # ═══════════════════════════════════════
    # SECTION 4: Position size distribution comparison
    # ═══════════════════════════════════════
    print_section("4. POSITION SIZE DISTRIBUTION")

    for label in ["V19.1", "V19.3", "V19.4"]:
        trades = versions[label]
        df = pd.DataFrame(trades)
        if len(df) == 0 or "position_size" not in df.columns:
            continue
        print(f"\n  {label}:")
        df["size_bucket"] = pd.cut(df["position_size"], bins=[0, 0.35, 0.55, 0.75, 1.01],
                                    labels=["0-35%", "35-55%", "55-75%", "75-100%"])
        for bucket, grp in df.groupby("size_bucket"):
            if len(grp) == 0:
                continue
            m = calc_metrics(grp.to_dict("records"))
            print(f"    {str(bucket):<10}: {m['trades']:>4} trades, WR={m['wr']:>5.1f}%, "
                  f"avg={m['avg_pnl']:>+6.2f}%, total={m['total_pnl']:>+8.1f}%, PF={m['pf']:>5.2f}")

    # ═══════════════════════════════════════
    # SECTION 5: Problem symbols - did they improve?
    # ═══════════════════════════════════════
    print_section("5. PROBLEM SYMBOLS - IMPROVEMENT CHECK")
    problem_syms = ["DGC", "AAS", "VND", "SSI", "ACB", "FPT", "REE"]
    print(f"  {'Sym':<6}| {'V19.1':>9} | {'V19.4':>9} | {'Rule':>9} | {'Delta':>8} | {'Status':>12}")
    print("  " + "-" * 70)
    for sym in problem_syms:
        v191_t = [t for t in versions["V19.1"] if t.get("symbol") == sym]
        v194_t = [t for t in versions["V19.4"] if t.get("symbol") == sym]
        rule_t = [t for t in versions["Rule"] if t.get("symbol") == sym]
        m191 = calc_metrics(v191_t)
        m194 = calc_metrics(v194_t)
        mr = calc_metrics(rule_t)
        delta = m194["total_pnl"] - m191["total_pnl"]
        gap_to_rule = m194["total_pnl"] - mr["total_pnl"]
        status = "IMPROVED" if delta > 5 else ("WORSE" if delta < -5 else "SAME")
        if gap_to_rule > 0:
            status += " +Rule"
        print(f"  {sym:<6}| {m191['total_pnl']:>+8.1f}% | {m194['total_pnl']:>+8.1f}% | {mr['total_pnl']:>+8.1f}% | "
              f"{delta:>+7.1f}% | {status:>12}")

    # ═══════════════════════════════════════
    # SECTION 6: Worst trades comparison
    # ═══════════════════════════════════════
    print_section("6. WORST TRADES COMPARISON")
    for label in ["V19.1", "V19.4"]:
        trades = versions[label]
        df = pd.DataFrame(trades)
        worst = df.nsmallest(10, "pnl_pct")
        print(f"\n  {label} - Top 10 worst trades:")
        for _, t in worst.iterrows():
            print(f"    {t.get('symbol','?')} {t.get('entry_date','')} -> {t.get('exit_date','')}: "
                  f"{t['pnl_pct']:>+6.2f}% (exit: {t.get('exit_reason','')}, hold: {t.get('holding_days',0)}d, "
                  f"trend: {t.get('entry_trend','')}, size: {t.get('position_size',0):.2f})")

    # ═══════════════════════════════════════
    # SECTION 7: By trend comparison
    # ═══════════════════════════════════════
    print_section("7. BY TREND COMPARISON")
    for label in ["V19.1", "V19.4"]:
        trades = versions[label]
        df = pd.DataFrame(trades)
        if "entry_trend" not in df.columns:
            continue
        print(f"\n  {label}:")
        for trend, grp in df.groupby("entry_trend"):
            m = calc_metrics(grp.to_dict("records"))
            print(f"    {trend:<12}: {m['trades']:>4} trades, WR={m['wr']:>5.1f}%, "
                  f"avg={m['avg_pnl']:>+6.2f}%, total={m['total_pnl']:>+8.1f}%, PF={m['pf']:>5.2f}")

    # ═══════════════════════════════════════
    # SECTION 8: VERDICT
    # ═══════════════════════════════════════
    print_section("8. VERDICT & RECOMMENDATION")

    best_label = None
    best_pnl = -999
    best_pf = 0
    for label, trades in versions.items():
        if label == "Rule":
            continue
        m = calc_metrics(trades)
        score = m["total_pnl"] * 0.4 + m["pf"] * 200 + m["wr"] * 5 - abs(m["max_loss"]) * 3
        if score > best_pnl:
            best_pnl = score
            best_label = label
            best_pf = m["pf"]

    print(f"\n  Best overall model (composite score): {best_label}")
    print()
    for label in ["V19.1", "V19.2", "V19.3", "V19.4"]:
        m = calc_metrics(versions[label])
        mr_m = calc_metrics(versions["Rule"])
        gap = m["total_pnl"] - mr_m["total_pnl"]
        score = m["total_pnl"] * 0.4 + m["pf"] * 200 + m["wr"] * 5 - abs(m["max_loss"]) * 3
        verdict = "*** BEST ***" if label == best_label else ""
        keep = "KEEP" if m["total_pnl"] > m191["total_pnl"] or m["pf"] > m191["pf"] else "DISCARD"
        if label == "V19.1":
            keep = "BASELINE"
        print(f"  {label}: TotPnL={m['total_pnl']:>+9.1f}% PF={m['pf']:>5.2f} WR={m['wr']:>5.1f}% "
              f"MaxLoss={m['max_loss']:>+7.1f}% Gap-Rule={gap:>+8.1f}% Score={score:>7.1f} -> {keep} {verdict}")

    print("\n" + "=" * 120)
    print("COMPARISON COMPLETE")
    print("=" * 120)
