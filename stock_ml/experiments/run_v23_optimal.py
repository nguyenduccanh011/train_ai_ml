"""
V23 OPTIMAL: Best of V19.1 + V19.3 + V22-Final.

Key fixes from analysis:
1. Graduated fast_exit_loss by trend strength (V19.3 too aggressive, V22 too lenient)
2. Restore peak_protect sensitivity (V22 dropped from 30 to 7 triggers)
3. Trend-specific signal_hard_cap (weak trend max -10%)
4. Accelerated time-decay for stagnant signal exits
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.backtest.engine import backtest_unified


def backtest_v23(y_pred, returns, df_test, feature_cols, **kwargs):
    """V23 Optimal: graduated exits + restored peak_protect + trend-specific caps."""
    return backtest_unified(y_pred, returns, df_test, feature_cols, **kwargs)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from src.evaluation.scoring import calc_metrics, composite_score
    from src.config_loader import get_pipeline_symbols
    from src.experiment_runner import run_rule_test, run_test as run_test_base
    from src.strategies.legacy import backtest_v19_3, backtest_v19_1
    from experiments.run_v22_final import backtest_v22

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 130)
    print("V23 OPTIMAL - COMPREHENSIVE COMPARISON")
    print("=" * 130)

    # Baselines
    print("\n  Running baselines...")
    t_v191 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=backtest_v19_1)
    t_v193 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=backtest_v19_3)

    def v22_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        return backtest_v22(y_pred, returns, df_test, feature_cols, **kwargs)
    t_v22 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=v22_fn)

    t_rule = run_rule_test(SYMBOLS)

    m_v191 = calc_metrics(t_v191)
    m_v193 = calc_metrics(t_v193)
    m_v22 = calc_metrics(t_v22)
    m_rule = calc_metrics(t_rule)

    # V23 grid search
    configs = [
        ("V23-default", {}),
        ("V23-fe_s=-7%", {"fast_exit_strong": -0.07}),
        ("V23-fe_s=-9%", {"fast_exit_strong": -0.09}),
        ("V23-fe_s=-10%", {"fast_exit_strong": -0.10}),
        ("V23-fe_m=-5%", {"fast_exit_moderate": -0.05}),
        ("V23-fe_m=-7%", {"fast_exit_moderate": -0.07}),
        ("V23-fe_w=-3%", {"fast_exit_weak": -0.03}),
        ("V23-fe_w=-5%", {"fast_exit_weak": -0.05}),
        ("V23-pp_s=0.12", {"peak_protect_strong_threshold": 0.12}),
        ("V23-pp_s=0.18", {"peak_protect_strong_threshold": 0.18}),
        ("V23-hc_w=-8%", {"hard_cap_weak": -0.08}),
        ("V23-hc_w=-12%", {"hard_cap_weak": -0.12}),
        ("V23-td=18", {"time_decay_bars": 18}),
        ("V23-td=25", {"time_decay_bars": 25}),
        ("V23-td_m=0.40", {"time_decay_mult": 0.40}),
        ("V23-td_m=0.60", {"time_decay_mult": 0.60}),
        # Promising combos
        ("V23-fe_s9+pp12", {"fast_exit_strong": -0.09, "peak_protect_strong_threshold": 0.12}),
        ("V23-fe_s10+pp12+hcw8", {"fast_exit_strong": -0.10, "peak_protect_strong_threshold": 0.12, "hard_cap_weak": -0.08}),
        ("V23-fe_s9+fem5+pp12", {"fast_exit_strong": -0.09, "fast_exit_moderate": -0.05, "peak_protect_strong_threshold": 0.12}),
        ("V23-optimal_A", {"fast_exit_strong": -0.09, "fast_exit_moderate": -0.05, "peak_protect_strong_threshold": 0.12, "hard_cap_weak": -0.08, "time_decay_bars": 18}),
        ("V23-optimal_B", {"fast_exit_strong": -0.10, "fast_exit_moderate": -0.06, "peak_protect_strong_threshold": 0.15, "hard_cap_weak": -0.10, "time_decay_bars": 20}),
    ]

    print(f"\n  {'Config':<30} | {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} "
          f"{'MaxLoss':>8} {'AvgHold':>8} | {'vs V191':>9} {'vs V193':>9} {'vs V22':>9} {'vs Rule':>9}")
    print("  " + "-" * 140)

    for lbl, m in [("V19.1", m_v191), ("V19.3", m_v193), ("V22-Final", m_v22), ("Rule", m_rule)]:
        print(f"  {lbl:<30} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d |")
    print("  " + "-" * 140)

    results = {}
    for label, params in configs:
        print(f"  Running {label}...", end="", flush=True)

        def make_fn(p):
            def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
                return backtest_v23(y_pred, returns, df_test, feature_cols, **{**kwargs, **p})
            return bt_fn

        trades = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                               backtest_fn=make_fn(params))
        m = calc_metrics(trades)
        results[label] = (m, trades)
        star = " ***" if m["total_pnl"] > m_v191["total_pnl"] else ""
        print(f"\r  {label:<30} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d | "
              f"{m['total_pnl']-m_v191['total_pnl']:>+8.1f}% {m['total_pnl']-m_v193['total_pnl']:>+8.1f}% "
              f"{m['total_pnl']-m_v22['total_pnl']:>+8.1f}% {m['total_pnl']-m_rule['total_pnl']:>+8.1f}%{star}")

    # Ranking
    print("\n\n" + "=" * 130)
    print("RANKING BY COMPOSITE SCORE")
    print("=" * 130)

    scored = []
    for name, (m, trades) in results.items():
        score = composite_score(m)
        scored.append((score, name, m, trades))
    scored.sort(reverse=True)

    s_v191 = composite_score(m_v191)
    s_v193 = composite_score(m_v193)
    s_v22 = composite_score(m_v22)

    print(f"\n  {'ref':>4} {'V19.1':<30} {m_v191['total_pnl']:>+9.1f}% {m_v191['pf']:>5.2f} {m_v191['wr']:>5.1f}% "
          f"{m_v191['max_loss']:>+7.1f}% {s_v191:>7.1f}")
    print(f"  {'ref':>4} {'V19.3':<30} {m_v193['total_pnl']:>+9.1f}% {m_v193['pf']:>5.2f} {m_v193['wr']:>5.1f}% "
          f"{m_v193['max_loss']:>+7.1f}% {s_v193:>7.1f}")
    print(f"  {'ref':>4} {'V22-Final':<30} {m_v22['total_pnl']:>+9.1f}% {m_v22['pf']:>5.2f} {m_v22['wr']:>5.1f}% "
          f"{m_v22['max_loss']:>+7.1f}% {s_v22:>7.1f}")
    print("  " + "-" * 100)

    for rank, (score, name, m, _) in enumerate(scored[:10], 1):
        marker = " <-- V23 OPTIMAL" if rank == 1 else ""
        print(f"  {rank:>4} {name:<30} {m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['wr']:>5.1f}% "
              f"{m['max_loss']:>+7.1f}% {score:>7.1f}{marker}")

    # Per-symbol of best
    best_score, best_name, best_m, best_trades = scored[0]
    print(f"\n\n{'='*130}")
    print(f"BEST V23 = {best_name}")
    print(f"{'='*130}")

    df_b = pd.DataFrame(best_trades)
    df_191 = pd.DataFrame(t_v191)
    df_193 = pd.DataFrame(t_v193)
    df_22 = pd.DataFrame(t_v22)
    df_r = pd.DataFrame(t_rule)

    print(f"\n  {'Sym':<6}| {'V19.1':>9} | {'V19.3':>9} | {'V22':>9} | {'V23':>9} | {'Rule':>9} | "
          f"{'V23-V191':>9} | {'V23-V193':>9} | {'V23-V22':>9} | {'V23-Rule':>9}")
    print("  " + "-" * 120)
    syms = sorted(set(df_b["symbol"].unique()) | set(df_191["symbol"].unique()))
    t_all = {"v191": 0, "v193": 0, "v22": 0, "v23": 0, "rule": 0}
    for sym in syms:
        m1 = calc_metrics(df_191[df_191["symbol"] == sym].to_dict("records"))
        m3 = calc_metrics(df_193[df_193["symbol"] == sym].to_dict("records"))
        m22 = calc_metrics(df_22[df_22["symbol"] == sym].to_dict("records"))
        mb = calc_metrics(df_b[df_b["symbol"] == sym].to_dict("records"))
        mr = calc_metrics(df_r[df_r["symbol"] == sym].to_dict("records"))
        t_all["v191"] += m1["total_pnl"]; t_all["v193"] += m3["total_pnl"]
        t_all["v22"] += m22["total_pnl"]; t_all["v23"] += mb["total_pnl"]; t_all["rule"] += mr["total_pnl"]
        print(f"  {sym:<6}| {m1['total_pnl']:>+8.1f}% | {m3['total_pnl']:>+8.1f}% | "
              f"{m22['total_pnl']:>+8.1f}% | {mb['total_pnl']:>+8.1f}% | {mr['total_pnl']:>+8.1f}% | "
              f"{mb['total_pnl']-m1['total_pnl']:>+8.1f}% | {mb['total_pnl']-m3['total_pnl']:>+8.1f}% | "
              f"{mb['total_pnl']-m22['total_pnl']:>+8.1f}% | {mb['total_pnl']-mr['total_pnl']:>+8.1f}%")
    print("  " + "-" * 120)
    print(f"  {'TOTAL':<6}| {t_all['v191']:>+8.1f}% | {t_all['v193']:>+8.1f}% | "
          f"{t_all['v22']:>+8.1f}% | {t_all['v23']:>+8.1f}% | {t_all['rule']:>+8.1f}% | "
          f"{t_all['v23']-t_all['v191']:>+8.1f}% | {t_all['v23']-t_all['v193']:>+8.1f}% | "
          f"{t_all['v23']-t_all['v22']:>+8.1f}% | {t_all['v23']-t_all['rule']:>+8.1f}%")

    # Exit reasons
    print(f"\n  EXIT REASONS ({best_name}):")
    for reason, grp in df_b.groupby("exit_reason"):
        wins = len(grp[grp["pnl_pct"] > 0])
        print(f"    {reason:<25}: {len(grp):>4} ({wins}W), WR={wins/len(grp)*100:>5.1f}%, "
              f"avg={grp['pnl_pct'].mean():>+6.2f}%, total={grp['pnl_pct'].sum():>+8.1f}%")

    # Compare exit reasons across models
    print(f"\n  EXIT REASON COMPARISON (total PnL per reason):")
    print(f"    {'Reason':<25}: {'V19.1':>10} {'V19.3':>10} {'V22':>10} {'V23':>10}")
    all_reasons = set()
    for df in [df_191, df_193, df_22, df_b]:
        all_reasons |= set(df["exit_reason"].unique())
    for reason in sorted(all_reasons):
        tots = []
        for df in [df_191, df_193, df_22, df_b]:
            g = df[df["exit_reason"] == reason]
            tots.append(g["pnl_pct"].sum() if len(g) > 0 else 0)
        print(f"    {reason:<25}: {tots[0]:>+9.1f}% {tots[1]:>+9.1f}% {tots[2]:>+9.1f}% {tots[3]:>+9.1f}%")

    # By year
    print(f"\n  BY YEAR:")
    df_bc = df_b.copy(); df_bc["year"] = df_bc["entry_date"].astype(str).str[:4]
    df_191c = df_191.copy(); df_191c["year"] = df_191c["entry_date"].astype(str).str[:4]
    df_193c = df_193.copy(); df_193c["year"] = df_193c["entry_date"].astype(str).str[:4]
    df_22c = df_22.copy(); df_22c["year"] = df_22c["entry_date"].astype(str).str[:4]
    for yr in sorted(df_bc["year"].unique()):
        m23 = calc_metrics(df_bc[df_bc["year"] == yr].to_dict("records"))
        m191y = calc_metrics(df_191c[df_191c["year"] == yr].to_dict("records"))
        m193y = calc_metrics(df_193c[df_193c["year"] == yr].to_dict("records"))
        m22y = calc_metrics(df_22c[df_22c["year"] == yr].to_dict("records"))
        print(f"    {yr}: V19.1={m191y['total_pnl']:>+8.1f}% V19.3={m193y['total_pnl']:>+8.1f}% "
              f"V22={m22y['total_pnl']:>+8.1f}% V23={m23['total_pnl']:>+8.1f}% "
              f"(vs V19.1: {m23['total_pnl']-m191y['total_pnl']:>+7.1f}%, "
              f"vs V22: {m23['total_pnl']-m22y['total_pnl']:>+7.1f}%)")

    # By trend
    print(f"\n  BY ENTRY TREND:")
    for trend in ["strong", "moderate", "weak"]:
        for lbl, df in [("V19.1", df_191), ("V19.3", df_193), ("V22", df_22), ("V23", df_b)]:
            if "entry_trend" not in df.columns:
                continue
            tt = df[df["entry_trend"] == trend]
            if len(tt) == 0:
                continue
            m = calc_metrics(tt.to_dict("records"))
            print(f"    {lbl:<6} {trend:<10}: #{m['trades']:>3} WR={m['wr']:>5.1f}% avg={m['avg_pnl']:>+6.2f}% "
                  f"tot={m['total_pnl']:>+8.1f}% PF={m['pf']:>5.2f}")
        print()

    # fast_exit_loss analysis
    print(f"\n  FAST_EXIT_LOSS COMPARISON:")
    for lbl, df in [("V19.3", df_193), ("V22", df_22), ("V23", df_b)]:
        fel = df[df["exit_reason"] == "fast_exit_loss"]
        print(f"    {lbl}: {len(fel)} trades, total={fel['pnl_pct'].sum():>+8.1f}%, avg={fel['pnl_pct'].mean():>+6.2f}%" if len(fel) > 0 else f"    {lbl}: 0 fast_exit_loss trades")

    # peak_protect comparison
    print(f"\n  PEAK_PROTECT COMPARISON:")
    for lbl, df in [("V19.1", df_191), ("V19.3", df_193), ("V22", df_22), ("V23", df_b)]:
        pp = df[df["exit_reason"].str.startswith("peak_protect")]
        print(f"    {lbl}: {len(pp)} trades, total={pp['pnl_pct'].sum():>+8.1f}%, avg={pp['pnl_pct'].mean():>+6.2f}%" if len(pp) > 0 else f"    {lbl}: 0 peak_protect trades")

    print("\n" + "=" * 130)
    print("DONE")
