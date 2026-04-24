"""
V22 FINAL: Deep diagnosis and targeted fix.

Key insight from tuning:
- V19.1 has 336 'signal' exits with avg +0.28% and total +95.2%
  -> Many weak signal exits that barely break even, but it works because
     peak_protect_dist catches 32 trades at +1041%
- V19.3 introduces fast_exit_loss which creates -901% damage,
  BUT improves 'signal' exits to 187 trades at +5.82% avg (+1087%)
  -> The signal quality filter (mod_h) is doing great work
  -> Problem: fast_exit_loss is too aggressive

Strategy for V22 FINAL:
1. Keep V19.3's signal quality (mod_h) - it's excellent
2. Fix fast_exit_loss: only trigger when trend is truly broken
   (not just a pullback in uptrend)
3. Keep signal_hard_cap but make it ATR-adaptive
4. Don't touch peak_protect (it's working perfectly)

The key difference: instead of saving ALL fast_exit trades (fix_A),
save only those where trend is intact. Let bad trades still exit fast.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.backtest.engine import backtest_unified


def backtest_v22(y_pred, returns, df_test, feature_cols, **kwargs):
    """V22 Final: delegates to unified engine with v22_mode=True."""
    v22_defaults = dict(
        v22_mode=True,
        v22_fast_exit_skip_strong=True,
        v22_fast_exit_vol_confirm=True,
        v22_fast_exit_threshold_hb=-0.07,
        v22_fast_exit_threshold_std=-0.05,
        v22_adaptive_hard_cap=True,
        v22_hard_cap_mult_hb=3.0,
        v22_hard_cap_mult_std=2.5,
        v22_hard_cap_floor=0.12,
        v22_hard_cap_floor_hb=0.15,
    )
    return backtest_unified(y_pred, returns, df_test, feature_cols, **{**v22_defaults, **kwargs})


if __name__ == "__main__":
    import pandas as pd
    from src.experiment_runner import run_test as run_test_base, run_rule_test
    from src.evaluation.scoring import calc_metrics, composite_score
    from src.config_loader import get_pipeline_symbols
    from src.strategies.legacy import backtest_v19_3, backtest_v19_1

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 130)
    print("V22 FINAL - PARAMETER GRID SEARCH")
    print("=" * 130)

    # Baselines
    print("\n  Running baselines...")
    t_v191 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=backtest_v19_1)
    t_v193 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=backtest_v19_3)
    t_rule = run_rule_test(SYMBOLS)
    m_v191 = calc_metrics(t_v191)
    m_v193 = calc_metrics(t_v193)
    m_rule = calc_metrics(t_rule)

    # Grid search over V22 params
    configs = [
        # Format: (label, params_dict)
        ("V22-base(smart_exit+adap_cap)", {}),
        ("V22-no_vol_confirm", {"v22_fast_exit_vol_confirm": False}),
        ("V22-no_skip_strong", {"v22_fast_exit_skip_strong": False}),
        ("V22-hb_thresh=-6%", {"v22_fast_exit_threshold_hb": -0.06}),
        ("V22-hb_thresh=-8%", {"v22_fast_exit_threshold_hb": -0.08}),
        ("V22-std_thresh=-6%", {"v22_fast_exit_threshold_std": -0.06}),
        ("V22-std_thresh=-4%", {"v22_fast_exit_threshold_std": -0.04}),
        ("V22-cap_mult_hb=2.5", {"v22_hard_cap_mult_hb": 2.5}),
        ("V22-cap_mult_hb=3.5", {"v22_hard_cap_mult_hb": 3.5}),
        ("V22-cap_floor_hb=0.18", {"v22_hard_cap_floor_hb": 0.18}),
        ("V22-no_adaptive_cap", {"v22_adaptive_hard_cap": False}),
        # Combo refinements
        ("V22-hb8+std6", {"v22_fast_exit_threshold_hb": -0.08, "v22_fast_exit_threshold_std": -0.06}),
        ("V22-hb8+std6+cap3.5", {"v22_fast_exit_threshold_hb": -0.08, "v22_fast_exit_threshold_std": -0.06, "v22_hard_cap_mult_hb": 3.5}),
        ("V22-hb7+novol", {"v22_fast_exit_vol_confirm": False, "v22_fast_exit_threshold_hb": -0.07}),
        ("V22-hb8+novol", {"v22_fast_exit_vol_confirm": False, "v22_fast_exit_threshold_hb": -0.08}),
        ("V22-hb7+std6+novol", {"v22_fast_exit_vol_confirm": False, "v22_fast_exit_threshold_hb": -0.07, "v22_fast_exit_threshold_std": -0.06}),
    ]

    print(f"\n  {'Config':<30} | {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} "
          f"{'MaxLoss':>8} {'AvgHold':>8} | {'vs V19.1':>9} {'vs V19.3':>9} {'vs Rule':>9}")
    print("  " + "-" * 135)

    for lbl, m in [("V19.1", m_v191), ("V19.3", m_v193), ("Rule", m_rule)]:
        print(f"  {lbl:<30} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d | "
              f"{m['total_pnl']-m_v191['total_pnl']:>+8.1f}% {m['total_pnl']-m_v193['total_pnl']:>+8.1f}% {m['total_pnl']-m_rule['total_pnl']:>+8.1f}%")
    print("  " + "-" * 135)

    results = {}
    for label, params in configs:
        print(f"  Running {label}...", end="", flush=True)

        def make_fn(p):
            def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
                return backtest_v22(y_pred, returns, df_test, feature_cols, **{**kwargs, **p})
            return bt_fn

        trades = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                               backtest_fn=make_fn(params))
        m = calc_metrics(trades)
        results[label] = (m, trades)
        d191 = m["total_pnl"] - m_v191["total_pnl"]
        d193 = m["total_pnl"] - m_v193["total_pnl"]
        drule = m["total_pnl"] - m_rule["total_pnl"]
        star = " ***" if d191 > 0 else (" ++" if d193 > 50 else ("" if d193 > 0 else " --"))
        print(f"\r  {label:<30} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d | "
              f"{d191:>+8.1f}% {d193:>+8.1f}% {drule:>+8.1f}%{star}")

    # Ranking
    print("\n\n" + "=" * 130)
    print("RANKING BY COMPOSITE SCORE")
    print("=" * 130)

    scored = []
    for name, (m, trades) in results.items():
        score = composite_score(m)
        scored.append((score, name, m, trades))
    scored.sort(reverse=True)

    print(f"\n  {'Rank':>4} {'Config':<30} {'TotPnL':>10} {'PF':>6} {'WR':>6} {'MaxLoss':>8} {'Score':>8} {'vs V19.1':>9} {'vs V19.3':>9}")
    print("  " + "-" * 100)

    s_v191 = composite_score(m_v191)
    s_v193 = composite_score(m_v193)
    print(f"  {'ref':>4} {'V19.1':<30} {m_v191['total_pnl']:>+9.1f}% {m_v191['pf']:>5.2f} {m_v191['wr']:>5.1f}% "
          f"{m_v191['max_loss']:>+7.1f}% {s_v191:>7.1f}")
    print(f"  {'ref':>4} {'V19.3':<30} {m_v193['total_pnl']:>+9.1f}% {m_v193['pf']:>5.2f} {m_v193['wr']:>5.1f}% "
          f"{m_v193['max_loss']:>+7.1f}% {s_v193:>7.1f}")
    print("  " + "-" * 100)

    for rank, (score, name, m, _) in enumerate(scored[:10], 1):
        marker = " <-- V22 FINAL" if rank == 1 else ""
        print(f"  {rank:>4} {name:<30} {m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['wr']:>5.1f}% "
              f"{m['max_loss']:>+7.1f}% {score:>7.1f} {m['total_pnl']-m_v191['total_pnl']:>+8.1f}% "
              f"{m['total_pnl']-m_v193['total_pnl']:>+8.1f}%{marker}")

    # Per-symbol of best
    best_score, best_name, best_m, best_trades = scored[0]
    print(f"\n\n{'='*130}")
    print(f"BEST V22 = {best_name}")
    print(f"{'='*130}")

    df_b = pd.DataFrame(best_trades)
    df_191 = pd.DataFrame(t_v191)
    df_193 = pd.DataFrame(t_v193)
    df_r = pd.DataFrame(t_rule)

    print(f"\n  {'Sym':<6}| {'V19.1':>9} | {'V19.3':>9} | {'V22':>9} | {'Rule':>9} | {'V22-V191':>9} | {'V22-V193':>9} | {'V22-Rule':>9}")
    print("  " + "-" * 100)
    syms = sorted(set(df_b["symbol"].unique()) | set(df_191["symbol"].unique()))
    t_all = {"v191": 0, "v193": 0, "v22": 0, "rule": 0}
    for sym in syms:
        m1 = calc_metrics(df_191[df_191["symbol"] == sym].to_dict("records"))
        m3 = calc_metrics(df_193[df_193["symbol"] == sym].to_dict("records"))
        mb = calc_metrics(df_b[df_b["symbol"] == sym].to_dict("records"))
        mr = calc_metrics(df_r[df_r["symbol"] == sym].to_dict("records"))
        t_all["v191"] += m1["total_pnl"]; t_all["v193"] += m3["total_pnl"]
        t_all["v22"] += mb["total_pnl"]; t_all["rule"] += mr["total_pnl"]
        print(f"  {sym:<6}| {m1['total_pnl']:>+8.1f}% | {m3['total_pnl']:>+8.1f}% | "
              f"{mb['total_pnl']:>+8.1f}% | {mr['total_pnl']:>+8.1f}% | "
              f"{mb['total_pnl']-m1['total_pnl']:>+8.1f}% | {mb['total_pnl']-m3['total_pnl']:>+8.1f}% | "
              f"{mb['total_pnl']-mr['total_pnl']:>+8.1f}%")
    print("  " + "-" * 100)
    print(f"  {'TOTAL':<6}| {t_all['v191']:>+8.1f}% | {t_all['v193']:>+8.1f}% | "
          f"{t_all['v22']:>+8.1f}% | {t_all['rule']:>+8.1f}% | "
          f"{t_all['v22']-t_all['v191']:>+8.1f}% | {t_all['v22']-t_all['v193']:>+8.1f}% | "
          f"{t_all['v22']-t_all['rule']:>+8.1f}%")

    # Exit reasons
    print(f"\n  EXIT REASONS ({best_name}):")
    for reason, grp in df_b.groupby("exit_reason"):
        wins = len(grp[grp["pnl_pct"] > 0])
        print(f"    {reason:<25}: {len(grp):>4} ({wins}W), WR={wins/len(grp)*100:>5.1f}%, "
              f"avg={grp['pnl_pct'].mean():>+6.2f}%, total={grp['pnl_pct'].sum():>+8.1f}%")

    # By year
    print(f"\n  BY YEAR:")
    df_bc = df_b.copy(); df_bc["year"] = df_bc["entry_date"].astype(str).str[:4]
    df_191c = df_191.copy(); df_191c["year"] = df_191c["entry_date"].astype(str).str[:4]
    df_193c = df_193.copy(); df_193c["year"] = df_193c["entry_date"].astype(str).str[:4]
    for yr in sorted(df_bc["year"].unique()):
        m22 = calc_metrics(df_bc[df_bc["year"] == yr].to_dict("records"))
        m191y = calc_metrics(df_191c[df_191c["year"] == yr].to_dict("records"))
        m193y = calc_metrics(df_193c[df_193c["year"] == yr].to_dict("records"))
        print(f"    {yr}: V19.1={m191y['total_pnl']:>+8.1f}% V19.3={m193y['total_pnl']:>+8.1f}% "
              f"V22={m22['total_pnl']:>+8.1f}% (vs V19.1: {m22['total_pnl']-m191y['total_pnl']:>+7.1f}%, "
              f"vs V19.3: {m22['total_pnl']-m193y['total_pnl']:>+7.1f}%)")

    print("\n" + "=" * 130)
    print("DONE")
