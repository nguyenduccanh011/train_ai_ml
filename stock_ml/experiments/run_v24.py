"""
V24: V23-best + 5 patches from V24 spec.

Patches on top of V23-best (pp_s=0.12):
5.1 Smart hard_cap (confirm 1 bar in strong/moderate trend)
5.2 Peak_protect restore V19.1 sensitivity (drop 3-in-1 requirement)
5.3 Long-horizon carry module (capture Rule big wins)
5.4 Symbol-specific tuning (REE/MBB/AAS)
5.5 Rule + ML ensemble (partial rule signal blend)
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.backtest.engine import backtest_unified


def backtest_v24(y_pred, returns, df_test, feature_cols, **kwargs):
    """V24 backtest — V23 base + all 5 patches enabled."""
    v24_defaults = dict(
        patch_smart_hardcap=True,
        patch_pp_restore=True,
        patch_long_horizon=True,
        patch_symbol_tuning=True,
        patch_rule_ensemble=True,
    )
    return backtest_unified(y_pred, returns, df_test, feature_cols, **{**v24_defaults, **kwargs})


if __name__ == "__main__":
    import argparse
    import sys as _sys
    import pandas as pd
    from src.experiment_runner import run_test as run_test_base, run_rule_test
    from src.evaluation.scoring import calc_metrics, composite_score
    from src.config_loader import get_pipeline_symbols
    from src.strategies.legacy import backtest_v19_1
    from experiments.run_v22_final import backtest_v22
    from experiments.run_v23_optimal import backtest_v23

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols. Empty means auto all clean symbols.")
    args = parser.parse_args()

    _sys.stdout = open(_sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)
    symbols_list = get_pipeline_symbols(args.symbols)
    if not symbols_list:
        raise RuntimeError("No symbols resolved for V24 backtest.")
    SYMBOLS = ",".join(symbols_list)

    print("=" * 130)
    print("V24 BACKTEST — V23 + 5 PATCHES vs ALL BASELINES")
    print("=" * 130)
    print(f"  Symbols: {len(symbols_list)}")

    # Baselines
    print("\n  Running baselines...")
    t_v191 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=backtest_v19_1)

    def v22_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        return backtest_v22(y_pred, returns, df_test, feature_cols, **kwargs)
    t_v22 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=v22_fn)

    def v23_best_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        return backtest_v23(y_pred, returns, df_test, feature_cols, peak_protect_strong_threshold=0.12, **kwargs)
    t_v23 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=v23_best_fn)

    t_rule = run_rule_test(SYMBOLS)

    # V24
    print("  Running V24...")
    t_v24 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=backtest_v24)

    m_v191 = calc_metrics(t_v191)
    m_v22 = calc_metrics(t_v22)
    m_v23 = calc_metrics(t_v23)
    m_rule = calc_metrics(t_rule)
    m_v24 = calc_metrics(t_v24)

    # Save V24 trades
    df_v24 = pd.DataFrame(t_v24)
    df_v24.to_csv(os.path.join(OUT, "trades_v24.csv"), index=False)
    print(f"  Saved {len(t_v24)} V24 trades to results/trades_v24.csv")

    # Summary table
    print(f"\n{'='*130}")
    print(f"  {'Model':<20} | {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgHold':>8}")
    print("  " + "-" * 80)
    for lbl, m in [("V19.1", m_v191), ("V22-Final", m_v22), ("V23-best", m_v23), ("Rule", m_rule), ("V24", m_v24)]:
        star = " <<<" if lbl == "V24" else ""
        print(f"  {lbl:<20} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d{star}")

    # Delta table
    print(f"\n  {'Model':<20} | {'vs V19.1':>10} {'vs V22':>10} {'vs V23':>10} {'vs Rule':>10}")
    print("  " + "-" * 60)
    for lbl, m in [("V24", m_v24), ("V23-best", m_v23), ("V19.1", m_v191), ("V22-Final", m_v22), ("Rule", m_rule)]:
        print(f"  {lbl:<20} | {m['total_pnl']-m_v191['total_pnl']:>+9.1f}% "
              f"{m['total_pnl']-m_v22['total_pnl']:>+9.1f}% "
              f"{m['total_pnl']-m_v23['total_pnl']:>+9.1f}% "
              f"{m['total_pnl']-m_rule['total_pnl']:>+9.1f}%")

    print(f"\n  COMPOSITE SCORES:")
    for lbl, m in [("V19.1", m_v191), ("V22", m_v22), ("V23-best", m_v23), ("Rule", m_rule), ("V24", m_v24)]:
        print(f"    {lbl:<20}: {composite_score(m):>8.1f}")

    # Per-symbol
    print(f"\n{'='*130}")
    print(f"  PER-SYMBOL COMPARISON")
    print(f"{'='*130}")
    df_191 = pd.DataFrame(t_v191); df_22 = pd.DataFrame(t_v22)
    df_23 = pd.DataFrame(t_v23); df_r = pd.DataFrame(t_rule)
    print(f"  {'Sym':<6}| {'V19.1':>9} | {'V22':>9} | {'V23':>9} | {'Rule':>9} | {'V24':>9} | {'V24-V23':>9} | {'V24-Rule':>9}")
    print("  " + "-" * 100)
    for sym in sorted(df_v24["symbol"].unique() if "symbol" in df_v24.columns else []):
        def sp(df, s): return calc_metrics(df[df["symbol"]==s].to_dict("records")) if "symbol" in df.columns else calc_metrics([])
        m1=sp(df_191,sym); m2=sp(df_22,sym); m3=sp(df_23,sym); mr=sp(df_r,sym); m4=sp(df_v24,sym)
        print(f"  {sym:<6}| {m1['total_pnl']:>+8.1f}% | {m2['total_pnl']:>+8.1f}% | "
              f"{m3['total_pnl']:>+8.1f}% | {mr['total_pnl']:>+8.1f}% | {m4['total_pnl']:>+8.1f}% | "
              f"{m4['total_pnl']-m3['total_pnl']:>+8.1f}% | {m4['total_pnl']-mr['total_pnl']:>+8.1f}%")

    # V24 exit reasons
    if len(df_v24) > 0 and "exit_reason" in df_v24.columns:
        print(f"\n  V24 EXIT REASONS:")
        for reason, grp in df_v24.groupby("exit_reason"):
            wins = len(grp[grp["pnl_pct"] > 0])
            print(f"    {reason:<25}: {len(grp):>4} ({wins}W), WR={wins/len(grp)*100:>5.1f}%, "
                  f"avg={grp['pnl_pct'].mean():>+6.2f}%, total={grp['pnl_pct'].sum():>+8.1f}%")

    print(f"\n  V24 PATCH VERIFICATION:")
    print(f"    Target: TotalPnL >= +2000%, MaxLoss <= -18%, signal_hard_cap improved, peak_protect increased")
    print(f"    Actual: TotalPnL = {m_v24['total_pnl']:>+.1f}%, MaxLoss = {m_v24['max_loss']:>+.1f}%")
    target_met = m_v24["total_pnl"] >= 2000
    print(f"    TotalPnL >= 2000%: {'YES ✓' if target_met else 'NO ✗'}")
    print(f"    MaxLoss <= -18%: {'YES ✓' if m_v24['max_loss'] >= -18 else 'NO ✗'}")
    print(f"    vs V23-best: {m_v24['total_pnl'] - m_v23['total_pnl']:>+.1f}%")
    print(f"    vs Rule: {m_v24['total_pnl'] - m_rule['total_pnl']:>+.1f}%")

    print("\n" + "=" * 130)
    print("DONE")
    print("=" * 130)
