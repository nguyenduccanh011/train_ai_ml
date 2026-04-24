"""
V25: Ablation study of V24 patches + new fixes.

Each V24 patch can be toggled independently to find the best combination.
New fixes: noise_filter, adaptive_hardcap, improved peak_protect (2-of-3).

Usage:
  python run_v25.py --default14          # ablation on 14 common symbols
  python run_v25.py --symbols ACB,VND    # specific symbols
  python run_v25.py                      # all clean symbols (slow)
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.backtest.engine import backtest_unified
from src.evaluation.scoring import calc_metrics, composite_score
from src.experiment_runner import run_rule_test, run_test as run_test_base


def backtest_v25(y_pred, returns, df_test, feature_cols, **kwargs):
    """V25 backtest — delegates to unified engine."""
    return backtest_unified(y_pred, returns, df_test, feature_cols, **kwargs)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from src.config_loader import get_pipeline_symbols
    from src.strategies.legacy import backtest_v19_1
    from experiments.run_v22_final import backtest_v22
    from experiments.run_v23_optimal import backtest_v23

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 140)
    print("V25 ABLATION STUDY + NEW FIXES")
    print("=" * 140)
    print(f"  Symbols: {SYMBOLS}")
    print()

    print("  [Phase 1] Running baselines...")

    t_v191 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                           backtest_fn=backtest_v19_1)
    m_v191 = calc_metrics(t_v191)
    print(f"    V19.1: {m_v191['trades']} trades, TotalPnL={m_v191['total_pnl']:+.1f}%")

    def v22_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        return backtest_v22(y_pred, returns, df_test, feature_cols, **kwargs)
    t_v22 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=v22_fn)
    m_v22 = calc_metrics(t_v22)
    print(f"    V22:   {m_v22['trades']} trades, TotalPnL={m_v22['total_pnl']:+.1f}%")

    def v23_best_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        return backtest_v23(y_pred, returns, df_test, feature_cols,
                           peak_protect_strong_threshold=0.12, **kwargs)
    t_v23 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=v23_best_fn)
    m_v23 = calc_metrics(t_v23)
    print(f"    V23:   {m_v23['trades']} trades, TotalPnL={m_v23['total_pnl']:+.1f}%")

    t_rule = run_rule_test(SYMBOLS)
    m_rule = calc_metrics(t_rule)
    print(f"    Rule:  {m_rule['trades']} trades, TotalPnL={m_rule['total_pnl']:+.1f}%")

    print("\n  [Phase 2] Running ablation study...")

    ALL_OFF = dict(patch_smart_hardcap=False, patch_pp_restore=False,
                   patch_long_horizon=False, patch_symbol_tuning=False,
                   patch_rule_ensemble=False, patch_noise_filter=False,
                   patch_adaptive_hardcap=False, patch_pp_2of3=False)

    ablation_configs = [
        ("V23-base (no patches)", ALL_OFF),
        ("P1: smart_hardcap",     {**ALL_OFF, "patch_smart_hardcap": True}),
        ("P2: pp_restore",        {**ALL_OFF, "patch_pp_restore": True}),
        ("P3: long_horizon",      {**ALL_OFF, "patch_long_horizon": True}),
        ("P4: symbol_tuning",     {**ALL_OFF, "patch_symbol_tuning": True}),
        ("P5: rule_ensemble",     {**ALL_OFF, "patch_rule_ensemble": True}),
        ("P1+P2",                 {**ALL_OFF, "patch_smart_hardcap": True, "patch_pp_restore": True}),
        ("P1+P2+P4",             {**ALL_OFF, "patch_smart_hardcap": True, "patch_pp_restore": True, "patch_symbol_tuning": True}),
        ("P1+P2+P4+P5",         {**ALL_OFF, "patch_smart_hardcap": True, "patch_pp_restore": True, "patch_symbol_tuning": True, "patch_rule_ensemble": True}),
        ("V24-all (P1-P5)",      {**ALL_OFF, "patch_smart_hardcap": True, "patch_pp_restore": True, "patch_long_horizon": True, "patch_symbol_tuning": True, "patch_rule_ensemble": True}),
        ("P1+P2+P3+P4",         {**ALL_OFF, "patch_smart_hardcap": True, "patch_pp_restore": True, "patch_long_horizon": True, "patch_symbol_tuning": True}),
        ("N1: noise_filter",      {**ALL_OFF, "patch_noise_filter": True}),
        ("N2: adaptive_hc",       {**ALL_OFF, "patch_adaptive_hardcap": True, "patch_smart_hardcap": True}),
        ("N3: pp_2of3",           {**ALL_OFF, "patch_pp_2of3": True}),
        ("P1+P2+N1",             {**ALL_OFF, "patch_smart_hardcap": True, "patch_pp_restore": True, "patch_noise_filter": True}),
        ("P1+P2+P4+N1",         {**ALL_OFF, "patch_smart_hardcap": True, "patch_pp_restore": True, "patch_symbol_tuning": True, "patch_noise_filter": True}),
        ("P1+N3+P4",            {**ALL_OFF, "patch_smart_hardcap": True, "patch_pp_2of3": True, "patch_symbol_tuning": True}),
        ("N2+P2+P4",            {**ALL_OFF, "patch_adaptive_hardcap": True, "patch_smart_hardcap": True, "patch_pp_restore": True, "patch_symbol_tuning": True}),
        ("N2+P2+P4+N1",         {**ALL_OFF, "patch_adaptive_hardcap": True, "patch_smart_hardcap": True, "patch_pp_restore": True, "patch_symbol_tuning": True, "patch_noise_filter": True}),
        ("N2+N3+P4+N1",         {**ALL_OFF, "patch_adaptive_hardcap": True, "patch_smart_hardcap": True, "patch_pp_2of3": True, "patch_symbol_tuning": True, "patch_noise_filter": True}),
    ]

    print(f"\n  {'Config':<30} | {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} "
          f"{'MaxLoss':>8} {'AvgHold':>8} | {'Comp':>8} | {'vs V23':>9} {'vs Rule':>9}")
    print("  " + "-" * 140)

    for lbl, m in [("V19.1 (baseline)", m_v191), ("V22 (baseline)", m_v22),
                   ("V23-best (baseline)", m_v23), ("Rule (baseline)", m_rule)]:
        print(f"  {lbl:<30} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d | "
              f"{composite_score(m):>7.0f} |")
    print("  " + "-" * 140)

    results = {}
    for label, patch_cfg in ablation_configs:
        print(f"    Running {label}...", end="", flush=True)

        def make_fn(cfg):
            def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
                return backtest_v25(y_pred, returns, df_test, feature_cols,
                                   peak_protect_strong_threshold=0.12,
                                   **cfg, **kwargs)
            return bt_fn

        trades = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                               backtest_fn=make_fn(patch_cfg))
        m = calc_metrics(trades)
        results[label] = (m, trades, patch_cfg)

        cs = composite_score(m)
        print(f"\r  {label:<30} | {m['trades']:>4} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>6.1f}d | "
              f"{cs:>7.0f} | {m['total_pnl']-m_v23['total_pnl']:>+8.1f}% "
              f"{m['total_pnl']-m_rule['total_pnl']:>+8.1f}%")

    print(f"\n{'='*140}")
    print("  RANKING BY COMPOSITE SCORE")
    print(f"{'='*140}")

    scored = []
    for name, (m, trades, cfg) in results.items():
        scored.append((composite_score(m), name, m, cfg))
    scored.sort(reverse=True)

    for rank, (sc, name, m, cfg) in enumerate(scored, 1):
        marker = " <<<" if rank == 1 else ""
        print(f"  #{rank:>2} {sc:>7.0f}  {name:<30} TotPnL={m['total_pnl']:>+9.1f}% "
              f"PF={m['pf']:>5.2f} WR={m['wr']:>5.1f}% MaxLoss={m['max_loss']:>+7.1f}%{marker}")

    best_score, best_name, best_m, best_cfg = scored[0]
    print(f"\n  BEST CONFIG: {best_name}")
    print(f"    Composite: {best_score:.0f}")
    print(f"    TotalPnL:  {best_m['total_pnl']:+.1f}%  (vs V23: {best_m['total_pnl']-m_v23['total_pnl']:+.1f}%,"
          f" vs Rule: {best_m['total_pnl']-m_rule['total_pnl']:+.1f}%)")
    print(f"    PF:        {best_m['pf']:.2f}")
    print(f"    WR:        {best_m['wr']:.1f}%")
    print(f"    MaxLoss:   {best_m['max_loss']:+.1f}%")
    print(f"    Patches:   {best_cfg}")

    print(f"\n{'='*140}")
    print(f"  PER-SYMBOL: V25-best vs baselines")
    print(f"{'='*140}")

    _, best_trades, _ = results[best_name]
    df_25 = pd.DataFrame(best_trades)
    df_23 = pd.DataFrame(t_v23)
    df_r = pd.DataFrame(t_rule)
    df_191 = pd.DataFrame(t_v191)

    print(f"  {'Sym':<6}| {'V19.1':>9} | {'V23':>9} | {'Rule':>9} | {'V25':>9} | {'V25-V23':>9} | {'V25-Rule':>9}")
    print("  " + "-" * 80)

    all_syms = sorted(set(
        list(df_25["symbol"].unique() if "symbol" in df_25.columns else []) +
        list(df_23["symbol"].unique() if "symbol" in df_23.columns else [])
    ))
    for sym in all_syms:
        def sp(df, s):
            if "symbol" not in df.columns:
                return calc_metrics([])
            return calc_metrics(df[df["symbol"]==s].to_dict("records"))
        m1 = sp(df_191, sym); m3 = sp(df_23, sym); mr = sp(df_r, sym); m5 = sp(df_25, sym)
        print(f"  {sym:<6}| {m1['total_pnl']:>+8.1f}% | {m3['total_pnl']:>+8.1f}% | "
              f"{mr['total_pnl']:>+8.1f}% | {m5['total_pnl']:>+8.1f}% | "
              f"{m5['total_pnl']-m3['total_pnl']:>+8.1f}% | {m5['total_pnl']-mr['total_pnl']:>+8.1f}%")

    if len(df_25) > 0:
        df_25.to_csv(os.path.join(OUT, "trades_v25.csv"), index=False)
        print(f"\n  Saved {len(df_25)} V25-best trades to results/trades_v25.csv")

    if len(df_25) > 0 and "exit_reason" in df_25.columns:
        print(f"\n  V25-best EXIT REASONS:")
        for reason, grp in df_25.groupby("exit_reason"):
            wins = len(grp[grp["pnl_pct"] > 0])
            print(f"    {reason:<25}: {len(grp):>4} ({wins}W), WR={wins/len(grp)*100:>5.1f}%, "
                  f"avg={grp['pnl_pct'].mean():>+6.2f}%, total={grp['pnl_pct'].sum():>+8.1f}%")

    print(f"\n{'='*140}")
    print("  DONE")
    print(f"{'='*140}")
