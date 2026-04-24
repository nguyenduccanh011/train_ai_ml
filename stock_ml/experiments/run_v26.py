"""
V26: Best improvements from V26 experiments applied on top of V25.

Winning config from experiments: B+C+E
  B) Relaxed entry for confirmed strong trends (bypass prev_pred when rule signals 3+ bars)
  C) Skip choppy regime trades entirely
  E) Stronger rule ensemble (moderate trend entry when rule signals 3+ bars)

Usage:
  python run_v26.py                      # all clean symbols
  python run_v26.py --symbols ACB,VND    # specific symbols
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.backtest.engine import backtest_unified


def backtest_v26(y_pred, returns, df_test, feature_cols, **kwargs):
    """V26 with the winning B+C+E configuration applied on V25 baseline."""
    v26_config = dict(
        patch_smart_hardcap=True,
        patch_pp_restore=True,
        patch_long_horizon=True,
        patch_symbol_tuning=True,
        patch_rule_ensemble=True,
        patch_noise_filter=False,
        patch_adaptive_hardcap=False,
        patch_pp_2of3=False,
        v26_wider_hardcap=False,
        v26_relaxed_entry=True,
        v26_skip_choppy=True,
        v26_extended_hold=False,
        v26_strong_rule_ensemble=True,
        v26_min_position=False,
        v26_score5_penalty=False,
        v26_hardcap_confirm_strong=False,
    )
    merged = {**v26_config, **kwargs}
    return backtest_unified(y_pred, returns, df_test, feature_cols, **merged)


if __name__ == "__main__":
    import argparse
    import time
    import numpy as np
    import pandas as pd
    from src.experiment_runner import run_test as run_test_base, run_rule_test
    from src.evaluation.scoring import calc_metrics, composite_score as comp_score
    from experiments.run_v25 import backtest_v25

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    DEFAULT_SYMBOLS = (
        "ACB,AAS,AAV,ACV,BCG,BCM,BID,BSR,BVH,CTG,DCM,DGC,DIG,DPM,EIB,"
        "FPT,FRT,GAS,GEX,GMD,HCM,HDB,HDG,HPG,HSG,KBC,KDH,LPB,MBB,MSN,"
        "MWG,NKG,NLG,NT2,NVL,OCB,PC1,PDR,PLX,PNJ,POW,PVD,PVS,REE,SAB,"
        "SBT,SHB,SSI,STB,TCB,TPB,VCB,VCI,VDS,VHM,VIC,VJC,VND,VNM,VPB,VTP"
    )
    SYMBOLS = args.symbols.strip() if args.symbols.strip() else DEFAULT_SYMBOLS
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

    print("=" * 120)
    print("V26 BACKTEST: B+C+E (relaxed_entry + skip_choppy + strong_rule_ensemble)")
    print("=" * 120)

    # V25 baseline
    print("\n  [1] V25 Baseline...")
    def make_v25_fn():
        def bt_fn(y_pred, returns, df_test, feature_cols, **kw):
            return backtest_v25(y_pred, returns, df_test, feature_cols,
                               peak_protect_strong_threshold=0.12,
                               patch_smart_hardcap=True, patch_pp_restore=True,
                               patch_long_horizon=True, patch_symbol_tuning=True,
                               patch_rule_ensemble=True,
                               patch_noise_filter=False, patch_adaptive_hardcap=False,
                               patch_pp_2of3=False, **kw)
        return bt_fn

    t_v25 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=make_v25_fn())
    m_v25 = calc_metrics(t_v25)
    cs25 = comp_score(m_v25)
    print(f"    V25: {m_v25['trades']} trades, TotalPnL={m_v25['total_pnl']:+.1f}%, "
          f"WR={m_v25['wr']:.1f}%, PF={m_v25['pf']:.2f}, Comp={cs25:.0f}")

    # Rule baseline
    t_rule = run_rule_test(SYMBOLS)
    m_rule = calc_metrics(t_rule)
    print(f"    Rule: {m_rule['trades']} trades, TotalPnL={m_rule['total_pnl']:+.1f}%")

    # V26
    print("\n  [2] V26 (B+C+E)...")
    t0 = time.time()
    t_v26 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v26)
    dt = time.time() - t0
    m_v26 = calc_metrics(t_v26)
    cs26 = comp_score(m_v26)

    print(f"\n{'='*120}")
    print(f"  {'Config':<20} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} "
          f"{'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}")
    print("  " + "-" * 100)
    print(f"  {'V25 Baseline':<20} | {m_v25['trades']:>5} {m_v25['wr']:>5.1f}% {m_v25['avg_pnl']:>+7.2f}% "
          f"{m_v25['total_pnl']:>+9.1f}% {m_v25['pf']:>5.2f} {m_v25['max_loss']:>+7.1f}% "
          f"{m_v25['avg_hold']:>5.1f}d | {cs25:>6.0f}")
    print(f"  {'V26 B+C+E':<20} | {m_v26['trades']:>5} {m_v26['wr']:>5.1f}% {m_v26['avg_pnl']:>+7.2f}% "
          f"{m_v26['total_pnl']:>+9.1f}% {m_v26['pf']:>5.2f} {m_v26['max_loss']:>+7.1f}% "
          f"{m_v26['avg_hold']:>5.1f}d | {cs26:>6.0f}")
    print("  " + "-" * 100)

    delta_pnl = m_v26['total_pnl'] - m_v25['total_pnl']
    delta_rule = m_v26['total_pnl'] - m_rule['total_pnl']
    print(f"  Delta vs V25:  TotPnL={delta_pnl:+.1f}%, WR={m_v26['wr']-m_v25['wr']:+.2f}%, "
          f"PF={m_v26['pf']-m_v25['pf']:+.3f}, Comp={cs26-cs25:+.0f}")
    print(f"  Delta vs Rule: TotPnL={delta_rule:+.1f}%")
    print(f"  Time: {dt:.1f}s")

    # Save trades
    df_v26 = pd.DataFrame(t_v26)
    if len(df_v26) > 0:
        df_v26.to_csv(os.path.join(OUT, "trades_v26.csv"), index=False)
        print(f"\n  Saved {len(df_v26)} V26 trades to results/trades_v26.csv")

    if len(df_v26) > 0 and "exit_reason" in df_v26.columns:
        print(f"\n  V26 EXIT REASONS:")
        for reason, grp in df_v26.groupby("exit_reason"):
            wins = len(grp[grp["pnl_pct"] > 0])
            print(f"    {reason:<25}: {len(grp):>4} ({wins}W), WR={wins/len(grp)*100:>5.1f}%, "
                  f"avg={grp['pnl_pct'].mean():>+6.2f}%, total={grp['pnl_pct'].sum():>+8.1f}%")

    print(f"\n{'='*120}")
    print("  DONE")
    print(f"{'='*120}")
