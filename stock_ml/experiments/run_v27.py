"""
V27: V26 baseline + selected V27 improvements.

Selected from ablation:
  J2 + J3 + J4 + J5
  - v27_hardcap_two_step
  - v27_rule_priority
  - v27_dynamic_score5_penalty
  - v27_trend_persistence_hold
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.backtest.engine import backtest_unified


def backtest_v27(y_pred, returns, df_test, feature_cols, **kwargs):
    v27_config = dict(
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
        v27_selective_choppy=False,
        v27_hardcap_two_step=True,
        v27_rule_priority=True,
        v27_dynamic_score5_penalty=True,
        v27_trend_persistence_hold=True,
    )
    merged = {**v27_config, **kwargs}
    return backtest_unified(y_pred, returns, df_test, feature_cols, **merged)


if __name__ == "__main__":
    import argparse
    import time
    import pandas as pd

    from src.experiment_runner import run_test as run_test_base, run_rule_test
    from src.evaluation.scoring import calc_metrics, composite_score as comp_score
    from experiments.run_v26 import backtest_v26

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    DEFAULT_SYMBOLS = (
        "ACB,AAS,AAV,ACV,BCG,BCM,BID,BSR,BVH,CTG,DCM,DGC,DIG,DPM,EIB,"
        "FPT,FRT,GAS,GEX,GMD,HCM,HDB,HDG,HPG,HSG,KBC,KDH,LPB,MBB,MSN,"
        "MWG,NKG,NLG,NT2,NVL,OCB,PC1,PDR,PLX,PNJ,POW,PVD,PVS,REE,SAB,"
        "SBT,SHB,SSI,STB,TCB,TPB,VCB,VCI,VDS,VHM,VIC,VJC,VND,VNM,VPB,VTP"
    )
    SYMBOLS = args.symbols.strip() if args.symbols.strip() else DEFAULT_SYMBOLS
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

    print("=" * 120)
    print("V27 BACKTEST: V26 baseline + J2+J3+J4+J5")
    print("=" * 120)

    t_rule = run_rule_test(SYMBOLS)
    m_rule = calc_metrics(t_rule)

    t0 = time.time()
    t_v26 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v26)
    m_v26 = calc_metrics(t_v26)
    cs26 = comp_score(m_v26)

    t_v27 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v27)
    dt = time.time() - t0
    m_v27 = calc_metrics(t_v27)
    cs27 = comp_score(m_v27)

    print(f"\n{'='*120}")
    print(f"  {'Config':<22} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}")
    print("  " + "-" * 102)
    print(f"  {'V26 baseline':<22} | {m_v26['trades']:>5} {m_v26['wr']:>5.1f}% {m_v26['avg_pnl']:>+7.2f}% {m_v26['total_pnl']:>+9.1f}% {m_v26['pf']:>5.2f} {m_v26['max_loss']:>+7.1f}% {m_v26['avg_hold']:>5.1f}d | {cs26:>6.0f}")
    print(f"  {'V27 selected':<22} | {m_v27['trades']:>5} {m_v27['wr']:>5.1f}% {m_v27['avg_pnl']:>+7.2f}% {m_v27['total_pnl']:>+9.1f}% {m_v27['pf']:>5.2f} {m_v27['max_loss']:>+7.1f}% {m_v27['avg_hold']:>5.1f}d | {cs27:>6.0f}")
    print("  " + "-" * 102)
    print(f"  Delta vs V26:  TotPnL={m_v27['total_pnl']-m_v26['total_pnl']:+.1f}%, WR={m_v27['wr']-m_v26['wr']:+.2f}%, PF={m_v27['pf']-m_v26['pf']:+.3f}, Comp={cs27-cs26:+.0f}")
    print(f"  Delta vs Rule: TotPnL={m_v27['total_pnl']-m_rule['total_pnl']:+.1f}%")
    print(f"  Time: {dt:.1f}s")

    df_v27 = pd.DataFrame(t_v27)
    if len(df_v27) > 0:
        df_v27.to_csv(os.path.join(OUT, "trades_v27.csv"), index=False)
        print(f"\n  Saved {len(df_v27)} V27 trades to results/trades_v27.csv")

    print(f"\n{'='*120}")
    print("  DONE")
    print(f"{'='*120}")
