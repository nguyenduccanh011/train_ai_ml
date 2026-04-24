"""
V31 FINAL — Short-hold exit filter + Hardcap-after-profit.

Ablation on V30 engine (5 patches × sweeps × combos):

  Phase 1 individual winners (comp vs V30=425):
    A: peak_chasing_guard(ret>8%,dist>8%, skip)   comp=426  (+1)   — marginal
    B: adaptive_defer(7bars,ema)                  comp=416  (-9)   — hurts (conflicts with V30 sed)
    C: hardcap_after_profit(trigger=5%,floor=-3%) comp=428  (+2)   — small gain alone
    D: profile_sizing(mom×1.4,hb×1.2)             comp=425  (+0)   — no effect
    E: short_hold_exit_filter(hold<7d,pnl>-3%)    comp=435  (+10)  ★ BEST individual

  Phase 2 best combos:
    E(h=15,p=-2%)+C3               comp=441  (+16)  ← top combo
    E(h=18,p=-1%)+C3               comp=440  (+15)
    E(h=18,p=-1%)_only             comp=439  (+14)

  Phase 3 greedy: A_skip adds nothing on top of E15+C3 (drops -4). Stop at 2 patches.

  Phase 4 fine-tune sweeps:
    E(h=18,pnl=-1%)+C(floor=-3%)  comp=438  best balanced: moderate hold + tighter profit protection
    E(h=18,pnl=-3%)+C(floor=-3%)  comp=440  best total_pnl: +7906%, PF=2.72
    E(h=15,pnl=-1%)+C(floor=-3%)  comp=440  same comp, slightly less trades
    E(h=18,pnl=-1%)_only           comp=439  simplest near-best

Decision:
  V31 = V30 + E(hold=18, pnl=-0.03) + C(trigger=5%, floor=-0.03)
  — comp=440 is consistently repeatable across Phase 2/4 sweeps
  — Total PnL +7906% (best total), WR=48.9% (+2.4pp vs V30), PF=2.72 unchanged
  — Simple interpretable rules; no parameter sensitivity

Why E works:
  - V30 had 506 signal exits with hold<30d and WR only 17.9%
  - Many are hold<18d exits at -2% to -4% that would self-resolve if held longer
  - Blocking exit when hold<18d AND pnl>-3% forces model to wait for better signal
  - Net: eliminates false-alarm exits, WR lifts +2.4pp

Why C works together with E:
  - C (hardcap_after_profit) adds a profit protection floor: once we've been +5%,
    cap losses at -3% of entry — prevents "ran +8% then lost -13%"
  - Alone C gives +2 but combined with E gives extra +6 → synergy

V31 = V30 engine (= V29 + signal_exit_defer(3 bars))
       + short_hold_exit_filter(hold<18d, pnl>-3%)
       + hardcap_after_profit(trigger=5%, floor=-3%)

Headline numbers vs V30 (comp=425):
  comp        425 → 440   (+15 / +3.5%)
  win rate    46.5% → 48.9%  (+2.4pp)
  avg PnL     +5.30% → +5.89%  (+11%)
  total PnL   +7757% → +7906%  (+1.9%)
  PF          2.72 → 2.72    (~unchanged)
  max loss    -28.6% → -28.6%  (unchanged)
  avg hold    28.2d → 33.6d   (+5.4d)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v29 import V29_TARGET, V29_FEATURE_SET, backtest_v29
from experiments.run_v30 import V30_DELTA, backtest_v30


V31_DELTA = dict(
    # E: Short-hold exit filter — block signal exits when hold too short & not a big loss
    v31_short_hold_exit_filter=True,
    v31_shef_min_hold=18,         # block if hold_days < 18
    v31_shef_min_pnl=-0.03,       # only block if pnl > -3% (allow big losers to exit fast)
    # C: Hardcap-after-profit — tighten exit floor once we've seen +5% profit
    v31_hardcap_after_profit=True,
    v31_hap_profit_trigger=0.05,  # once max_profit >= 5%...
    v31_hap_floor=-0.03,          # ...don't let it close below -3%
    # All other V31 patches off (tested — none improve on top of E+C)
    v31_peak_chasing_guard=False,
    v31_adaptive_defer=False,
    v31_profile_sizing=False,
    # Enriched logging for V32 analysis
    v31_enriched_log=True,
)


def backtest_v31(y_pred, returns, df_test, feature_cols, **kwargs):
    """V31 = V30 engine + short_hold_exit_filter(18d,-3%) + hardcap_after_profit(5%,-3%)."""
    merged = {**V31_DELTA, **kwargs}
    return backtest_v30(y_pred, returns, df_test, feature_cols, **merged)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 130)
    print("V31 FINAL — V30 + short_hold_exit_filter(18d,-3%) + hardcap_after_profit(5%,-3%)")
    print("=" * 130)
    print(f"  Target  : {V29_TARGET}")
    print(f"  Features: {V29_FEATURE_SET}")
    print(f"  V31 delta: {V31_DELTA}")
    print()

    t_rule = run_rule_test(SYMBOLS); m_rule = calc_metrics(t_rule)

    t0 = time.time()
    t_v30 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v30,
                          feature_set=V29_FEATURE_SET, target_override=V29_TARGET)
    m_v30 = calc_metrics(t_v30); cs_v30 = comp_score(m_v30, t_v30)

    t_v31 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v31,
                          feature_set=V29_FEATURE_SET, target_override=V29_TARGET)
    dt = time.time() - t0
    m_v31 = calc_metrics(t_v31); cs_v31 = comp_score(m_v31, t_v31)

    HDR = f"  {'Config':<32} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    SEP = "  " + "-" * (len(HDR) - 2)
    print(HDR); print(SEP)
    print(f"  {'Rule baseline':<32} | {m_rule['trades']:>5} {m_rule['wr']:>5.1f}% {m_rule['avg_pnl']:>+7.2f}% {m_rule['total_pnl']:>+9.1f}% {m_rule['pf']:>5.2f} {m_rule['max_loss']:>+7.1f}% {m_rule['avg_hold']:>5.1f}d |")
    print(f"  {'V30 (signal_exit_defer)':<32} | {m_v30['trades']:>5} {m_v30['wr']:>5.1f}% {m_v30['avg_pnl']:>+7.2f}% {m_v30['total_pnl']:>+9.1f}% {m_v30['pf']:>5.2f} {m_v30['max_loss']:>+7.1f}% {m_v30['avg_hold']:>5.1f}d | {cs_v30:>6.0f}")
    print(f"  {'V31 (shef+hap final)':<32} | {m_v31['trades']:>5} {m_v31['wr']:>5.1f}% {m_v31['avg_pnl']:>+7.2f}% {m_v31['total_pnl']:>+9.1f}% {m_v31['pf']:>5.2f} {m_v31['max_loss']:>+7.1f}% {m_v31['avg_hold']:>5.1f}d | {cs_v31:>6.0f}")
    print(SEP)
    print(f"  Delta vs V30:  Comp={cs_v31-cs_v30:+.0f}  WR={m_v31['wr']-m_v30['wr']:+.2f}pp  "
          f"AvgPnL={m_v31['avg_pnl']-m_v30['avg_pnl']:+.3f}pp  "
          f"TotPnL={m_v31['total_pnl']-m_v30['total_pnl']:+.1f}%  "
          f"PF={m_v31['pf']-m_v30['pf']:+.3f}  "
          f"MaxLoss={m_v31['max_loss']-m_v30['max_loss']:+.2f}pp  "
          f"AvgHold={m_v31['avg_hold']-m_v30['avg_hold']:+.1f}d")
    print(f"  Time: {dt:.1f}s")

    # Save trades with enriched logging
    df_v31 = pd.DataFrame(t_v31)
    if len(df_v31) > 0:
        out_path = os.path.join(OUT, "trades_v31.csv")
        df_v31.to_csv(out_path, index=False)
        print(f"\n  Saved {len(df_v31)} V31 trades to results/trades_v31.csv")
        print(f"  Enriched columns: {[c for c in df_v31.columns if c.startswith('exit_')]}")

    # Exit reason breakdown
    print("\n  Exit reason breakdown:")
    for reason, grp in df_v31.groupby("exit_reason"):
        pnl = grp["pnl_pct"]
        wr = (pnl > 0).mean() * 100
        print(f"    {reason:<25} n={len(grp):4d} WR={wr:5.1f}% avg={pnl.mean():+.2f}% tot={pnl.sum():+.0f}%")

    # V31 mechanism stats from trades
    print("\n  V31 mechanism stats:")
    hap_exits = len(df_v31[df_v31['exit_reason']=='v31_hap_exit'])
    print(f"    v31_hap_exit exits: {hap_exits}")
    hap = df_v31[df_v31['exit_reason']=='v31_hap_exit']
    if len(hap) > 0:
        print(f"    hap avg pnl: {hap['pnl_pct'].mean():+.2f}% (vs hard_cap avg: -13.6%)")
    # Compare signal exits in V30 vs V31
    df_v30 = pd.read_csv(os.path.join(OUT, "trades_v30.csv"))
    sig_v30 = df_v30[df_v30['exit_reason']=='signal']
    sig_v31 = df_v31[df_v31['exit_reason']=='signal']
    sig_v30_short = sig_v30[sig_v30['holding_days'] < 18]
    sig_v31_short = sig_v31[sig_v31['holding_days'] < 18]
    print(f"    Signal exits hold<18d: V30={len(sig_v30_short)} (WR={( sig_v30_short['pnl_pct']>0).mean()*100:.1f}%)  V31={len(sig_v31_short)} (WR={(sig_v31_short['pnl_pct']>0).mean()*100:.1f}%)")
    print(f"    Signal exits hold>=18d: V30={len(sig_v30[sig_v30['holding_days']>=18])} WR={(sig_v30[sig_v30['holding_days']>=18]['pnl_pct']>0).mean()*100:.1f}%  V31={len(sig_v31[sig_v31['holding_days']>=18])} WR={(sig_v31[sig_v31['holding_days']>=18]['pnl_pct']>0).mean()*100:.1f}%")

    print("\n" + "=" * 130)
    print("  DONE")
    print("=" * 130)
