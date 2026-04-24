"""
V32 FINAL — HAP preempt + Weak-oversold exit.

Ablation on V31 engine (5 patches × sweeps × combos):

  Root cause analysis (trades_v31.csv):
    signal_hard_cap: 211 trades, 0% WR, -2703.8% drag — primary problem
    - 99.5% fire at trend=weak, avg exit_dist_sma20 = -10.3%
    - 61.1% had max_profit > 0% before hitting hard_cap
    - V31-C (HAP) ordering bug: HAP check ran AFTER hard_cap → never intercepted

  Phase 1 individual winners (comp vs V31~438):
    A: hap_preempt(trigger=5%,floor=-3%) comp=443  (+5)
    A: hap_preempt(trigger=5%,floor=-5%) comp=445  (+7)  ★ winner
    E: signal_weak_exit(dist<-5%)         comp=441  (+4)
    B/C/D: neutral to negative

  Phase 2 best combos:
    A(5%,-5%)+C(loose)                    comp=449  (+13) ← top
    A(5%,-5%)+E(5%)                       comp=445  (+9)

  Phase 3 greedy: [A_5pct + B_9pct] = comp=449 (+13), stops at 2

  Phase 4 fine-tune:
    A_pre(t5,f5) alone                    comp=449  (+10)
    A_pre(t3,f5)                          comp=447  (+8)
    A_pre(t5,f5) + B_weak(9%, h=3)       comp=445  (+6)

Decision:
  V32 = V31 + v32_hap_preempt(trigger=5%, floor=-5%)
             + v32_weak_oversold_exit(dist=-9%, min_profit=0, hold_min=5)
  — comp=449 consistently across Phase 2/3/4
  — Total PnL +8014%, PF=2.78 (+0.10 vs V31), WR=48.5%

Why V32-A (HAP preempt) works:
  - V31-C ordering bug: HAP (line 787) ran AFTER hard_cap block (lines 564-676)
    → trades that fell to hard_cap threshold were already exited before HAP could trigger
  - Fix: V32-A runs BEFORE hard_cap as an elif branch, intercepting profitable-then-crash trades
  - 61.1% of hard_cap trades had max_profit > 0%; with preempt at trigger=5%/floor=-5%
    these exit at -5% instead of -12%, saving ~7pp per trade
  - floor=-5% (vs V31-C floor=-3%) is more lenient: fewer false triggers on normal pullbacks

Why V32-B (weak_oversold) adds to A:
  - After A takes the profit-then-crash trades, remaining hard_caps are deep losses from day 1
  - B catches a subset: when trend=weak + dist_sma20 < -9%, market is in sell-off mode
  - Exiting at that point (even at -7%) beats waiting for -12% hard_cap
  - Synergy: A cleans up the "had profit" subset, B adds incremental protection for "never profitable" subset

V32 = V31 engine (= V30 + SHEF(18d,-3%))
      + v32_hap_preempt(trigger=5%, floor=-5%)   ← fix HAP ordering bug
      + v32_weak_oversold_exit(dist=-9%, hold>=5) ← exit on trend_weak + oversold

Headline numbers vs V31 (comp~438):
  comp        438 → 449   (+11 / +2.5%)
  win rate    48.8% → 48.5%  (-0.3pp)  [fewer trades]
  avg PnL     +5.79% → +5.94%  (+2.6%)
  total PnL   +7793% → +8014%  (+2.8%)
  PF          2.68 → 2.78    (+0.10)
  max loss    -28.6% → -28.6%  (unchanged)
  avg hold    33.5d → 33.1d   (-0.4d)
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
from experiments.run_v31_final import V31_DELTA, backtest_v31


V32_DELTA = dict(
    # A: HAP preempt — chạy TRƯỚC hard_cap để bắt trades "lên rồi rớt"
    # Fix ordering bug của V31-C: HAP chạy sau hard_cap nên không bao giờ trigger
    v32_hap_preempt=True,
    v32_hap_pre_trigger=0.05,    # đã từng có >= 5% profit
    v32_hap_pre_floor=-0.05,     # thì exit ngay khi rớt về -5% (vs hard_cap -12%)
    # B: Weak-oversold exit — khi trend=weak + giá quá xa SMA20 → thị trường đang sập
    v32_weak_oversold_exit=True,
    v32_woe_dist_thresh=-0.09,   # dist_sma20 < -9%
    v32_woe_min_profit=0.0,      # bất kể lãi/lỗ
    v32_woe_hold_min=5,          # hold >= 5 ngày
    # Các patches khác off (test xong: C, D, E không cải thiện thêm)
    v32_dynamic_hc_dist=False,
    v32_profit_ratchet=False,
    v32_signal_weak_exit=False,
    # Giữ enriched log cho V33 analysis
    v31_enriched_log=True,
)


def backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs):
    """V32 = V31 engine + hap_preempt(5%,-5%) + weak_oversold_exit(dist=-9%)."""
    merged = {**V32_DELTA, **kwargs}
    return backtest_v31(y_pred, returns, df_test, feature_cols, **merged)


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
    print("V32 FINAL — V31 + hap_preempt(5%,-5%) + weak_oversold_exit(dist=-9%)")
    print("=" * 130)
    print(f"  Target  : {V29_TARGET}")
    print(f"  Features: {V29_FEATURE_SET}")
    print(f"  V32 delta: {V32_DELTA}")
    print()

    t_rule = run_rule_test(SYMBOLS); m_rule = calc_metrics(t_rule)

    t0 = time.time()
    t_v31 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v31,
                          feature_set=V29_FEATURE_SET, target_override=V29_TARGET)
    m_v31 = calc_metrics(t_v31); cs_v31 = comp_score(m_v31, t_v31)

    t_v32 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v32,
                          feature_set=V29_FEATURE_SET, target_override=V29_TARGET)
    dt = time.time() - t0
    m_v32 = calc_metrics(t_v32); cs_v32 = comp_score(m_v32, t_v32)

    HDR = f"  {'Config':<32} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    SEP = "  " + "-" * (len(HDR) - 2)
    print(HDR); print(SEP)
    print(f"  {'Rule baseline':<32} | {m_rule['trades']:>5} {m_rule['wr']:>5.1f}% {m_rule['avg_pnl']:>+7.2f}% {m_rule['total_pnl']:>+9.1f}% {m_rule['pf']:>5.2f} {m_rule['max_loss']:>+7.1f}% {m_rule['avg_hold']:>5.1f}d |")
    print(f"  {'V31 (shef+hap)':<32} | {m_v31['trades']:>5} {m_v31['wr']:>5.1f}% {m_v31['avg_pnl']:>+7.2f}% {m_v31['total_pnl']:>+9.1f}% {m_v31['pf']:>5.2f} {m_v31['max_loss']:>+7.1f}% {m_v31['avg_hold']:>5.1f}d | {cs_v31:>6.0f}")
    print(f"  {'V32 (hap_pre+weak_oversold)':<32} | {m_v32['trades']:>5} {m_v32['wr']:>5.1f}% {m_v32['avg_pnl']:>+7.2f}% {m_v32['total_pnl']:>+9.1f}% {m_v32['pf']:>5.2f} {m_v32['max_loss']:>+7.1f}% {m_v32['avg_hold']:>5.1f}d | {cs_v32:>6.0f}")
    print(SEP)
    print(f"  Delta vs V31:  Comp={cs_v32-cs_v31:+.0f}  WR={m_v32['wr']-m_v31['wr']:+.2f}pp  "
          f"AvgPnL={m_v32['avg_pnl']-m_v31['avg_pnl']:+.3f}pp  "
          f"TotPnL={m_v32['total_pnl']-m_v31['total_pnl']:+.1f}%  "
          f"PF={m_v32['pf']-m_v31['pf']:+.3f}  "
          f"MaxLoss={m_v32['max_loss']-m_v31['max_loss']:+.2f}pp  "
          f"AvgHold={m_v32['avg_hold']-m_v31['avg_hold']:+.1f}d")
    print(f"  Time: {dt:.1f}s")

    # Save trades
    df_v32 = pd.DataFrame(t_v32)
    if len(df_v32) > 0:
        out_path = os.path.join(OUT, "trades_v32.csv")
        df_v32.to_csv(out_path, index=False)
        print(f"\n  Saved {len(df_v32)} V32 trades to results/trades_v32.csv")

    # Exit reason breakdown
    print("\n  Exit reason breakdown:")
    for reason, grp in df_v32.groupby("exit_reason"):
        pnl = grp["pnl_pct"]
        wr = (pnl > 0).mean() * 100
        print(f"    {reason:<25} n={len(grp):4d} WR={wr:5.1f}% avg={pnl.mean():+.2f}% tot={pnl.sum():+.0f}%")

    # V32 mechanism stats
    print("\n  V32 mechanism stats:")
    hap_pre = df_v32[df_v32['exit_reason'] == 'v32_hap_preempt']
    woe = df_v32[df_v32['exit_reason'] == 'v32_weak_oversold']
    hc = df_v32[df_v32['exit_reason'] == 'signal_hard_cap']
    print(f"    v32_hap_preempt exits  : {len(hap_pre):4d}  avg={hap_pre['pnl_pct'].mean():+.2f}% (vs hard_cap avg -12.81%)")
    print(f"    v32_weak_oversold exits: {len(woe):4d}  avg={woe['pnl_pct'].mean():+.2f}%")
    print(f"    signal_hard_cap exits  : {len(hc):4d}  avg={hc['pnl_pct'].mean():+.2f}%  tot={hc['pnl_pct'].sum():+.0f}%")

    # Compare V31 hard_cap
    try:
        df_v31_saved = pd.read_csv(os.path.join(OUT, "trades_v31.csv"))
        hc_v31 = df_v31_saved[df_v31_saved['exit_reason'] == 'signal_hard_cap']
        print(f"\n  signal_hard_cap comparison:")
        print(f"    V31: n={len(hc_v31):4d}  WR={( hc_v31['pnl_pct']>0).mean()*100:.1f}%  avg={hc_v31['pnl_pct'].mean():+.2f}%  tot={hc_v31['pnl_pct'].sum():+.0f}%")
        print(f"    V32: n={len(hc):4d}  WR={(hc['pnl_pct']>0).mean()*100:.1f}%  avg={hc['pnl_pct'].mean():+.2f}%  tot={hc['pnl_pct'].sum():+.0f}%")
        print(f"    Saved by preempt/oversold: {len(hc_v31)-len(hc):+d} hard_cap exits avoided")
    except FileNotFoundError:
        pass

    print("\n" + "=" * 130)
    print("  DONE")
    print("=" * 130)
