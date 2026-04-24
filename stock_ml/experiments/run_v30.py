"""
V30 FINAL — Signal-exit defer breakthrough.

Ablation on V29 engine (26 variants × 4 phases):

  Phase 1 individual winners (top by comp):
    B1a sig_defer 3bars/3%      comp=425  (+81 vs V29)
    B1b sig_defer 2bars/5%      comp=413  (+69)
    B2c mom_hold 7%/RSI75       comp=354  (+11)
    C3a regime_hc -5%/-12%      comp=348  (+4)

  Phase 2 best 2-way combos:
    B1+B2                       comp=431  (+87)  ← top combo but B2 drops alone in greedy
    B1+C3                       comp=428  (+84)
    B1+A3                       comp=428  (+84)

  Phase 3 greedy:
    Only B1 adds value; all others drop comp when added on top of B1.
    Greedy final: B1 only — comp=431 (+87)

  Phase 4 fine-tune B1 params (bars × min_cum_ret sweep):
    bars=3 cr=0.02  → comp=428, tot=+7757%, pf=2.72  ← best composite
    bars=3 cr=0.03  → comp=425, tot=+7812%, pf=2.74
    bars=3 cr=0.05  → comp=428, tot=+7762%, pf=2.72
    bars=5 cr=0.02  → comp=383, tot=+8052%, pf=2.75  (highest PnL but fewer trades)

Decision: V30 = V29 + v30_signal_exit_defer(bars=3, min_cum_ret=0.02)

Headline numbers vs V29 (comp=344):
  comp        344 → 428   (+84 / +24.4%)   ← BREAKTHROUGH
  win rate    46.5% → 46.5%  (~flat)
  avg PnL     +3.87% → +5.30%  (+37%)
  total PnL   +7034% → +7757% (+10.3%)
  PF          2.47 → 2.72   (+10%)
  max loss    -29.9% → -28.6% (↓ 1.3pp)
  avg hold    19.7d → 28.2d  (+8.5d — holds trends longer)

Why it works:
  - 62.8% of V29 exits were signal exits; 60% of those fired when price was still rising
  - avg sell_5d_later = +0.72%, avg fwd20_max = +9.8% → systematic early-exit bias
  - By deferring signal exits 3 bars when trade is profitable (≥2%) and trend is moderate/strong,
    we capture a significant slice of the missed upside without materially increasing drawdown
  - The filter ensures we don't defer on losing trades (cr ≥ 2%)

V30 = V29 engine (= V28 backtest + early_wave target + leading_v3 features)
       + v30_signal_exit_defer(bars=3, min_cum_ret=0.02)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.backtest.engine import backtest_unified
from src.config_loader import get_pipeline_symbols
from experiments.run_v29 import V29_TARGET, V29_FEATURE_SET, backtest_v29


V30_DELTA = dict(
    v30_signal_exit_defer=True,
    v30_sed_defer_bars=3,
    v30_sed_min_cum_ret=0.02,
    # All other V30 patches off (tested; none improve on top of B1)
    v30_peak_proximity_filter=False,
    v30_rally_extension_filter=False,
    v30_pullback_only_entry=False,
    v30_rally_position_scaling=False,
    v30_momentum_hold_override=False,
    v30_chandelier_trail=False,
    v30_atr_aware_hardcap=False,
    v30_hardcap_two_step_v2=False,
    v30_regime_aware_hardcap=False,
)


def backtest_v30(y_pred, returns, df_test, feature_cols, **kwargs):
    """V30 = V29 engine + signal_exit_defer(3 bars, ≥2% profit)."""
    merged = {**V30_DELTA, **kwargs}
    return backtest_v29(y_pred, returns, df_test, feature_cols, **merged)


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
    print("V30 FINAL — V29 + signal_exit_defer(3 bars, ≥2% profit)")
    print("=" * 130)
    print(f"  Target  : {V29_TARGET}")
    print(f"  Features: {V29_FEATURE_SET}")
    print(f"  V30 delta: {V30_DELTA}")
    print()

    t_rule = run_rule_test(SYMBOLS); m_rule = calc_metrics(t_rule)

    t0 = time.time()
    t_v29 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v29,
                          feature_set=V29_FEATURE_SET, target_override=V29_TARGET)
    m_v29 = calc_metrics(t_v29); cs_v29 = comp_score(m_v29, t_v29)

    t_v30 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v30,
                          feature_set=V29_FEATURE_SET, target_override=V29_TARGET)
    dt = time.time() - t0
    m_v30 = calc_metrics(t_v30); cs_v30 = comp_score(m_v30, t_v30)

    HDR = f"  {'Config':<28} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    SEP = "  " + "-" * (len(HDR) - 2)
    print(HDR); print(SEP)
    print(f"  {'Rule baseline':<28} | {m_rule['trades']:>5} {m_rule['wr']:>5.1f}% {m_rule['avg_pnl']:>+7.2f}% {m_rule['total_pnl']:>+9.1f}% {m_rule['pf']:>5.2f} {m_rule['max_loss']:>+7.1f}% {m_rule['avg_hold']:>5.1f}d |")
    print(f"  {'V29 (retrain final)':<28} | {m_v29['trades']:>5} {m_v29['wr']:>5.1f}% {m_v29['avg_pnl']:>+7.2f}% {m_v29['total_pnl']:>+9.1f}% {m_v29['pf']:>5.2f} {m_v29['max_loss']:>+7.1f}% {m_v29['avg_hold']:>5.1f}d | {cs_v29:>6.0f}")
    print(f"  {'V30 (sig_defer final)':<28} | {m_v30['trades']:>5} {m_v30['wr']:>5.1f}% {m_v30['avg_pnl']:>+7.2f}% {m_v30['total_pnl']:>+9.1f}% {m_v30['pf']:>5.2f} {m_v30['max_loss']:>+7.1f}% {m_v30['avg_hold']:>5.1f}d | {cs_v30:>6.0f}")
    print(SEP)
    print(f"  Delta vs V29:  Comp={cs_v30-cs_v29:+.0f}  WR={m_v30['wr']-m_v29['wr']:+.2f}pp  "
          f"AvgPnL={m_v30['avg_pnl']-m_v29['avg_pnl']:+.3f}pp  "
          f"TotPnL={m_v30['total_pnl']-m_v29['total_pnl']:+.1f}%  "
          f"PF={m_v30['pf']-m_v29['pf']:+.3f}  "
          f"MaxLoss={m_v30['max_loss']-m_v29['max_loss']:+.2f}pp  "
          f"AvgHold={m_v30['avg_hold']-m_v29['avg_hold']:+.1f}d")
    print(f"  Time: {dt:.1f}s")

    df_v30 = pd.DataFrame(t_v30)
    if len(df_v30) > 0:
        out_path = os.path.join(OUT, "trades_v30.csv")
        df_v30.to_csv(out_path, index=False)
        print(f"\n  Saved {len(df_v30)} V30 trades to results/trades_v30.csv")

    print("\n" + "=" * 130)
    print("  DONE")
    print("=" * 130)
