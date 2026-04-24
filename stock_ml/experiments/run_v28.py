"""
V28: V27 baseline + K4b (early loss cut -4% within 5d) + K1 (early wave filter).

Ablation results (vs V27 comp=314):
  K4b alone:   comp=324 (+10) — cut loss at -4% within first 5 days
  K4b + K1:    comp=322 (+9)  — same + block mature wave entries
  K4b + K1+K2: comp=319 (+6)  — adding crash_guard reduces signal coverage
  K5, K3:      negative         — cycle peak exit and wave accel hurt

Decision: V28 = K4b + K1
  - K4b raises avg_pnl +0.07%, WR 45.5->46.0%, PF 2.33->2.36
  - K1 adds marginal filtering, combined +9 comp pts
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401

from src.backtest.engine import backtest_unified


def backtest_v28(y_pred, returns, df_test, feature_cols, **kwargs):
    v28_config = dict(
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
        # V28 selected improvements
        v28_early_wave_filter=True,           # K1: block mature wave entries (days_since_low_10 > 7 + ret_5d > 8%)
        v28_crash_guard=False,                # K2: no improvement individually
        v28_wave_acceleration_entry=False,    # K3: negative
        v28_early_loss_cut=True,              # K4b: cut loss at -4% within 5 days (before hard_cap fires)
        v28_cycle_peak_exit=False,            # K5: negative
        v28_early_loss_cut_threshold=-0.04,
        v28_early_loss_cut_days=5,
    )
    merged = {**v28_config, **kwargs}
    return backtest_unified(y_pred, returns, df_test, feature_cols, **merged)


if __name__ == "__main__":
    import argparse
    import time
    import pandas as pd

    from src.experiment_runner import run_test as run_test_base, run_rule_test
    from src.evaluation.scoring import calc_metrics, composite_score as comp_score
    from experiments.run_v27 import backtest_v27
    from src.config_loader import get_pipeline_symbols

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

    print("=" * 120)
    print("V28 BACKTEST: V27 baseline + K4b (early_loss_cut -4%/5d) + K1 (early_wave_filter)")
    print("=" * 120)

    t_rule = run_rule_test(SYMBOLS)
    m_rule = calc_metrics(t_rule)

    t0 = time.time()
    t_v27 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v27)
    m_v27 = calc_metrics(t_v27)
    cs27 = comp_score(m_v27, t_v27)

    t_v28 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v28)
    dt = time.time() - t0
    m_v28 = calc_metrics(t_v28)
    cs28 = comp_score(m_v28, t_v28)

    print(f"\n{'='*120}")
    print(f"  {'Config':<22} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}")
    print("  " + "-" * 102)
    print(f"  {'Rule baseline':<22} | {m_rule['trades']:>5} {m_rule['wr']:>5.1f}% {m_rule['avg_pnl']:>+7.2f}% {m_rule['total_pnl']:>+9.1f}% {m_rule['pf']:>5.2f} {m_rule['max_loss']:>+7.1f}% {m_rule['avg_hold']:>5.1f}d |")
    print(f"  {'V27 baseline':<22} | {m_v27['trades']:>5} {m_v27['wr']:>5.1f}% {m_v27['avg_pnl']:>+7.2f}% {m_v27['total_pnl']:>+9.1f}% {m_v27['pf']:>5.2f} {m_v27['max_loss']:>+7.1f}% {m_v27['avg_hold']:>5.1f}d | {cs27:>6.0f}")
    print(f"  {'V28 selected':<22} | {m_v28['trades']:>5} {m_v28['wr']:>5.1f}% {m_v28['avg_pnl']:>+7.2f}% {m_v28['total_pnl']:>+9.1f}% {m_v28['pf']:>5.2f} {m_v28['max_loss']:>+7.1f}% {m_v28['avg_hold']:>5.1f}d | {cs28:>6.0f}")
    print("  " + "-" * 102)
    print(f"  Delta vs V27:  TotPnL={m_v28['total_pnl']-m_v27['total_pnl']:+.1f}%, WR={m_v28['wr']-m_v27['wr']:+.2f}%, PF={m_v28['pf']-m_v27['pf']:+.3f}, Comp={cs28-cs27:+.0f}")
    print(f"  Delta vs Rule: TotPnL={m_v28['total_pnl']-m_rule['total_pnl']:+.1f}%")
    print(f"  Time: {dt:.1f}s")

    df_v28 = pd.DataFrame(t_v28)
    if len(df_v28) > 0:
        df_v28.to_csv(os.path.join(OUT, "trades_v28.csv"), index=False)
        print(f"\n  Saved {len(df_v28)} V28 trades to results/trades_v28.csv")

    print(f"\n{'='*120}")
    print("  DONE")
    print(f"{'='*120}")
