"""
V28 Experiments: Test each proposed improvement individually then in combinations.
Measures delta vs V27 baseline to decide what goes into V28.

Improvements being tested:
  K1: early_wave_filter  — block entry when wave already mature (days_since_low_10 > 7 + ret_5d > 8%)
  K2: crash_guard        — block entry when symbol ret_20d < -12% (broad crash period)
  K3: wave_acceleration  — bonus entry signal when wave just starting (ret_2d momentum)
  K4: early_loss_cut     — cut loss faster within first 5 days (threshold -5%)
  K5: cycle_peak_exit    — exit when ret_3d turns negative after being up 8%+

Phase 1: Individual tests (K1..K5)
Phase 2: Best 2-way combos
Phase 3: Best 3-way combo
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from experiments.run_v27 import backtest_v27
from src.backtest.engine import backtest_unified
from src.config_loader import get_pipeline_symbols


# ─── V27 base config ──────────────────────────────────────────────────────────
V27_BASE = dict(
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

# V28 flags all off
V28_OFF = dict(
    v28_early_wave_filter=False,
    v28_crash_guard=False,
    v28_wave_acceleration_entry=False,
    v28_early_loss_cut=False,
    v28_cycle_peak_exit=False,
    v28_early_loss_cut_threshold=-0.05,
    v28_early_loss_cut_days=5,
)


def make_backtest_fn(**v28_overrides):
    """Create a backtest function with V27 base + selective V28 flags."""
    v28_cfg = {**V28_OFF, **v28_overrides}
    def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        merged = {**V27_BASE, **v28_cfg, **kwargs}
        return backtest_unified(y_pred, returns, df_test, feature_cols, **merged)
    return bt_fn


def run(symbols_str, backtest_fn):
    return run_test_base(symbols_str, True, True, False, False, True, True, True, True, True, True,
                         backtest_fn=backtest_fn)


def fmt(name, m, cs, baseline_cs=None):
    delta = f" ({cs - baseline_cs:+.0f})" if baseline_cs is not None else ""
    return (f"  {name:<30} | {m['trades']:>5} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
            f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% "
            f"{m['avg_hold']:>4.1f}d | {cs:>6.0f}{delta}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    HDR = f"  {'Config':<30} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>5} | {'Comp':>9}"
    SEP = "  " + "-" * (len(HDR) - 2)

    print("=" * 130)
    print("V28 ABLATION EXPERIMENTS")
    print("=" * 130)

    # ─── Phase 0: V27 Baseline ───────────────────────────────────────────────
    print("\n[Phase 0] V27 Baseline...")
    t0 = time.time()
    t_v27 = run(SYMBOLS, make_backtest_fn())
    m_v27 = calc_metrics(t_v27)
    cs_v27 = comp_score(m_v27, t_v27)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(HDR); print(SEP)
    print(fmt("V27 baseline", m_v27, cs_v27))
    print(SEP)

    # ─── Phase 1: Individual experiments ────────────────────────────────────
    print("\n[Phase 1] Individual K1..K5...")

    experiments = [
        ("K1: early_wave_filter",       dict(v28_early_wave_filter=True)),
        ("K2: crash_guard",              dict(v28_crash_guard=True)),
        ("K3: wave_accel_entry",         dict(v28_wave_acceleration_entry=True)),
        ("K4a: eloss -3% 3d",           dict(v28_early_loss_cut=True,
                                              v28_early_loss_cut_threshold=-0.03,
                                              v28_early_loss_cut_days=3)),
        ("K4b: eloss -4% 5d",           dict(v28_early_loss_cut=True,
                                              v28_early_loss_cut_threshold=-0.04,
                                              v28_early_loss_cut_days=5)),
        ("K4c: eloss -5% 7d",           dict(v28_early_loss_cut=True,
                                              v28_early_loss_cut_threshold=-0.05,
                                              v28_early_loss_cut_days=7)),
        ("K5: cycle_peak_exit",          dict(v28_cycle_peak_exit=True)),
    ]

    results_phase1 = {}
    print(HDR); print(SEP)
    print(fmt("V27 baseline", m_v27, cs_v27))
    print(SEP)

    for name, flags in experiments:
        t_exp = run(SYMBOLS, make_backtest_fn(**flags))
        m_exp = calc_metrics(t_exp)
        cs_exp = comp_score(m_exp, t_exp)
        results_phase1[name] = (flags, t_exp, m_exp, cs_exp)
        print(fmt(name, m_exp, cs_exp, baseline_cs=cs_v27))

    print(SEP)

    # Select positives from phase 1
    positive_k = [(n, r) for n, r in results_phase1.items() if r[3] > cs_v27]
    positive_k_sorted = sorted(positive_k, key=lambda x: x[1][3], reverse=True)
    print(f"\n  Positive changes vs V27: {len(positive_k)} / {len(experiments)}")
    for n, r in positive_k_sorted:
        print(f"    {n}: +{r[3]-cs_v27:.0f} pts (comp={r[3]:.0f})")

    # ─── Phase 2: K4b combos + fine-tuning ─────────────────────────────────
    K4B = dict(v28_early_loss_cut=True, v28_early_loss_cut_threshold=-0.04, v28_early_loss_cut_days=5)
    combos_2 = [
        ("K4b+K1",         {**K4B, **dict(v28_early_wave_filter=True)}),
        ("K4b+K2",         {**K4B, **dict(v28_crash_guard=True)}),
        ("K4b+K1+K2",      {**K4B, **dict(v28_early_wave_filter=True, v28_crash_guard=True)}),
        ("K4: -3.5% 4d",   dict(v28_early_loss_cut=True, v28_early_loss_cut_threshold=-0.035, v28_early_loss_cut_days=4)),
        ("K4: -4% 4d",     dict(v28_early_loss_cut=True, v28_early_loss_cut_threshold=-0.04,  v28_early_loss_cut_days=4)),
        ("K4: -4.5% 5d",   dict(v28_early_loss_cut=True, v28_early_loss_cut_threshold=-0.045, v28_early_loss_cut_days=5)),
        ("K4: -4% 6d",     dict(v28_early_loss_cut=True, v28_early_loss_cut_threshold=-0.04,  v28_early_loss_cut_days=6)),
    ]

    print("\n[Phase 2] K4b combos and fine-tuning...")
    results_phase2 = {}
    print(HDR); print(SEP)
    print(fmt("V27 baseline", m_v27, cs_v27))
    print(fmt("K4b: eloss -4% 5d (best)", results_phase1.get("K4b: eloss -4% 5d", [None,None,m_v27,cs_v27])[2],
              results_phase1.get("K4b: eloss -4% 5d", [None,None,m_v27,cs_v27])[3], baseline_cs=cs_v27))
    print(SEP)
    for name, flags in combos_2:
        t_exp = run(SYMBOLS, make_backtest_fn(**flags))
        m_exp = calc_metrics(t_exp)
        cs_exp = comp_score(m_exp, t_exp)
        results_phase2[name] = (flags, t_exp, m_exp, cs_exp)
        print(fmt(name[:30], m_exp, cs_exp, baseline_cs=cs_v27))
    print(SEP)
    best_combo2 = max(results_phase2.items(), key=lambda x: x[1][3])
    print(f"\n  Best Phase-2 config: {best_combo2[0]} (comp={best_combo2[1][3]:.0f})")

    # ─── Phase 3: 3-way combos (top 3 positives) ────────────────────────────
    if len(positive_k_sorted) >= 3:
        print("\n[Phase 3] 3-way combination of top-3 positives...")
        top3 = [(n, r[0]) for n, r in positive_k_sorted[:3]]
        merged_3 = {}
        for _, f in top3:
            merged_3.update(f)
        combo_name = " + ".join([n.split(":")[0].strip() for n, _ in top3])

        t_exp = run(SYMBOLS, make_backtest_fn(**merged_3))
        m_exp = calc_metrics(t_exp)
        cs_exp = comp_score(m_exp, t_exp)

        print(HDR); print(SEP)
        print(fmt("V27 baseline", m_v27, cs_v27))
        print(fmt(combo_name[:30], m_exp, cs_exp, baseline_cs=cs_v27))
        print(SEP)
    else:
        combo_name = None; m_exp = None; cs_exp = None

    # ─── Phase 4: All positives combined ────────────────────────────────────
    if len(positive_k_sorted) >= 2:
        print("\n[Phase 4] All positives combined...")
        all_positive_flags = {}
        for _, r in positive_k_sorted:
            all_positive_flags.update(r[0])

        t_all = run(SYMBOLS, make_backtest_fn(**all_positive_flags))
        m_all = calc_metrics(t_all)
        cs_all = comp_score(m_all, t_all)

        print(HDR); print(SEP)
        print(fmt("V27 baseline", m_v27, cs_v27))
        print(fmt("All positives", m_all, cs_all, baseline_cs=cs_v27))
        print(SEP)

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 130)
    print("  SUMMARY — Ranked by composite score (delta vs V27)")
    print("=" * 130)
    print(HDR); print(SEP)
    print(fmt("V27 baseline", m_v27, cs_v27))
    print(SEP)

    all_results = list(results_phase1.items())
    if results_phase2:
        all_results += list(results_phase2.items())
    all_results_sorted = sorted(all_results, key=lambda x: x[1][3], reverse=True)

    for name, (flags, t_exp, m_exp, cs_exp) in all_results_sorted:
        marker = " <-- BEST" if cs_exp == all_results_sorted[0][1][3] else ""
        print(fmt(name[:30], m_exp, cs_exp, baseline_cs=cs_v27) + marker)
    print(SEP)

    # ─── Save best candidate ─────────────────────────────────────────────────
    best_name, (best_flags, best_trades, best_m, best_cs) = all_results_sorted[0]
    if best_cs > cs_v27:
        df_best = pd.DataFrame(best_trades)
        out_path = os.path.join(OUT, "trades_v28_candidate.csv")
        df_best.to_csv(out_path, index=False)
        print(f"\n  Best candidate saved: {out_path}")
        print(f"  Config: {best_flags}")
    else:
        print("\n  No improvement found vs V27.")

    print(f"\n{'='*130}")
    print("  DONE")
    print(f"{'='*130}")
