"""
V29 Round 2: Sau Round 1 không tìm thấy improvement positive trên full universe,
thử các variant chọn lọc hơn (per-profile) và các threshold khác.
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.backtest.engine import backtest_unified
from src.config_loader import get_pipeline_symbols


V28_BASE = dict(
    patch_smart_hardcap=True, patch_pp_restore=True, patch_long_horizon=True,
    patch_symbol_tuning=True, patch_rule_ensemble=True, patch_noise_filter=False,
    patch_adaptive_hardcap=False, patch_pp_2of3=False,
    v26_wider_hardcap=False, v26_relaxed_entry=True, v26_skip_choppy=True,
    v26_extended_hold=False, v26_strong_rule_ensemble=True, v26_min_position=False,
    v26_score5_penalty=False, v26_hardcap_confirm_strong=False,
    v27_selective_choppy=False, v27_hardcap_two_step=True, v27_rule_priority=True,
    v27_dynamic_score5_penalty=True, v27_trend_persistence_hold=True,
    v28_early_wave_filter=True,
    v28_crash_guard=False, v28_wave_acceleration_entry=False,
    v28_early_loss_cut=True, v28_cycle_peak_exit=False,
    v28_early_loss_cut_threshold=-0.04, v28_early_loss_cut_days=5,
)

V29_OFF = dict(
    v29_adaptive_peak_lock=False,
    v29_atr_velocity_exit=False,
    v29_tighter_trail_high_profit=False,
    v29_reversal_after_peak=False,
    v29_breakout_strength_entry=False,
    v29_relstrength_filter=False,
    v29_peak_lock_high_beta_only=False,
    v29_profit_safety_net=False,
    v29_hardcap_after_peak=False,
)


def make_bt(**overrides):
    full = {**V28_BASE, **V29_OFF, **overrides}
    def bt(y_pred, returns, df_test, feature_cols, **kwargs):
        return backtest_unified(y_pred, returns, df_test, feature_cols, **{**full, **kwargs})
    return bt


def run(symbols_str, bt_fn):
    return run_test_base(symbols_str, True, True, False, False, True, True, True, True, True, True,
                         backtest_fn=bt_fn)


def fmt(name, m, cs, base_cs=None):
    delta = f"  ({cs-base_cs:+.0f})" if base_cs is not None else ""
    return (f"  {name:<40} | {m['trades']:>5} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
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

    HDR = f"  {'Config':<40} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>5} | {'Comp':>9}"
    SEP = "  " + "-" * (len(HDR) - 2)

    print("=" * 140)
    print("V29 ROUND 2 — Selective patches and higher thresholds")
    print("=" * 140)

    # Phase 0
    print("\n[Phase 0] V28 baseline...")
    t0 = time.time()
    t_v28 = run(SYMBOLS, make_bt())
    m_v28 = calc_metrics(t_v28)
    cs_v28 = comp_score(m_v28, t_v28)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(HDR); print(SEP)
    print(fmt("V28 baseline", m_v28, cs_v28))
    print(SEP)

    experiments = [
        # Higher trigger / keep more for peak lock
        ("P1d: peak_lock 15%/45%",
            dict(v29_adaptive_peak_lock=True,
                 v29_adaptive_peak_lock_trigger=0.15, v29_adaptive_peak_lock_keep=0.45)),
        ("P1e: peak_lock 20%/50%",
            dict(v29_adaptive_peak_lock=True,
                 v29_adaptive_peak_lock_trigger=0.20, v29_adaptive_peak_lock_keep=0.50)),
        # P7: peak_lock for high_beta/momentum only
        ("P7a: peak_lock HB-only 10%/40%",
            dict(v29_adaptive_peak_lock=True, v29_peak_lock_high_beta_only=True,
                 v29_adaptive_peak_lock_trigger=0.10, v29_adaptive_peak_lock_keep=0.40)),
        ("P7b: peak_lock HB-only 12%/45%",
            dict(v29_adaptive_peak_lock=True, v29_peak_lock_high_beta_only=True,
                 v29_adaptive_peak_lock_trigger=0.12, v29_adaptive_peak_lock_keep=0.45)),
        ("P7c: peak_lock HB-only 08%/35%",
            dict(v29_adaptive_peak_lock=True, v29_peak_lock_high_beta_only=True,
                 v29_adaptive_peak_lock_trigger=0.08, v29_adaptive_peak_lock_keep=0.35)),
        # P8: profit safety net
        ("P8a: safety net @25%",
            dict(v29_profit_safety_net=True, v29_profit_safety_trigger=0.25)),
        ("P8b: safety net @20%",
            dict(v29_profit_safety_net=True, v29_profit_safety_trigger=0.20)),
        ("P8c: safety net @30%",
            dict(v29_profit_safety_net=True, v29_profit_safety_trigger=0.30)),
        # P9: hardcap after peak
        ("P9a: hcap-after-peak 15%/-3%",
            dict(v29_hardcap_after_peak=True,
                 v29_hardcap_after_peak_trigger=0.15, v29_hardcap_after_peak_floor=-0.03)),
        ("P9b: hcap-after-peak 20%/-5%",
            dict(v29_hardcap_after_peak=True,
                 v29_hardcap_after_peak_trigger=0.20, v29_hardcap_after_peak_floor=-0.05)),
        ("P9c: hcap-after-peak 25%/-3%",
            dict(v29_hardcap_after_peak=True,
                 v29_hardcap_after_peak_trigger=0.25, v29_hardcap_after_peak_floor=-0.03)),
        # P6 with looser threshold (less aggressive filter)
        ("P6b: relstrength -8%",
            dict(v29_relstrength_filter=True, v29_rs_ret20_threshold=-0.08)),
        ("P6c: relstrength -10%",
            dict(v29_relstrength_filter=True, v29_rs_ret20_threshold=-0.10)),
    ]

    print("\n[Phase 1] Round-2 individual patches...")
    print(HDR); print(SEP)
    print(fmt("V28 baseline", m_v28, cs_v28))
    print(SEP)

    results = {}
    for name, flags in experiments:
        t_exp = run(SYMBOLS, make_bt(**flags))
        m = calc_metrics(t_exp)
        cs = comp_score(m, t_exp)
        results[name] = (flags, t_exp, m, cs)
        print(fmt(name, m, cs, base_cs=cs_v28))
    print(SEP)

    pos = [(n, r) for n, r in results.items() if r[3] > cs_v28]
    pos_sorted = sorted(pos, key=lambda x: x[1][3], reverse=True)
    print(f"\n  {len(pos)}/{len(experiments)} positive variants:")
    for n, r in pos_sorted:
        print(f"    {n}: comp={r[3]:.0f} (+{r[3]-cs_v28:.0f})  tot={r[2]['total_pnl']:+.1f}%  wr={r[2]['wr']:.1f}%")

    if not pos_sorted:
        print("\n  NO positive variant. V29 == V28.")
    else:
        # Try combo of top-2 if from different families
        if len(pos_sorted) >= 2:
            n1, r1 = pos_sorted[0]
            n2, r2 = pos_sorted[1]
            f1, f2 = r1[0], r2[0]
            # Detect family clash (same patch flag)
            shared_keys = set(f1) & set(f2)
            different_family = not any(
                (k.startswith("v29_") and not k.endswith(("trigger","keep","floor","threshold","k","min_profit","high_beta_only"))
                 and f1.get(k) == f2.get(k) == True) for k in shared_keys
            )
            print(f"\n[Phase 2] Combo top-2: {n1} + {n2}")
            merged = {**f1, **f2}
            t_exp = run(SYMBOLS, make_bt(**merged))
            m = calc_metrics(t_exp); cs = comp_score(m, t_exp)
            print(HDR); print(SEP)
            print(fmt("V28 baseline", m_v28, cs_v28))
            print(fmt(f"{n1[:20]}+{n2[:20]}", m, cs, base_cs=cs_v28))
            print(SEP)

            # Save winner
            winner_flags = merged if cs > pos_sorted[0][1][3] else f1
            winner_trades = t_exp if cs > pos_sorted[0][1][3] else r1[1]
            winner_cs = max(cs, pos_sorted[0][1][3])
        else:
            winner_flags = pos_sorted[0][1][0]
            winner_trades = pos_sorted[0][1][1]
            winner_cs = pos_sorted[0][1][3]

        df_w = pd.DataFrame(winner_trades)
        df_w.to_csv(os.path.join(OUT, "trades_v29_round2_candidate.csv"), index=False)
        print(f"\n[WINNER] comp={winner_cs:.0f} (+{winner_cs-cs_v28:.0f} vs V28)")
        print(f"  Flags: {winner_flags}")
        print(f"  Saved to results/trades_v29_round2_candidate.csv")

    print("\n" + "=" * 140)
    print("  DONE")
    print("=" * 140)
