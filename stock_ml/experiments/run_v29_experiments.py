"""
V29 Experiments: Test exit/entry improvements individually then in combinations.

Root-cause analysis of V28 trades (see analysis):
  - 124 signal_hard_cap trades avg -14.3%, max_profit during trade avg +178%
  - 100% big losers had intra-trade max >= +5% above exit price
  - 97.9% could have exited 3%+ higher in last 5 days before exit
  → "bán hớ" is the biggest pain point, not late entry.

Proposed patches:
  P1 adaptive_peak_lock     — ratchet a profit floor once max_profit >= 10%
  P2 atr_velocity_exit      — exit on 2-day decline > 1.6*ATR after +3% profit
  P3 tighter_trail_high_prof — cap trail at 12% when max_profit >= 20%
  P4 reversal_after_peak    — exit when max_profit>=10% AND ret_2d<=-4%
  P5 breakout_strength_entry — allow breakout entry with vol dry-then-spike
  P6 relstrength_filter     — skip entry when ret_20d < -5% and trend != strong
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.backtest.engine import backtest_unified
from src.config_loader import get_pipeline_symbols


# V28 final config = base to beat
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
    return (f"  {name:<32} | {m['trades']:>5} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
            f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% "
            f"{m['avg_hold']:>4.1f}d | {cs:>6.0f}{delta}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--phase", type=str, default="all")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    HDR = f"  {'Config':<32} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>5} | {'Comp':>9}"
    SEP = "  " + "-" * (len(HDR) - 2)

    print("=" * 130)
    print("V29 ABLATION — exit fixes first (root cause of V28 losers)")
    print("=" * 130)

    # Phase 0: V28 baseline
    print("\n[Phase 0] V28 baseline...")
    t0 = time.time()
    t_v28 = run(SYMBOLS, make_bt())
    m_v28 = calc_metrics(t_v28)
    cs_v28 = comp_score(m_v28, t_v28)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(HDR); print(SEP)
    print(fmt("V28 baseline", m_v28, cs_v28))
    print(SEP)

    # Phase 1: individual patches (default settings)
    experiments = [
        ("P1a: peak_lock 10%/40%",
            dict(v29_adaptive_peak_lock=True,
                 v29_adaptive_peak_lock_trigger=0.10, v29_adaptive_peak_lock_keep=0.40)),
        ("P1b: peak_lock 12%/50%",
            dict(v29_adaptive_peak_lock=True,
                 v29_adaptive_peak_lock_trigger=0.12, v29_adaptive_peak_lock_keep=0.50)),
        ("P1c: peak_lock 08%/35%",
            dict(v29_adaptive_peak_lock=True,
                 v29_adaptive_peak_lock_trigger=0.08, v29_adaptive_peak_lock_keep=0.35)),
        ("P2a: atr_vel 1.6x/3%",
            dict(v29_atr_velocity_exit=True,
                 v29_atr_velocity_k=1.6, v29_atr_velocity_min_profit=0.03)),
        ("P2b: atr_vel 2.0x/5%",
            dict(v29_atr_velocity_exit=True,
                 v29_atr_velocity_k=2.0, v29_atr_velocity_min_profit=0.05)),
        ("P3: tighter trail 20%/12%",
            dict(v29_tighter_trail_high_profit=True,
                 v29_high_profit_trigger=0.20, v29_high_profit_trail=0.12)),
        ("P4a: rev peak 10%/-4%",
            dict(v29_reversal_after_peak=True,
                 v29_reversal_peak_trigger=0.10, v29_reversal_ret2_threshold=-0.04)),
        ("P4b: rev peak 08%/-3%",
            dict(v29_reversal_after_peak=True,
                 v29_reversal_peak_trigger=0.08, v29_reversal_ret2_threshold=-0.03)),
        ("P5: bo_strength entry",
            dict(v29_breakout_strength_entry=True)),
        ("P6: relstrength filter",
            dict(v29_relstrength_filter=True)),
    ]

    print("\n[Phase 1] Individual patches...")
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

    # Pick positives
    pos = [(n, r) for n, r in results.items() if r[3] > cs_v28]
    pos_sorted = sorted(pos, key=lambda x: x[1][3], reverse=True)
    print(f"\n  {len(pos)}/{len(experiments)} positive. Sorted:")
    for n, r in pos_sorted:
        print(f"    {n}: comp={r[3]:.0f} (+{r[3]-cs_v28:.0f})  tot={r[2]['total_pnl']:+.1f}%  wr={r[2]['wr']:.1f}%")

    # Dedupe per-family (keep best variant)
    best_per_family = {}
    def family(name):
        return name.split(":")[0].strip()[:2]
    for name, r in pos_sorted:
        fam = family(name)
        if fam not in best_per_family:
            best_per_family[fam] = (name, r)

    # Phase 2: 2-way combos of families
    fams = list(best_per_family.keys())
    print(f"\n[Phase 2] 2-way combos between families {fams}...")
    print(HDR); print(SEP)
    print(fmt("V28 baseline", m_v28, cs_v28))
    print(SEP)

    combo_results = {}
    for i in range(len(fams)):
        for j in range(i+1, len(fams)):
            name_i, (f1, _, _, _) = best_per_family[fams[i]]
            name_j, (f2, _, _, _) = best_per_family[fams[j]]
            merged = {**f1, **f2}
            cname = f"{fams[i]}+{fams[j]}"
            t_exp = run(SYMBOLS, make_bt(**merged))
            m = calc_metrics(t_exp)
            cs = comp_score(m, t_exp)
            combo_results[cname] = (merged, t_exp, m, cs)
            print(fmt(cname, m, cs, base_cs=cs_v28))
    print(SEP)

    # Phase 3: all positive families combined
    print("\n[Phase 3] All-positive-families combo...")
    print(HDR); print(SEP)
    print(fmt("V28 baseline", m_v28, cs_v28))
    all_flags = {}
    for fam, (name, (f, _, _, _)) in best_per_family.items():
        all_flags.update(f)
    t_all = run(SYMBOLS, make_bt(**all_flags))
    m_all = calc_metrics(t_all); cs_all = comp_score(m_all, t_all)
    print(fmt(f"ALL: {'+'.join(fams)}", m_all, cs_all, base_cs=cs_v28))

    # Phase 4: greedy — start from best single, add families that improve
    print("\n[Phase 4] Greedy build from best single...")
    print(HDR); print(SEP)
    print(fmt("V28 baseline", m_v28, cs_v28))

    greedy_flags = {}
    greedy_families = []
    current_cs = cs_v28
    current_metrics = m_v28
    # sorted by best per-family single comp
    pack = sorted(best_per_family.items(), key=lambda kv: kv[1][1][3], reverse=True)
    for fam, (name, (f, _, m, cs)) in pack:
        trial_flags = {**greedy_flags, **f}
        t_exp = run(SYMBOLS, make_bt(**trial_flags))
        mm = calc_metrics(t_exp); ccs = comp_score(mm, t_exp)
        added = "KEEP" if ccs > current_cs else "DROP"
        print(fmt(f"+ {fam} ({name})", mm, ccs, base_cs=cs_v28) + f"  [{added}]")
        if ccs > current_cs:
            greedy_flags = trial_flags
            greedy_families.append(fam)
            current_cs = ccs
            current_metrics = mm
    print(SEP)
    print(f"  Greedy final families: {greedy_families}")
    print(f"  Greedy final comp: {current_cs:.0f} (+{current_cs-cs_v28:.0f} vs V28)")

    # Save best config
    best_configs = [("greedy", greedy_flags, current_metrics, current_cs, None)]
    best_configs.append(("all_positives", all_flags, m_all, cs_all, None))
    for n, r in pos_sorted:
        best_configs.append((n, r[0], r[2], r[3], r[1]))

    winner = max(best_configs, key=lambda x: x[3])
    print(f"\n[WINNER] {winner[0]} — comp={winner[3]:.0f}")
    print(f"  Flags: {winner[1]}")

    # Save trades of winner
    if winner[4] is not None:
        trades_df = pd.DataFrame(winner[4])
    else:
        # rerun winner to get trades
        t_best = run(SYMBOLS, make_bt(**winner[1]))
        trades_df = pd.DataFrame(t_best)
    if len(trades_df) > 0:
        out_path = os.path.join(OUT, "trades_v29_candidate.csv")
        trades_df.to_csv(out_path, index=False)
        print(f"  Saved candidate trades to {out_path}")

    print("\n" + "=" * 130)
    print("  DONE")
    print("=" * 130)
