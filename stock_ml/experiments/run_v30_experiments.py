"""
V30 Ablation — Systematic search for best engine-layer patches on top of V29.

V29 baseline: early_wave target + leading_v3 features + V28 engine.
This script tests V30 engine patches WITHOUT changing model/features
(to isolate engine contribution).

Patches tested:
  ENTRY:
    A1: peak_proximity_filter
    A2: rally_extension_filter
    A3: pullback_only_entry
    A4: rally_position_scaling

  EXIT (hold longer / smarter):
    B1: signal_exit_defer
    B2: momentum_hold_override
    B3: chandelier_trail

  HARD-CAP FIX:
    C1: atr_aware_hardcap
    C2: hardcap_two_step_v2
    C3: regime_aware_hardcap

Procedure:
  Phase 0 — V29 baseline (V28 engine + early_wave target + leading_v3)
  Phase 1 — All individuals
  Phase 2 — 2-way combos of positive families
  Phase 3 — Greedy build
  Phase 4 — Best combo fine-tune (threshold sweep on top-2)
"""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.backtest.engine import backtest_unified
from src.config_loader import get_pipeline_symbols
from experiments.run_v29 import V29_TARGET, V29_FEATURE_SET


# ── V29 engine config (= V28 engine) ────────────────────────────────────────
V29_ENGINE = dict(
    patch_smart_hardcap=True, patch_pp_restore=True, patch_long_horizon=True,
    patch_symbol_tuning=True, patch_rule_ensemble=True, patch_noise_filter=False,
    patch_adaptive_hardcap=False, patch_pp_2of3=False,
    v26_wider_hardcap=False, v26_relaxed_entry=True, v26_skip_choppy=True,
    v26_extended_hold=False, v26_strong_rule_ensemble=True, v26_min_position=False,
    v26_score5_penalty=False, v26_hardcap_confirm_strong=False,
    v27_selective_choppy=False, v27_hardcap_two_step=True, v27_rule_priority=True,
    v27_dynamic_score5_penalty=True, v27_trend_persistence_hold=True,
    v28_early_wave_filter=True, v28_crash_guard=False, v28_wave_acceleration_entry=False,
    v28_early_loss_cut=True, v28_cycle_peak_exit=False,
    v28_early_loss_cut_threshold=-0.04, v28_early_loss_cut_days=5,
    v29_adaptive_peak_lock=False, v29_atr_velocity_exit=False,
    v29_tighter_trail_high_profit=False, v29_reversal_after_peak=False,
    v29_breakout_strength_entry=False, v29_relstrength_filter=False,
    v29_peak_lock_high_beta_only=False, v29_profit_safety_net=False, v29_hardcap_after_peak=False,
)

V30_OFF = dict(
    v30_peak_proximity_filter=False, v30_rally_extension_filter=False,
    v30_pullback_only_entry=False, v30_rally_position_scaling=False,
    v30_signal_exit_defer=False, v30_momentum_hold_override=False,
    v30_chandelier_trail=False,
    v30_atr_aware_hardcap=False, v30_hardcap_two_step_v2=False, v30_regime_aware_hardcap=False,
)


def make_bt(**overrides):
    full = {**V29_ENGINE, **V30_OFF, **overrides}
    def bt(y_pred, returns, df_test, feature_cols, **kwargs):
        return backtest_unified(y_pred, returns, df_test, feature_cols, **{**full, **kwargs})
    return bt


def run(symbols_str, bt_fn):
    return run_test_base(symbols_str, True, True, False, False, True, True, True, True, True, True,
                         backtest_fn=bt_fn,
                         feature_set=V29_FEATURE_SET,
                         target_override=V29_TARGET)


def fmt(name, m, cs, base_cs=None):
    delta = f"  ({cs-base_cs:+.0f})" if base_cs is not None else ""
    return (f"  {name:<42} | {m['trades']:>5} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
            f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% "
            f"{m['avg_hold']:>5.1f}d | {cs:>6.0f}{delta}")


def HDR():
    return f"  {'Config':<42} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>9}"

def SEP(hdr):
    return "  " + "-" * (len(hdr) - 2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--phase", type=str, default="all", help="all|1|2|3|4")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    hdr = HDR(); sep = SEP(hdr)

    print("=" * 140)
    print("V30 ABLATION — Engine patches on top of V29 (early_wave + leading_v3)")
    print("=" * 140)

    # Phase 0: V29 baseline
    print("\n[Phase 0] V29 baseline...")
    t0 = time.time()
    t_v29 = run(SYMBOLS, make_bt())
    m_v29 = calc_metrics(t_v29)
    cs_v29 = comp_score(m_v29, t_v29)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(hdr); print(sep)
    print(fmt("V29 baseline", m_v29, cs_v29))
    print(sep)

    # ── Phase 1: Individual patches ──────────────────────────────────────────
    experiments = [
        # ── ENTRY filters ──
        ("A1a: peak_prox -2%/8%",
            dict(v30_peak_proximity_filter=True,
                 v30_peak_prox_dist_threshold=-0.02, v30_peak_prox_rally10_min=0.08)),
        ("A1b: peak_prox -2%/6%",
            dict(v30_peak_proximity_filter=True,
                 v30_peak_prox_dist_threshold=-0.02, v30_peak_prox_rally10_min=0.06)),
        ("A1c: peak_prox -3%/8%",
            dict(v30_peak_proximity_filter=True,
                 v30_peak_prox_dist_threshold=-0.03, v30_peak_prox_rally10_min=0.08)),
        ("A2a: rally_ext 12%/18%",
            dict(v30_rally_extension_filter=True,
                 v30_rally10_hard_block=0.12, v30_rally20_hard_block=0.18)),
        ("A2b: rally_ext 10%/15%",
            dict(v30_rally_extension_filter=True,
                 v30_rally10_hard_block=0.10, v30_rally20_hard_block=0.15)),
        ("A2c: rally_ext 15%/22%",
            dict(v30_rally_extension_filter=True,
                 v30_rally10_hard_block=0.15, v30_rally20_hard_block=0.22)),
        ("A3a: pullback_only 3%",
            dict(v30_pullback_only_entry=True, v30_pullback_min_pct=0.03)),
        ("A3b: pullback_only 2%",
            dict(v30_pullback_only_entry=True, v30_pullback_min_pct=0.02)),
        ("A4a: rally_scaling 5%/10%",
            dict(v30_rally_position_scaling=True,
                 v30_rps_tier1_rally=0.05, v30_rps_tier2_rally=0.10)),
        ("A4b: rally_scaling 6%/12%",
            dict(v30_rally_position_scaling=True,
                 v30_rps_tier1_rally=0.06, v30_rps_tier2_rally=0.12)),
        # ── EXIT improvements ──
        ("B1a: sig_defer 3bars/3%",
            dict(v30_signal_exit_defer=True,
                 v30_sed_defer_bars=3, v30_sed_min_cum_ret=0.03)),
        ("B1b: sig_defer 2bars/5%",
            dict(v30_signal_exit_defer=True,
                 v30_sed_defer_bars=2, v30_sed_min_cum_ret=0.05)),
        ("B2a: mom_hold 5%/RSI72",
            dict(v30_momentum_hold_override=True,
                 v30_mho_min_profit=0.05, v30_mho_rsi_max=72)),
        ("B2b: mom_hold 3%/RSI70",
            dict(v30_momentum_hold_override=True,
                 v30_mho_min_profit=0.03, v30_mho_rsi_max=70)),
        ("B2c: mom_hold 7%/RSI75",
            dict(v30_momentum_hold_override=True,
                 v30_mho_min_profit=0.07, v30_mho_rsi_max=75)),
        ("B3a: chandelier 3ATR/5%",
            dict(v30_chandelier_trail=True,
                 v30_chand_atr_mult=3.0, v30_chand_profit_trigger=0.05)),
        ("B3b: chandelier 2.5ATR/5%",
            dict(v30_chandelier_trail=True,
                 v30_chand_atr_mult=2.5, v30_chand_profit_trigger=0.05)),
        ("B3c: chandelier 3ATR/8%",
            dict(v30_chandelier_trail=True,
                 v30_chand_atr_mult=3.0, v30_chand_profit_trigger=0.08)),
        # ── HARD CAP fixes ──
        ("C1a: atr_hc 1.5x/8-20%",
            dict(v30_atr_aware_hardcap=True,
                 v30_atr_hc_mult=1.5, v30_atr_hc_floor=0.08, v30_atr_hc_ceiling=0.20)),
        ("C1b: atr_hc 2.0x/8-20%",
            dict(v30_atr_aware_hardcap=True,
                 v30_atr_hc_mult=2.0, v30_atr_hc_floor=0.08, v30_atr_hc_ceiling=0.20)),
        ("C2a: hc2step -4%/-8%",
            dict(v30_hardcap_two_step_v2=True,
                 v30_hc2_step1_loss=-0.04, v30_hc2_step2_loss=-0.08)),
        ("C2b: hc2step -3%/-7%",
            dict(v30_hardcap_two_step_v2=True,
                 v30_hc2_step1_loss=-0.03, v30_hc2_step2_loss=-0.07)),
        ("C2c: hc2step -5%/-10%",
            dict(v30_hardcap_two_step_v2=True,
                 v30_hc2_step1_loss=-0.05, v30_hc2_step2_loss=-0.10)),
        ("C3a: regime_hc -5%/-12%",
            dict(v30_regime_aware_hardcap=True,
                 v30_rah_choppy_cap=-0.05, v30_rah_trending_cap=-0.12)),
        ("C3b: regime_hc -4%/-10%",
            dict(v30_regime_aware_hardcap=True,
                 v30_rah_choppy_cap=-0.04, v30_rah_trending_cap=-0.10)),
    ]

    print("\n[Phase 1] Individual patches...")
    print(hdr); print(sep)
    print(fmt("V29 baseline", m_v29, cs_v29))
    print(sep)

    results = {}
    for name, flags in experiments:
        t1 = time.time()
        t_exp = run(SYMBOLS, make_bt(**flags))
        m = calc_metrics(t_exp)
        cs = comp_score(m, t_exp)
        results[name] = (flags, t_exp, m, cs)
        print(fmt(name, m, cs, base_cs=cs_v29))
    print(sep)

    pos = [(n, r) for n, r in results.items() if r[3] > cs_v29]
    pos_sorted = sorted(pos, key=lambda x: x[1][3], reverse=True)
    print(f"\n  {len(pos)}/{len(experiments)} positive. Top positives:")
    for n, r in pos_sorted[:8]:
        print(f"    {n}: comp={r[3]:.0f} (+{r[3]-cs_v29:.0f})  tot={r[2]['total_pnl']:+.1f}%  pf={r[2]['pf']:.2f}  maxloss={r[2]['max_loss']:+.1f}%")

    # Best per family
    def family(nm): return nm.split(":")[0].strip()[:2]
    best_per_family = {}
    for nm, r in pos_sorted:
        fam = family(nm)
        if fam not in best_per_family:
            best_per_family[fam] = (nm, r)

    # ── Phase 2: 2-way combos ────────────────────────────────────────────────
    if args.phase in ("all", "2", "3", "4"):
        fams = list(best_per_family.keys())
        print(f"\n[Phase 2] 2-way combos of positive families {fams}...")
        print(hdr); print(sep)
        print(fmt("V29 baseline", m_v29, cs_v29))
        print(sep)

        combo_results = {}
        for fa, fb in itertools.combinations(fams, 2):
            nm_a, (fl_a, _, _, _) = best_per_family[fa]
            nm_b, (fl_b, _, _, _) = best_per_family[fb]
            merged = {**fl_a, **fl_b}
            cname = f"{fa}+{fb}"
            t_exp = run(SYMBOLS, make_bt(**merged))
            m = calc_metrics(t_exp); cs = comp_score(m, t_exp)
            combo_results[cname] = (merged, t_exp, m, cs)
            print(fmt(cname, m, cs, base_cs=cs_v29))
        print(sep)

        combo_pos = sorted(combo_results.items(), key=lambda x: x[1][3], reverse=True)
        print("  Top combos:")
        for cname, (fl, _, m, cs) in combo_pos[:5]:
            print(f"    {cname}: comp={cs:.0f} (+{cs-cs_v29:.0f})  tot={m['total_pnl']:+.1f}%  pf={m['pf']:.2f}  maxloss={m['max_loss']:+.1f}%")

    # ── Phase 3: Greedy build ────────────────────────────────────────────────
    if args.phase in ("all", "3", "4"):
        print(f"\n[Phase 3] Greedy build from best single family...")
        print(hdr); print(sep)
        print(fmt("V29 baseline", m_v29, cs_v29))
        print(sep)

        greedy_flags = {}
        greedy_families = []
        current_cs = cs_v29

        pack = sorted(best_per_family.items(), key=lambda kv: kv[1][1][3], reverse=True)
        for fam, (nm, (fl, _, m_single, cs_single)) in pack:
            trial_flags = {**greedy_flags, **fl}
            t_exp = run(SYMBOLS, make_bt(**trial_flags))
            mm = calc_metrics(t_exp); ccs = comp_score(mm, t_exp)
            added = "KEEP" if ccs > current_cs else "DROP"
            print(fmt(f"+ {fam} ({nm[:22]})", mm, ccs, base_cs=cs_v29) + f"  [{added}]")
            if ccs > current_cs:
                greedy_flags = trial_flags
                greedy_families.append(fam)
                current_cs = ccs
        print(sep)
        print(f"  Greedy final families: {greedy_families}")
        print(f"  Greedy final comp: {current_cs:.0f} (+{current_cs-cs_v29:.0f} vs V29)")

    # ── Phase 4: Fine-tune best combo ────────────────────────────────────────
    if args.phase in ("all", "4") and greedy_families:
        print(f"\n[Phase 4] Fine-tune top params of greedy winner...")
        print(hdr); print(sep)
        print(fmt("Greedy baseline", calc_metrics(run(SYMBOLS, make_bt(**greedy_flags))),
                  current_cs))
        print(sep)

        # Build fine-tune variants for each active family
        finetune_variants = []
        if "A1" in greedy_families:
            for d, r in [(-0.02, 0.07), (-0.02, 0.09), (-0.025, 0.08), (-0.015, 0.08)]:
                finetune_variants.append((f"A1 tune d={d} r={r}",
                    {**greedy_flags, "v30_peak_prox_dist_threshold": d, "v30_peak_prox_rally10_min": r}))
        if "A2" in greedy_families:
            for r10, r20 in [(0.10, 0.16), (0.11, 0.17), (0.13, 0.20)]:
                finetune_variants.append((f"A2 tune r10={r10} r20={r20}",
                    {**greedy_flags, "v30_rally10_hard_block": r10, "v30_rally20_hard_block": r20}))
        if "A4" in greedy_families:
            for t1, t2 in [(0.04, 0.09), (0.05, 0.11), (0.06, 0.13)]:
                finetune_variants.append((f"A4 tune t1={t1} t2={t2}",
                    {**greedy_flags, "v30_rps_tier1_rally": t1, "v30_rps_tier2_rally": t2}))
        if "B2" in greedy_families:
            for mp, rm in [(0.04, 70), (0.05, 73), (0.06, 72), (0.08, 74)]:
                finetune_variants.append((f"B2 tune mp={mp} rsi={rm}",
                    {**greedy_flags, "v30_mho_min_profit": mp, "v30_mho_rsi_max": rm}))
        if "B3" in greedy_families:
            for am, pt in [(2.5, 0.05), (3.0, 0.06), (3.5, 0.05), (3.0, 0.04)]:
                finetune_variants.append((f"B3 tune atr={am} trig={pt}",
                    {**greedy_flags, "v30_chand_atr_mult": am, "v30_chand_profit_trigger": pt}))
        if "C2" in greedy_families:
            for s1, s2 in [(-0.03, -0.06), (-0.04, -0.07), (-0.05, -0.09)]:
                finetune_variants.append((f"C2 tune s1={s1} s2={s2}",
                    {**greedy_flags, "v30_hc2_step1_loss": s1, "v30_hc2_step2_loss": s2}))

        best_ft_cs = current_cs
        best_ft_flags = greedy_flags
        best_ft_trades = None
        for ftname, ftflags in finetune_variants:
            t_exp = run(SYMBOLS, make_bt(**ftflags))
            m = calc_metrics(t_exp); cs = comp_score(m, t_exp)
            print(fmt(ftname, m, cs, base_cs=cs_v29))
            if cs > best_ft_cs:
                best_ft_cs = cs; best_ft_flags = ftflags; best_ft_trades = t_exp
        print(sep)
        if best_ft_cs > current_cs:
            print(f"  Fine-tune improved: {current_cs:.0f} -> {best_ft_cs:.0f} (+{best_ft_cs-current_cs:.0f})")
            greedy_flags = best_ft_flags; current_cs = best_ft_cs
        else:
            print(f"  No fine-tune improvement. Greedy winner stands: {current_cs:.0f}")

    # ── Summary & save ────────────────────────────────────────────────────────
    print("\n" + "="*140)
    print("  SUMMARY")
    print("="*140)
    print(hdr); print(sep)
    print(fmt("V29 baseline", m_v29, cs_v29))
    # rerun greedy final
    t_final = run(SYMBOLS, make_bt(**greedy_flags))
    m_final = calc_metrics(t_final)
    cs_final = comp_score(m_final, t_final)
    print(fmt("V30 winner (greedy)", m_final, cs_final, base_cs=cs_v29))
    print(sep)
    print(f"\n  Winner flags: {greedy_flags}")

    df_win = pd.DataFrame(t_final)
    win_path = os.path.join(OUT, "trades_v30_candidate.csv")
    df_win.to_csv(win_path, index=False)
    print(f"\n  Saved {len(df_win)} trades to {win_path}")
    print("\n" + "="*140)
    print("  DONE")
    print("="*140)
