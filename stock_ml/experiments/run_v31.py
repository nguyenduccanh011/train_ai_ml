"""
V31 EXPERIMENT — Multi-patch ablation on top of V30 engine.

Hypotheses (from V30 trade analysis):
  A: Peak-chasing guard     — 130 hard_cap losses (avg -13.6%) entered after ret5d>8% + dist_sma20>8%
  B: Adaptive signal defer  — 506 short-hold signal losers (<30d, WR 17.9%); smarter than V30 flat-3-bars
  C: Hardcap-after-profit   — prevent "ran +5% then lost -13%" by tightening floor once profitable
  D: Profile sizing         — momentum/high_beta avg +11-12% but same size as balanced +4.8%
  E: Short-hold exit filter — block signal exit at hold<10d unless big loss

Ablation plan:
  Phase 1: Individual (A, B, C, D, E) vs V30 baseline
  Phase 2: Best 2-way combos of Phase-1 winners
  Phase 3: Greedy build — best single + try adding each remaining
  Phase 4: Fine-tune params of winning combination

Decision rule: composite_score (higher = better)
"""
import os, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.backtest.engine import backtest_unified
from src.config_loader import get_pipeline_symbols
from experiments.run_v29 import V29_TARGET, V29_FEATURE_SET, backtest_v29
from experiments.run_v30 import V30_DELTA, backtest_v30


# ── V31 patch deltas (each can be switched on/off independently) ──────────────

# A: Peak-chasing guard
V31_A_half = dict(v31_peak_chasing_guard=True, v31_pcg_ret5d_thresh=0.08, v31_pcg_dist_thresh=8.0, v31_pcg_action="half_size")
V31_A_skip = dict(v31_peak_chasing_guard=True, v31_pcg_ret5d_thresh=0.08, v31_pcg_dist_thresh=8.0, v31_pcg_action="skip")
V31_A_tight = dict(v31_peak_chasing_guard=True, v31_pcg_ret5d_thresh=0.06, v31_pcg_dist_thresh=6.0, v31_pcg_action="half_size")

# B: Adaptive defer (replaces / stacks with V30-B1)
V31_B_5bars = dict(v31_adaptive_defer=True, v31_ad_min_cum_ret=0.02, v31_ad_max_bars=5, v31_ad_use_ema_confirm=True)
V31_B_7bars = dict(v31_adaptive_defer=True, v31_ad_min_cum_ret=0.02, v31_ad_max_bars=7, v31_ad_use_ema_confirm=True)
V31_B_10bars = dict(v31_adaptive_defer=True, v31_ad_min_cum_ret=0.02, v31_ad_max_bars=10, v31_ad_use_ema_confirm=True)
V31_B_noema = dict(v31_adaptive_defer=True, v31_ad_min_cum_ret=0.02, v31_ad_max_bars=7, v31_ad_use_ema_confirm=False)

# C: Hardcap-after-profit
V31_C_3pct = dict(v31_hardcap_after_profit=True, v31_hap_profit_trigger=0.05, v31_hap_floor=-0.03)
V31_C_5pct = dict(v31_hardcap_after_profit=True, v31_hap_profit_trigger=0.05, v31_hap_floor=-0.05)
V31_C_strict = dict(v31_hardcap_after_profit=True, v31_hap_profit_trigger=0.03, v31_hap_floor=-0.02)

# D: Profile sizing
V31_D = dict(v31_profile_sizing=True, v31_ps_momentum_mult=1.4, v31_ps_highbeta_mult=1.2,
             v31_ps_defensive_mult=0.80, v31_ps_bank_mult=0.85)
V31_D_mild = dict(v31_profile_sizing=True, v31_ps_momentum_mult=1.2, v31_ps_highbeta_mult=1.1,
                  v31_ps_defensive_mult=0.85, v31_ps_bank_mult=0.90)

# E: Short-hold exit filter
V31_E_10d = dict(v31_short_hold_exit_filter=True, v31_shef_min_hold=10, v31_shef_min_pnl=-0.03)
V31_E_7d = dict(v31_short_hold_exit_filter=True, v31_shef_min_hold=7, v31_shef_min_pnl=-0.03)
V31_E_15d = dict(v31_short_hold_exit_filter=True, v31_shef_min_hold=15, v31_shef_min_pnl=-0.02)


def backtest_v31(y_pred, returns, df_test, feature_cols, **kwargs):
    """V31 = V30 engine + V31 patches (passed via kwargs)."""
    merged = {**V30_DELTA, **kwargs}
    return backtest_v29(y_pred, returns, df_test, feature_cols, **merged)


def run_v31(symbols, patches: dict, label: str):
    """Run one V31 variant and return (metrics, trades, comp_score, label)."""
    t = run_test_base(symbols, True, True, False, False, True, True, True, True, True, True,
                      backtest_fn=lambda *a, **kw: backtest_v31(*a, **kw, **patches),
                      feature_set=V29_FEATURE_SET, target_override=V29_TARGET)
    m = calc_metrics(t)
    cs = comp_score(m, t)
    return m, t, cs, label


def fmt_row(label, m, cs, delta=None):
    d = f"  Δcomp={delta:+.0f}" if delta is not None else ""
    return (f"  {label:<38} | {m['trades']:>5} {m['wr']:>5.1f}% "
            f"{m['avg_pnl']:>+7.2f}% {m['total_pnl']:>+9.1f}% "
            f"{m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>5.1f}d | "
            f"{cs:>6.0f}{d}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--phase", type=int, default=0, help="0=all, 1-4=specific phase")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    HDR = f"  {'Config':<38} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    SEP = "  " + "-" * (len(HDR) - 2)

    print("=" * 130)
    print("V31 ABLATION — 5 patches × parameter sweep × combinations")
    print("=" * 130)

    # ── Baseline ──────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\nBaseline V30...")
    m_v30, t_v30, cs_v30, _ = run_v31(SYMBOLS, {}, "V30 baseline")

    print(HDR); print(SEP)
    print(fmt_row("V30 baseline", m_v30, cs_v30))
    print(SEP)

    # ── Phase 1: Individual patches ──────────────────────────────────────────
    if args.phase in (0, 1):
        print("\n── PHASE 1: Individual patches ──")
        print(HDR); print(SEP)

        p1_variants = [
            # A variants
            ("A_half(ret>8,dist>8)", V31_A_half),
            ("A_skip(ret>8,dist>8)", V31_A_skip),
            ("A_tight(ret>6,dist>6)", V31_A_tight),
            # B variants
            ("B_5bars_ema", V31_B_5bars),
            ("B_7bars_ema", V31_B_7bars),
            ("B_10bars_ema", V31_B_10bars),
            ("B_7bars_noema", V31_B_noema),
            # C variants
            ("C_hap_floor-3pct", V31_C_3pct),
            ("C_hap_floor-5pct", V31_C_5pct),
            ("C_strict_floor-2pct", V31_C_strict),
            # D variants
            ("D_profile_sizing", V31_D),
            ("D_profile_mild", V31_D_mild),
            # E variants
            ("E_shef_10d", V31_E_10d),
            ("E_shef_7d", V31_E_7d),
            ("E_shef_15d", V31_E_15d),
        ]

        p1_results = []
        for label, patches in p1_variants:
            m, t, cs, lbl = run_v31(SYMBOLS, patches, label)
            delta = cs - cs_v30
            print(fmt_row(label, m, cs, delta))
            p1_results.append((label, patches, m, t, cs, delta))

        print(SEP)
        p1_results.sort(key=lambda x: x[4], reverse=True)

        # Best per group
        best_A = max((r for r in p1_results if r[0].startswith("A")), key=lambda x: x[4])
        best_B = max((r for r in p1_results if r[0].startswith("B")), key=lambda x: x[4])
        best_C = max((r for r in p1_results if r[0].startswith("C")), key=lambda x: x[4])
        best_D = max((r for r in p1_results if r[0].startswith("D")), key=lambda x: x[4])
        best_E = max((r for r in p1_results if r[0].startswith("E")), key=lambda x: x[4])

        print(f"\n  Phase 1 winners:")
        for r in [best_A, best_B, best_C, best_D, best_E]:
            print(f"    {r[0]:<38} comp={r[4]:.0f} Δ={r[5]:+.0f}")

        # Keep only positive-delta winners
        p1_winners = [r for r in [best_A, best_B, best_C, best_D, best_E] if r[5] > 0]
        print(f"\n  Positive-delta winners: {[r[0] for r in p1_winners]}")

    # ── Phase 2: Best 2-way combos ────────────────────────────────────────────
    if args.phase in (0, 2):
        print("\n── PHASE 2: Top 2-way combos ──")
        if args.phase == 2:
            # Re-run phase 1 to get winners
            p1_variants = [
                ("A_half", V31_A_half), ("A_skip", V31_A_skip), ("A_tight", V31_A_tight),
                ("B_5bars_ema", V31_B_5bars), ("B_7bars_ema", V31_B_7bars), ("B_10bars_ema", V31_B_10bars),
                ("C_hap_floor-3pct", V31_C_3pct), ("C_hap_floor-5pct", V31_C_5pct),
                ("D_profile_sizing", V31_D), ("E_shef_10d", V31_E_10d), ("E_shef_15d", V31_E_15d),
            ]
            p1_results = []
            for label, patches in p1_variants:
                m, t, cs, lbl = run_v31(SYMBOLS, patches, label)
                p1_results.append((label, patches, m, t, cs, cs - cs_v30))
            best_A = max((r for r in p1_results if r[0].startswith("A")), key=lambda x: x[4])
            best_B = max((r for r in p1_results if r[0].startswith("B")), key=lambda x: x[4])
            best_C = max((r for r in p1_results if r[0].startswith("C")), key=lambda x: x[4])
            best_D = max((r for r in p1_results if r[0].startswith("D")), key=lambda x: x[4])
            best_E = max((r for r in p1_results if r[0].startswith("E")), key=lambda x: x[4])
            p1_winners = [r for r in [best_A, best_B, best_C, best_D, best_E] if r[4] - cs_v30 > 0]

        print(HDR); print(SEP)
        p2_results = []
        for (la, pa, *_), (lb, pb, *_) in itertools.combinations(p1_winners, 2):
            combo_patches = {**pa, **pb}
            label = f"{la}+{lb}"[:38]
            m, t, cs, _ = run_v31(SYMBOLS, combo_patches, label)
            delta = cs - cs_v30
            print(fmt_row(label, m, cs, delta))
            p2_results.append((label, combo_patches, m, t, cs, delta))

        print(SEP)
        p2_results.sort(key=lambda x: x[4], reverse=True)
        print(f"\n  Top 3 combos:")
        for r in p2_results[:3]:
            print(f"    {r[0]:<38} comp={r[4]:.0f} Δ={r[5]:+.0f}")

    # ── Phase 3: Greedy build ─────────────────────────────────────────────────
    if args.phase in (0, 3):
        print("\n── PHASE 3: Greedy build ──")
        if args.phase == 3 or args.phase == 0:
            # Use the best from phase 1 & 2 analysis
            # Start with best single patch, try adding each other
            candidates = {
                "A_half": V31_A_half, "A_tight": V31_A_tight, "A_skip": V31_A_skip,
                "B_7bars": V31_B_7bars, "B_5bars": V31_B_5bars, "B_10bars": V31_B_10bars,
                "C_3pct": V31_C_3pct, "C_5pct": V31_C_5pct,
                "D": V31_D, "D_mild": V31_D_mild,
                "E_10d": V31_E_10d, "E_15d": V31_E_15d,
            }

        greedy_patches = {}
        greedy_cs = cs_v30
        greedy_label = "V30 base"
        greedy_selected = []

        print(HDR); print(SEP)
        max_rounds = 5
        for rnd in range(max_rounds):
            round_best = None
            for name, patch in candidates.items():
                if name in greedy_selected:
                    continue
                trial = {**greedy_patches, **patch}
                m, t, cs, _ = run_v31(SYMBOLS, trial, name)
                delta = cs - greedy_cs
                if round_best is None or cs > round_best[2]:
                    round_best = (name, patch, cs, m, t, delta)

            if round_best is None or round_best[5] <= 0:
                print(f"  Round {rnd+1}: no improvement, stopping greedy.")
                break

            name, patch, cs, m, t, delta = round_best
            greedy_patches = {**greedy_patches, **patch}
            greedy_cs = cs
            greedy_selected.append(name)
            print(fmt_row(f"Greedy+{name}", m, cs, delta))
            print(f"  → Added {name} (Δcomp={delta:+.0f}). Stack: {greedy_selected}")

        print(SEP)
        print(f"\n  Greedy final: {greedy_selected}  comp={greedy_cs:.0f} Δ={greedy_cs-cs_v30:+.0f}")

    # ── Phase 4: Fine-tune best combo ─────────────────────────────────────────
    if args.phase in (0, 4):
        print("\n── PHASE 4: Fine-tune winning combo ──")
        # Tune the B (adaptive defer) max_bars sweep since it's the most impactful param
        sweep_variants = []
        for bars in [3, 5, 7, 10, 14]:
            for cr in [0.01, 0.02, 0.03]:
                for ema in [True, False]:
                    label = f"B_bars{bars}_cr{int(cr*100)}{'_ema' if ema else ''}"
                    patch = dict(v31_adaptive_defer=True, v31_ad_max_bars=bars,
                                v31_ad_min_cum_ret=cr, v31_ad_use_ema_confirm=ema)
                    sweep_variants.append((label, patch))

        print(HDR); print(SEP)
        ft_results = []
        for label, patch in sweep_variants:
            m, t, cs, _ = run_v31(SYMBOLS, patch, label)
            delta = cs - cs_v30
            print(fmt_row(label, m, cs, delta))
            ft_results.append((label, patch, m, t, cs, delta))

        print(SEP)
        ft_results.sort(key=lambda x: x[4], reverse=True)
        print(f"\n  Top 5 fine-tune:")
        for r in ft_results[:5]:
            print(f"    {r[0]:<40} comp={r[4]:.0f} tot={r[2]['total_pnl']:+.0f}% PF={r[2]['pf']:.2f} Δ={r[5]:+.0f}")

    dt = time.time() - t0
    print(f"\n  Total time: {dt:.1f}s")
    print("=" * 130)
    print("  DONE — Run with --phase 1/2/3/4 to run individual phases")
    print("=" * 130)
