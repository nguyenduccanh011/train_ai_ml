"""
V32 ABLATION — Multi-patch ablation on top of V31 engine.

Root cause analysis (from trades_v31.csv):
  signal_hard_cap: 211 trades, 0% WR, -2703.8% drag — #1 problem
  - 99.5% exit at trend=weak, avg exit_dist_sma20=-10.3%
  - 61.1% had max_profit > 0% before hitting hard_cap
  - 43 trades had max_profit > 5% then closed < -10% = -535% drag
  Bug: V31-C (HAP) check runs AFTER hard_cap → never intercepts

  Signal exits trend=weak: 435 trades, WR 29%, avg -0.39%
  v31_hap_exit: only 1 trade (threshold too high or ordering bug)
  quick_reentry: 0 trades (feature not triggering)
  Missed upside: 59.3% profitable trades had avg missed 13.47%

Hypotheses:
  A: HAP preempt — run V32-hap_preempt BEFORE hard_cap (fix ordering bug)
  B: Trend-weak oversold exit — exit early when trend=weak + dist_sma20 < -7%
  C: Dynamic hard_cap tighten — when dist_sma20 < -8% exit at -7% instead of -12%
  D: Profit ratchet exit — when max_profit >= 8%, floor = 30% of max_profit
  E: Signal-weak passthrough — allow signal exit when trend=weak + oversold even if short hold

Ablation plan:
  Phase 1: Individual (A, B, C, D, E) vs V31 baseline
  Phase 2: Best 2-way combos
  Phase 3: Greedy build
  Phase 4: Fine-tune params of winning combination
"""
import os, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v29 import V29_TARGET, V29_FEATURE_SET, backtest_v29
from experiments.run_v30 import V30_DELTA, backtest_v30
from experiments.run_v31_final import V31_DELTA, backtest_v31


# ── V32 patch deltas ──────────────────────────────────────────────────────────

# A: HAP preempt — thoát TRƯỚC hard_cap khi đã có profit rồi rớt
V32_A_3pct = dict(v32_hap_preempt=True, v32_hap_pre_trigger=0.05, v32_hap_pre_floor=-0.03)
V32_A_5pct = dict(v32_hap_preempt=True, v32_hap_pre_trigger=0.05, v32_hap_pre_floor=-0.05)
V32_A_tight = dict(v32_hap_preempt=True, v32_hap_pre_trigger=0.03, v32_hap_pre_floor=-0.02)

# B: Trend-weak oversold exit
V32_B_7pct = dict(v32_weak_oversold_exit=True, v32_woe_dist_thresh=-0.07, v32_woe_min_profit=0.0, v32_woe_hold_min=5)
V32_B_9pct = dict(v32_weak_oversold_exit=True, v32_woe_dist_thresh=-0.09, v32_woe_min_profit=0.0, v32_woe_hold_min=5)
V32_B_profit = dict(v32_weak_oversold_exit=True, v32_woe_dist_thresh=-0.07, v32_woe_min_profit=0.02, v32_woe_hold_min=5)
V32_B_long = dict(v32_weak_oversold_exit=True, v32_woe_dist_thresh=-0.07, v32_woe_min_profit=0.0, v32_woe_hold_min=10)

# C: Dynamic hard_cap tighten when far below SMA20
V32_C_7pct = dict(v32_dynamic_hc_dist=True, v32_dhc_dist_thresh=-0.08, v32_dhc_tight_cap=-0.07)
V32_C_8pct = dict(v32_dynamic_hc_dist=True, v32_dhc_dist_thresh=-0.08, v32_dhc_tight_cap=-0.08)
V32_C_loose = dict(v32_dynamic_hc_dist=True, v32_dhc_dist_thresh=-0.10, v32_dhc_tight_cap=-0.07)

# D: Profit ratchet exit
V32_D_30 = dict(v32_profit_ratchet=True, v32_pr_trigger=0.08, v32_pr_keep=0.30)
V32_D_50 = dict(v32_profit_ratchet=True, v32_pr_trigger=0.10, v32_pr_keep=0.50)
V32_D_tight = dict(v32_profit_ratchet=True, v32_pr_trigger=0.06, v32_pr_keep=0.40)

# E: Signal-weak passthrough
V32_E_5pct = dict(v32_signal_weak_exit=True, v32_swe_dist_thresh=-0.05)
V32_E_7pct = dict(v32_signal_weak_exit=True, v32_swe_dist_thresh=-0.07)


def backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs):
    """V32 = V31 engine + V32 patches (passed via kwargs)."""
    merged = {**V31_DELTA, **kwargs}
    return backtest_v30(y_pred, returns, df_test, feature_cols, **merged)


def run_v32(symbols, patches: dict, label: str):
    t = run_test_base(symbols, True, True, False, False, True, True, True, True, True, True,
                      backtest_fn=lambda *a, **kw: backtest_v32(*a, **kw, **patches),
                      feature_set=V29_FEATURE_SET, target_override=V29_TARGET)
    m = calc_metrics(t)
    cs = comp_score(m, t)
    return m, t, cs, label


def fmt_row(label, m, cs, delta=None):
    d = f"  D={delta:+.0f}" if delta is not None else ""
    return (f"  {label:<50} | {m['trades']:>5} {m['wr']:>5.1f}% "
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

    HDR = f"  {'Config':<50} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    SEP = "  " + "-" * (len(HDR) - 2)

    print("=" * 140)
    print("V32 ABLATION — Fix signal_hard_cap (-2703% drag) via 5 targeted patches")
    print("=" * 140)

    t0 = time.time()
    print("\nBaseline V31...")
    m_v31, t_v31, cs_v31, _ = run_v32(SYMBOLS, {}, "V31 baseline")

    print(HDR); print(SEP)
    print(fmt_row("V31 baseline", m_v31, cs_v31))
    print(SEP)

    # ── Phase 1: Individual patches ──────────────────────────────────────────
    if args.phase in (0, 1):
        print("\n== PHASE 1: Individual patches ==")
        print(HDR); print(SEP)

        p1_variants = [
            # A: HAP preempt
            ("A_hap_pre_3pct", V32_A_3pct),
            ("A_hap_pre_5pct", V32_A_5pct),
            ("A_hap_pre_tight", V32_A_tight),
            # B: Trend-weak oversold exit
            ("B_weak_oversold_7pct", V32_B_7pct),
            ("B_weak_oversold_9pct", V32_B_9pct),
            ("B_weak_oversold_profit", V32_B_profit),
            ("B_weak_oversold_long", V32_B_long),
            # C: Dynamic hard_cap tighten
            ("C_dyn_hc_7pct", V32_C_7pct),
            ("C_dyn_hc_8pct", V32_C_8pct),
            ("C_dyn_hc_loose", V32_C_loose),
            # D: Profit ratchet
            ("D_ratchet_30", V32_D_30),
            ("D_ratchet_50", V32_D_50),
            ("D_ratchet_tight", V32_D_tight),
            # E: Signal-weak passthrough
            ("E_sig_weak_5pct", V32_E_5pct),
            ("E_sig_weak_7pct", V32_E_7pct),
        ]

        p1_results = []
        for label, patches in p1_variants:
            m, t, cs, lbl = run_v32(SYMBOLS, patches, label)
            delta = cs - cs_v31
            print(fmt_row(label, m, cs, delta))
            p1_results.append((label, patches, m, t, cs, delta))

        print(SEP)
        p1_results.sort(key=lambda x: x[4], reverse=True)

        best_A = max((r for r in p1_results if r[0].startswith("A")), key=lambda x: x[4])
        best_B = max((r for r in p1_results if r[0].startswith("B")), key=lambda x: x[4])
        best_C = max((r for r in p1_results if r[0].startswith("C")), key=lambda x: x[4])
        best_D = max((r for r in p1_results if r[0].startswith("D")), key=lambda x: x[4])
        best_E = max((r for r in p1_results if r[0].startswith("E")), key=lambda x: x[4])

        print(f"\n  Phase 1 winners:")
        for r in [best_A, best_B, best_C, best_D, best_E]:
            print(f"    {r[0]:<50} comp={r[4]:.0f} D={r[5]:+.0f}")

        p1_winners = [r for r in [best_A, best_B, best_C, best_D, best_E] if r[5] > 0]
        print(f"\n  Positive-delta: {[r[0] for r in p1_winners]}")

    # ── Phase 2: Best 2-way combos ────────────────────────────────────────────
    if args.phase in (0, 2):
        print("\n== PHASE 2: Top 2-way combos ==")
        if args.phase == 2:
            # Re-run individual best variants to get winners
            p1_variants_min = [
                ("A_hap_pre_3pct", V32_A_3pct), ("A_hap_pre_5pct", V32_A_5pct),
                ("B_weak_oversold_7pct", V32_B_7pct), ("B_weak_oversold_9pct", V32_B_9pct),
                ("C_dyn_hc_7pct", V32_C_7pct), ("C_dyn_hc_loose", V32_C_loose),
                ("D_ratchet_30", V32_D_30), ("D_ratchet_50", V32_D_50),
                ("E_sig_weak_5pct", V32_E_5pct),
            ]
            p1_results = []
            for label, patches in p1_variants_min:
                m, t, cs, lbl = run_v32(SYMBOLS, patches, label)
                p1_results.append((label, patches, m, t, cs, cs - cs_v31))
            best_A = max((r for r in p1_results if r[0].startswith("A")), key=lambda x: x[4])
            best_B = max((r for r in p1_results if r[0].startswith("B")), key=lambda x: x[4])
            best_C = max((r for r in p1_results if r[0].startswith("C")), key=lambda x: x[4])
            best_D = max((r for r in p1_results if r[0].startswith("D")), key=lambda x: x[4])
            best_E = max((r for r in p1_results if r[0].startswith("E")), key=lambda x: x[4])
            p1_winners = [r for r in [best_A, best_B, best_C, best_D, best_E] if r[4] - cs_v31 > 0]

        print(HDR); print(SEP)
        p2_results = []
        for (la, pa, *_), (lb, pb, *_) in itertools.combinations(p1_winners, 2):
            combo_patches = {**pa, **pb}
            label = f"{la}+{lb}"[:50]
            m, t, cs, _ = run_v32(SYMBOLS, combo_patches, label)
            delta = cs - cs_v31
            print(fmt_row(label, m, cs, delta))
            p2_results.append((label, combo_patches, m, t, cs, delta))

        print(SEP)
        p2_results.sort(key=lambda x: x[4], reverse=True)
        print(f"\n  Top 3 combos:")
        for r in p2_results[:3]:
            print(f"    {r[0]:<50} comp={r[4]:.0f} D={r[5]:+.0f}")

    # ── Phase 3: Greedy build ─────────────────────────────────────────────────
    if args.phase in (0, 3):
        print("\n== PHASE 3: Greedy build ==")
        candidates = {
            "A_3pct": V32_A_3pct, "A_5pct": V32_A_5pct, "A_tight": V32_A_tight,
            "B_7pct": V32_B_7pct, "B_9pct": V32_B_9pct, "B_profit": V32_B_profit,
            "C_7pct": V32_C_7pct, "C_8pct": V32_C_8pct, "C_loose": V32_C_loose,
            "D_30": V32_D_30, "D_50": V32_D_50,
            "E_5pct": V32_E_5pct, "E_7pct": V32_E_7pct,
        }

        greedy_patches = {}
        greedy_cs = cs_v31
        greedy_selected = []

        print(HDR); print(SEP)
        for rnd in range(5):
            round_best = None
            for name, patch in candidates.items():
                if name in greedy_selected:
                    continue
                trial = {**greedy_patches, **patch}
                m, t, cs, _ = run_v32(SYMBOLS, trial, name)
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
            print(f"  → Added {name} (D={delta:+.0f}). Stack: {greedy_selected}")

        print(SEP)
        print(f"\n  Greedy final: {greedy_selected}  comp={greedy_cs:.0f} D={greedy_cs-cs_v31:+.0f}")

    # ── Phase 4: Fine-tune best combo ─────────────────────────────────────────
    if args.phase in (0, 4):
        print("\n== PHASE 4: Fine-tune best combo ==")
        # Sweep A thresholds (most likely winner)
        sweep_variants = []
        for trigger in [0.03, 0.05, 0.07, 0.10]:
            for floor_pct in [0.02, 0.03, 0.05]:
                label = f"A_pre_t{int(trigger*100)}_f{int(floor_pct*100)}"
                patch = dict(v32_hap_preempt=True, v32_hap_pre_trigger=trigger, v32_hap_pre_floor=-floor_pct)
                sweep_variants.append((label, patch))
        # Sweep A + B combos
        for dist in [0.06, 0.07, 0.08, 0.09]:
            for hold_min in [3, 5, 7]:
                label = f"A3+B_d{int(dist*100)}_h{hold_min}"
                patch = {**V32_A_3pct,
                         **dict(v32_weak_oversold_exit=True, v32_woe_dist_thresh=-dist,
                                v32_woe_min_profit=0.0, v32_woe_hold_min=hold_min)}
                sweep_variants.append((label, patch))

        print(HDR); print(SEP)
        ft_results = []
        for label, patch in sweep_variants:
            m, t, cs, _ = run_v32(SYMBOLS, patch, label)
            delta = cs - cs_v31
            print(fmt_row(label, m, cs, delta))
            ft_results.append((label, patch, m, t, cs, delta))

        print(SEP)
        ft_results.sort(key=lambda x: x[4], reverse=True)
        print(f"\n  Top 5:")
        for r in ft_results[:5]:
            print(f"    {r[0]:<50} comp={r[4]:.0f} tot={r[2]['total_pnl']:+.0f}% PF={r[2]['pf']:.2f} WR={r[2]['wr']:.1f}% D={r[5]:+.0f}")

    dt = time.time() - t0
    print(f"\n  Total time: {dt:.1f}s")
    print("=" * 140)
    print("  DONE — Run with --phase 1/2/3/4 to run individual phases")
    print("=" * 140)
