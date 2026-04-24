"""
V33 Phase 2-4 — Deep sweep on F_sig_confirm_hi winner + combos.

Phase 1 result:
  V32 baseline: comp=441 trades=1464 WR=46.5% avg=+5.40% tot=+7908% PF=2.80
  F_sig_confirm_hi: comp=450 (+10) trades=1032 WR=36.0% avg=+9.28% tot=+9573% PF=3.28
  All others: D <= -2 (A_loose/C near-neutral; B,D,E negative)

Note: F_sig_confirm_std/loose have max_loss=-83.2% (zombie issue).
  F_hi (min_pnl=-0.01, min_profit=0.05) dodges this — narrower scope.

Strategy for Phase 2+:
  1. Deep sweep F params to find optimal min_pnl / min_profit_seen threshold
  2. F + near-neutral patches (A_loose, C_10pct/15pct) — can they synergize?
  3. F + exit quality patches (A trailing, B trend_rev)
  4. Greedy: start from F_hi, try adding each other patch
"""
import os, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v29 import V29_TARGET, V29_FEATURE_SET
from experiments.run_v30 import backtest_v30
from experiments.run_v32_final import V32_DELTA


# Known winner from Phase 1
V33_F_HI = dict(v33_signal_confirm_exit=True, v33_sce_min_pnl=-0.01,
                v33_sce_min_profit_seen=0.05)

# Other patches
V33_A_LOOSE = dict(v33_trailing_ratchet=True,
                   v33_tr_tier1_trigger=0.15, v33_tr_tier1_keep=0.35,
                   v33_tr_tier2_trigger=0.25, v33_tr_tier2_keep=0.50,
                   v33_tr_tier3_trigger=0.40, v33_tr_tier3_keep=0.60)
V33_C_10 = dict(v33_recovery_peak_filter=True, v33_rpf_ret10_thresh=0.10,
                v33_rpf_dist_sma20_thresh=0.03, v33_rpf_require_weak=True)
V33_C_15 = dict(v33_recovery_peak_filter=True, v33_rpf_ret10_thresh=0.15,
                v33_rpf_dist_sma20_thresh=0.02, v33_rpf_require_weak=True)
V33_B_RHI = dict(v33_trend_rev_exit=True, v33_tre_min_profit=0.12,
                 v33_tre_rsi_thresh=50.0, v33_tre_hold_min=5)
V33_D_3D = dict(v33_hap_consec_drop=True, v33_hcd_min_days=3)


def backtest_v33(y_pred, returns, df_test, feature_cols, **kwargs):
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)


def run_v33(symbols, patches: dict, label: str):
    t = run_test_base(symbols, True, True, False, False, True, True, True, True, True, True,
                      backtest_fn=lambda *a, **kw: backtest_v33(*a, **kw, **patches),
                      feature_set=V29_FEATURE_SET, target_override=V29_TARGET)
    m = calc_metrics(t)
    cs = comp_score(m, t)
    return m, t, cs, label


def fmt_row(label, m, cs, delta=None):
    d = f"  D={delta:+.0f}" if delta is not None else ""
    return (f"  {label:<60} | {m['trades']:>5} {m['wr']:>5.1f}% "
            f"{m['avg_pnl']:>+7.2f}% {m['total_pnl']:>+9.1f}% "
            f"{m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>5.1f}d | "
            f"{cs:>6.0f}{d}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    HDR = f"  {'Config':<60} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    SEP = "  " + "-" * (len(HDR) - 2)

    print("=" * 160)
    print("V33 Phase 2-4 — Deep sweep on F_sig_confirm_hi winner")
    print("=" * 160)

    t0 = time.time()
    print("\nBaselines...")
    m_v32, t_v32, cs_v32, _ = run_v33(SYMBOLS, {}, "V32 baseline")
    m_fhi, t_fhi, cs_fhi, _ = run_v33(SYMBOLS, V33_F_HI, "F_sig_confirm_hi (Phase1 winner)")

    print(HDR); print(SEP)
    print(fmt_row("V32 baseline", m_v32, cs_v32))
    print(fmt_row("F_sig_confirm_hi", m_fhi, cs_fhi, cs_fhi - cs_v32))
    print(SEP)

    # ── Phase 2A: Fine-tune F params alone ───────────────────────────────────
    print("\n== PHASE 2A: Fine-tune F params ==")
    print(HDR); print(SEP)

    f_sweep = []
    for min_pnl in [-0.005, -0.01, -0.015, -0.02, -0.03]:
        for min_profit in [0.03, 0.05, 0.07, 0.10]:
            label = f"F_pnl{int(min_pnl*1000)}_mp{int(min_profit*100)}"
            patch = dict(v33_signal_confirm_exit=True,
                         v33_sce_min_pnl=min_pnl,
                         v33_sce_min_profit_seen=min_profit)
            f_sweep.append((label, patch))

    f_results = []
    for label, patch in f_sweep:
        m, t, cs, _ = run_v33(SYMBOLS, patch, label)
        delta_v32 = cs - cs_v32
        delta_fhi = cs - cs_fhi
        print(fmt_row(label[:60], m, cs, delta_v32))
        f_results.append((label, patch, m, t, cs, delta_v32))

    print(SEP)
    f_results.sort(key=lambda x: x[4], reverse=True)
    print(f"\n  Top 5 F variants:")
    for r in f_results[:5]:
        print(f"    {r[0]:<60} comp={r[4]:.0f} D={r[5]:+.0f} trades={r[2]['trades']} maxloss={r[2]['max_loss']:+.1f}%")

    # Best F without extreme max_loss (filter -83%)
    safe_f = [r for r in f_results if r[2]['max_loss'] > -50]
    best_f = safe_f[0] if safe_f else f_results[0]
    print(f"\n  Best safe F: {best_f[0]} comp={best_f[4]:.0f}")

    # ── Phase 2B: F_hi + each other patch ────────────────────────────────────
    print("\n== PHASE 2B: F_hi + each other single patch ==")
    print(HDR); print(SEP)

    combos_2b = [
        ("F_hi+A_loose",  {**V33_F_HI, **V33_A_LOOSE}),
        ("F_hi+A_std",    {**V33_F_HI,
                           **dict(v33_trailing_ratchet=True,
                                  v33_tr_tier1_trigger=0.12, v33_tr_tier1_keep=0.40,
                                  v33_tr_tier2_trigger=0.20, v33_tr_tier2_keep=0.55,
                                  v33_tr_tier3_trigger=0.35, v33_tr_tier3_keep=0.65)}),
        ("F_hi+B_rsi50",  {**V33_F_HI,
                           **dict(v33_trend_rev_exit=True, v33_tre_min_profit=0.08,
                                  v33_tre_rsi_thresh=50.0, v33_tre_hold_min=5)}),
        ("F_hi+B_hi",     {**V33_F_HI, **V33_B_RHI}),
        ("F_hi+C_10pct",  {**V33_F_HI, **V33_C_10}),
        ("F_hi+C_15pct",  {**V33_F_HI, **V33_C_15}),
        ("F_hi+D_3d",     {**V33_F_HI, **V33_D_3D}),
        ("F_hi+D_2d",     {**V33_F_HI,
                           **dict(v33_hap_consec_drop=True, v33_hcd_min_days=2)}),
        ("F_hi+E_30",     {**V33_F_HI,
                           **dict(v33_rsi_oversold_block=True, v33_rob_rsi_thresh=30.0)}),
    ]

    p2b_results = []
    for label, patches in combos_2b:
        m, t, cs, _ = run_v33(SYMBOLS, patches, label)
        delta_fhi = cs - cs_fhi
        delta_v32 = cs - cs_v32
        print(fmt_row(label, m, cs, delta_v32))
        p2b_results.append((label, patches, m, t, cs, delta_v32, delta_fhi))

    print(SEP)
    p2b_results.sort(key=lambda x: x[4], reverse=True)
    p2b_safe = [r for r in p2b_results if r[2]['max_loss'] > -50]
    print(f"\n  Top combos (excl. max_loss < -50):")
    for r in (p2b_safe or p2b_results)[:5]:
        print(f"    {r[0]:<60} comp={r[4]:.0f} Dvs32={r[5]:+.0f} DvsF={r[6]:+.0f} "
              f"trades={r[2]['trades']} maxloss={r[2]['max_loss']:+.1f}%")

    # ── Phase 2C: Best F variant + best combo ─────────────────────────────────
    print("\n== PHASE 2C: Best F param + best combo from 2B ==")
    print(HDR); print(SEP)

    best_combo_2b = (p2b_safe or p2b_results)[0]
    # Try best_f params + the extra patch from best combo
    extra_key = best_combo_2b[0].replace("F_hi+", "")
    extra_map = {
        "A_loose": V33_A_LOOSE, "C_10pct": V33_C_10, "C_15pct": V33_C_15,
        "B_hi": V33_B_RHI, "D_3d": V33_D_3D,
    }
    extra_patch = extra_map.get(extra_key, {})

    if best_f[0] != "F_pnl-10_mp5":  # if best_f differs from F_hi
        variants_2c = []
        for r in f_results[:5]:
            if r[2]['max_loss'] < -50:
                continue
            for extra_name, extra in [("A_loose", V33_A_LOOSE), ("C_10pct", V33_C_10),
                                      ("C_15pct", V33_C_15), ("D_3d", V33_D_3D)]:
                label = f"{r[0][:30]}+{extra_name}"
                patch = {**r[1], **extra}
                variants_2c.append((label[:60], patch))

        for label, patch in variants_2c[:12]:
            m, t, cs, _ = run_v33(SYMBOLS, patch, label)
            delta = cs - cs_v32
            print(fmt_row(label, m, cs, delta))

    # ── Phase 3: Greedy starting from F_hi ───────────────────────────────────
    print("\n== PHASE 3: Greedy build starting from F_hi ==")
    print(HDR); print(SEP)

    greedy_patches = dict(**V33_F_HI)
    greedy_cs = cs_fhi
    greedy_selected = ["F_hi"]

    remaining = {
        "A_loose":  V33_A_LOOSE,
        "C_10pct":  V33_C_10,
        "C_15pct":  V33_C_15,
        "B_rsi50":  dict(v33_trend_rev_exit=True, v33_tre_min_profit=0.08,
                         v33_tre_rsi_thresh=50.0, v33_tre_hold_min=5),
        "B_hi":     V33_B_RHI,
        "D_3d":     V33_D_3D,
        "E_30":     dict(v33_rsi_oversold_block=True, v33_rob_rsi_thresh=30.0),
    }

    for rnd in range(5):
        round_best = None
        for name, patch in remaining.items():
            if name in greedy_selected:
                continue
            trial = {**greedy_patches, **patch}
            m, t, cs, _ = run_v33(SYMBOLS, trial, name)
            if round_best is None or cs > round_best[2]:
                round_best = (name, patch, cs, m, t)

        if round_best is None:
            break
        name, patch, cs, m, t = round_best
        delta = cs - greedy_cs
        if delta <= 0:
            print(f"  Round {rnd+1}: best candidate {name} D={delta:+.0f} — stopping.")
            break
        greedy_patches = {**greedy_patches, **patch}
        greedy_cs = cs
        greedy_selected.append(name)
        print(fmt_row(f"Greedy+{name}", m, cs, delta))
        print(f"  → Added {name} (D={delta:+.0f}). Stack: {greedy_selected}")

    print(SEP)
    print(f"\n  Greedy final: {greedy_selected}")
    print(f"  comp={greedy_cs:.0f} D_vs_v32={greedy_cs-cs_v32:+.0f} D_vs_Fhi={greedy_cs-cs_fhi:+.0f}")

    # ── Phase 4: Fine-tune F params on greedy stack ───────────────────────────
    print("\n== PHASE 4: Fine-tune F on greedy stack ==")
    greedy_no_f = {k: v for k, v in greedy_patches.items()
                   if not k.startswith("v33_sce")}

    print(HDR); print(SEP)
    ft_results = []
    for min_pnl in [-0.005, -0.01, -0.015, -0.02]:
        for min_profit in [0.03, 0.05, 0.07, 0.10]:
            label = f"Stack+F_pnl{int(min_pnl*1000)}_mp{int(min_profit*100)}"
            patch = {**greedy_no_f,
                     **dict(v33_signal_confirm_exit=True,
                            v33_sce_min_pnl=min_pnl,
                            v33_sce_min_profit_seen=min_profit)}
            m, t, cs, _ = run_v33(SYMBOLS, patch, label)
            delta = cs - cs_v32
            print(fmt_row(label[:60], m, cs, delta))
            ft_results.append((label, patch, m, t, cs, delta))

    print(SEP)
    ft_results.sort(key=lambda x: x[4], reverse=True)
    ft_safe = [r for r in ft_results if r[2]['max_loss'] > -50]
    print(f"\n  Top 5 fine-tune (safe max_loss):")
    for r in (ft_safe or ft_results)[:5]:
        print(f"    {r[0]:<60} comp={r[4]:.0f} D={r[5]:+.0f} "
              f"trades={r[2]['trades']} tot={r[2]['total_pnl']:+.0f}% "
              f"PF={r[2]['pf']:.2f} maxloss={r[2]['max_loss']:+.1f}%")

    # ── Save best candidate trades ─────────────────────────────────────────────
    all_candidates = f_results + p2b_results + ft_results
    all_safe = [r for r in all_candidates if r[2]['max_loss'] > -50]
    overall_best = max(all_safe, key=lambda x: x[4]) if all_safe else max(all_candidates, key=lambda x: x[4])

    print(f"\n== OVERALL BEST CANDIDATE ==")
    print(f"  {overall_best[0]}")
    print(f"  comp={overall_best[4]:.0f} D_vs_v32={overall_best[5]:+.0f}")
    print(f"  trades={overall_best[2]['trades']} WR={overall_best[2]['wr']:.1f}% "
          f"avg={overall_best[2]['avg_pnl']:+.2f}% tot={overall_best[2]['total_pnl']:+.0f}% "
          f"PF={overall_best[2]['pf']:.2f} maxloss={overall_best[2]['max_loss']:+.1f}%")
    print(f"  Patches: {overall_best[1]}")

    # Save trades
    df_best = pd.DataFrame(overall_best[3])
    if len(df_best) > 0:
        out_path = os.path.join(OUT, "trades_v33_candidate.csv")
        df_best.to_csv(out_path, index=False)
        print(f"\n  Saved {len(df_best)} trades → results/trades_v33_candidate.csv")

    dt = time.time() - t0
    print(f"\n  Total time: {dt:.1f}s")
    print("=" * 160)
    print("  DONE")
    print("=" * 160)
