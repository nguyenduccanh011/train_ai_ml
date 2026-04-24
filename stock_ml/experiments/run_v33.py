"""
V33 ABLATION — 6 targeted patches trên V32 engine.

Root cause analysis (từ trades_v32.csv + timing analysis):
  1. signal_hard_cap: 169 trades, 0% WR, avg -12.88%, total -2177%
     → 100% đã có max_profit>5%/10% rồi đóng lỗ nặng (bán quá muộn)
  2. v32_hap_preempt: 115 trades, 0% WR, avg -6.93%, total -797%
     → 85 lệnh trong đó giá tăng >8% trong 5 ngày sau bán → bán đúng đáy spike
  3. Entry timing: 73.8% lệnh thua có giá rẻ hơn nếu mua sớm 5 ngày
     → fomo_top + recovery_peak: 96 trades, WR 33%, vs tổng 48.5%
  4. Signal exits bán muộn: 556 lệnh đỉnh đã qua 5d trước >5%
  5. Bán sớm: 322 lệnh giá tăng >5% trong 5d sau bán

Hypotheses:
  A: Multi-tier trailing ratchet — fix signal_hard_cap drain (-2177%)
     Sàn tiến triển: +12%→40%, +20%→55%, +35%→65% của max_profit
  B: Trend-reversal exit — bán ngay khi profit đạt ngưỡng + 2d below ema8 + rsi<50
     Fix bán muộn (hap_preempt bán đúng đáy)
  C: Recovery-peak filter — block entry khi ret10d>12% + dist_sma20>3% + trend≠strong
     Fix mua đỉnh hồi phục (96 lệnh WR 33%)
  D: HAP consecutive drop — HAP chỉ trigger sau 2+ ngày close < ema8 liên tiếp
     Fix HAP bán đúng đáy spike đơn lẻ
  E: RSI oversold block — không thoát signal/hap khi rsi<32 (đáy điều chỉnh)
     Fix 85 lệnh bán đáy rồi giá bật mạnh
  F: Signal confirm exit — cần 2 bar liên tiếp mới signal exit (khi có profit)
     Fix false signal exits

Ablation plan:
  Phase 1: Individual A-F vs V32 baseline
  Phase 2: Best 2-way combos
  Phase 3: Greedy build
  Phase 4: Fine-tune params of winning combo
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
from experiments.run_v32_final import V32_DELTA, backtest_v32


# ── V33 patch deltas ──────────────────────────────────────────────────────────

# A: Multi-tier trailing ratchet — progressive floor tighten
V33_A_std   = dict(v33_trailing_ratchet=True,
                   v33_tr_tier1_trigger=0.12, v33_tr_tier1_keep=0.40,
                   v33_tr_tier2_trigger=0.20, v33_tr_tier2_keep=0.55,
                   v33_tr_tier3_trigger=0.35, v33_tr_tier3_keep=0.65)
V33_A_tight = dict(v33_trailing_ratchet=True,
                   v33_tr_tier1_trigger=0.10, v33_tr_tier1_keep=0.45,
                   v33_tr_tier2_trigger=0.18, v33_tr_tier2_keep=0.60,
                   v33_tr_tier3_trigger=0.30, v33_tr_tier3_keep=0.70)
V33_A_loose = dict(v33_trailing_ratchet=True,
                   v33_tr_tier1_trigger=0.15, v33_tr_tier1_keep=0.35,
                   v33_tr_tier2_trigger=0.25, v33_tr_tier2_keep=0.50,
                   v33_tr_tier3_trigger=0.40, v33_tr_tier3_keep=0.60)
V33_A_2tier = dict(v33_trailing_ratchet=True,
                   v33_tr_tier1_trigger=0.12, v33_tr_tier1_keep=0.45,
                   v33_tr_tier2_trigger=0.25, v33_tr_tier2_keep=0.60,
                   v33_tr_tier3_trigger=99.0, v33_tr_tier3_keep=0.60)  # no tier3

# B: Trend-reversal exit — 2d below ema8 + rsi<50 after profit
V33_B_rsi50 = dict(v33_trend_rev_exit=True, v33_tre_min_profit=0.08,
                   v33_tre_rsi_thresh=50.0, v33_tre_hold_min=5)
V33_B_rsi45 = dict(v33_trend_rev_exit=True, v33_tre_min_profit=0.08,
                   v33_tre_rsi_thresh=45.0, v33_tre_hold_min=5)
V33_B_rsi55 = dict(v33_trend_rev_exit=True, v33_tre_min_profit=0.06,
                   v33_tre_rsi_thresh=55.0, v33_tre_hold_min=3)
V33_B_hi    = dict(v33_trend_rev_exit=True, v33_tre_min_profit=0.12,
                   v33_tre_rsi_thresh=50.0, v33_tre_hold_min=5)

# C: Recovery-peak entry filter
V33_C_12pct = dict(v33_recovery_peak_filter=True, v33_rpf_ret10_thresh=0.12,
                   v33_rpf_dist_sma20_thresh=0.03, v33_rpf_require_weak=True)
V33_C_10pct = dict(v33_recovery_peak_filter=True, v33_rpf_ret10_thresh=0.10,
                   v33_rpf_dist_sma20_thresh=0.03, v33_rpf_require_weak=True)
V33_C_15pct = dict(v33_recovery_peak_filter=True, v33_rpf_ret10_thresh=0.15,
                   v33_rpf_dist_sma20_thresh=0.02, v33_rpf_require_weak=True)
V33_C_all   = dict(v33_recovery_peak_filter=True, v33_rpf_ret10_thresh=0.12,
                   v33_rpf_dist_sma20_thresh=0.03, v33_rpf_require_weak=False)  # block even strong

# D: HAP consecutive drop
V33_D_2d = dict(v33_hap_consec_drop=True, v33_hcd_min_days=2)
V33_D_3d = dict(v33_hap_consec_drop=True, v33_hcd_min_days=3)

# E: RSI oversold block
V33_E_32 = dict(v33_rsi_oversold_block=True, v33_rob_rsi_thresh=32.0)
V33_E_35 = dict(v33_rsi_oversold_block=True, v33_rob_rsi_thresh=35.0)
V33_E_30 = dict(v33_rsi_oversold_block=True, v33_rob_rsi_thresh=30.0)

# F: Signal confirm exit
V33_F_std  = dict(v33_signal_confirm_exit=True, v33_sce_min_pnl=-0.02,
                  v33_sce_min_profit_seen=0.03)
V33_F_hi   = dict(v33_signal_confirm_exit=True, v33_sce_min_pnl=-0.01,
                  v33_sce_min_profit_seen=0.05)
V33_F_loose = dict(v33_signal_confirm_exit=True, v33_sce_min_pnl=-0.03,
                   v33_sce_min_profit_seen=0.02)


def backtest_v33(y_pred, returns, df_test, feature_cols, **kwargs):
    """V33 = V32 engine + V33 patches (passed via kwargs).
    Sử dụng backtest_v32 (đúng chain V31→V30) thay vì backtest_v30 trực tiếp."""
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
    return (f"  {label:<55} | {m['trades']:>5} {m['wr']:>5.1f}% "
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

    HDR = f"  {'Config':<55} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    SEP = "  " + "-" * (len(HDR) - 2)

    print("=" * 150)
    print("V33 ABLATION — 6 patches targeting: hard_cap drain, HAP false exits, recovery-peak entries")
    print("=" * 150)

    t0 = time.time()
    print("\nBaseline V32...")
    m_v32, t_v32, cs_v32, _ = run_v33(SYMBOLS, {}, "V32 baseline")

    print(HDR); print(SEP)
    print(fmt_row("V32 baseline", m_v32, cs_v32))
    print(SEP)

    # ── Phase 1: Individual patches ──────────────────────────────────────────
    if args.phase in (0, 1):
        print("\n== PHASE 1: Individual patches ==")
        print(HDR); print(SEP)

        p1_variants = [
            # A: Multi-tier trailing ratchet
            ("A_trailing_std",    V33_A_std),
            ("A_trailing_tight",  V33_A_tight),
            ("A_trailing_loose",  V33_A_loose),
            ("A_trailing_2tier",  V33_A_2tier),
            # B: Trend-reversal exit
            ("B_trend_rev_rsi50", V33_B_rsi50),
            ("B_trend_rev_rsi45", V33_B_rsi45),
            ("B_trend_rev_rsi55", V33_B_rsi55),
            ("B_trend_rev_hi",    V33_B_hi),
            # C: Recovery-peak filter
            ("C_rpf_12pct",       V33_C_12pct),
            ("C_rpf_10pct",       V33_C_10pct),
            ("C_rpf_15pct",       V33_C_15pct),
            ("C_rpf_all",         V33_C_all),
            # D: HAP consecutive drop
            ("D_hap_consec_2d",   V33_D_2d),
            ("D_hap_consec_3d",   V33_D_3d),
            # E: RSI oversold block
            ("E_rsi_ob_32",       V33_E_32),
            ("E_rsi_ob_35",       V33_E_35),
            ("E_rsi_ob_30",       V33_E_30),
            # F: Signal confirm exit
            ("F_sig_confirm_std", V33_F_std),
            ("F_sig_confirm_hi",  V33_F_hi),
            ("F_sig_confirm_loose", V33_F_loose),
        ]

        p1_results = []
        for label, patches in p1_variants:
            m, t, cs, lbl = run_v33(SYMBOLS, patches, label)
            delta = cs - cs_v32
            print(fmt_row(label, m, cs, delta))
            p1_results.append((label, patches, m, t, cs, delta))

        print(SEP)
        p1_results.sort(key=lambda x: x[4], reverse=True)

        # Pick best per group
        def best_of(prefix):
            cands = [r for r in p1_results if r[0].startswith(prefix)]
            return max(cands, key=lambda x: x[4]) if cands else None

        best_A = best_of("A")
        best_B = best_of("B")
        best_C = best_of("C")
        best_D = best_of("D")
        best_E = best_of("E")
        best_F = best_of("F")

        print(f"\n  Phase 1 winners:")
        for r in [best_A, best_B, best_C, best_D, best_E, best_F]:
            if r:
                print(f"    {r[0]:<55} comp={r[4]:.0f} D={r[5]:+.0f}")

        p1_winners = [r for r in [best_A, best_B, best_C, best_D, best_E, best_F]
                      if r and r[5] > 0]
        print(f"\n  Positive-delta patches: {[r[0] for r in p1_winners]}")

    # ── Phase 2: Best 2-way combos ────────────────────────────────────────────
    if args.phase in (0, 2):
        print("\n== PHASE 2: Top 2-way combos ==")
        if args.phase == 2:
            # Re-derive winners minimal set
            min_variants = [
                ("A_std", V33_A_std), ("A_tight", V33_A_tight),
                ("B_rsi50", V33_B_rsi50), ("B_rsi45", V33_B_rsi45),
                ("C_12pct", V33_C_12pct), ("C_10pct", V33_C_10pct),
                ("D_2d", V33_D_2d), ("D_3d", V33_D_3d),
                ("E_32", V33_E_32), ("E_35", V33_E_35),
                ("F_std", V33_F_std), ("F_hi", V33_F_hi),
            ]
            p1_results = []
            for label, patches in min_variants:
                m, t, cs, _ = run_v33(SYMBOLS, patches, label)
                p1_results.append((label, patches, m, t, cs, cs - cs_v32))
            def best_of(prefix):
                cands = [r for r in p1_results if r[0].startswith(prefix)]
                return max(cands, key=lambda x: x[4]) if cands else None
            best_A = best_of("A"); best_B = best_of("B"); best_C = best_of("C")
            best_D = best_of("D"); best_E = best_of("E"); best_F = best_of("F")
            p1_winners = [r for r in [best_A, best_B, best_C, best_D, best_E, best_F]
                          if r and r[4] - cs_v32 > 0]

        print(HDR); print(SEP)
        p2_results = []
        for (la, pa, *_), (lb, pb, *_) in itertools.combinations(p1_winners, 2):
            combo_patches = {**pa, **pb}
            label = f"{la}+{lb}"[:55]
            m, t, cs, _ = run_v33(SYMBOLS, combo_patches, label)
            delta = cs - cs_v32
            print(fmt_row(label, m, cs, delta))
            p2_results.append((label, combo_patches, m, t, cs, delta))

        print(SEP)
        p2_results.sort(key=lambda x: x[4], reverse=True)
        print(f"\n  Top 5 combos:")
        for r in p2_results[:5]:
            print(f"    {r[0]:<55} comp={r[4]:.0f} D={r[5]:+.0f}")

    # ── Phase 3: Greedy build ─────────────────────────────────────────────────
    if args.phase in (0, 3):
        print("\n== PHASE 3: Greedy build ==")
        all_candidates = {
            "A_std":   V33_A_std, "A_tight":  V33_A_tight, "A_2tier":  V33_A_2tier,
            "B_rsi50": V33_B_rsi50, "B_rsi45": V33_B_rsi45, "B_hi": V33_B_hi,
            "C_12pct": V33_C_12pct, "C_10pct": V33_C_10pct, "C_15pct": V33_C_15pct,
            "D_2d":    V33_D_2d, "D_3d":    V33_D_3d,
            "E_32":    V33_E_32, "E_35":    V33_E_35,
            "F_std":   V33_F_std, "F_hi":    V33_F_hi,
        }

        greedy_patches = {}
        greedy_cs = cs_v32
        greedy_selected = []

        print(HDR); print(SEP)
        for rnd in range(6):
            round_best = None
            for name, patch in all_candidates.items():
                # Skip if already selected OR same group already selected
                group = name.split("_")[0]
                already_group = any(s.split("_")[0] == group for s in greedy_selected)
                if name in greedy_selected or already_group:
                    continue
                trial = {**greedy_patches, **patch}
                m, t, cs, _ = run_v33(SYMBOLS, trial, name)
                if round_best is None or cs > round_best[2]:
                    round_best = (name, patch, cs, m, t)

            if round_best is None:
                print(f"  Round {rnd+1}: no candidates left, stopping."); break

            name, patch, cs, m, t = round_best
            delta = cs - greedy_cs
            if delta <= 0:
                print(f"  Round {rnd+1}: best is {name} but delta={delta:+.0f} — stopping greedy."); break

            greedy_patches = {**greedy_patches, **patch}
            greedy_cs = cs
            greedy_selected.append(name)
            print(fmt_row(f"Greedy+{name}", m, cs, delta))
            print(f"  → Added {name} (D={delta:+.0f}). Stack: {greedy_selected}")

        print(SEP)
        print(f"\n  Greedy final: {greedy_selected}  comp={greedy_cs:.0f} D={greedy_cs-cs_v32:+.0f}")

    # ── Phase 4: Fine-tune best combo ─────────────────────────────────────────
    if args.phase in (0, 4):
        print("\n== PHASE 4: Fine-tune A+B and A+C params ==")
        sweep_variants = []

        # Sweep A tiers
        for t1 in [0.10, 0.12, 0.15]:
            for t1k in [0.35, 0.40, 0.50]:
                for t2 in [0.20, 0.25]:
                    for t2k in [0.50, 0.60]:
                        label = f"A_t1={int(t1*100)}_k1={int(t1k*100)}_t2={int(t2*100)}_k2={int(t2k*100)}"
                        patch = dict(v33_trailing_ratchet=True,
                                     v33_tr_tier1_trigger=t1, v33_tr_tier1_keep=t1k,
                                     v33_tr_tier2_trigger=t2, v33_tr_tier2_keep=t2k,
                                     v33_tr_tier3_trigger=0.40, v33_tr_tier3_keep=0.70)
                        sweep_variants.append((label[:55], patch))

        # Sweep A+B combos
        for rsi in [45.0, 50.0, 55.0]:
            for mp in [0.06, 0.08, 0.10]:
                label = f"A_std+B_rsi{int(rsi)}_mp{int(mp*100)}"
                patch = {**V33_A_std,
                         **dict(v33_trend_rev_exit=True, v33_tre_min_profit=mp,
                                v33_tre_rsi_thresh=rsi, v33_tre_hold_min=5)}
                sweep_variants.append((label[:55], patch))

        # Sweep A+C combos
        for ret10 in [0.10, 0.12, 0.15]:
            for dist in [0.02, 0.03, 0.05]:
                label = f"A_std+C_r{int(ret10*100)}_d{int(dist*100)}"
                patch = {**V33_A_std,
                         **dict(v33_recovery_peak_filter=True, v33_rpf_ret10_thresh=ret10,
                                v33_rpf_dist_sma20_thresh=dist, v33_rpf_require_weak=True)}
                sweep_variants.append((label[:55], patch))

        print(HDR); print(SEP)
        ft_results = []
        for label, patch in sweep_variants:
            m, t, cs, _ = run_v33(SYMBOLS, patch, label)
            delta = cs - cs_v32
            print(fmt_row(label, m, cs, delta))
            ft_results.append((label, patch, m, t, cs, delta))

        print(SEP)
        ft_results.sort(key=lambda x: x[4], reverse=True)
        print(f"\n  Top 10 fine-tune:")
        for r in ft_results[:10]:
            print(f"    {r[0]:<55} comp={r[4]:.0f} tot={r[2]['total_pnl']:+.0f}% "
                  f"PF={r[2]['pf']:.2f} WR={r[2]['wr']:.1f}% D={r[5]:+.0f}")

    dt = time.time() - t0
    print(f"\n  Total time: {dt:.1f}s")
    print("=" * 150)
    print("  DONE — Run with --phase 1/2/3/4 to run individual phases")
    print("=" * 150)
