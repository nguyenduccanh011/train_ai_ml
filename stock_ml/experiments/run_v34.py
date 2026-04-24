"""
V34 ABLATION — Heikin-Ashi features (leading_v4) retrain experiment.

Root cause analysis (từ trades_v32.csv):
  1. Losers avg entry_ret_5d=+3.55%, entry_dist_sma20=+3.14%
     → Model mua khi giá đã tăng: fomo top + recovery peak (rule-based đã xử lý một phần)
  2. 317 losing signal exits với avg max_profit=+150% → massive profit giveback
     → Signal exit bán quá muộn (hoặc quá sớm sau đó miss big move)
  3. signal_hard_cap 169 trades avg -12.88% → gap risk + crash entries
  4. weak trend entries avg -8.12% → sai hướng market

V34 strategy: RETRAIN MODEL với leading_v4 = leading_v3 + Heikin-Ashi features
  - HA features giúp model học "wave position" (đầu sóng vs cuối sóng/fomo)
  - HA shadow ratios → tín hiệu phân phối sớm hơn MACD/RSI
  - HA streak + early/late wave signals → label quality improvement

Key HA features added (src/features/engine.py):
  ha_green, ha_green_streak, ha_red_streak, ha_color_switch
  ha_upper_shadow_ratio, ha_lower_shadow_ratio, ha_no_lower_shadow, ha_no_upper_shadow
  ha_upper_shadow_growing, ha_lower_shadow_growing
  ha_body_ratio, ha_body_shrinking
  ha_streak_position (0=đầu sóng, 1=đỉnh)
  ha_doji
  ha_bearish_reversal_signal (green_streak>=4 + upper shadow growing + body shrinking)
  ha_bullish_reversal_signal (red_streak>=3 + lower shadow growing + color_switch)
  ha_early_wave (streak<=2 + no_lower_shadow + body>0.5)
  ha_late_wave (streak>=5 + upper_shadow>0.3 + body_shrinking)

Ablation plan:
  Phase 1: leading_v4 retrain vs leading_v3 baseline (V29_TARGET unchanged)
  Phase 2: leading_v4 + modified target (early_wave với shorter window)
  Phase 3: leading_v4 + engine patches từ V32 (best combo)
  Phase 4: fine-tune winning config

V34 = V33 engine + leading_v4 retrain
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v29 import V29_TARGET, V29_FEATURE_SET
from experiments.run_v32_final import V32_DELTA, backtest_v32
from experiments.run_v33_final import V33_DELTA, backtest_v33


# ── Feature sets ──────────────────────────────────────────────────────────────
V34_FEATURE_SET = "leading_v4"   # leading_v3 + HA features

# ── Target variants to test ───────────────────────────────────────────────────
# V29 baseline target
V29_TARGET_CFG = V29_TARGET  # early_wave, fwd=8, short=8, long=20

# V34-T1: shorter forward window (tránh label bị nhiễu bởi late-wave)
V34_TARGET_SHORTER = dict(
    type="early_wave",
    forward_window=6,
    short_window=6,
    long_window=15,
    gain_threshold=0.05,
    loss_threshold=0.04,
    classes=3,
)

# V34-T2: tighter gain threshold (nhắm vào sóng ngắn, chất lượng cao hơn)
V34_TARGET_TIGHT = dict(
    type="early_wave",
    forward_window=8,
    short_window=8,
    long_window=20,
    gain_threshold=0.06,
    loss_threshold=0.04,
    classes=3,
)

# ── Backtest engine for V34 ───────────────────────────────────────────────────
def backtest_v34(y_pred, returns, df_test, feature_cols, **kwargs):
    """V34 = V33 engine (unchanged). Improvement comes from model layer (retrain)."""
    return backtest_v33(y_pred, returns, df_test, feature_cols, **kwargs)


def run_v34(symbols, feature_set, target_cfg, backtest_fn=None, label="", extra_patches=None):
    if backtest_fn is None:
        backtest_fn = backtest_v34
    if extra_patches:
        fn = lambda *a, **kw: backtest_fn(*a, **{**extra_patches, **kw})
    else:
        fn = backtest_fn
    t = run_test_base(
        symbols, True, True, False, False, True, True, True, True, True, True,
        backtest_fn=fn,
        feature_set=feature_set,
        target_override=target_cfg,
    )
    m = calc_metrics(t)
    cs = comp_score(m, t)
    return m, t, cs, label


def fmt_row(label, m, cs, delta=None):
    d = f"  D={delta:+.0f}" if delta is not None else ""
    return (f"  {label:<62} | {m['trades']:>5} {m['wr']:>5.1f}% "
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

    HDR = (f"  {'Config':<62} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} "
           f"{'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}")
    SEP = "  " + "-" * (len(HDR) - 2)

    print("=" * 170)
    print("V34 ABLATION — Heikin-Ashi features (leading_v4) retrain experiment")
    print("=" * 170)
    print(f"  V34 feature set: {V34_FEATURE_SET}")
    print(f"  Engine: V33 (= V32 + recovery_peak_filter)")
    print()

    t0 = time.time()

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("Computing baselines (V33 engine + leading_v3 = current production)...")
    m_v33, t_v33, cs_v33, _ = run_v34(
        SYMBOLS, V29_FEATURE_SET, V29_TARGET_CFG,
        backtest_fn=backtest_v33,
        extra_patches=V33_DELTA,
        label="V33 baseline (leading_v3 + V33 engine)",
    )

    print(HDR); print(SEP)
    print(fmt_row("V33 baseline (leading_v3+V33engine)", m_v33, cs_v33))
    print(SEP)

    # ── Phase 1: leading_v4 retrain, same target ───────────────────────────
    if args.phase in (0, 1):
        print("\n== PHASE 1: leading_v4 retrain (HA features) — same V29 target ==")
        print("Hypothesis: HA streak/shadow features help model avoid fomo entries")
        print("  and detect distribution patterns earlier (fix +150% avg max_profit giveback)")
        print(HDR); print(SEP)

        # 1A: V34 engine (= V33) + leading_v4 + V29 target
        m_1a, t_1a, cs_1a, _ = run_v34(
            SYMBOLS, V34_FEATURE_SET, V29_TARGET_CFG,
            backtest_fn=backtest_v33,
            extra_patches=V33_DELTA,
            label="V34A: leading_v4 + V33engine + V29target",
        )
        print(fmt_row("V34A: leading_v4+V33engine+V29target", m_1a, cs_1a, cs_1a - cs_v33))

        # 1B: V34 engine (= V33) + leading_v4 + V34 target shorter
        m_1b, t_1b, cs_1b, _ = run_v34(
            SYMBOLS, V34_FEATURE_SET, V34_TARGET_SHORTER,
            backtest_fn=backtest_v33,
            extra_patches=V33_DELTA,
            label="V34B: leading_v4+V33engine+target_shorter",
        )
        print(fmt_row("V34B: leading_v4+V33engine+target_shorter", m_1b, cs_1b, cs_1b - cs_v33))

        # 1C: V34 engine + leading_v4 + tighter gain target
        m_1c, t_1c, cs_1c, _ = run_v34(
            SYMBOLS, V34_FEATURE_SET, V34_TARGET_TIGHT,
            backtest_fn=backtest_v33,
            extra_patches=V33_DELTA,
            label="V34C: leading_v4+V33engine+target_tight",
        )
        print(fmt_row("V34C: leading_v4+V33engine+target_tight", m_1c, cs_1c, cs_1c - cs_v33))

        # 1D: V32 engine (no V33 RPF) + leading_v4 + V29 target
        m_1d, t_1d, cs_1d, _ = run_v34(
            SYMBOLS, V34_FEATURE_SET, V29_TARGET_CFG,
            backtest_fn=backtest_v32,
            label="V34D: leading_v4+V32engine+V29target",
        )
        print(fmt_row("V34D: leading_v4+V32engine+V29target", m_1d, cs_1d, cs_1d - cs_v33))

        print(SEP)
        p1 = [
            ("V34A", cs_1a, cs_1a - cs_v33, m_1a, t_1a),
            ("V34B", cs_1b, cs_1b - cs_v33, m_1b, t_1b),
            ("V34C", cs_1c, cs_1c - cs_v33, m_1c, t_1c),
            ("V34D", cs_1d, cs_1d - cs_v33, m_1d, t_1d),
        ]
        p1.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Phase 1 ranking:")
        for name, cs, d, m, _ in p1:
            print(f"    {name:<8} comp={cs:.0f} D={d:+.0f} WR={m['wr']:.1f}% "
                  f"avg={m['avg_pnl']:+.2f}% tot={m['total_pnl']:+.0f}% PF={m['pf']:.2f}")

        best_p1 = p1[0]
        print(f"\n  Phase 1 winner: {best_p1[0]} (comp={best_p1[1]:.0f}, D={best_p1[2]:+.0f})")

    # ── Phase 2: Sweep HA-specific target configurations ──────────────────
    if args.phase in (0, 2):
        print("\n== PHASE 2: Sweep target configs with leading_v4 ==")
        print(HDR); print(SEP)

        target_sweep = []
        for fwd in [5, 6, 7, 8, 10]:
            for gain in [0.04, 0.05, 0.06]:
                for loss in [0.03, 0.04]:
                    target_sweep.append((
                        f"fwd{fwd}_g{int(gain*100)}_l{int(loss*100)}",
                        dict(type="early_wave", forward_window=fwd, short_window=fwd,
                             long_window=fwd*2+4, gain_threshold=gain,
                             loss_threshold=loss, classes=3)
                    ))

        p2_results = []
        for label, tgt in target_sweep:
            m, t, cs, _ = run_v34(
                SYMBOLS, V34_FEATURE_SET, tgt,
                backtest_fn=backtest_v33,
                extra_patches=V33_DELTA,
                label=label,
            )
            delta = cs - cs_v33
            print(fmt_row(label, m, cs, delta))
            p2_results.append((label, tgt, m, t, cs, delta))

        print(SEP)
        p2_results.sort(key=lambda x: x[4], reverse=True)
        print(f"\n  Top 5 target configs:")
        for r in p2_results[:5]:
            print(f"    {r[0]:<40} comp={r[4]:.0f} D={r[5]:+.0f} "
                  f"trades={r[2]['trades']} WR={r[2]['wr']:.1f}% "
                  f"avg={r[2]['avg_pnl']:+.2f}% PF={r[2]['pf']:.2f}")

        best_p2_target = p2_results[0][1] if p2_results else V29_TARGET_CFG
        best_p2_cs = p2_results[0][4] if p2_results else cs_v33

    # ── Phase 3: leading_v4 + engine patches sweep ────────────────────────
    if args.phase in (0, 3):
        print("\n== PHASE 3: leading_v4 + engine patch combos ==")
        print("Testing V33 engine patches on top of leading_v4 retrain")
        print(HDR); print(SEP)

        # Best target from Phase 2 (fallback to V29 if Phase 2 skipped)
        best_target = best_p2_target if args.phase == 0 else V29_TARGET_CFG

        # Baseline: leading_v4 alone (best of Phase 1/2)
        m_base_v4, t_base_v4, cs_base_v4, _ = run_v34(
            SYMBOLS, V34_FEATURE_SET, best_target,
            backtest_fn=backtest_v33,
            extra_patches=V33_DELTA,
            label="leading_v4+best_target (base)",
        )
        print(fmt_row("leading_v4+best_target (base)", m_base_v4, cs_base_v4, cs_base_v4 - cs_v33))

        # Test with V32 engine (no V33 RPF patch) vs V33 engine
        # Sometimes the RPF blocks good entries that HA already handles via model
        m_v32e, t_v32e, cs_v32e, _ = run_v34(
            SYMBOLS, V34_FEATURE_SET, best_target,
            backtest_fn=backtest_v32,
            label="leading_v4+best_target+V32engine(no_rpf)",
        )
        print(fmt_row("leading_v4+V32engine+best_target", m_v32e, cs_v32e, cs_v32e - cs_v33))

        # Test additional V33 engine patch variants
        extra_patches_list = [
            ("RPF_loose",  dict(v33_recovery_peak_filter=True, v33_rpf_ret10_thresh=0.08,
                                v33_rpf_dist_sma20_thresh=0.005, v33_rpf_require_weak=True)),
            ("RPF_tight",  dict(v33_recovery_peak_filter=True, v33_rpf_ret10_thresh=0.12,
                                v33_rpf_dist_sma20_thresh=0.02, v33_rpf_require_weak=True)),
            ("RPF_off",    {}),  # no engine patches (pure model improvement)
        ]
        for name, patches in extra_patches_list:
            m, t, cs, _ = run_v34(
                SYMBOLS, V34_FEATURE_SET, best_target,
                backtest_fn=backtest_v33 if patches else backtest_v32,
                extra_patches=patches if patches else None,
                label=f"leading_v4+{name}",
            )
            print(fmt_row(f"leading_v4+{name}", m, cs, cs - cs_v33))

        print(SEP)

    # ── Phase 4: HA feature ablation (which HA features matter most) ──────
    if args.phase in (0, 4):
        print("\n== PHASE 4: HA feature importance analysis ==")
        print("Which HA features contribute most? Test by disabling feature groups.")
        print("(Uses same retrain — different features selected at model level)")
        print()
        print("  Note: leading_v4 always computes all HA features.")
        print("  Model selection (XGBoost/RF) will naturally weight important ones.")
        print("  To explicitly ablate: would need separate feature_set variants.")
        print()
        print("  Key HA features to watch in model.feature_importances_:")
        ha_features = [
            "ha_green_streak", "ha_red_streak", "ha_color_switch",
            "ha_upper_shadow_ratio", "ha_lower_shadow_ratio",
            "ha_no_lower_shadow", "ha_no_upper_shadow",
            "ha_upper_shadow_growing", "ha_lower_shadow_growing",
            "ha_body_ratio", "ha_body_shrinking",
            "ha_streak_position",
            "ha_doji",
            "ha_bearish_reversal_signal", "ha_bullish_reversal_signal",
            "ha_early_wave", "ha_late_wave",
        ]
        for f in ha_features:
            print(f"    {f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    dt = time.time() - t0
    print("\n" + "=" * 170)
    print("  V34 ABLATION SUMMARY")
    print("=" * 170)
    print(f"  V33 baseline: comp={cs_v33:.0f} WR={m_v33['wr']:.1f}% avg={m_v33['avg_pnl']:+.2f}% "
          f"tot={m_v33['total_pnl']:+.0f}% PF={m_v33['pf']:.2f}")
    print()

    # Collect all results for final ranking
    all_results = []
    try:
        all_results += [(f"P1_{r[0]}", r[1], r[2], r[3], r[4]) for r in p1]
    except Exception:
        pass
    try:
        all_results += [(f"P2_{r[0]}", r[4], r[5], r[2], r[3]) for r in p2_results[:5]]
    except Exception:
        pass

    if all_results:
        all_results.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top overall configs:")
        for name, cs, d, m, t in all_results[:5]:
            print(f"    {name:<50} comp={cs:.0f} D={d:+.0f} trades={m['trades']} "
                  f"WR={m['wr']:.1f}% avg={m['avg_pnl']:+.2f}% PF={m['pf']:.2f}")

        # Save best candidate trades
        best = all_results[0]
        df_best = pd.DataFrame(best[4])
        if len(df_best) > 0:
            out_path = os.path.join(OUT, "trades_v34_candidate.csv")
            df_best.to_csv(out_path, index=False)
            print(f"\n  Saved best candidate trades → results/trades_v34_candidate.csv")

    print(f"\n  Total time: {dt:.1f}s")
    print("=" * 170)
    print("  DONE — V34 Ablation")
    print("=" * 170)
