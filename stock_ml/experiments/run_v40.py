"""V40 — V37a engine + retrain với forward_window=15-20d.

Root cause analysis từ V39 series:
  - V39a/V39b đều cố fix bias thoát sớm ở 21-30d bằng ENGINE heuristic
    (min_hold, HAP reform, rule_confirm_exit)
  - Nhưng root cause thực sự là: target fw=8d quá ngắn → model học rằng
    "sóng kết thúc sau 8-15d" → bias signal exit trước 35d

V40 fix ở TRAINING LAYER: retrain với fw=15-20d dạy model nhận
đầu sóng dài hơn ngay từ đầu, không cần engine heuristic.

Sweep:
  V40a: fw=15, gain=6%, loss=3%   (ngắn nhất, conservative)
  V40b: fw=17, gain=7%, loss=3%   (trung bình)
  V40c: fw=20, gain=8%, loss=4%   (dài nhất, cao gain threshold)

Engine: V37a (per-profile dispatch, V35 relax flags cho bank/defensive/balanced)
Feature: leading_v4 (HA features, giữ nguyên từ V34)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v34_final import V34_TARGET, V34_FEATURE_SET
from experiments.run_v37a import backtest_v37a


V40_FEATURE_SET = V34_FEATURE_SET  # leading_v4

# --- Target variants ---
V40_TARGET_BASE = V34_TARGET  # fw=8, g6%, l3% — baseline (V37a hiện tại)

V40_TARGET_A = dict(
    type="early_wave",
    forward_window=15,
    short_window=8,
    long_window=20,
    gain_threshold=0.06,
    loss_threshold=0.03,
    classes=3,
)

V40_TARGET_B = dict(
    type="early_wave",
    forward_window=17,
    short_window=8,
    long_window=20,
    gain_threshold=0.07,
    loss_threshold=0.03,
    classes=3,
)

V40_TARGET_C = dict(
    type="early_wave",
    forward_window=20,
    short_window=8,
    long_window=20,
    gain_threshold=0.08,
    loss_threshold=0.04,
    classes=3,
)


def backtest_v40(y_pred, returns, df_test, feature_cols, **kwargs):
    """V40 = V37a engine, target thay đổi ở training layer."""
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **kwargs)


def _run(symbols, target, label):
    t = run_test_base(
        symbols, True, True, False, False, True, True, True, True, True, True,
        backtest_fn=backtest_v40,
        feature_set=V40_FEATURE_SET,
        target_override=target,
    )
    m = calc_metrics(t)
    cs = comp_score(m, t)
    return m, t, cs, label


def _fmt_row(label, m, cs, base_cs=None):
    delta = f"  ({cs - base_cs:+.0f})" if base_cs is not None else ""
    return (
        f"  {label:<42} | {m['trades']:>5} {m['wr']:>5.1f}%"
        f" {m['avg_pnl']:>+7.2f}% {m['total_pnl']:>+9.1f}%"
        f" {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>5.1f}d"
        f" | {cs:>6.0f}{delta}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--variants", type=str, default="base,a,b,c",
                        help="Comma-separated variants to run: base,a,b,c")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    requested = {v.strip().lower() for v in args.variants.split(",")}

    print("=" * 140)
    print("V40 — V37a engine + retrain fw=15-20d (giải quyết root cause bias thoát sớm)")
    print("=" * 140)
    print(f"  Engine  : V37a (per-profile V35 relax flags)")
    print(f"  Features: {V40_FEATURE_SET}")
    print(f"  Variants: fw=8 (base/V37a), fw=15 (V40a), fw=17 (V40b), fw=20 (V40c)")
    print()

    t_rule = run_rule_test(SYMBOLS)
    m_rule = calc_metrics(t_rule)

    HDR = (
        f"  {'Config':<42} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10}"
        f" {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    )
    SEP = "  " + "-" * (len(HDR) - 2)

    results = {}

    t0 = time.time()

    if "base" in requested:
        m, t, cs, lbl = _run(SYMBOLS, V40_TARGET_BASE, "V37a-base (fw=8,g6%,l3%)")
        results["base"] = (m, t, cs, lbl)

    if "a" in requested:
        m, t, cs, lbl = _run(SYMBOLS, V40_TARGET_A, "V40a (fw=15,g6%,l3%)")
        results["a"] = (m, t, cs, lbl)

    if "b" in requested:
        m, t, cs, lbl = _run(SYMBOLS, V40_TARGET_B, "V40b (fw=17,g7%,l3%)")
        results["b"] = (m, t, cs, lbl)

    if "c" in requested:
        m, t, cs, lbl = _run(SYMBOLS, V40_TARGET_C, "V40c (fw=20,g8%,l4%)")
        results["c"] = (m, t, cs, lbl)

    dt = time.time() - t0

    print(HDR)
    print(SEP)
    print(
        f"  {'Rule baseline':<42} | {m_rule['trades']:>5} {m_rule['wr']:>5.1f}%"
        f" {m_rule['avg_pnl']:>+7.2f}% {m_rule['total_pnl']:>+9.1f}%"
        f" {m_rule['pf']:>5.2f} {m_rule['max_loss']:>+7.1f}% {m_rule['avg_hold']:>5.1f}d |"
    )

    base_cs = results["base"][2] if "base" in results else None
    for key in ["base", "a", "b", "c"]:
        if key in results:
            m, t, cs, lbl = results[key]
            ref = None if key == "base" else base_cs
            print(_fmt_row(lbl, m, cs, ref))

    print(SEP)
    print(f"  Time: {dt:.1f}s")

    # Save trades and exit breakdown for each variant
    for key, (m, t, cs, lbl) in results.items():
        df_t = pd.DataFrame(t)
        if len(df_t) == 0:
            continue
        fname = f"trades_v40{'_' + key if key != 'base' else '_base'}.csv"
        out_path = os.path.join(OUT, fname)
        df_t.to_csv(out_path, index=False)
        print(f"\n  Saved {len(df_t)} trades ({lbl}) → results/{fname}")

        if "exit_reason" in df_t.columns:
            print(f"\n  Exit reason breakdown ({lbl}):")
            for reason, grp in df_t.groupby("exit_reason"):
                pnl = grp["pnl_pct"]
                wr = (pnl > 0).mean() * 100
                print(
                    f"    {reason:<28} n={len(grp):<5} WR={wr:>5.1f}%"
                    f"  avg={pnl.mean():>+7.2f}%  tot={pnl.sum():>+9.1f}%"
                )
