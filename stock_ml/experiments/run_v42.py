"""V42 — V37a engine + separate exit model B.

Architecture:
  - Model A (entry): early_wave target {-1, 0, 1}, forward_window=15, gain=6%, loss=3%
  - Model B (exit): binary target_sell, predict EXIT point
    label=1 nếu trong forward_window ngày tới giá giảm >= loss_threshold

Engine: V37a (per-profile dispatch, V35 relax flags cho bank/defensive/balanced)
Feature: leading_v4 (HA features phù hợp cho cả entry và exit)

Model B override engine: khi predict exit=1 và hold >= 3 ngày → thoát ngay
Chỉ hard_stop (8%) ưu tiên hơn model B.

So sánh:
  V42_base: V37a + early_wave fw=15 (không có model B)
  V42_a:    V37a + early_wave_dual fw=15 + model B exit
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v34_final import V34_FEATURE_SET
from experiments.run_v37a import backtest_v37a


V42_FEATURE_SET = V34_FEATURE_SET  # leading_v4

V42_TARGET_BASE = dict(
    type="early_wave",
    forward_window=15,
    short_window=8,
    long_window=20,
    gain_threshold=0.06,
    loss_threshold=0.03,
    classes=3,
)

V42_TARGET_DUAL = dict(
    type="early_wave_dual",
    forward_window=15,
    short_window=8,
    long_window=20,
    gain_threshold=0.06,
    loss_threshold=0.03,
    classes=3,
)


def backtest_v42(y_pred, returns, df_test, feature_cols, **kwargs):
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **kwargs)


def _run(symbols, target, label, use_exit_model=False):
    t = run_test_base(
        symbols, True, True, False, False, True, True, True, True, True, True,
        backtest_fn=backtest_v42,
        feature_set=V42_FEATURE_SET,
        target_override=target,
        train_exit_model=use_exit_model,
    )
    m = calc_metrics(t)
    cs = comp_score(m, t)
    return m, t, cs, label


def _fmt_row(label, m, cs, base_cs=None):
    delta = f"  ({cs - base_cs:+.0f})" if base_cs is not None else ""
    return (
        f"  {label:<46} | {m['trades']:>5} {m['wr']:>5.1f}%"
        f" {m['avg_pnl']:>+7.2f}% {m['total_pnl']:>+9.1f}%"
        f" {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>5.1f}d"
        f" | {cs:>6.0f}{delta}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--variants", type=str, default="base,a",
                        help="Comma-separated variants: base,a")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    requested = {v.strip().lower() for v in args.variants.split(",")}

    print("=" * 150)
    print("V42 — V37a engine + separate exit model B")
    print("=" * 150)
    print(f"  Engine  : V37a (per-profile V35 relax flags)")
    print(f"  Features: {V42_FEATURE_SET}")
    print(f"  Model A : early_wave fw=15, gain=6%, loss=3%")
    print(f"  Model B : binary exit — drawdown >= 3% trong 15 ngày tới (min_hold=3d)")
    print()

    t_rule = run_rule_test(SYMBOLS)
    m_rule = calc_metrics(t_rule)

    HDR = (
        f"  {'Config':<46} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10}"
        f" {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    )
    SEP = "  " + "-" * (len(HDR) - 2)

    results = {}
    t0 = time.time()

    if "base" in requested:
        m, t, cs, lbl = _run(SYMBOLS, V42_TARGET_BASE, "V42_base (V37a, fw=15, no exit model)")
        results["base"] = (m, t, cs, lbl)

    if "a" in requested:
        m, t, cs, lbl = _run(SYMBOLS, V42_TARGET_DUAL, "V42_a (V37a + exit model B, fw=15)", use_exit_model=True)
        results["a"] = (m, t, cs, lbl)

    dt = time.time() - t0

    print(HDR)
    print(SEP)
    print(
        f"  {'Rule baseline':<46} | {m_rule['trades']:>5} {m_rule['wr']:>5.1f}%"
        f" {m_rule['avg_pnl']:>+7.2f}% {m_rule['total_pnl']:>+9.1f}%"
        f" {m_rule['pf']:>5.2f} {m_rule['max_loss']:>+7.1f}% {m_rule['avg_hold']:>5.1f}d |"
    )

    base_cs = results["base"][2] if "base" in results else None
    for key in ["base", "a"]:
        if key in results:
            m, t, cs, lbl = results[key]
            ref = None if key == "base" else base_cs
            print(_fmt_row(lbl, m, cs, ref))

    print(SEP)
    print(f"  Time: {dt:.1f}s")

    for key, (m, t, cs, lbl) in results.items():
        df_t = pd.DataFrame(t)
        if len(df_t) == 0:
            continue
        fname = f"trades_v42_{key}.csv"
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
