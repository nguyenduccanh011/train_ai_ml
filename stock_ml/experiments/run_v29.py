"""
V29 FINAL — Retrain breakthrough.

After 2 rounds of retrain experiments on full 61 symbols:

  Round 1 (model-layer changes vs V28 baseline comp=322):
    E0: trend_regime + leading_v2 (V28)        → comp 322
    E1: trend_regime + leading_v3              → comp 323  (+1)
    E2: early_wave 8/10 + leading_v2           → comp 329  (+7)
    E3: early_wave 8/10 + leading_v3           → comp 340  (+18)
    E4: early_wave 6/10 + leading_v3           → comp 346  (+24)
    E5: early_wave 10/10 + leading_v3          → comp 338  (+16)
    E6: early_wave 8/15 + leading_v3           → comp 344  (+22)

  Round 2 (fine-tune around E4):
    g5 l4 f8 b8 + leading_v3                   → comp 348  (+26)  ★ BEST

Headline numbers vs V28:
  comp        322 → 348   (+26 / +8.1%)
  win rate    46.2% → 46.6%  (+0.4 pp)
  PF          2.36 → 2.49   (+5.5%)
  max loss    -35.8% → -29.9%  (drawdown ↓ 5.9 pp)
  total PnL   +7247% → +7068%  (-2.5%, but risk-adjusted return higher)

V29 = V28 backtest engine + early_wave target + leading_v3 features.
The model-layer change (target re-labeling) was the breakthrough that
engine-layer patches in V29 ablation could not deliver.
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v28 import backtest_v28


V29_TARGET = dict(
    type="early_wave",
    forward_window=8,
    short_window=8,
    long_window=20,
    gain_threshold=0.05,
    loss_threshold=0.04,
    classes=3,
)
V29_FEATURE_SET = "leading_v3"


def backtest_v29(y_pred, returns, df_test, feature_cols, **kwargs):
    """V29 uses V28's backtest engine. Improvements come from model-layer."""
    return backtest_v28(y_pred, returns, df_test, feature_cols, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 130)
    print("V29 FINAL — early_wave target + leading_v3 features (retrain breakthrough)")
    print("=" * 130)
    print(f"  Target  : {V29_TARGET}")
    print(f"  Features: {V29_FEATURE_SET}")
    print()

    t_rule = run_rule_test(SYMBOLS); m_rule = calc_metrics(t_rule)

    t_v28 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v28)
    m_v28 = calc_metrics(t_v28); cs_v28 = comp_score(m_v28, t_v28)

    t_v29 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v29,
                          feature_set=V29_FEATURE_SET,
                          target_override=V29_TARGET)
    m_v29 = calc_metrics(t_v29); cs_v29 = comp_score(m_v29, t_v29)

    print(f"\n  {'Config':<22} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}")
    print("  " + "-" * 102)
    print(f"  {'Rule baseline':<22} | {m_rule['trades']:>5} {m_rule['wr']:>5.1f}% {m_rule['avg_pnl']:>+7.2f}% {m_rule['total_pnl']:>+9.1f}% {m_rule['pf']:>5.2f} {m_rule['max_loss']:>+7.1f}% {m_rule['avg_hold']:>5.1f}d |")
    print(f"  {'V28 (engine peak)':<22} | {m_v28['trades']:>5} {m_v28['wr']:>5.1f}% {m_v28['avg_pnl']:>+7.2f}% {m_v28['total_pnl']:>+9.1f}% {m_v28['pf']:>5.2f} {m_v28['max_loss']:>+7.1f}% {m_v28['avg_hold']:>5.1f}d | {cs_v28:>6.0f}")
    print(f"  {'V29 (retrain final)':<22} | {m_v29['trades']:>5} {m_v29['wr']:>5.1f}% {m_v29['avg_pnl']:>+7.2f}% {m_v29['total_pnl']:>+9.1f}% {m_v29['pf']:>5.2f} {m_v29['max_loss']:>+7.1f}% {m_v29['avg_hold']:>5.1f}d | {cs_v29:>6.0f}")
    print("  " + "-" * 102)
    print(f"  Δ comp   vs V28: {cs_v29 - cs_v28:+.0f}")
    print(f"  Δ WR     vs V28: {m_v29['wr'] - m_v28['wr']:+.2f} pp")
    print(f"  Δ PF     vs V28: {m_v29['pf'] - m_v28['pf']:+.3f}")
    print(f"  Δ MaxLoss vs V28: {m_v29['max_loss'] - m_v28['max_loss']:+.2f} pp (less negative = better)")

    df_v29 = pd.DataFrame(t_v29)
    if len(df_v29) > 0:
        df_v29.to_csv(os.path.join(OUT, "trades_v29.csv"), index=False)
        print(f"\n  Saved {len(df_v29)} V29 trades to results/trades_v29.csv")
