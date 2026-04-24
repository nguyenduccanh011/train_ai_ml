"""
V29 RETRAIN — model-layer experiments.

Hypothesis: V28 has saturated engine-level optimization. Breakthrough requires:
  1. Re-label "early-wave" target instead of "trend_regime"
  2. Add accumulation features (vol contraction, range compression, vol dry+spike)
  3. Add cross-sectional relative-strength features

Experiments (all use V28's backtest_fn — same engine, different model output):

  E0: V28 baseline                        [trend_regime + leading_v2]
  E1: feature_set leading_v3              [trend_regime + leading_v3 (accumulation)]
  E2: target early_wave (gain 8% / 10d)   [early_wave + leading_v2]
  E3: target early_wave + leading_v3      [early_wave + leading_v3]
  E4: target early_wave (gain 6% / 10d)   [looser threshold]
  E5: target early_wave (gain 10% / 10d)  [stricter threshold]
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v28 import backtest_v28


TREND_TGT = dict(
    type="trend_regime", trend_method="dual_ma",
    short_window=5, long_window=20, classes=3,
)


def early_wave_tgt(gain=0.08, loss=0.05, fwd=10, back=10, long_w=20, classes=3):
    return dict(
        type="early_wave",
        forward_window=fwd, short_window=back, long_window=long_w,
        gain_threshold=gain, loss_threshold=loss, classes=classes,
    )


def fmt(name, m, cs, base_cs=None):
    delta = f"  ({cs-base_cs:+.0f})" if base_cs is not None else ""
    return (f"  {name:<40} | {m['trades']:>5} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
            f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% "
            f"{m['avg_hold']:>4.1f}d | {cs:>6.0f}{delta}")


def run(symbols_str, feature_set, target_override):
    return run_test_base(
        symbols_str, True, True, False, False, True, True, True, True, True, True,
        backtest_fn=backtest_v28,
        feature_set=feature_set,
        target_override=target_override,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    HDR = (f"  {'Config':<40} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} "
           f"{'PF':>6} {'MaxLoss':>8} {'AvgH':>5} | {'Comp':>9}")
    SEP = "  " + "-" * (len(HDR) - 2)

    print("=" * 130)
    print("V29 RETRAIN — model-layer experiments (early_wave target + accumulation features)")
    print("=" * 130)

    experiments = [
        ("E0: V28 baseline (trend, v2)",      "leading_v2", TREND_TGT),
        ("E1: trend + v3 (accumulation)",     "leading_v3", TREND_TGT),
        ("E2: early_wave 8%/10d + v2",        "leading_v2", early_wave_tgt(0.08, 0.05, 10)),
        ("E3: early_wave 8%/10d + v3",        "leading_v3", early_wave_tgt(0.08, 0.05, 10)),
        ("E4: early_wave 6%/10d + v3",        "leading_v3", early_wave_tgt(0.06, 0.05, 10)),
        ("E5: early_wave 10%/10d + v3",       "leading_v3", early_wave_tgt(0.10, 0.05, 10)),
        ("E6: early_wave 8%/15d + v3",        "leading_v3", early_wave_tgt(0.08, 0.06, 15)),
    ]

    print(HDR); print(SEP)
    base_cs = None
    results = {}
    for name, fs, tgt in experiments:
        t0 = time.time()
        try:
            trades = run(SYMBOLS, fs, tgt)
            m = calc_metrics(trades); cs = comp_score(m, trades)
        except Exception as e:
            print(f"  {name:<40} | ERROR: {e}")
            continue
        if base_cs is None:
            base_cs = cs
        results[name] = (fs, tgt, trades, m, cs)
        elapsed = time.time() - t0
        print(fmt(name, m, cs, base_cs=base_cs), f" [{elapsed:.0f}s]")
    print(SEP)

    if not results:
        print("\n  No experiment succeeded.")
        sys.exit(1)

    # Sort
    ranked = sorted(results.items(), key=lambda x: x[1][4], reverse=True)
    print("\n  Ranked by composite:")
    for n, (fs, tgt, _, m, cs) in ranked:
        print(f"    {n}: comp={cs:.0f}  tot={m['total_pnl']:+.1f}%  wr={m['wr']:.1f}%  pf={m['pf']:.2f}")

    winner = ranked[0]
    win_name, (win_fs, win_tgt, win_trades, win_m, win_cs) = winner
    print(f"\n[WINNER] {win_name} — comp={win_cs:.0f}")
    print(f"  feature_set={win_fs}")
    print(f"  target={win_tgt}")

    df_w = pd.DataFrame(win_trades)
    out_path = os.path.join(OUT, "trades_v29_retrain_winner.csv")
    df_w.to_csv(out_path, index=False)
    print(f"  Saved {len(df_w)} trades to {out_path}")

    print("\n" + "=" * 130)
    print("  DONE")
    print("=" * 130)
