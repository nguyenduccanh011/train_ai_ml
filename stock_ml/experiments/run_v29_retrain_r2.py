"""
V29 RETRAIN ROUND 2 — fine-tune around E4 winner.

Winner from Round 1: early_wave gain=6%/10d + leading_v3 (comp=346, +24 vs V28).

Sweep directions:
  - gain_threshold  [0.05, 0.06, 0.07]
  - loss_threshold  [0.04, 0.05, 0.06]
  - forward_window  [8, 10, 12]
  - short_window (accumulation lookback) [8, 10, 12]
  - classes 2 vs 3 (binary vs 3-class)
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v28 import backtest_v28


def tgt(gain=0.06, loss=0.05, fwd=10, back=10, long_w=20, classes=3):
    return dict(
        type="early_wave",
        forward_window=fwd, short_window=back, long_window=long_w,
        gain_threshold=gain, loss_threshold=loss, classes=classes,
    )


def fmt(name, m, cs, base_cs=None):
    delta = f"  ({cs-base_cs:+.0f})" if base_cs is not None else ""
    return (f"  {name:<42} | {m['trades']:>5} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
            f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% "
            f"{m['avg_hold']:>4.1f}d | {cs:>6.0f}{delta}")


def run(symbols_str, target_override, feature_set="leading_v3"):
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

    HDR = f"  {'Config':<42} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>5} | {'Comp':>9}"
    SEP = "  " + "-" * (len(HDR) - 2)

    print("=" * 135)
    print("V29 RETRAIN ROUND 2 — fine-tune around early_wave + leading_v3")
    print("=" * 135)

    experiments = [
        ("R1-win: g6 l5 f10 b10 c3 (anchor)", tgt(0.06, 0.05, 10, 10, 20, 3)),
        # gain sweep
        ("g=0.05",                            tgt(0.05, 0.05, 10, 10, 20, 3)),
        ("g=0.07",                            tgt(0.07, 0.05, 10, 10, 20, 3)),
        # loss sweep
        ("l=0.04 (tighter)",                  tgt(0.06, 0.04, 10, 10, 20, 3)),
        ("l=0.06 (looser)",                   tgt(0.06, 0.06, 10, 10, 20, 3)),
        # forward window sweep
        ("fwd=8",                             tgt(0.06, 0.05,  8, 10, 20, 3)),
        ("fwd=12",                            tgt(0.06, 0.05, 12, 10, 20, 3)),
        # accumulation lookback sweep
        ("back=8",                            tgt(0.06, 0.05, 10,  8, 20, 3)),
        ("back=12",                           tgt(0.06, 0.05, 10, 12, 20, 3)),
        # binary
        ("binary (2-class)",                  tgt(0.06, 0.05, 10, 10, 20, 2)),
        # combos
        ("g5 l4 f8 b8",                       tgt(0.05, 0.04,  8,  8, 20, 3)),
        ("g6 l4 f10 b10",                     tgt(0.06, 0.04, 10, 10, 20, 3)),
        ("g7 l5 f12 b10",                     tgt(0.07, 0.05, 12, 10, 20, 3)),
    ]

    print(HDR); print(SEP)
    base_cs = None
    results = {}
    for name, t in experiments:
        t0 = time.time()
        try:
            trades = run(SYMBOLS, t)
            m = calc_metrics(trades); cs = comp_score(m, trades)
        except Exception as e:
            print(f"  {name:<42} | ERROR: {e}")
            continue
        if base_cs is None:
            base_cs = cs
        results[name] = (t, trades, m, cs)
        print(fmt(name, m, cs, base_cs=base_cs), f"[{time.time()-t0:.0f}s]")
    print(SEP)

    ranked = sorted(results.items(), key=lambda x: x[1][3], reverse=True)
    print("\n  Ranked by composite:")
    for n, (t, _, m, cs) in ranked:
        print(f"    {n}: comp={cs:.0f}  tot={m['total_pnl']:+.1f}%  wr={m['wr']:.1f}%  pf={m['pf']:.2f}  ml={m['max_loss']:+.1f}%")

    winner = ranked[0]
    win_name, (win_tgt, win_trades, win_m, win_cs) = winner
    print(f"\n[WINNER-R2] {win_name} — comp={win_cs:.0f}")
    print(f"  target={win_tgt}")

    df_w = pd.DataFrame(win_trades)
    out_path = os.path.join(OUT, "trades_v29_retrain_r2_winner.csv")
    df_w.to_csv(out_path, index=False)
    print(f"  Saved {len(df_w)} trades to {out_path}")
