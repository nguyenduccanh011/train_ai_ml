"""
V26 Experiments: Test each proposed improvement individually and in combinations.
Measures delta vs V25 baseline to decide what goes into V26.

Improvements being tested:
  A) Adaptive hard_cap with wider ATR multipliers for volatile stocks
  B) Relaxed entry for confirmed strong trends (bypass prev_pred requirement)
  C) Skip choppy regime trades entirely
  D) Extended hold for winning trades in strong trends
  E) Stronger rule ensemble (rule_signal 3+ bars = force entry)
  F) Minimum position threshold (skip low-conviction trades)
  G) Score-5 penalty (reduce position for overfit high-score entries)
  H) Wider hard_cap confirm bars in strong trend
"""
import sys, os, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from experiments.run_v25 import backtest_v25
from src.backtest.engine import backtest_unified


def backtest_v26(y_pred, returns, df_test, feature_cols, **kwargs):
    """V26 candidate backtest — delegates to unified engine."""
    return backtest_unified(y_pred, returns, df_test, feature_cols, **kwargs)


# ============================================================
# MAIN: Run experiments
# ============================================================
if __name__ == "__main__":
    import argparse
    from src.config_loader import get_pipeline_symbols
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 150)
    print("V26 EXPERIMENTS: Testing improvements individually and in combinations")
    print("=" * 150)

    # V25 baseline patches (all V24 patches ON, V25 new patches OFF)
    V25_BASE = dict(
        patch_smart_hardcap=True, patch_pp_restore=True,
        patch_long_horizon=True, patch_symbol_tuning=True,
        patch_rule_ensemble=True,
        patch_noise_filter=False, patch_adaptive_hardcap=False, patch_pp_2of3=False,
    )

    V26_OFF = dict(
        v26_wider_hardcap=False, v26_relaxed_entry=False,
        v26_skip_choppy=False, v26_extended_hold=False,
        v26_strong_rule_ensemble=False, v26_min_position=False,
        v26_score5_penalty=False, v26_hardcap_confirm_strong=False,
    )

    # ---- PHASE 1: V25 Baseline ----
    print("\n  [Phase 1] V25 Baseline...")
    def make_v25_fn():
        def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
            return backtest_v25(y_pred, returns, df_test, feature_cols,
                               peak_protect_strong_threshold=0.12,
                               **V25_BASE, **kwargs)
        return bt_fn

    t_v25 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=make_v25_fn())
    m_v25 = calc_metrics(t_v25)
    print(f"    V25 Baseline: {m_v25['trades']} trades, TotalPnL={m_v25['total_pnl']:+.1f}%, "
          f"WR={m_v25['wr']:.1f}%, PF={m_v25['pf']:.2f}, AvgPnL={m_v25['avg_pnl']:+.2f}%")

    # Rule baseline
    t_rule = run_rule_test(SYMBOLS)
    m_rule = calc_metrics(t_rule)
    print(f"    Rule:         {m_rule['trades']} trades, TotalPnL={m_rule['total_pnl']:+.1f}%")

    # ---- PHASE 2: Individual improvements ----
    print("\n  [Phase 2] Individual V26 improvements...")

    experiments = [
        # Individual
        ("A: Wider hardcap",         {**V26_OFF, "v26_wider_hardcap": True}),
        ("B: Relaxed entry",         {**V26_OFF, "v26_relaxed_entry": True}),
        ("C: Skip choppy",           {**V26_OFF, "v26_skip_choppy": True}),
        ("D: Extended hold",         {**V26_OFF, "v26_extended_hold": True}),
        ("E: Strong rule ensemble",  {**V26_OFF, "v26_strong_rule_ensemble": True}),
        ("F: Min position (0.28)",   {**V26_OFF, "v26_min_position": True}),
        ("G: Score-5 penalty",       {**V26_OFF, "v26_score5_penalty": True}),
        ("H: HC confirm strong+1",   {**V26_OFF, "v26_hardcap_confirm_strong": True}),
        # Combinations of 2
        ("A+C",                      {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True}),
        ("A+D",                      {**V26_OFF, "v26_wider_hardcap": True, "v26_extended_hold": True}),
        ("A+H",                      {**V26_OFF, "v26_wider_hardcap": True, "v26_hardcap_confirm_strong": True}),
        ("B+E",                      {**V26_OFF, "v26_relaxed_entry": True, "v26_strong_rule_ensemble": True}),
        ("C+F",                      {**V26_OFF, "v26_skip_choppy": True, "v26_min_position": True}),
        ("C+G",                      {**V26_OFF, "v26_skip_choppy": True, "v26_score5_penalty": True}),
        # Combinations of 3
        ("A+C+D",                    {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True, "v26_extended_hold": True}),
        ("A+C+H",                    {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True, "v26_hardcap_confirm_strong": True}),
        ("A+D+H",                    {**V26_OFF, "v26_wider_hardcap": True, "v26_extended_hold": True, "v26_hardcap_confirm_strong": True}),
        ("B+C+E",                    {**V26_OFF, "v26_relaxed_entry": True, "v26_skip_choppy": True, "v26_strong_rule_ensemble": True}),
        ("A+C+F",                    {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True, "v26_min_position": True}),
        # Big combos
        ("A+C+D+H",                  {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True, "v26_extended_hold": True, "v26_hardcap_confirm_strong": True}),
        ("A+B+C+D+H",               {**V26_OFF, "v26_wider_hardcap": True, "v26_relaxed_entry": True, "v26_skip_choppy": True, "v26_extended_hold": True, "v26_hardcap_confirm_strong": True}),
        ("A+B+C+E+H",               {**V26_OFF, "v26_wider_hardcap": True, "v26_relaxed_entry": True, "v26_skip_choppy": True, "v26_strong_rule_ensemble": True, "v26_hardcap_confirm_strong": True}),
        ("A+C+D+F+G+H",             {**V26_OFF, "v26_wider_hardcap": True, "v26_skip_choppy": True, "v26_extended_hold": True, "v26_min_position": True, "v26_score5_penalty": True, "v26_hardcap_confirm_strong": True}),
        ("ALL",                      {k: True for k in V26_OFF}),
        ("ALL except F",             {**{k: True for k in V26_OFF}, "v26_min_position": False}),
        ("ALL except G",             {**{k: True for k in V26_OFF}, "v26_score5_penalty": False}),
    ]

    print(f"\n  {'Config':<30} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} "
          f"{'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7} | {'vs V25':>9} {'vs Rule':>9}")
    print("  " + "-" * 145)

    # Print V25 baseline
    cs25 = comp_score(m_v25)
    print(f"  {'V25-BASELINE':<30} | {m_v25['trades']:>5} {m_v25['wr']:>5.1f}% {m_v25['avg_pnl']:>+7.2f}% "
          f"{m_v25['total_pnl']:>+9.1f}% {m_v25['pf']:>5.2f} {m_v25['max_loss']:>+7.1f}% {m_v25['avg_hold']:>5.1f}d | "
          f"{cs25:>6.0f} |      --- {'':>9}")
    print("  " + "-" * 145)

    results = {}
    for label, v26_cfg in experiments:
        t0 = time.time()
        print(f"    Running {label}...", end="", flush=True)

        def make_fn(cfg):
            def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
                return backtest_v26(y_pred, returns, df_test, feature_cols,
                                   peak_protect_strong_threshold=0.12,
                                   **V25_BASE, **cfg, **kwargs)
            return bt_fn

        trades = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                               backtest_fn=make_fn(v26_cfg))
        m = calc_metrics(trades)
        results[label] = (m, trades, v26_cfg)

        cs = comp_score(m)
        dt = time.time() - t0
        delta_v25 = m['total_pnl'] - m_v25['total_pnl']
        delta_rule = m['total_pnl'] - m_rule['total_pnl']
        marker = " +++" if delta_v25 > 200 else " ++" if delta_v25 > 50 else " +" if delta_v25 > 0 else " ---" if delta_v25 < -200 else " --" if delta_v25 < -50 else ""
        print(f"\r  {label:<30} | {m['trades']:>5} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>5.1f}d | "
              f"{cs:>6.0f} | {delta_v25:>+8.1f}% {delta_rule:>+8.1f}%{marker}")

    # ---- PHASE 3: Ranking ----
    print(f"\n{'='*150}")
    print("  RANKING BY COMPOSITE SCORE (higher = better)")
    print(f"{'='*150}")

    scored = []
    for name, (m, trades, cfg) in results.items():
        scored.append((comp_score(m), name, m, cfg))
    scored.sort(reverse=True)

    for rank, (sc, name, m, cfg) in enumerate(scored, 1):
        delta = m['total_pnl'] - m_v25['total_pnl']
        marker = " <<<" if rank <= 3 else ""
        print(f"  #{rank:>2} {sc:>7.0f}  {name:<30} TotPnL={m['total_pnl']:>+9.1f}% "
              f"PF={m['pf']:>5.2f} WR={m['wr']:>5.1f}% MaxLoss={m['max_loss']:>+7.1f}% "
              f"vs_V25={delta:>+8.1f}%{marker}")

    # Best config
    best_score, best_name, best_m, best_cfg = scored[0]
    print(f"\n  BEST: {best_name}")
    print(f"    Composite: {best_score:.0f} (V25 baseline: {cs25:.0f}, delta: {best_score-cs25:+.0f})")
    print(f"    TotalPnL:  {best_m['total_pnl']:+.1f}% (vs V25: {best_m['total_pnl']-m_v25['total_pnl']:+.1f}%)")
    print(f"    PF:        {best_m['pf']:.2f}")
    print(f"    WR:        {best_m['wr']:.1f}%")
    active_patches = [k for k, v in best_cfg.items() if v]
    print(f"    Patches:   {active_patches}")

    # ---- Improvement summary ----
    print(f"\n{'='*150}")
    print("  INDIVIDUAL IMPROVEMENT DELTA vs V25")
    print(f"{'='*150}")
    for name in ["A: Wider hardcap", "B: Relaxed entry", "C: Skip choppy",
                  "D: Extended hold", "E: Strong rule ensemble",
                  "F: Min position (0.28)", "G: Score-5 penalty", "H: HC confirm strong+1"]:
        if name in results:
            m, _, _ = results[name]
            dt = m['total_pnl'] - m_v25['total_pnl']
            dwr = m['wr'] - m_v25['wr']
            dpf = m['pf'] - m_v25['pf']
            verdict = "KEEP" if dt > 0 and dpf >= 0 else "KEEP(marginal)" if dt > 0 else "SKIP" if dt < -50 else "NEUTRAL"
            print(f"  {name:<30} dTotPnL={dt:>+8.1f}%  dWR={dwr:>+5.2f}%  dPF={dpf:>+5.3f}  => {verdict}")

    # Save best trades
    if scored:
        _, best_name_2, _, _ = scored[0]
        _, best_trades, _ = results[best_name_2]
        df_best = pd.DataFrame(best_trades)
        if len(df_best) > 0:
            df_best.to_csv(os.path.join(OUT, "trades_v26_best.csv"), index=False)
            print(f"\n  Saved {len(df_best)} V26-best trades to results/trades_v26_best.csv")

    print(f"\n{'='*150}")
    print("  DONE")
    print(f"{'='*150}")
