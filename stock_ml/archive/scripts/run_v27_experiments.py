"""
V27 Experiments: Implement and validate V27 patch ideas on top of V26 baseline.

Goals:
- Run real backtests for V27 ideas and combinations.
- Eliminate combinations that reduce performance vs V26.
- Select final V27 patch set.
"""
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_v19_1_compare import run_test as run_test_base, run_rule_test, calc_metrics
from run_v25 import comp_score
from run_v26_experiments import backtest_v26


def make_backtest_fn(v25_base: dict, v26_base: dict, v27_cfg: dict):
    def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        merged = {}
        merged.update(v25_base)
        merged.update(v26_base)
        merged.update(v27_cfg)
        merged.update(kwargs)
        return backtest_v26(
            y_pred,
            returns,
            df_test,
            feature_cols,
            peak_protect_strong_threshold=0.12,
            **merged,
        )

    return bt_fn


def print_row(name, m, score, delta_v26=None, delta_rule=None):
    base = (
        f"  {name:<34} | {m['trades']:>5} {m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% "
        f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>5.1f}d | {score:>6.0f}"
    )
    if delta_v26 is not None and delta_rule is not None:
        base += f" | {delta_v26:>+8.1f}% {delta_rule:>+8.1f}%"
    print(base)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    DEFAULT_SYMBOLS = (
        "ACB,AAS,AAV,ACV,BCG,BCM,BID,BSR,BVH,CTG,DCM,DGC,DIG,DPM,EIB,"
        "FPT,FRT,GAS,GEX,GMD,HCM,HDB,HDG,HPG,HSG,KBC,KDH,LPB,MBB,MSN,"
        "MWG,NKG,NLG,NT2,NVL,OCB,PC1,PDR,PLX,PNJ,POW,PVD,PVS,REE,SAB,"
        "SBT,SHB,SSI,STB,TCB,TPB,VCB,VCI,VDS,VHM,VIC,VJC,VND,VNM,VPB,VTP"
    )
    SYMBOLS = args.symbols.strip() if args.symbols.strip() else DEFAULT_SYMBOLS
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    print("=" * 150)
    print("V27 EXPERIMENTS: V27 patch ideas over V26 baseline")
    print("=" * 150)

    v25_base = dict(
        patch_smart_hardcap=True,
        patch_pp_restore=True,
        patch_long_horizon=True,
        patch_symbol_tuning=True,
        patch_rule_ensemble=True,
        patch_noise_filter=False,
        patch_adaptive_hardcap=False,
        patch_pp_2of3=False,
    )

    # V26 locked baseline (B+C+E)
    v26_base = dict(
        v26_wider_hardcap=False,
        v26_relaxed_entry=True,
        v26_skip_choppy=True,
        v26_extended_hold=False,
        v26_strong_rule_ensemble=True,
        v26_min_position=False,
        v26_score5_penalty=False,
        v26_hardcap_confirm_strong=False,
    )

    v27_off = dict(
        v27_selective_choppy=False,
        v27_hardcap_two_step=False,
        v27_rule_priority=False,
        v27_dynamic_score5_penalty=False,
        v27_trend_persistence_hold=False,
    )

    print("\n  [1] Baseline: Rule + V26...")
    t_rule = run_rule_test(SYMBOLS)
    m_rule = calc_metrics(t_rule)

    t_v26 = run_test_base(
        SYMBOLS,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        backtest_fn=make_backtest_fn(v25_base, v26_base, v27_off),
    )
    m_v26 = calc_metrics(t_v26)
    cs_v26 = comp_score(m_v26)

    print(f"    Rule: trades={m_rule['trades']}, TotPnL={m_rule['total_pnl']:+.1f}%")
    print(f"    V26 : trades={m_v26['trades']}, TotPnL={m_v26['total_pnl']:+.1f}%, PF={m_v26['pf']:.2f}, Comp={cs_v26:.0f}")

    experiments = [
        ("J1 selective_choppy", {**v27_off, "v27_selective_choppy": True, "v26_skip_choppy": False}),
        ("J2 hardcap_two_step", {**v27_off, "v27_hardcap_two_step": True}),
        ("J3 rule_priority", {**v27_off, "v27_rule_priority": True}),
        ("J4 dynamic_score5_penalty", {**v27_off, "v27_dynamic_score5_penalty": True}),
        ("J5 trend_persistence_hold", {**v27_off, "v27_trend_persistence_hold": True}),
        ("J1+J2", {**v27_off, "v27_selective_choppy": True, "v26_skip_choppy": False, "v27_hardcap_two_step": True}),
        ("J1+J2+J3", {**v27_off, "v27_selective_choppy": True, "v26_skip_choppy": False, "v27_hardcap_two_step": True, "v27_rule_priority": True}),
        ("J1+J2+J3+J4", {**v27_off, "v27_selective_choppy": True, "v26_skip_choppy": False, "v27_hardcap_two_step": True, "v27_rule_priority": True, "v27_dynamic_score5_penalty": True}),
        ("J1+J2+J3+J5", {**v27_off, "v27_selective_choppy": True, "v26_skip_choppy": False, "v27_hardcap_two_step": True, "v27_rule_priority": True, "v27_trend_persistence_hold": True}),
        ("J2+J3+J4+J5", {**v27_off, "v27_hardcap_two_step": True, "v27_rule_priority": True, "v27_dynamic_score5_penalty": True, "v27_trend_persistence_hold": True}),
        ("J1+J2+J3+J4+J5", {**v27_off, "v27_selective_choppy": True, "v26_skip_choppy": False, "v27_hardcap_two_step": True, "v27_rule_priority": True, "v27_dynamic_score5_penalty": True, "v27_trend_persistence_hold": True}),
    ]

    print("\n  [2] V27 ablation runs...")
    print(
        f"\n  {'Config':<34} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7} | {'vs V26':>9} {'vs Rule':>9}"
    )
    print("  " + "-" * 145)
    print_row("V26 baseline (B+C+E)", m_v26, cs_v26, 0.0, m_v26["total_pnl"] - m_rule["total_pnl"])
    print("  " + "-" * 145)

    rows = []
    best = None

    for name, cfg in experiments:
        t0 = time.time()
        trades = run_test_base(
            SYMBOLS,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            backtest_fn=make_backtest_fn(v25_base, v26_base, cfg),
        )
        m = calc_metrics(trades)
        cs = comp_score(m)
        dt = time.time() - t0

        d_v26 = m["total_pnl"] - m_v26["total_pnl"]
        d_rule = m["total_pnl"] - m_rule["total_pnl"]
        print_row(name, m, cs, d_v26, d_rule)
        print(f"    time={dt:.1f}s")

        row = {
            "config": name,
            "trades": m["trades"],
            "wr": m["wr"],
            "avg_pnl": m["avg_pnl"],
            "total_pnl": m["total_pnl"],
            "pf": m["pf"],
            "max_loss": m["max_loss"],
            "avg_hold": m["avg_hold"],
            "comp": cs,
            "delta_vs_v26": d_v26,
            "delta_vs_rule": d_rule,
            "cfg": cfg,
            "trades_raw": trades,
        }
        rows.append(row)

        if best is None or cs > best["comp"]:
            best = row

    # Eliminate regressions
    kept = [
        r
        for r in rows
        if (r["delta_vs_v26"] > 0) and (r["pf"] >= m_v26["pf"] - 1e-9)
    ]

    print("\n" + "=" * 150)
    print("  ELIMINATION")
    print("=" * 150)
    print(f"  Total tested: {len(rows)}")
    print(f"  Kept (delta_vs_v26>0 and PF>=V26): {len(kept)}")

    if kept:
        kept_sorted = sorted(kept, key=lambda x: (x["comp"], x["total_pnl"], x["pf"]), reverse=True)
        chosen = kept_sorted[0]
    else:
        # fallback: choose by composite score even if no strict-kept config
        chosen = sorted(rows, key=lambda x: x["comp"], reverse=True)[0]

    print("\n  TOP 5 CONFIGS BY COMPOSITE SCORE:")
    for r in sorted(rows, key=lambda x: x["comp"], reverse=True)[:5]:
        print(
            f"    {r['config']:<24} Comp={r['comp']:.0f} TotPnL={r['total_pnl']:+.1f}% "
            f"PF={r['pf']:.3f} dV26={r['delta_vs_v26']:+.1f}%"
        )

    print("\n" + "=" * 150)
    print("  FINAL V27 CHOICE")
    print("=" * 150)
    print(f"  Config:   {chosen['config']}")
    print(f"  TotPnL:   {chosen['total_pnl']:+.1f}% (vs V26: {chosen['delta_vs_v26']:+.1f}%)")
    print(f"  WR/PF:    {chosen['wr']:.1f}% / {chosen['pf']:.3f}")
    print(f"  Comp:     {chosen['comp']:.0f} (V26={cs_v26:.0f}, delta={chosen['comp']-cs_v26:+.0f})")
    print(f"  Patches:  {sorted([k for k, v in chosen['cfg'].items() if isinstance(v, bool) and v])}")

    # Persist outputs
    df_summary = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if k not in ("cfg", "trades_raw")
            }
            for r in rows
        ]
    ).sort_values("comp", ascending=False)
    df_summary.to_csv(os.path.join(OUT, "v27_experiments_summary.csv"), index=False)

    df_best = pd.DataFrame(chosen["trades_raw"])
    if len(df_best) > 0:
        df_best.to_csv(os.path.join(OUT, "trades_v27.csv"), index=False)
        print(f"\n  Saved {len(df_best)} trades to results/trades_v27.csv")

    with open(os.path.join(OUT, "v27_selected_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"selected={chosen['config']}\n")
        for k, v in chosen["cfg"].items():
            f.write(f"{k}={v}\n")

    print("  Saved summary to results/v27_experiments_summary.csv")
    print("  Saved selected config to results/v27_selected_config.txt")
    print("\n" + "=" * 150)
    print("  DONE")
    print("=" * 150)
