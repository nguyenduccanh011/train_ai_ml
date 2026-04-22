"""
Test leading_v2 features across ALL active models (v22-v27).
Compares each model's performance with leading (67 feat) vs leading_v2 (99 feat).
"""
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.safe_io  # noqa: F401

from run_v19_1_compare import run_test, calc_metrics
from run_pipeline import get_backtest_function
from run_v25 import comp_score
from run_v24 import resolve_symbols
from src.config_loader import get_model_config, get_training_device
from src.models.registry import detect_device


MODELS_TO_TEST = ["v22", "v23", "v24", "v25", "v26", "v27"]


def load_existing_trades(version_key):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", f"trades_{version_key}.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


def run_model_with_features(version_key, feature_set, device, symbols_str):
    model_cfg = get_model_config(version_key)
    strategy = model_cfg.get("strategy", version_key)
    backtest_fn = get_backtest_function(strategy)

    mods = model_cfg.get("mods", {})
    params = model_cfg.get("params", {})

    if params:
        def wrapped_fn(y_pred, returns, df_test, feature_cols, **kwargs):
            merged = {**kwargs, **params}
            return backtest_fn(y_pred, returns, df_test, feature_cols, **merged)
        fn = wrapped_fn
    else:
        fn = backtest_fn

    trades = run_test(symbols_str,
                      mods.get("a", True), mods.get("b", True),
                      mods.get("c", False), mods.get("d", False),
                      mods.get("e", True), mods.get("f", True),
                      mods.get("g", True), mods.get("h", True),
                      mods.get("i", True), mods.get("j", True),
                      backtest_fn=fn, device=device,
                      feature_set=feature_set)
    return trades


def get_metrics(trades):
    if not trades:
        return {"trades": 0, "wr": 0, "avg_pnl": 0, "total_pnl": 0,
                "pf": 0, "max_loss": 0, "avg_hold": 0, "score": 0}
    m = calc_metrics(trades)
    m["score"] = comp_score(m)
    return m


def main():
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    device = get_training_device()
    resolved = detect_device(device)
    print(f"Device: {resolved.upper()}")

    print(f"\nResolving symbols...")
    symbols_list = resolve_symbols("", min_rows=2000)
    symbols_str = ",".join(symbols_list)
    print(f"Using {len(symbols_list)} symbols")

    print(f"\n{'=' * 130}")
    print(f"FEATURE UPGRADE TEST: leading (67 feat) vs leading_v2 (99 feat) across ALL models")
    print(f"{'=' * 130}")

    results = {}
    total_start = time.time()

    for vk in MODELS_TO_TEST:
        print(f"\n{'─' * 100}")
        print(f"  MODEL: {vk.upper()}")
        print(f"{'─' * 100}")

        # Load existing leading results
        existing = load_existing_trades(vk)
        if existing:
            m_old = get_metrics(existing)
            print(f"  leading  (cached): Trades={m_old['trades']}, WR={m_old['wr']:.1f}%, "
                  f"AvgPnL={m_old['avg_pnl']:+.2f}%, TotPnL={m_old['total_pnl']:+.1f}%, "
                  f"PF={m_old['pf']:.2f}, Score={m_old['score']:.0f}")
        else:
            print(f"  leading  (cached): NO CSV FOUND — will run fresh")
            t0 = time.time()
            trades_old = run_model_with_features(vk, "leading", device, symbols_str)
            dt = time.time() - t0
            m_old = get_metrics(trades_old)
            print(f"  leading  ({dt:.0f}s): Trades={m_old['trades']}, WR={m_old['wr']:.1f}%, "
                  f"AvgPnL={m_old['avg_pnl']:+.2f}%, TotPnL={m_old['total_pnl']:+.1f}%, "
                  f"PF={m_old['pf']:.2f}, Score={m_old['score']:.0f}")

        # Run with leading_v2
        t0 = time.time()
        trades_new = run_model_with_features(vk, "leading_v2", device, symbols_str)
        dt = time.time() - t0
        m_new = get_metrics(trades_new)
        print(f"  leading_v2 ({dt:.0f}s): Trades={m_new['trades']}, WR={m_new['wr']:.1f}%, "
              f"AvgPnL={m_new['avg_pnl']:+.2f}%, TotPnL={m_new['total_pnl']:+.1f}%, "
              f"PF={m_new['pf']:.2f}, Score={m_new['score']:.0f}")

        # Delta
        d_wr = m_new["wr"] - m_old["wr"]
        d_pnl = m_new["total_pnl"] - m_old["total_pnl"]
        d_pf = m_new["pf"] - m_old["pf"]
        d_score = m_new["score"] - m_old["score"]
        improved = d_score > 0 and d_pnl > 0

        tag = "IMPROVED" if improved else "WORSE"
        print(f"  Delta: WR {d_wr:+.1f}%, TotPnL {d_pnl:+.1f}%, PF {d_pf:+.2f}, "
              f"Score {d_score:+.0f}  [{tag}]")

        results[vk] = {
            "old": m_old, "new": m_new,
            "d_wr": d_wr, "d_pnl": d_pnl, "d_pf": d_pf, "d_score": d_score,
            "improved": improved, "time": dt,
        }

    # Summary
    total_time = time.time() - total_start
    print(f"\n\n{'=' * 130}")
    print(f"SUMMARY: leading vs leading_v2 across {len(MODELS_TO_TEST)} models")
    print(f"{'=' * 130}")

    header = (f"  {'Model':<8} | {'leading WR':>10} {'leading PnL':>12} {'leading PF':>10} {'Score':>7} | "
              f"{'v2 WR':>8} {'v2 PnL':>10} {'v2 PF':>8} {'Score':>7} | "
              f"{'dWR':>6} {'dPnL':>10} {'dPF':>6} {'dScore':>7} | {'Result':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    n_improved = 0
    for vk, r in results.items():
        o, n = r["old"], r["new"]
        tag = "UP" if r["improved"] else "DOWN"
        if r["improved"]:
            n_improved += 1
        print(f"  {vk:<8} | {o['wr']:>9.1f}% {o['total_pnl']:>+11.1f}% {o['pf']:>9.2f} {o['score']:>6.0f} | "
              f"{n['wr']:>7.1f}% {n['total_pnl']:>+9.1f}% {n['pf']:>7.2f} {n['score']:>6.0f} | "
              f"{r['d_wr']:>+5.1f}% {r['d_pnl']:>+9.1f}% {r['d_pf']:>+5.2f} {r['d_score']:>+6.0f} | "
              f"{tag:>8}")

    print(f"\n  {n_improved}/{len(MODELS_TO_TEST)} models improved with leading_v2")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save results CSV
    rows = []
    for vk, r in results.items():
        for variant, m in [("leading", r["old"]), ("leading_v2", r["new"])]:
            rows.append({
                "model": vk, "feature_set": variant,
                "trades": m["trades"], "win_rate": round(m["wr"], 2),
                "avg_pnl": round(m["avg_pnl"], 3), "total_pnl": round(m["total_pnl"], 2),
                "profit_factor": round(m["pf"], 3), "max_loss": round(m.get("max_loss", 0), 2),
                "score": round(m["score"], 1),
            })
    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "feature_upgrade_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")

    print(f"\n{'=' * 130}")


if __name__ == "__main__":
    main()
