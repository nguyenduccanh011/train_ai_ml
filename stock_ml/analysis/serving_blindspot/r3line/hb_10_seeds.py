# -*- coding: utf-8 -*-
"""hb_10: multi-seed t2943 (pr5_dynclean_mh25 = t2936-stack + head csrank dyn e3
+ mh25). Thu tu: 42 (CANARY — phai bit-exact vs dump pr5_70, neu khong ABORT),
7, 99, 555, roi 42 lan cuoi (row canonical dep — run_id t2943 bi ghi de moi seed).
Dump hb_2943_s{seed}_trades.csv + nh_nav2 2 che do x full/f22/f23/f25 + seed-mean.
PHAI chay cwd=repo root (head csrank/xsec — bug loader hard-code path tuong doi).
KHONG dung canonical, KHONG commit. Workdir ngoai repo (corpus r3line bi refactor
926f4c6b xoa 22:13 — xem HB2943_VALIDATION.md)."""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ["STOCK_DATA_DIR"] = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"

REPO = Path("F:/PROJECTS/train_ai_ml")
sys.path.insert(0, str(REPO))
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402

from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
TID = 2943  # pr5_dynclean_mh25
SEEDS = [42, 7, 99, 555]  # 42 dau = canary bit-exact
SLICES = (("2020-01-01", "full"), ("2022-01-01", "f22"),
          ("2023-01-01", "f23"), ("2025-01-01", "f25"))
REF42 = HERE / "pr5_dynclean_mh25_s42_trades.csv"  # dump tu run pr5_70 (DB)


def run_seed(seed: int):
    csv_path = HERE / f"hb_2943_s{seed}_trades.csv"
    if not csv_path.exists():
        r = run_template_experiment(template_id=TID, seed=seed)
        con = psycopg2.connect(**PG)
        cur = con.cursor()
        cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, "
                    "avg_hold FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
        row = cur.fetchone()
        print(f"HB10RUN hb_2943 s{seed} comp={row[0]:.1f} pnl={row[1]:.1f} pf={row[2]:.2f} "
              f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f}", flush=True)
        tdf = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                          "holding_days, pnl_pct, exit_reason from run_trades "
                          "where run_id=%s", con, params=(r.get("run_id"),))
        con.close()
        tdf.to_csv(csv_path, index=False)
        print(f"dumped {len(tdf)} -> {csv_path.name}", flush=True)
    return csv_path


def main():
    acc = {}
    for seed in SEEDS:
        csv_path = run_seed(seed)
        if seed == 42:
            old = pd.read_csv(REF42)
            new = pd.read_csv(csv_path)
            same = old.equals(new)
            print(f"HB10CANARY s42 vs pr5_70: n_old={len(old)} n_new={len(new)} "
                  f"bit_exact={same}", flush=True)
            if not same:
                print("HB10ABORT: pipeline KHONG tai lap s42 — dung, khong chay seed khac",
                      flush=True)
                return
        for lo, tag in SLICES:
            sim = NavSim2(str(csv_path), date_lo=lo)
            a = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2,
                              advance_fee=0.0008, n=20)
            n = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2,
                              advance_fee=None, n=20)
            acc.setdefault((tag, "adv"), []).append(a["mean"])
            acc.setdefault((tag, "noadv"), []).append(n["mean"])
            print(f"HB10NAV hb_2943 s{seed} {tag}: adv x{a['mean']:.2f}±{a['sd']:.2f} "
                  f"DDw {a['dd_worst']*100:.1f}% | noadv x{n['mean']:.2f}±{n['sd']:.2f} "
                  f"DDw {n['dd_worst']*100:.1f}%", flush=True)
    print("\n=== SEED-MEAN (4 seeds 42/7/99/555) ===", flush=True)
    for lo, tag in SLICES:
        a = acc[(tag, "adv")]
        n = acc[(tag, "noadv")]
        print(f"HB10MEAN {tag}: adv x{sum(a)/len(a):.2f} "
              f"(min x{min(a):.2f} max x{max(a):.2f}) | "
              f"noadv x{sum(n)/len(n):.2f} (min x{min(n):.2f} max x{max(n):.2f})", flush=True)
    # chay lai s42 CUOI cung de row canonical t2943 = seed 42 (run_id bi ghi de)
    r = run_template_experiment(template_id=TID, seed=42)
    con = psycopg2.connect(**PG)
    tdf = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                      "holding_days, pnl_pct, exit_reason from run_trades "
                      "where run_id=%s", con, params=(r.get("run_id"),))
    con.close()
    same = pd.read_csv(REF42).equals(
        pd.read_csv(HERE / "hb_2943_s42_trades.csv")) and len(tdf) > 0
    print(f"HB10RESTORE row canonical s42: n={len(tdf)} "
          f"bit_exact_final={pd.read_csv(REF42).reset_index(drop=True).equals(tdf) if len(tdf) else False}",
          flush=True)
    print("HB_10_DONE", flush=True)


if __name__ == "__main__":
    main()
