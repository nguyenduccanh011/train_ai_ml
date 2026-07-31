# -*- coding: utf-8 -*-
"""hb_00: khoi phuc trades CSV tu Postgres run_trades (dia bi xoa boi refactor
926f4c6b — CSV goc khong tracked). Dump lai: t2943 s42, t2936 s42/7/99/555,
t2929 s42, t2783 (gb, run canonical seed 42 — CHI DOC, khong re-run).
Verify anchor NAV sau khi dump (hb_01)."""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import psycopg2

HERE = Path(__file__).parent
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml",
          password="stockml_dev")

WANT = [  # (template_id, seed, out_csv)
    (2943, 42, "pr5_dynclean_mh25_s42_trades.csv"),
    (2936, 42, "ab_noT_s42_trades.csv"),
    (2936, 7, "ab_noT_s7_trades.csv"),
    (2936, 99, "ab_noT_s99_trades.csv"),
    (2936, 555, "ab_noT_s555_trades.csv"),
    (2929, 42, "xg_dyn_mh25_s42_trades.csv"),
    (2783, 42, "gb_t2783_s42_trades.csv"),
]


def main():
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    for tid, seed, out in WANT:
        cur.execute(
            "SELECT run_id, composite_score, trades, total_pnl, created_at "
            "FROM leaderboard_runs WHERE template_id=%s AND run_seed=%s "
            "ORDER BY created_at DESC", (tid, seed))
        rows = cur.fetchall()
        if not rows:
            print(f"HB00MISS t{tid} s{seed}: KHONG co run trong DB", flush=True)
            continue
        run_id, comp, ntr, pnl, ts = rows[0]
        print(f"HB00 t{tid} s{seed}: {len(rows)} run, dung run_id={run_id} "
              f"comp={comp:.1f} tr={ntr} pnl={pnl:.1f} ({ts})", flush=True)
        tdf = pd.read_sql(
            "select symbol, entry_date, exit_date, entry_price, exit_price, "
            "holding_days, pnl_pct, exit_reason from run_trades where run_id=%s",
            con, params=(run_id,))
        if len(tdf) == 0:
            print(f"HB00EMPTY t{tid} s{seed}: run_trades trong!", flush=True)
            continue
        tdf.to_csv(HERE / out, index=False)
        print(f"  dumped {len(tdf)} -> {out}", flush=True)
    con.close()
    print("HB_00_DONE", flush=True)


if __name__ == "__main__":
    main()
