# -*- coding: utf-8 -*-
"""P1-E2 buoc 2 — chay template xr_* (hoac smoke clone champion) theo seed.

Usage: python xr_02_run.py <template_name> <seed> [<seed> ...]
In metrics tu leaderboard_runs + dump trades CSV (xr_<name>_s<seed>_trades.csv).
"""
from __future__ import annotations
import csv, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)

import psycopg2  # noqa: E402

from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).resolve().parent


def tmpl_id(name: str) -> int:
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT id FROM strategy_templates WHERE name=%s", (name,))
    r = cur.fetchone(); con.close()
    if not r:
        raise SystemExit(f"template not found: {name}")
    return r[0]


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, avg_hold "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def dump_trades(run_id: str, name: str, seed: int) -> int:
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT symbol, entry_date, entry_price, exit_date, exit_price, holding_days, "
                "pnl_pct, exit_reason FROM run_trades WHERE run_id=%s ORDER BY entry_date, symbol",
                (run_id,))
    rows = cur.fetchall(); con.close()
    out = HERE / f"xr_{name}_s{seed}_trades.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "entry_date", "entry_price", "exit_date", "exit_price",
                    "holding_days", "pnl_pct", "exit_reason"])
        w.writerows(rows)
    return len(rows)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: xr_02_run.py <template_name> <seed> [<seed> ...]")
    name = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]]
    tid = tmpl_id(name)
    for sd in seeds:
        r = run_template_experiment(template_id=tid, seed=sd)
        if not r.get("success"):
            print(f"XR_RUN_FAIL {name} seed={sd} err={r.get('error')}", flush=True)
            sys.exit(1)
        rid = r["run_id"]
        row = read_row(rid)
        n = dump_trades(rid, name, sd)
        print(f"XR_RUN {name} seed={sd} comp={float(row[0]):.1f} pnl={row[1]:.1f} "
              f"pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} "
              f"hold={row[6]:.1f} ntr_dump={n} run_id={rid}", flush=True)
    print("XR_RUN_DONE")


if __name__ == "__main__":
    main()
