# -*- coding: utf-8 -*-
"""pm2 vong 2: multi-seed 7/99/555/123 cho cac o tot nhat vong 1 (seed 42 da co).

Dump trades CSV per seed (run_trades bi ghi de moi seed vi run_id on dinh theo template).
Usage: python pm2_02_seeds.py pm2_hs10_zx25 [pm2_hs12_zx25 ...]
"""
from __future__ import annotations
import csv, logging, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)

import psycopg2  # noqa: E402

from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).resolve().parent
SEEDS = [7, 99, 555, 123]
CHAMP = {42: 729.6, 7: 731.5, 99: 722.5, 555: 730.4, 123: 728.3}
GBX08 = {42: 735.0, 7: 736.7, 99: 728.1, 555: 735.7, 123: 733.3}


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
    out = HERE / f"pm2_{name}_s{seed}_trades.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "entry_date", "entry_price", "exit_date", "exit_price",
                    "holding_days", "pnl_pct", "exit_reason"])
        w.writerows(rows)
    return len(rows)


def main():
    names = sys.argv[1:]
    if not names:
        raise SystemExit("usage: pm2_02_seeds.py <template_name> ...")
    for name in names:
        tid = tmpl_id(name)
        comps = {}
        for sd in SEEDS:
            r = run_template_experiment(template_id=tid, seed=sd)
            rid = r.get("run_id")
            row = read_row(rid)
            n = dump_trades(rid, name, sd)
            comps[sd] = float(row[0])
            print(f"PM2_SEED {name} seed={sd} comp={comps[sd]} dChamp={comps[sd]-CHAMP[sd]:+.1f} "
                  f"dGB={comps[sd]-GBX08[sd]:+.1f} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} "
                  f"tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f} ntr_dump={n}", flush=True)
        cv = list(comps.values())
        print(f"PM2_SEED_SUMMARY {name} mean4={statistics.mean(cv):.1f} seeds={comps} "
              f"(seed42 xem vong 1)", flush=True)
    print("PM2_SEEDS_DONE")


if __name__ == "__main__":
    main()
