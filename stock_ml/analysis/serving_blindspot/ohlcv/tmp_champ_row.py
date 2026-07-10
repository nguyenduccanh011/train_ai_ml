# -*- coding: utf-8 -*-
"""Champion 2646 leaderboard row (read-only) for per-metric comparison."""
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()
cur.execute("SELECT run_id, run_seed, composite_score, total_pnl, pf, mdd_per_symbol, trades, "
            "wr, avg_hold, yearly_consistency FROM leaderboard_runs WHERE template_id=2646 "
            "ORDER BY run_seed")
for r in cur.fetchall():
    print(r)
con.close()
