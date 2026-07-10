# -*- coding: utf-8 -*-
"""pr4_01: diff engine_config t2429 vs t2783 (era cu vs hien dai) + t2907 (r3_mh16)
+ lich su seed cua ho (leaderboard_runs.run_seed, leaderboard_seed_stats)."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(PG and None or None, **PG) if False else psycopg2.connect(**PG)
cur = con.cursor()

def jload(x):
    return json.loads(x) if isinstance(x, str) else x

cfgs = {}
for tid in (2429, 2783, 2907, 2910):
    cur.execute("SELECT name, engine_config FROM strategy_templates WHERE id=%s", (tid,))
    nm, eng = cur.fetchone()
    cfgs[tid] = (nm, jload(eng) or {})

a_id, b_id = 2429, 2783
(a_nm, a), (b_nm, b) = cfgs[a_id], cfgs[b_id]
keys = sorted(set(a) | set(b))
print(f"DIFF engine_config t{a_id} ({a_nm})  vs  t{b_id} ({b_nm})")
same = []
for k in keys:
    va, vb = a.get(k, "<ABSENT>"), b.get(k, "<ABSENT>")
    if va == vb:
        same.append(k)
    else:
        print(f"  {k}: {va!r}  |  {vb!r}")
print(f"  (giong nhau: {len(same)} keys: {', '.join(same)})")

print("\nDIFF t2907 r3_mh16 vs t2429 (phai chi la max_hold):")
c = cfgs[2907][1]
for k in sorted(set(a) | set(c)):
    va, vc = a.get(k, "<ABSENT>"), c.get(k, "<ABSENT>")
    if va != vc:
        print(f"  {k}: {va!r} -> {vc!r}")

print("\n===== RUNS ho 2429/2516/2903/2907/2783/2910 (run_seed)")
cur.execute(
    "SELECT r.template_id, t.name, r.run_seed, r.composite_score, r.total_pnl, r.pf, "
    "r.trades, r.avg_hold, r.superseded, r.state, r.created_at::date "
    "FROM leaderboard_runs r JOIN strategy_templates t ON t.id=r.template_id "
    "WHERE r.template_id IN (2429,2516,2903,2907,2783,2910) "
    "ORDER BY r.template_id, r.run_seed, r.created_at")
for row in cur.fetchall():
    print(" ", row)

print("\n===== seed_stats")
cur.execute("SELECT * FROM leaderboard_seed_stats WHERE template_id IN "
            "(2429,2516,2903,2907,2783,2910)")
for row in cur.fetchall():
    print(" ", row)
con.close()
