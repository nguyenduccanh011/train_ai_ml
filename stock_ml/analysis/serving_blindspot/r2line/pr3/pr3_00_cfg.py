# -*- coding: utf-8 -*-
"""pr3_00: cong to — dump config t2900 (r2c_oxt04_p42) + fill types + sqlite schema."""
import json
import sqlite3

import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()

cur.execute("SELECT id, name, engine_config FROM strategy_templates WHERE name IN ('fc_rule2','r2c_oxt04_p42') ORDER BY id")
rows = cur.fetchall()
cfgs = {}
for tid, name, eng in rows:
    eng = json.loads(eng) if isinstance(eng, str) else eng
    cfgs[name] = eng
    print(f"{name}: id={tid}, {len(eng)} keys")

base = cfgs.get("fc_rule2", {})
cand = cfgs.get("r2c_oxt04_p42", {})
print("\n=== DIFF r2c_oxt04_p42 vs fc_rule2 ===")
for k in sorted(set(base) | set(cand)):
    a, b = base.get(k, "<missing>"), cand.get(k, "<missing>")
    if a != b:
        print(f"  {k}: {a} -> {b}")

print("\n=== fill/entry/exit types + trail/overext/snr keys ===")
for k in sorted(cand):
    if any(s in k for s in ("fill", "overext", "trail", "snr", "pullback", "dtstop",
                            "downtrend", "entry_gate", "signal_exit", "min_hold",
                            "market", "nonbull", "downleg", "atr")):
        print(f"  {k} = {cand[k]}")
con.close()

print("\n=== sqlite ohlcv schema ===")
sc = sqlite3.connect("C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db")
c2 = sc.cursor()
c2.execute("SELECT sql FROM sqlite_master WHERE type='table'")
for r in c2.fetchall():
    print(r[0])
c2.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT symbol) FROM ohlcv")
print("range:", c2.fetchone())
sc.close()
