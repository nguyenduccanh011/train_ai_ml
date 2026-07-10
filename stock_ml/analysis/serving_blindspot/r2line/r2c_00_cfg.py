# -*- coding: utf-8 -*-
"""r2c_00: doc config chinh xac cua r2b_oxtrail04 trong DB + diff vs r2_c2_pb40snr / fc_rule2."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()

cfgs = {}
for name in ("fc_rule2", "r2_c2_pb40snr", "r2b_oxtrail04"):
    cur.execute("SELECT id, engine_config FROM strategy_templates WHERE name=%s", (name,))
    row = cur.fetchone()
    if not row:
        print(f"NOT FOUND: {name}")
        continue
    tid, eng = row
    eng = json.loads(eng) if isinstance(eng, str) else eng
    cfgs[name] = eng
    print(f"{name}: id={tid}, {len(eng)} keys")

base = cfgs["fc_rule2"]
for name in ("r2_c2_pb40snr", "r2b_oxtrail04"):
    print(f"\n=== DIFF {name} vs fc_rule2 ===")
    for k in sorted(set(base) | set(cfgs[name])):
        a, b = base.get(k, "<missing>"), cfgs[name].get(k, "<missing>")
        if a != b:
            print(f"  {k}: {a} -> {b}")

print("\n=== fc_rule2 exit/trail/overext keys (nen mac dinh) ===")
for k in sorted(base):
    if any(s in k for s in ("overext", "trail", "snr", "pullback", "dtstop", "downtrend",
                            "stop", "hold", "exit", "entry")):
        print(f"  {k} = {base[k]}")
con.close()
