# -*- coding: utf-8 -*-
"""r3_00: dump engine_config cua cac template goc cho tuyen R3/maxhold."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()

IDS = [2429, 2516, 2643, 2646, 2783, 2903]
for tid in IDS:
    cur.execute(
        "SELECT id, name, market, strategy, model_mode, signal_mode, signal_threshold, "
        "entry_threshold, exit_threshold, engine_config, universe_slug, seed "
        "FROM strategy_templates WHERE id=%s", (tid,))
    row = cur.fetchone()
    if not row:
        print(f"NOT FOUND: {tid}")
        continue
    (i, nm, mk, strat, mm, sm, st, et, xt, eng, uni, seed) = row
    print(f"\n=== t{i} {nm} market={mk} strategy={strat} mode={mm} sig={sm}/{st} "
          f"entry={et} exit={xt} uni={uni} seed={seed}")
    eng = json.loads(eng) if isinstance(eng, str) else eng
    print(json.dumps(eng, indent=1, sort_keys=True))
con.close()
