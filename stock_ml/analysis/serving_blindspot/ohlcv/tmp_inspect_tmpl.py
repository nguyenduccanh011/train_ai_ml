# -*- coding: utf-8 -*-
"""Inspect templates 2646 / 2730: slots + engine_config keys (read-only)."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()
for tid in (2646, 2730):
    cur.execute("SELECT name, strategy, engine_config, signal_threshold, entry_threshold, "
                "exit_threshold, seed FROM strategy_templates WHERE id=%s", (tid,))
    name, strat, eng, st, et, xt, seed = cur.fetchone()
    if isinstance(eng, str):
        eng = json.loads(eng)
    print(f"=== {tid} {name} strat={strat} sig_thr={st} ent_thr={et} exit_thr={xt} seed={seed}")
    print(json.dumps(eng, indent=1, sort_keys=True))
    cur.execute("SELECT slot_type, ml_component_id, rule_component_id, feature_set_name, "
                "target_config FROM component_slots WHERE template_id=%s", (tid,))
    for r in cur.fetchall():
        print("SLOT", r)
con.close()
