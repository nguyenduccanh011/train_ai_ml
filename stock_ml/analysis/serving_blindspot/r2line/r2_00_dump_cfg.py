# -*- coding: utf-8 -*-
"""R2 line step 0: dump full template config cua fc_rule2 (t2835) + t2005 de biet key."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()
for name in ("fc_rule2", "n3_dtstop06"):
    cur.execute(
        "SELECT id, name, market, strategy, model_mode, signal_mode, signal_threshold, "
        "entry_threshold, exit_threshold, engine_config, split_config, universe_slug, seed "
        "FROM strategy_templates WHERE name=%s", (name,))
    row = cur.fetchone()
    if not row:
        print(f"NOT FOUND: {name}")
        continue
    (tid, nm, mk, strat, mm, sm, st, et, xt, eng, split, uni, seed) = row
    print(f"=== {nm} id={tid} market={mk} strategy={strat} model_mode={mm} "
          f"sig_mode={sm} sig_thr={st} entry_thr={et} exit_thr={xt} uni={uni} seed={seed}")
    eng = json.loads(eng) if isinstance(eng, str) else eng
    print(json.dumps(eng, indent=1, sort_keys=True))
    cur.execute("SELECT slot_type, ml_component_id, rule_component_id, feature_set_name, target_config "
                "FROM template_component_slots WHERE template_id=%s", (tid,))
    for sl in cur.fetchall():
        print("SLOT:", sl)
con.close()
