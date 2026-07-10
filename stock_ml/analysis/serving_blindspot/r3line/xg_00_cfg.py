# -*- coding: utf-8 -*-
"""xg_00: dump config 2 base cho cross-apply max_hold: t2531 dyncsr88, t1799 v19_fullwave.
In engine_config, exit_priority, target_config (label horizon), description, slots.
"""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()

IDS = [2531, 1799, 2429]
for tid in IDS:
    cur.execute(
        "SELECT id, name, market, strategy, model_mode, signal_mode, signal_threshold, "
        "entry_threshold, exit_threshold, engine_config, universe_slug, seed, description, "
        "hypothesis, target_id, feature_set_id "
        "FROM strategy_templates WHERE id=%s", (tid,))
    row = cur.fetchone()
    if not row:
        print(f"NOT FOUND: {tid}")
        continue
    (i, nm, mk, strat, mm, sm, st, et, xt, eng, uni, seed, desc, hyp, tgt, fs) = row
    print(f"\n=== t{i} {nm} market={mk} strategy={strat} mode={mm} sig={sm}/{st} "
          f"entry={et} exit={xt} uni={uni} seed={seed} target_id={tgt} fs={fs}")
    print(f"desc: {desc}")
    print(f"hyp: {hyp}")
    eng = json.loads(eng) if isinstance(eng, str) else eng
    print("engine_config:", json.dumps(eng, indent=1, sort_keys=True))
    cur.execute(
        "SELECT slot_type, ml_component_id, rule_component_id, feature_set_name, target_config "
        "FROM component_slots WHERE template_id=%s", (tid,))
    for sl in cur.fetchall():
        tc = sl[4]
        tc = json.loads(tc) if isinstance(tc, str) else tc
        print(f"  slot {sl[0]} ml={sl[1]} rule={sl[2]} fs={sl[3]} target={json.dumps(tc)}")
con.close()
print("XG00_DONE")
