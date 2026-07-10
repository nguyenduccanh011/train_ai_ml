"""un00: dump engine_config + slots cua t2783 (gb_x08) va t2005/t2835 (fc_rule2)."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()
for tid in (2783, 2005, 2835):
    cur.execute("select id, name, strategy, model_mode, signal_mode, signal_threshold, "
                "entry_threshold, exit_threshold, engine_config, split_config, seed, universe_slug "
                "from strategy_templates where id=%s", (tid,))
    r = cur.fetchone()
    if not r:
        print(f"=== {tid}: NOT FOUND ==="); continue
    eng = r[8]
    eng = json.loads(eng) if isinstance(eng, str) else eng
    print(f"=== t{r[0]} {r[1]} strategy={r[2]} model_mode={r[3]} signal_mode={r[4]} "
          f"sig_thr={r[5]} entry_thr={r[6]} exit_thr={r[7]} seed={r[10]} universe={r[11]}")
    print(json.dumps(eng, indent=1, sort_keys=True))
    cur.execute("select slot_type, ml_component_id, rule_component_id, feature_set_name, "
                "target_config from component_slots where template_id=%s order by id", (tid,))
    for sl in cur.fetchall():
        tc = sl[4]
        tc = json.loads(tc) if isinstance(tc, str) else tc
        print(f"  slot {sl[0]} ml={sl[1]} rule={sl[2]} fs={sl[3]} target={json.dumps(tc)[:200]}")
    print()
con.close()
