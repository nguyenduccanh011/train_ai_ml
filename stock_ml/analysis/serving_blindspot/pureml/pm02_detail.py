# -*- coding: utf-8 -*-
"""pureml step 2: dump full config + slots cho top-3 nhom (a) va (b)."""
import json

import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
IDS = [1058, 1029, 903, 1059,  # top-4 class a
       1053, 1052, 1054, 743]  # top class b

con = psycopg2.connect(**PG)
for tid in IDS:
    t = pd.read_sql(f"select * from strategy_templates where id={tid}", con).iloc[0]
    ec = t["engine_config"]
    ec = json.loads(ec) if isinstance(ec, str) else ec
    print(f"\n===== template {tid} {t['name']} =====")
    print(f"strategy={t['strategy']} model_mode={t['model_mode']} "
          f"sig_thr={t['signal_threshold']} entry_thr={t['entry_threshold']} exit_thr={t['exit_threshold']}")
    print(f"desc: {str(t['description'])[:300]}")
    print(f"created: {t['created_at']}")
    print("engine_config:", json.dumps({k: v for k, v in (ec or {}).items()}, indent=1)[:2200])
    slots = pd.read_sql(
        f"select slot_type, ml_component_id, rule_component_id, feature_set_name, target_config "
        f"from component_slots where template_id={tid}", con)
    for _, s in slots.iterrows():
        tc = s["target_config"]
        tc = json.loads(tc) if isinstance(tc, str) and tc else tc
        print(f"  slot {s['slot_type']}: feat={s['feature_set_name']} target={tc} "
              f"ml={s['ml_component_id']} rule={s['rule_component_id']}")
    # global feature set / target
    fs = pd.read_sql(f"select name from feature_set where id={t['feature_set_id']}", con)
    tg = pd.read_sql(f"select name, type, params from target_catalog where id={t['target_id']}", con)
    print(f"  global feature_set={fs.iloc[0]['name'] if len(fs) else '?'} "
          f"target={tg.iloc[0].to_dict() if len(tg) else '?'}")
con.close()
print("PM02_DONE")
