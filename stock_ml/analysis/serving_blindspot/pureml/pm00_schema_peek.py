# -*- coding: utf-8 -*-
"""pureml step 0: peek schema + engine_config key frequency across all templates."""
import json
from collections import Counter

import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

con = psycopg2.connect(**PG)

for tbl in ("strategy_templates", "leaderboard_runs"):
    cols = pd.read_sql(
        "select column_name, data_type from information_schema.columns "
        f"where table_name='{tbl}' order by ordinal_position", con)
    print(f"== {tbl} ==")
    print(cols.to_string())

tpl = pd.read_sql("select id, name, strategy, engine_config from strategy_templates", con)
print("n templates:", len(tpl))

keyfreq = Counter()
exit_prio = Counter()
strategies = Counter()
for _, r in tpl.iterrows():
    ec = r["engine_config"]
    if isinstance(ec, str):
        ec = json.loads(ec) if ec else {}
    ec = ec or {}
    for k in ec:
        keyfreq[k] += 1
    ep = ec.get("exit_priority")
    exit_prio[json.dumps(ep) if ep is not None else "<default>"] += 1
    strategies[r["strategy"]] += 1

print("\n== engine_config key frequency ==")
for k, v in sorted(keyfreq.items(), key=lambda x: -x[1]):
    print(f"{v:5d}  {k}")

print("\n== exit_priority variants ==")
for k, v in exit_prio.most_common(30):
    print(f"{v:5d}  {k}")

print("\n== strategies ==")
for k, v in strategies.most_common(40):
    print(f"{v:5d}  {k}")

con.close()
print("PM00_DONE")
