# -*- coding: utf-8 -*-
"""ab_00: DIFF CHINH XAC t2783 (gb_x08) vs t2429 (n2_consw20...) tu DB truoc khi ablation.
Full engine_config hai chieu + slots (exit feature-set) + threshold + catalog features."""
import json
import sys
from pathlib import Path

import psycopg2

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()


def jload(x):
    return json.loads(x) if isinstance(x, str) else x


info = {}
for tid in (2429, 2783, 2907, 2910):
    cur.execute(
        "SELECT name, strategy, model_mode, signal_mode, signal_threshold, entry_threshold, "
        "exit_threshold, universe_slug, seed, split_config, validation_config, engine_config "
        "FROM strategy_templates WHERE id=%s", (tid,))
    row = cur.fetchone()
    cur.execute(
        "SELECT slot_type, ml_component_id, rule_component_id, feature_set_name, target_config "
        "FROM component_slots WHERE template_id=%s ORDER BY slot_type", (tid,))
    slots = cur.fetchall()
    info[tid] = dict(name=row[0], strategy=row[1], model_mode=row[2], signal_mode=row[3],
                     sig_thr=row[4], ent_thr=row[5], exit_thr=row[6], uni=row[7], seed=row[8],
                     split=jload(row[9]), val=jload(row[10]), eng=jload(row[11]) or {},
                     slots=slots)

for pair in ((2429, 2783), (2910, 2907)):
    a_id, b_id = pair
    A, B = info[a_id], info[b_id]
    print(f"\n===== DIFF t{a_id} ({A['name']})  vs  t{b_id} ({B['name']}) =====")
    for f in ("strategy", "model_mode", "signal_mode", "sig_thr", "ent_thr", "exit_thr",
              "uni", "seed", "split", "val"):
        if A[f] != B[f]:
            print(f"  META {f}: {A[f]!r}  |  {B[f]!r}")
    a, b = A["eng"], B["eng"]
    n_same = 0
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<ABSENT>"), b.get(k, "<ABSENT>")
        if va == vb:
            n_same += 1
        else:
            print(f"  ENG {k}: {va!r}  |  {vb!r}")
    print(f"  (engine keys giong nhau: {n_same})")
    for sa, sb in zip(A["slots"], B["slots"]):
        if sa != sb:
            print(f"  SLOT A: {sa}")
            print(f"  SLOT B: {sb}")

print("\n===== catalog: exit_vol_market vs exit_vol_downpress =====")
from stock_ml.src.features.catalog import SETS  # noqa: E402
vm = SETS["exit_vol_market"][1]
dp = SETS["exit_vol_downpress"][1]
print(f"exit_vol_market ({len(vm)}): {vm}")
print(f"exit_vol_downpress ({len(dp)}): {dp}")
print(f"downpress - market ({len(set(dp) - set(vm))}): {sorted(set(dp) - set(vm))}")
print(f"market - downpress ({len(set(vm) - set(dp))}): {sorted(set(vm) - set(dp))}")
con.close()
print("AB00_DONE")
