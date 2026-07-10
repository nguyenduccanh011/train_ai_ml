"""Dump template configs for operating-point retest: champion 2646, gb_x08 2783,
pv_full 2779, xl_swal06, xl_swal2 (looked up by name). Print engine_config diff vs 2646."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()

# resolve xl_ template ids by name
cur.execute("SELECT id, name FROM strategy_templates WHERE name IN ('xl_swal06','xl_swal2','gb_x08','pv_full') OR id IN (2646,2779,2783)")
rows = cur.fetchall()
print("== template ids ==")
for r in rows:
    print(r)
ids = [r[0] for r in rows]

cfg = {}
for tid in ids:
    cur.execute("SELECT id, name, signal_mode, signal_threshold, entry_threshold, exit_threshold, engine_config FROM strategy_templates WHERE id=%s", (tid,))
    r = cur.fetchone()
    eng = r[6] if isinstance(r[6], dict) else json.loads(r[6])
    cfg[r[1]] = dict(id=r[0], signal_mode=r[2], signal_threshold=r[3], entry_threshold=r[4], exit_threshold=r[5], eng=eng)
    # component slots
    cur.execute("SELECT slot_type, feature_set_name, target_config FROM component_slots WHERE template_id=%s", (tid,))
    slots = cur.fetchall()
    cfg[r[1]]["slots"] = [(s[0], s[1], s[2] if isinstance(s[2], dict) else json.loads(s[2])) for s in slots]

base = None
for name, c in cfg.items():
    if c["id"] == 2646:
        base = c
for name, c in sorted(cfg.items(), key=lambda kv: kv[1]["id"]):
    print(f"\n==== {name} (id={c['id']}) ====")
    print(f"signal_mode={c['signal_mode']} signal_threshold={c['signal_threshold']} entry_threshold={c['entry_threshold']} exit_threshold={c['exit_threshold']}")
    for st, fs, tc in c["slots"]:
        print(f"  slot {st}: fs={fs} target={json.dumps(tc)}")
    if base is not None and c["id"] != 2646:
        added = {k: v for k, v in c["eng"].items() if k not in base["eng"] or base["eng"][k] != v}
        removed = {k: v for k, v in base["eng"].items() if k not in c["eng"]}
        print(f"  eng DIFF vs 2646: added/changed={json.dumps(added)} removed={json.dumps(removed)}")
    else:
        # print exit/z/norm-relevant keys of the champion in full
        keys = {k: v for k, v in c["eng"].items() if any(s in k.lower() for s in ("exit", "z_norm", "znorm", "norm", "scale", "thresh", "snr", "recombine", "weight"))}
        print(f"  eng exit/norm-relevant keys: {json.dumps(keys, indent=1)}")

con.close()
