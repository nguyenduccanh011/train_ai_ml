# -*- coding: utf-8 -*-
"""pr4_00: CONG TO tuyen R3/maxhold — dump era-forensics t2429 vs t2783.
So sanh: strategy, split_config, slots (heads/labels/features), universe, seed,
validation_config; + lich su seed cua ho 2429/2516/2903/r3_*."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()

def jload(x):
    if x is None:
        return None
    return json.loads(x) if isinstance(x, str) else x

IDS = [2429, 2516, 2783, 2903]
# them cac clone r3_
cur.execute("SELECT id, name FROM strategy_templates WHERE name LIKE 'r3\\_%' ORDER BY id")
r3s = cur.fetchall()
print("R3 clones:", r3s)
IDS += [r[0] for r in r3s if "mh16" in r[1] and "gb" not in r[1] and "2643" not in r[1]]

for tid in IDS:
    cur.execute(
        "SELECT id, name, market, strategy, model_mode, signal_mode, signal_threshold, "
        "entry_threshold, exit_threshold, universe_slug, seed, split_config, "
        "validation_config, feature_set_id, target_id, created_at "
        "FROM strategy_templates WHERE id=%s", (tid,))
    row = cur.fetchone()
    if not row:
        print(f"NOT FOUND t{tid}")
        continue
    (i, nm, mk, strat, mm, sm, st, et, xt, uni, seed, spl, val, fsid, tgid, ca) = row
    print(f"\n===== t{i} {nm}")
    print(f" strategy={strat} mode={mm} sig={sm}/{st} entry={et} exit={xt} uni={uni} "
          f"seed={seed} fs={fsid} tgt={tgid} created={ca}")
    print(" split_config:", json.dumps(jload(spl), sort_keys=True))
    print(" validation_config:", json.dumps(jload(val), sort_keys=True))
    # slots
    cur.execute(
        "SELECT slot_type, ml_component_id, rule_component_id, feature_set_name, target_config "
        "FROM component_slots WHERE template_id=%s ORDER BY id", (tid,))
    for (styp, mlid, rlid, fsn, tcfg) in cur.fetchall():
        tc = jload(tcfg)
        tcs = json.dumps(tc, sort_keys=True)
        if len(tcs) > 400:
            tcs = tcs[:400] + "..."
        print(f"  slot {styp}: ml={mlid} rule={rlid} fs={fsn} target={tcs}")
        if mlid:
            cur.execute("SELECT name, role, algorithm, params, component_type "
                        "FROM model_components WHERE id=%s", (mlid,))
            r = cur.fetchone()
            if r:
                cfg = json.dumps(jload(r[3]), sort_keys=True)
                if len(cfg) > 500:
                    cfg = cfg[:500] + "..."
                print(f"    ml_comp {mlid}: {r[0]} role={r[1]} algo={r[2]} type={r[4]} params={cfg}")

# lich su seed / runs cua ho
print("\n===== RUNS ho 2429/maxhold")
cur.execute(
    "SELECT r.run_id, r.template_id, t.name, r.seed, r.composite_score, r.total_pnl, r.pf, "
    "r.trades, r.avg_hold, r.status FROM leaderboard_runs r "
    "JOIN strategy_templates t ON t.id=r.template_id "
    "WHERE r.template_id IN (2429,2516,2903,2783) OR t.name LIKE 'r3\\_%' "
    "ORDER BY r.template_id, r.seed")
for row in cur.fetchall():
    print(" ", row)
con.close()
