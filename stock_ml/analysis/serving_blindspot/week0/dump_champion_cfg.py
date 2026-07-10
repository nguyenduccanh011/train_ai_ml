"""Week-0 Task 1/3/4: dump champion 2646 engine_config, SNR run recon, per-seed history."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
OUT = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/week0/champion_engine_config.json"

con = psycopg2.connect(**PG)
cur = con.cursor()

# --- Task 1: engine_config dump ---
cur.execute("SELECT id, name, engine_config FROM strategy_templates WHERE id=2646")
tid, name, eng = cur.fetchone()
if isinstance(eng, str):
    eng = json.loads(eng)
with open(OUT, "w") as f:
    json.dump(eng, f, indent=2, sort_keys=True)
print(f"TEMPLATE id={tid} name={name} n_keys={len(eng)}")
CHECK = ["entry_pullback_pct", "entry_pullback_window", "trailing_activate_pct",
         "signal_exit_protect_release_drop_k", "exit_priority",
         "pyramid_add_units", "pyramid_add_bars", "pyramid_add_min_ret",
         "mfe_act_k", "exit_snr_extend_threshold", "exit_snr_extend_window",
         "exit_snr_min_gain", "entry_pullback_cancel_below_ma", "resume_reentry_win",
         "signal_exit_hold_legage_scale", "signal_exit_hold_legage_pct",
         "signal_exit_hold_legamp_scale"]
for k in CHECK:
    print(f"  {k}: {'PRESENT=' + repr(eng[k]) if k in eng else 'ABSENT'}")

# --- Task 3a: prior runs with 'snr' in run_name ---
print("\n=== SNR runs in leaderboard_runs (incl. superseded) ===")
cur.execute("""SELECT run_name, template_id, run_seed, composite_score, total_pnl, pf,
                      mdd_per_symbol, trades, superseded
               FROM leaderboard_runs WHERE run_name ILIKE '%snr%'
               ORDER BY run_name, run_seed""")
rows = cur.fetchall()
if not rows:
    print("  (none)")
tmpl_ids = set()
for r in rows:
    print(f"  name={r[0]} tmpl={r[1]} seed={r[2]} comp={r[3]} pnl={r[4]} pf={r[5]} mdd={r[6]} tr={r[7]} superseded={r[8]}")
    if r[1]:
        tmpl_ids.add(r[1])
for t in sorted(tmpl_ids):
    cur.execute("SELECT id, name, engine_config FROM strategy_templates WHERE id=%s", (t,))
    row = cur.fetchone()
    if not row:
        print(f"  template {t}: DELETED")
        continue
    e = row[2]
    e = json.loads(e) if isinstance(e, str) else e
    snr_keys = {k: v for k, v in e.items() if "snr" in k.lower()}
    print(f"  template {t} ({row[1]}) snr keys: {snr_keys}")

# --- Task 4: per-seed history of 2646 ---
print("\n=== template 2646 leaderboard history (incl. superseded) ===")
cur.execute("""SELECT run_seed, composite_score, total_pnl, pf, mdd_per_symbol, trades,
                      superseded, run_id, created_at
               FROM leaderboard_runs WHERE template_id=2646
               ORDER BY run_seed, created_at""")
for r in cur.fetchall():
    print(f"  seed={r[0]} comp={r[1]} pnl={r[2]} pf={r[3]} mdd={r[4]} tr={r[5]} superseded={r[6]} run_id={r[7]} at={r[8]}")

con.close()
print("DUMP_DONE")
