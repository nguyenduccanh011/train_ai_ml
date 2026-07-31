"""Overfit audit: the whole loop tuned the engine modulators on the SAME 3 seeds (42/7/99). Confirm
the champion 2417's gain over baseline 2409 HOLDS on UNSEEN seeds (123/555). The engine params
(vol-gate/combo/head-blend) are seed-independent transforms; only the ML heads differ per seed. If
2417 still beats 2409 by ~+11 on fresh seeds, the result is robust (not seed-overfit).
Usage: python stock_ml/scripts/audit_champion_newseeds.py
"""

from __future__ import annotations
import sys, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import psycopg2  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
NEW_SEEDS = [123, 555]


def comp(rid):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score,mdd_per_symbol FROM leaderboard_runs WHERE run_id=%s", (rid,)
    )
    r = cur.fetchone()
    con.close()
    return (float(r[0]), float(r[1])) if r and r[0] is not None else (None, None)


res = {2409: {}, 2417: {}}
for sd in NEW_SEEDS:
    for tid in (2409, 2417):
        r = run_template_experiment(template_id=tid, seed=sd)
        c, m = comp(r.get("run_id"))
        res[tid][sd] = c
        print(f"  tmpl {tid} seed {sd}: comp={c} mdd={m:.3f}", flush=True)
print("\n=== NEW-SEED ROBUSTNESS (2417 head-blend champ vs 2409 baseline) ===")
print(f"  seeds tested: {NEW_SEEDS}  (training seeds were 42/7/99)")
print(
    f"  2409 baseline: per-seed {res[2409]}  mean {statistics.mean([v for v in res[2409].values() if v]):.1f}"
)
print(
    f"  2417 champion: per-seed {res[2417]}  mean {statistics.mean([v for v in res[2417].values() if v]):.1f}"
)
g = statistics.mean([v for v in res[2417].values() if v]) - statistics.mean(
    [v for v in res[2409].values() if v]
)
print(f"  GAIN on unseen seeds: {g:+.1f}  (training-seed gain was +11.5)")
print("  -> gain holds (~+11) = robust; gain shrinks = seed-overfit")
print("AUDIT_DONE")
