"""Multi-seed confirm: does the additive cross_sectional_entry direction head (e5_csent_z09, t3255)
beat the frontier (x2_struct_to, t3185) across seeds {42,7,99}? seed42 showed +0.22 NAV at same DD
— inside the noise band, so REQUIRE seed-robustness (the xr_liq false-positive lesson). Runs both
templates at seeds 7 & 99, NAV-scores the fresh run_ids, prints per-seed + mean nav_adv/cagr_adv.
"""
from __future__ import annotations
import subprocess, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import psycopg2
from stock_ml.scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = {"x2_struct_to": 3185, "e5_csent_z09": 3255}
SEEDS = [7, 99]

env = dict(os.environ, STOCK_DATA_DIR="F:/PROJECTS/train_ai_ml/market_data/market.duckdb",
           NH_NAV2_DIR="F:/PROJECTS/hb2943_work")

con = psycopg2.connect(**PG); cur = con.cursor()


def score_and_read(rid):
    subprocess.run([str(ROOT / "venv/Scripts/python.exe"),
                    str(ROOT / "stock_ml/scripts/ops/score_nav_leaderboard.py"),
                    "--run-like", rid, "--force"], env=env, cwd=str(ROOT),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    con.commit()  # refresh snapshot
    cur.execute("SELECT nav_adv,cagr_adv,maxdd_nav FROM leaderboard_nav WHERE run_id=%s", (rid,))
    return cur.fetchone()


# run_id excludes seed -> each seed OVERWRITES the row; must score+read BEFORE next seed runs.
results = {nm: {} for nm in TEMPLATES}
for sd in [42, 7, 99]:
    for nm, tid in TEMPLATES.items():
        try:
            r = run_template_experiment(template_id=tid, seed=sd)
            rid = r.get("run_id")
            nav = score_and_read(rid)
            results[nm][sd] = nav
            print(f"{nm} seed{sd}: NAV={nav[0]:.3f} CAGR={nav[1]*100:.2f}% DD={nav[2]*100:.2f}% ({rid})", flush=True)
        except Exception as e:
            print(f"ERR {nm} seed{sd}: {type(e).__name__} {str(e)[:200]}", flush=True)

print("\n=== MULTISEED e5_csent_z09 (additive direction head) vs frontier — 3 seeds {42,7,99} ===")
for nm in TEMPLATES:
    navs = [results[nm][s][0] for s in (42, 7, 99) if s in results[nm]]
    cagrs = [results[nm][s][1] for s in (42, 7, 99) if s in results[nm]]
    if not navs:
        print(f"{nm}: no rows"); continue
    per = " ".join(f"{n:.2f}" for n in navs)
    print(f"{nm:16s}: navs=[{per}] mean_NAV={sum(navs)/len(navs):.3f} mean_CAGR={100*sum(cagrs)/len(cagrs):.2f}%")
con.close()
print("MULTISEED_E5_DONE")
