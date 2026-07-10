"""Week-0 finalization: export seed-42 trade frames (CSV) for best candidate
w0_pyr_u10_r02 (template 2669) and champion control 2646, then restore the
champion seed-555 leaderboard row.

Fold caches live at {out_dir}/{run_id}/folds, so all existing fp-dirs for
templates 2669/2646 are copied from results/ into best_trades/ first to
guarantee cache hits (no retraining).

Usage: python stock_ml/analysis/serving_blindspot/week0/export_best_trades.py
       (cwd must be repo root f:/PROJECTS/train_ai_ml)
"""
from __future__ import annotations
import json, shutil, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402

from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BEST_DIR = Path(__file__).resolve().parent / "best_trades"
RESULTS = REPO / "results"
GUARD_S = 8 * 60


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def copy_fold_caches(template_id: int):
    for src in RESULTS.glob(f"tmpl_{template_id}_*"):
        folds = src / "folds"
        if not folds.is_dir():
            continue
        dst = BEST_DIR / src.name / "folds"
        if dst.is_dir():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folds, dst)
        print(f"copied fold cache {src.name} -> best_trades/")


def run_one(template_id: int, seed: int, out_dir=None, export_csv=False, tag=""):
    t0 = time.time()
    r = run_template_experiment(template_id=template_id, seed=seed,
                                out_dir=out_dir, export_csv=export_csv)
    dt = time.time() - t0
    row = read_row(r.get("run_id"))
    rec = {"tag": tag, "template_id": template_id, "seed": seed, "runtime_s": round(dt, 1),
           "composite": float(row[0]), "total_pnl": float(row[1]), "pf": float(row[2]),
           "mdd": float(row[3]), "trades": int(row[4])}
    print("ROW " + json.dumps(rec), flush=True)
    if dt > GUARD_S:
        print(f"WARNING runtime {dt:.0f}s exceeded guard", flush=True)
    return rec


def main():
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    copy_fold_caches(2669)
    copy_fold_caches(2646)
    # 1) best candidate seed 42, export trades
    run_one(2669, 42, out_dir=BEST_DIR, export_csv=True, tag="best_w0_pyr_u10_r02_s42_export")
    # 2) champion control seed 42, export trades (overwrites champion row -> restored below)
    run_one(2646, 42, out_dir=BEST_DIR, export_csv=True, tag="champion_2646_s42_export")
    # 3) restore champion seed-555 leaderboard row
    rec = run_one(2646, 555, tag="champion_2646_s555_restore")
    ok = (round(rec["composite"], 1) == 730.4 and rec["trades"] == 1386)
    print(f"RESTORE_CHECK {'PASS' if ok else 'FAIL'} comp={rec['composite']} trades={rec['trades']}")
    print("EXPORT_DONE")


if __name__ == "__main__":
    main()
