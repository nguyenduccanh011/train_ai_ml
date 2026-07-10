"""Week-0 multi-seed check for candidate w0_pyr_u10_r03 (template_id=2670, clone of champion 2646).

Seed 42 already run this session (composite=912.8). This script runs seeds 7, 99, 555
sequentially via run_template_experiment(template_id=2670, seed=<s>) and reads each
leaderboard row. Each seed runs in a child subprocess with a 480s guard: if it exceeds
that, it is retraining models (cache-miss) -> killed and marked, per harness rules.

Pattern = stock_ml/scripts/deploy_wavestruct.py / week0/run_line_b.py. No cloning here:
template 2670 already exists and is reused (verified name=w0_pyr_u10_r03).

Usage (cwd=f:/PROJECTS/train_ai_ml):
  python stock_ml/analysis/serving_blindspot/week0/ms_w0_pyr_u10_r03.py            # parent
  python stock_ml/analysis/serving_blindspot/week0/ms_w0_pyr_u10_r03.py --child 7  # internal
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
NAME = "w0_pyr_u10_r03"
TEMPLATE_ID = 2670
SEEDS = [7, 99, 555]
GUARD_S = 480  # kill child beyond this: cache-miss / retraining

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")


def child(seed: int) -> None:
    sys.path.insert(0, str(REPO))
    import psycopg2
    from stock_ml.scripts.run_template import run_template_experiment

    r = run_template_experiment(template_id=TEMPLATE_ID, seed=seed)
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
    row = cur.fetchone()
    con.close()
    out = {"run_id": r.get("run_id"),
           "composite": float(row[0]) if row and row[0] is not None else None,
           "total_pnl": float(row[1]) if row and row[1] is not None else None,
           "pf": float(row[2]) if row and row[2] is not None else None,
           "mdd": float(row[3]) if row and row[3] is not None else None,
           "trades": int(row[4]) if row and row[4] is not None else None}
    print("RESULT " + json.dumps(out), flush=True)


def main() -> None:
    results = []
    for seed in SEEDS:
        t0 = time.time()
        rec = {"name": NAME, "template_id": TEMPLATE_ID, "seed": seed}
        try:
            p = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", str(seed)],
                cwd=str(REPO), capture_output=True, text=True, timeout=GUARD_S)
            dt = time.time() - t0
            rec["runtime_s"] = round(dt, 1)
            res_line = next((ln for ln in (p.stdout or "").splitlines()
                             if ln.startswith("RESULT ")), None)
            if p.returncode == 0 and res_line:
                rec.update(json.loads(res_line[len("RESULT "):]))
            else:
                tail = ((p.stderr or "").strip().splitlines() or ["<no stderr>"])[-5:]
                rec["note"] = f"child failed rc={p.returncode}: " + " | ".join(tail)
        except subprocess.TimeoutExpired:
            rec["runtime_s"] = round(time.time() - t0, 1)
            rec["note"] = f"cache-miss: killed after {GUARD_S}s (retraining guard)"
        results.append(rec)
        print("ROW " + json.dumps(rec), flush=True)
    print("MS_DONE " + json.dumps(results))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        child(int(sys.argv[2]))
    else:
        main()
