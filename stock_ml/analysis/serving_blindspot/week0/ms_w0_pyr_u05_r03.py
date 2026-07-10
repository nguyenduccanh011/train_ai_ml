"""Week-0 multi-seed check: candidate w0_pyr_u05_r03 (existing clone template_id=2667).

Runs run_template_experiment(template_id=2667, seed=s) sequentially for seeds 7, 99, 555
(seed 42 already done this session: composite=825.9) and reads each leaderboard row.
Config-only clone of champion 2646 -> reuses cached predictions (should be well under
~5 min per run). Pattern = stock_ml/scripts/deploy_wavestruct.py / week0/run_line_b.py.

Usage: python stock_ml/analysis/serving_blindspot/week0/ms_w0_pyr_u05_r03.py
"""
from __future__ import annotations
import asyncio, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
TMPL_ID = 2667
TMPL_NAME = "w0_pyr_u05_r03"
SEEDS = [7, 99, 555]
OUT_JSON = Path(__file__).with_name("ms_w0_pyr_u05_r03_results.json")


async def check_template() -> None:
    """Sanity: template 2667 must exist and be the expected clone name. No mutation."""
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        t = await repo.get_by_id(TMPL_ID)
        assert t is not None, f"template {TMPL_ID} not found"
        assert t.name == TMPL_NAME, f"template {TMPL_ID} name mismatch: {t.name!r}"
        print(f"template ok: id={TMPL_ID} name={t.name}", flush=True)


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    asyncio.run(check_template())
    asyncio.run(async_engine.dispose())
    results = []
    for seed in SEEDS:
        t0 = time.time()
        r = run_template_experiment(template_id=TMPL_ID, seed=seed)
        dt = time.time() - t0
        row = read_row(r.get("run_id"))
        rec = {"name": TMPL_NAME, "template_id": TMPL_ID, "seed": seed, "runtime_s": round(dt, 1),
               "composite": float(row[0]) if row[0] is not None else None,
               "total_pnl": float(row[1]) if row[1] is not None else None,
               "pf": float(row[2]) if row[2] is not None else None,
               "mdd": float(row[3]) if row[3] is not None else None,
               "trades": int(row[4]) if row[4] is not None else None}
        results.append(rec)
        print("ROW " + json.dumps(rec), flush=True)
        OUT_JSON.write_text(json.dumps(results, indent=2))
    print("MS_DONE")


if __name__ == "__main__":
    main()
