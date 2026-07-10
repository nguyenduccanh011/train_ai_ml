"""Week-0 multi-seed run for candidate w0_pyr_u10_r04 (existing clone template_id=2671).

Runs run_template_experiment for seeds 7, 99, 555 sequentially (seed 42 already done this
session: composite=886.1) and reads each leaderboard row. NO clone creation, NO training —
runs must reuse cached predictions. Results dumped as JSON lines + summary file.

Usage: python stock_ml/analysis/serving_blindspot/week0/ms_w0_pyr_u10_r04.py
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
NAME = "w0_pyr_u10_r04"
EXPECTED_TMPL = 2671
SEEDS = [7, 99, 555]
KNOWN = {42: 886.1}  # composite from this session's seed-42 run
OUT = Path(__file__).with_suffix(".results.json")


async def resolve_template() -> int:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NAME)
        if not ex:
            raise SystemExit(f"FATAL: template {NAME} not found — expected existing id {EXPECTED_TMPL}")
        if ex.id != EXPECTED_TMPL:
            print(f"WARN: {NAME} resolved to id={ex.id} (expected {EXPECTED_TMPL}) — using resolved id")
        return ex.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
        "FROM leaderboard_runs WHERE run_id=%s",
        (run_id,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    tid = asyncio.run(resolve_template())
    asyncio.run(async_engine.dispose())
    print(f"template {NAME} id={tid}; seeds={SEEDS}", flush=True)

    rows = []
    for sd in SEEDS:
        t0 = time.time()
        r = run_template_experiment(template_id=tid, seed=sd)
        dt = time.time() - t0
        row = read_row(r.get("run_id"))
        rec = {
            "name": NAME,
            "template_id": tid,
            "seed": sd,
            "composite": float(row[0]) if row and row[0] is not None else None,
            "total_pnl": float(row[1]) if row and row[1] is not None else None,
            "pf": float(row[2]) if row and row[2] is not None else None,
            "mdd": float(row[3]) if row and row[3] is not None else None,
            "trades": int(row[4]) if row and row[4] is not None else None,
            "runtime_s": round(dt, 1),
            "run_id": r.get("run_id"),
        }
        rows.append(rec)
        print("ROW " + json.dumps(rec), flush=True)

    comps = [rec["composite"] for rec in rows if rec["composite"] is not None]
    all4 = comps + [KNOWN[42]]
    mean4 = statistics.mean(all4) if all4 else None
    summary = {"rows": rows, "known_seed42_composite": KNOWN[42],
               "mean_composite_4seeds": round(mean4, 2) if mean4 is not None else None}
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"MEAN_COMPOSITE_4SEEDS={summary['mean_composite_4seeds']} (seeds 42,7,99,555)", flush=True)
    print("MS_W0_PYR_U10_R04_DONE", flush=True)


if __name__ == "__main__":
    main()
