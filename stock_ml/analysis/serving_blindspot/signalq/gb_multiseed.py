"""Multi-seed cho gb_x08 (2730 + exit_snr_defer_min_giveback=0.08).

Template da duoc tao boi agent truoc (run gb_x08 seed 42 = 735.0).
Chay not seeds 7/99/555/123, in delta so voi champion 2646 va 2730 per-seed.
"""
from __future__ import annotations
import asyncio, statistics, sys
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
NAME = "gb_x08"
SEEDS = [int(x) for x in sys.argv[1:]] or [7, 99, 555, 123]
CHAMP = {42: 729.6, 7: 731.5, 99: 722.5, 555: 730.4, 123: 728.3}


async def get_id() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        t = await StrategyTemplateRepository(s).get_by_name(NAME)
        if not t:
            raise SystemExit(f"template {NAME} not found")
        return t.id


def snr08_baseline():
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT run_seed, composite_score FROM leaderboard_runs WHERE run_name='xq_snr_t08_g27' ORDER BY run_seed")
    r = dict(cur.fetchall()); con.close(); return r


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    snr = snr08_baseline()
    print(f"2730 per-seed baseline: {snr}", flush=True)
    tid = asyncio.run(get_id()); asyncio.run(async_engine.dispose())
    print(f"template {NAME} id={tid}", flush=True)
    comps = {42: 735.0}
    for sd in SEEDS:
        r = run_template_experiment(template_id=tid, seed=sd)
        row = read_row(r.get("run_id"))
        comps[sd] = float(row[0]) if row and row[0] is not None else None
        b2730 = snr.get(sd); bch = CHAMP.get(sd)
        d1 = f"{comps[sd]-float(b2730):+.1f}" if b2730 is not None else "n/a"
        d2 = f"{comps[sd]-bch:+.1f}" if bch is not None else "n/a"
        print(f"  {NAME} seed={sd}: comp={comps[sd]} (vs2730 {d1}, vsChamp {d2}) "
              f"pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}", flush=True)
    cv = [v for v in comps.values() if v is not None]
    print(f"\n== {NAME} MEAN-5={statistics.mean(cv):.2f} seeds={comps}")
    sv = [float(snr[sd]) for sd in comps if snr.get(sd) is not None]
    if sv:
        print(f"== 2730 MEAN={statistics.mean(sv):.2f} -> Δ={statistics.mean(cv)-statistics.mean(sv):+.2f}")
    print(f"== champ MEAN={statistics.mean(CHAMP.values()):.2f} -> Δ={statistics.mean(cv)-statistics.mean(CHAMP.values()):+.2f}")
    print("GB_MULTISEED_DONE")


if __name__ == "__main__":
    main()
