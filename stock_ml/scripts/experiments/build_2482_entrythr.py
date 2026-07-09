"""PnL-frame (user accepts a bit more MDD): the genuine PnL lever = LOOSEN the entry z-threshold
(champ -1.9) so more buy signals pass -> more dip-buy entries (net-positive edge) -> +PnL at +MDD.
entry_threshold is baked into the cached signals, so this needs a retrain. Clone the structure-ride
champion 2482 + sweep entry_threshold, run seed 555, report PnL/MDD (not composite). Keep if PnL up.
Usage: python stock_ml/scripts/build_2482_entrythr.py
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = 2482
THRS = [-2.2, -2.6, -3.0]
CHAMP = dict(comp=720.7, pnl=125.4, mdd=0.179)


async def make(thr):
    name = f"n2_2482_et{str(thr).replace('.','').replace('-','m')}"
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            return ex.id, name
        b = await repo.get_by_id(BASE)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))} for sl in b.component_slots]
        eng = b.engine_config; eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        t = await repo.create(
            name=name, market=b.market, strategy=b.strategy, feature_set_id=b.feature_set_id,
            target_id=b.target_id, component_slots=copy.deepcopy(slots), direction=b.direction,
            signal_mode=b.signal_mode, signal_threshold=b.signal_threshold,
            entry_threshold=thr, exit_threshold=b.exit_threshold,
            split_config=b.split_config, engine_config=eng, validation_config=b.validation_config,
            seed=b.seed, description=f"2482 structure-ride + LOOSER entry_threshold {thr} (more dip-buy "
            f"entries for +PnL, accepting a bit more MDD - PnL-frame).",
            hypothesis="looser entry z lets more net-positive dip-buys through -> +PnL at +MDD.",
            universe_slug=b.universe_slug, model_mode=b.model_mode)
        await s.commit(); return t.id, name


def rd(rid):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades FROM leaderboard_runs WHERE run_id=%s", (rid,))
    r = cur.fetchone(); con.close(); return r


def deactivate(tid):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("UPDATE leaderboard_runs SET superseded=true WHERE template_id=%s", (tid,))
    cur.execute("UPDATE strategy_templates SET is_active=false WHERE id=%s", (tid,))
    con.commit(); con.close()


def main():
    res = []
    for thr in THRS:
        tid, name = asyncio.run(make(thr)); asyncio.run(async_engine.dispose())
        r = run_template_experiment(template_id=tid, seed=555); row = rd(r.get("run_id"))
        res.append((thr, tid, name, row))
        print(f"  et{thr} (id {tid}): comp={row[0]} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}", flush=True)
    print(f"\n== entry_threshold sweep on 2482 (champ pnl={CHAMP['pnl']} mdd={CHAMP['mdd']} comp={CHAMP['comp']}) ==")
    for thr, tid, name, row in res:
        dpnl = float(row[1]) - CHAMP['pnl']; dmdd = float(row[3]) - CHAMP['mdd']; dcomp = float(row[0]) - CHAMP['comp']
        keep = dpnl > 2.0  # meaningful PnL gain (user accepts MDD)
        print(f"  et{thr}: Δpnl={dpnl:+.1f} Δmdd={dmdd:+.3f} Δcomp={dcomp:+.1f} -> {'KEEP (PnL-frame)' if keep else 'deactivate'}")
        if not keep:
            deactivate(tid)
    print("BUILD_ENTRYTHR_DONE")


if __name__ == "__main__":
    main()
