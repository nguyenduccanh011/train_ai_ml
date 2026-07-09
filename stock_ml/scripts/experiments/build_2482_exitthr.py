"""Exploit the niche the structure-ride champion (2482) opened: signal-exit now carries 100% of PnL
(the wide Donchian trail disabled the tight %-trail; winners ride to the signal exit). So the EXIT
z-threshold is now load-bearing. Clone 2482 + sweep exit_threshold (champ=2.0; higher=hold longer,
lower=sell sooner), run seed 555, keep only an improver (>720.7), deactivate losers.
Usage: python stock_ml/scripts/build_2482_exitthr.py
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
THRS = [1.75, 2.25, 2.5, 3.0]
CHAMP_555 = 720.7


async def make(thr):
    name = f"n2_2482_xt{str(thr).replace('.','')}"
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
            entry_threshold=b.entry_threshold, exit_threshold=thr,
            split_config=b.split_config, engine_config=eng, validation_config=b.validation_config,
            seed=b.seed, description=f"2482 structure-ride + exit_threshold {thr} (signal-exit is now "
            f"load-bearing 100% of PnL; tune the now-binding exit z-threshold).",
            hypothesis="with the tight %-trail disabled (structure-ride), winners ride to the signal "
            "exit which now carries all PnL -> its z-threshold becomes the binding exit lever.",
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
    results = []
    for thr in THRS:
        tid, name = asyncio.run(make(thr)); asyncio.run(async_engine.dispose())
        r = run_template_experiment(template_id=tid, seed=555); row = rd(r.get("run_id"))
        comp = float(row[0]) if row and row[0] is not None else None
        results.append((thr, tid, name, comp, row))
        print(f"  xt{thr} (id {tid}): comp={comp} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}", flush=True)
    print(f"\n== exit_threshold sweep on 2482 (champ seed555={CHAMP_555}) ==")
    best = max((r for r in results if r[3] is not None), key=lambda z: z[3], default=None)
    for thr, tid, name, comp, row in results:
        verdict = "KEEP" if (comp is not None and comp > CHAMP_555 + 0.5) else "deactivate"
        print(f"  xt{thr}: comp={comp} Δ={comp-CHAMP_555:+.1f} -> {verdict}")
        if verdict == "deactivate":
            deactivate(tid)
    if best and best[3] > CHAMP_555 + 0.5:
        print(f"\n  >> NEW BEST: xt{best[0]} comp={best[3]} (Δ{best[3]-CHAMP_555:+.1f}) — multi-seed next")
    else:
        print(f"\n  >> no exit_threshold beats champ 2482 ({CHAMP_555}); 2.0 stays optimal")
    print("BUILD_EXITTHR_DONE")


if __name__ == "__main__":
    main()
