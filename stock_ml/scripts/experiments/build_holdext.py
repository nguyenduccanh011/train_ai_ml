"""ENGINE test: conviction-conditional SIGNAL-defer. holdext_diag showed the 'signal' exit cuts
hi-conviction (extended) winners that keep running (+0.45/+0.57% fwd 5/10) while correctly cutting
lo-conviction losers (-1.06/-1.12%). New engine knob signal_exit_skip_if_entry_distma20 (winner_only)
defers the signal exit for positions whose ENTRY-bar dist_ma20 >= threshold. Clone t3185 with the knob
at a few thresholds, run 3-seed, score board NAV (K=25 shuffle) + f22. WIN = beats frontier 3/3.
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path
sys.path.insert(0, "F:/PROJECTS/hb2943_work")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import psycopg2, pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from nh_nav2 import NavSim2, shuffle_stats
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository
from stock_ml.scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
WORK = Path("F:/PROJECTS/hb2943_work/navboard"); WORK.mkdir(exist_ok=True)
BASE = 3185; SEEDS = [42, 7, 99]

# name -> engine_config overrides (winner_only defer of the signal exit by entry dist_ma20)
VARIANTS = {
    "he_d04w": {"signal_exit_skip_if_entry_distma20": 0.04, "signal_exit_skip_if_entry_distma20_winner_only": True},
    "he_d08w": {"signal_exit_skip_if_entry_distma20": 0.08, "signal_exit_skip_if_entry_distma20_winner_only": True},
}


async def make_all():
    ids = {}
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s); base = await repo.get_by_id(BASE)
        bs = []
        for sl in base.component_slots:
            tc = sl.target_config; tc = json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
            bs.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                       "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                       "target_config": tc})
        be = base.engine_config; be = json.loads(be) if isinstance(be, str) else dict(be)
        for nm, ov in VARIANTS.items():
            ex = await repo.get_by_name(nm)
            if ex:
                print(f"exists {nm} {ex.id}", flush=True); ids[nm] = ex.id; continue
            ec = copy.deepcopy(be); ec.update(ov)
            t = await repo.create(
                name=nm, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
                target_id=base.target_id, component_slots=copy.deepcopy(bs), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=ec,
                validation_config=base.validation_config, seed=base.seed,
                description=f"frontier + conviction signal-defer {ov}",
                hypothesis="hold hi-conviction (entry dist_ma20) winners through the signal exit",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit(); print(f"created {nm} {t.id}", flush=True); ids[nm] = t.id
    return ids


def nav_of(rid, con):
    tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price FROM run_trades "
                     "WHERE run_id=%s AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
                     "AND exit_price IS NOT NULL", con, params=(rid,))
    csv = WORK / f"_he_{rid.replace('/', '_')}.csv"; tr.to_csv(csv, index=False)
    full = shuffle_stats(NavSim2(str(csv), "2020-01-01"), K=25, roundtrip=0.006, advance_fee=0.0008, n=20)
    f22 = shuffle_stats(NavSim2(str(csv), "2022-01-01"), K=25, roundtrip=0.006, advance_fee=0.0008, n=20)
    return full["mean"], full["dd_mean"], f22["mean"], len(tr)


ids = asyncio.run(make_all()); asyncio.run(async_engine.dispose())
con = psycopg2.connect(**PG)
res = {"base": {}, **{nm: {} for nm in VARIANTS}}
todo = [("base", BASE)] + [(nm, ids[nm]) for nm in VARIANTS]
for sd in SEEDS:
    for nm, tid in todo:
        try:
            r = run_template_experiment(template_id=tid, seed=sd); rid = r.get("run_id")
            nav, dd, f22, nt = nav_of(rid, con)
            res[nm][sd] = (nav, dd, f22, nt)
            print(f"seed{sd} {nm}(t{tid}): NAV={nav:.2f} DD={dd*100:.1f}% f22={f22:.2f} tr={nt}", flush=True)
        except Exception as e:
            print(f"seed{sd} {nm}: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
con.close()

print("\n=== conviction signal-defer vs frontier (board K25, sign 3/3) ===")
base = res["base"]
for nm in VARIANTS:
    navs = [res[nm][s][0] for s in SEEDS if s in res[nm]]
    d = [res[nm][s][0] - base[s][0] for s in SEEDS if s in res[nm] and s in base]
    df = [res[nm][s][2] - base[s][2] for s in SEEDS if s in res[nm] and s in base]
    signs = "".join("+" if x > 0 else "-" for x in d)
    signs22 = "".join("+" if x > 0 else "-" for x in df)
    if navs:
        print(f"{nm}: NAVmean={sum(navs)/len(navs):.3f} Δ={[f'{x:+.2f}' for x in d]} {signs} | f22Δ={[f'{x:+.2f}' for x in df]} {signs22}")
bn = [base[s][0] for s in SEEDS if s in base]
print(f"base : NAVmean={sum(bn)/len(bn):.3f}")
print("HOLDEXT_DONE")
