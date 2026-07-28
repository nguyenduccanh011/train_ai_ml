"""score4 AMPLITUDE-hold veto (Q2/Q4): hold the position while the mfe/amplitude head (score4)
still predicts a big move remaining. The frontier holds on score3 (continuation) + RS + leg but
NOT on the amplitude head — signal_exit_skip_if_score4_z is OFF. New engine knob. Sweep 3 thresholds
x 3 seeds on t3185. Frontier 3-seed mean = NAV 26.405 / CAGR 65.29%. WIN only if sign-consistent 3/3.
"""
from __future__ import annotations
import asyncio, copy, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import psycopg2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository
from stock_ml.scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
ROOT = Path(__file__).resolve().parents[3]
BASE = 3185
VARIANTS = {"s4h_08": 0.8, "s4h_13": 1.3, "s4h_18": 1.8}   # name -> signal_exit_skip_if_score4_z
env = dict(os.environ, STOCK_DATA_DIR="F:/PROJECTS/train_ai_ml/market_data/market.duckdb",
           NH_NAV2_DIR="F:/PROJECTS/hb2943_work")


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
        be0 = base.engine_config; be0 = json.loads(be0) if isinstance(be0, str) else dict(be0)
        for nm, thr in VARIANTS.items():
            ex = await repo.get_by_name(nm)
            if ex:
                print(f"exists {nm} {ex.id}"); ids[nm] = ex.id; continue
            be = copy.deepcopy(be0); be["signal_exit_skip_if_score4_z"] = thr
            slots = copy.deepcopy(bs)
            t = await repo.create(
                name=nm, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
                target_id=base.target_id, component_slots=slots, direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=be,
                validation_config=base.validation_config, seed=base.seed,
                description=f"frontier + score4 amplitude-hold veto z>={thr}",
                hypothesis="hold while amplitude head predicts big remaining move (Q2 amplitude->exit)",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit(); print(f"created {nm} {t.id}"); ids[nm] = t.id
    return ids


ids = asyncio.run(make_all()); asyncio.run(async_engine.dispose())
con = psycopg2.connect(**PG); cur = con.cursor()


def score_read(rid):
    subprocess.run([str(ROOT / "venv/Scripts/python.exe"),
                    str(ROOT / "stock_ml/scripts/ops/score_nav_leaderboard.py"),
                    "--run-like", rid, "--force"], env=env, cwd=str(ROOT),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    con.commit()
    cur.execute("SELECT nav_adv,cagr_adv,maxdd_nav FROM leaderboard_nav WHERE run_id=%s", (rid,))
    return cur.fetchone()


results = {nm: {} for nm in VARIANTS}
for sd in [42, 7, 99]:
    for nm, tid in ids.items():
        try:
            r = run_template_experiment(template_id=tid, seed=sd)
            rid = r.get("run_id"); nav = score_read(rid)
            results[nm][sd] = nav
            print(f"{nm} seed{sd}: NAV={nav[0]:.3f} CAGR={nav[1]*100:.2f}% DD={nav[2]*100:.2f}%", flush=True)
        except Exception as e:
            print(f"ERR {nm} seed{sd}: {type(e).__name__}: {str(e)[:260]}", flush=True)
            results[nm][sd] = None

FR = {42: 27.14, 7: 26.31, 99: 25.76}
print("\n=== score4 AMPLITUDE-HOLD vs frontier (mean 26.405/65.29%) — sign-consistency 3/3 ===")
for nm in VARIANTS:
    navs = [results[nm][s][0] for s in (42, 7, 99) if results[nm].get(s)]
    if len(navs) < 3:
        print(f"{nm}: incomplete ({len(navs)}/3)"); continue
    deltas = [results[nm][s][0] - FR[s] for s in (42, 7, 99)]
    signs = "".join("+" if d > 0 else "-" for d in deltas)
    print(f"{nm}: navs={[f'{n:.2f}' for n in navs]} mean={sum(navs)/3:.3f} | Δ={[f'{d:+.2f}' for d in deltas]} signs={signs}")
con.close()
print("S4HOLD_DONE")
