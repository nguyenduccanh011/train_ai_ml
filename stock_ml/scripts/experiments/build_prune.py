"""FEATURE-CROWDING masker test: entry head = 48 features but num_leaves=7 + feature_fraction=0.6
can only use ~6/tree -> new features get crowded out (why "everything dilutes"). Prune 48 -> 11 / 18
core features. If pruned matches/beats the full 48, crowding is a real masker and PRUNING is the lever.
2 arms x 3 seeds on t3185. Frontier 3-seed mean = NAV 26.405 / CAGR 65.29%.
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
VARIANTS = {"ep_core11": "entry_rs_core11", "ep_core18": "entry_rs_core18"}
env = dict(
    os.environ,
    STOCK_DATA_DIR="F:/PROJECTS/train_ai_ml/market_data/market.duckdb",
    NH_NAV2_DIR="F:/PROJECTS/hb2943_work",
)


async def make_all():
    ids = {}
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE)
        bs = []
        for sl in base.component_slots:
            tc = sl.target_config
            tc = json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
            bs.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": sl.feature_set_name,
                    "target_config": tc,
                }
            )
        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else dict(be)
        for nm, fs in VARIANTS.items():
            ex = await repo.get_by_name(nm)
            if ex:
                print(f"exists {nm} {ex.id}")
                ids[nm] = ex.id
                continue
            slots = copy.deepcopy(bs)
            for sl in slots:
                if sl["slot_type"] == "entry":
                    sl["feature_set_name"] = fs
            t = await repo.create(
                name=nm,
                market=base.market,
                strategy=base.strategy,
                feature_set_id=base.feature_set_id,
                target_id=base.target_id,
                component_slots=slots,
                direction=base.direction,
                signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=copy.deepcopy(be),
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"frontier + PRUNED entry feature set {fs} (crowding test)",
                hypothesis="pruning 48->core removes noise the shallow tree wastes splits on",
                universe_slug=base.universe_slug,
                model_mode=base.model_mode,
            )
            await s.commit()
            print(f"created {nm} {t.id}")
            ids[nm] = t.id
    return ids


ids = asyncio.run(make_all())
asyncio.run(async_engine.dispose())
con = psycopg2.connect(**PG)
cur = con.cursor()


def score_read(rid):
    subprocess.run(
        [
            str(ROOT / "venv/Scripts/python.exe"),
            str(ROOT / "stock_ml/scripts/ops/score_nav_leaderboard.py"),
            "--run-like",
            rid,
            "--force",
        ],
        env=env,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    con.commit()
    cur.execute("SELECT nav_adv,cagr_adv,maxdd_nav FROM leaderboard_nav WHERE run_id=%s", (rid,))
    return cur.fetchone()


results = {nm: {} for nm in VARIANTS}
for sd in [42, 7, 99]:
    for nm, tid in ids.items():
        try:
            r = run_template_experiment(template_id=tid, seed=sd)
            rid = r.get("run_id")
            nav = score_read(rid)
            results[nm][sd] = nav
            print(
                f"{nm} seed{sd}: NAV={nav[0]:.3f} CAGR={nav[1] * 100:.2f}% DD={nav[2] * 100:.2f}%",
                flush=True,
            )
        except Exception as e:
            print(f"ERR {nm} seed{sd}: {type(e).__name__}: {str(e)[:260]}", flush=True)
            results[nm][sd] = None

FR = {42: 27.14, 7: 26.31, 99: 25.76}
print("\n=== PRUNED entry (crowding test) vs frontier 48-feat (mean 26.405/65.29%) ===")
for nm in VARIANTS:
    navs = [results[nm][s][0] for s in (42, 7, 99) if results[nm].get(s)]
    if len(navs) < 3:
        print(f"{nm}: incomplete ({len(navs)}/3)")
        continue
    deltas = [results[nm][s][0] - FR[s] for s in (42, 7, 99)]
    signs = "".join("+" if d > 0 else "-" for d in deltas)
    print(
        f"{nm}: navs={[f'{n:.2f}' for n in navs]} mean={sum(navs) / 3:.3f} | Δ={[f'{d:+.2f}' for d in deltas]} signs={signs}"
    )
con.close()
print("PRUNE_DONE")
