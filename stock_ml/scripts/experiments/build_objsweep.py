"""Learner-OBJECTIVE sweep on the frontier x2_struct_to (t3185).

The entry ensemble (primary + ensemble2..6 heads) all train LGBMRegressor with the DEFAULT
L2 objective = regress-to-MEAN forward outcome. Selection is top-K by score, but the goal is
to pick the few names that RUN (upper tail). Test whether a tail/robust objective re-orders the
raw score toward runners — the one CORE-PREDICT lever (the learner itself) never varied.

Clones entry model-component 82 with an added `objective` param; points a cloned t3185 at it.
Baseline (L2) = frontier itself: seed42 comp ~654, NAV 27.14, CAGR 66.0%.
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import psycopg2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository
from stock_ml.scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = 3185
BASE_COMP = 82
BASE_PARAMS = {'n_estimators': 300, 'learning_rate': 0.02, 'num_leaves': 7, 'min_data_in_leaf': 200,
               'feature_fraction': 0.6, 'bagging_fraction': 0.7, 'bagging_freq': 5,
               'lambda_l1': 1.0, 'lambda_l2': 1.0}

# name -> extra params merged onto BASE_PARAMS (objective lever)
VARIANTS = {
    "obj_q70":  {"objective": "quantile", "alpha": 0.70},
    "obj_q80":  {"objective": "quantile", "alpha": 0.80},
    "obj_q90":  {"objective": "quantile", "alpha": 0.90},
    "obj_huber": {"objective": "huber", "alpha": 0.90},
}


def ensure_component(nm, extra):
    """Create (idempotently) a cloned entry model-component with the objective params. Returns id."""
    con = psycopg2.connect(**PG); cur = con.cursor()
    cn = f"entry_plain_{nm}"
    cur.execute("SELECT id FROM model_components WHERE name=%s", (cn,))
    r = cur.fetchone()
    if r:
        con.close(); return r[0]
    params = {**BASE_PARAMS, **extra}
    cur.execute(
        "INSERT INTO model_components (name, role, algorithm, params, description, is_default, "
        "is_active, created_at, updated_at, component_type) "
        "VALUES (%s,'entry','lightgbm',%s,%s,false,true,now(),now(),'ml') RETURNING id",
        (cn, json.dumps(params), f"entry_plain + {extra}"),
    )
    cid = cur.fetchone()[0]; con.commit(); con.close()
    return cid


async def make_all(comp_ids):
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
        for nm, extra in VARIANTS.items():
            ex = await repo.get_by_name(nm)
            if ex:
                print(f"exists {nm} {ex.id}"); ids[nm] = ex.id; continue
            slots = copy.deepcopy(bs)
            for sl in slots:
                if sl["slot_type"] == "entry":
                    sl["ml_component_id"] = comp_ids[nm]
            t = await repo.create(
                name=nm, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
                target_id=base.target_id, component_slots=slots, direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=copy.deepcopy(be),
                validation_config=base.validation_config, seed=base.seed,
                description=f"frontier + entry objective {extra}",
                hypothesis="tail/robust objective re-orders raw score toward runners (dead-year)",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit(); print(f"created {nm} {t.id}"); ids[nm] = t.id
    return ids


def read(rid):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score,total_pnl,trades FROM leaderboard_runs WHERE run_id=%s", (rid,))
    r = cur.fetchone(); con.close(); return r


comp_ids = {nm: ensure_component(nm, extra) for nm, extra in VARIANTS.items()}
print("comp_ids", comp_ids, flush=True)
ids = asyncio.run(make_all(comp_ids)); asyncio.run(async_engine.dispose())
for nm, tid in ids.items():
    try:
        r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id"); row = read(rid)
        print(f"  {nm}(t{tid}): comp={row[0]} pnl={row[1]:.1f} tr={row[2]} run_id={rid}", flush=True)
    except Exception as e:
        print(f"  {nm}: ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)
print("OBJSWEEP_DONE")
