"""CO-ADAPTATION test: is the entry feature-count optimum TARGET-conditional? Under triple_barrier,
48 features >> 11 (seed42 27.14 vs 25.23). Pair 2 ALTERNATIVE primary targets with BOTH feature
counts. If a new target FLIPS the count-response (11 >= 48) or any config beats frontier seed42
(27.14), co-adaptation is real and target-swaps-were-null was a false-null (tested with mismatched
features). seed42 SCREEN (4 configs); promote to 3-seed only if a config approaches/beats 27.14.
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
TARGETS = {
    "mfe": {"type": "mfe_regression", "horizon": 20},
    "fret": {"type": "forward_return_regression", "horizon": 20},
}
FEATS = {"48": "entry_recov_rs", "11": "entry_rs_core11"}
# name -> (target_key, feat_key)
VARIANTS = {f"ca_{tk}_{fk}": (tk, fk) for tk in TARGETS for fk in FEATS}
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
        for nm, (tk, fk) in VARIANTS.items():
            ex = await repo.get_by_name(nm)
            if ex:
                print(f"exists {nm} {ex.id}")
                ids[nm] = ex.id
                continue
            slots = copy.deepcopy(bs)
            for sl in slots:
                if sl["slot_type"] == "entry":
                    sl["target_config"] = copy.deepcopy(TARGETS[tk])
                    sl["feature_set_name"] = FEATS[fk]
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
                description=f"co-adapt: primary target {tk} x {fk}-feat",
                hypothesis="feature-count optimum is target-conditional (co-adaptation)",
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


res = {}
for nm, tid in ids.items():
    try:
        r = run_template_experiment(template_id=tid, seed=42)
        rid = r.get("run_id")
        nav = score_read(rid)
        res[nm] = nav
        print(
            f"{nm} seed42: NAV={nav[0]:.3f} CAGR={nav[1] * 100:.2f}% DD={nav[2] * 100:.2f}%",
            flush=True,
        )
    except Exception as e:
        print(f"ERR {nm}: {type(e).__name__}: {str(e)[:260]}", flush=True)
        res[nm] = None

print("\n=== CO-ADAPTATION seed42 screen (ref: triple_barrier 48=27.14, 11=25.23) ===")
print("target | 48-feat | 11-feat | count-response")
for tk in TARGETS:
    n48 = res.get(f"ca_{tk}_48")
    n11 = res.get(f"ca_{tk}_11")
    v48 = f"{n48[0]:.2f}" if n48 else "ERR"
    v11 = f"{n11[0]:.2f}" if n11 else "ERR"
    resp = ""
    if n48 and n11:
        resp = "11>=48 FLIP!" if n11[0] >= n48[0] else f"48 wins +{n48[0] - n11[0]:.2f}"
    print(f"{tk:5s} | {v48} | {v11} | {resp}")
print("triple| 27.14 | 25.23 | 48 wins +1.91 (baseline)")
print("COADAPT_DONE")
