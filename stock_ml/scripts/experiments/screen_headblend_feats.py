"""LOOP iter-11 (autonomous): re-open the entry feature-set question with the CORRECT metric. iter-1
judged feature-add by the head's DIRECT composite (masked -> all < recov). But the head's realizable
value is via the iter-8 head-BLEND. So a feature that sharpens the head's forward-peak prediction may
now win when blended. Clone the head-blend champion 2417, swap ONLY the entry feature_set, train, and
compare composite (the head-blend amplifies any real prediction gain). Seed-42 screen; multi-seed the
winners after. Usage: python stock_ml/scripts/screen_headblend_feats.py
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
BASE = 2417
# candidate entry feature sets that add momentum-SHAPE / volume signal the head could use to sharpen
# its forward-peak prediction (now judged via the blend, not direct use).
FEATS = [
    "entry_recov_mp",
    "entry_recov_mp_lean",
    "entry_recov_newsig",
    "entry_lvup126_volstruct",
    "entry_lvup126_recov_shape",
    "entry_recov_eff",
]


async def clone_all():
    """Create ALL clones in ONE event loop (async_engine is loop-bound; asyncio.run per item binds
    it to a closed loop on the 2nd call)."""
    out = {}
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        b = await repo.get_by_id(BASE)
        for fs in FEATS:
            name = f"hb_{fs[:24]}"
            ex = await repo.get_by_name(name)
            if ex:
                out[fs] = ex.id
                continue
            slots = [
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": (fs if sl.slot_type == "entry" else sl.feature_set_name),
                    "target_config": (
                        json.loads(sl.target_config)
                        if isinstance(sl.target_config, str)
                        else copy.deepcopy(sl.target_config)
                    ),
                }
                for sl in b.component_slots
            ]
            eng = b.engine_config
            eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
            t = await repo.create(
                name=name,
                market=b.market,
                strategy=b.strategy,
                feature_set_id=b.feature_set_id,
                target_id=b.target_id,
                component_slots=copy.deepcopy(slots),
                direction=b.direction,
                signal_mode=b.signal_mode,
                signal_threshold=b.signal_threshold,
                entry_threshold=b.entry_threshold,
                exit_threshold=b.exit_threshold,
                split_config=b.split_config,
                engine_config=eng,
                validation_config=b.validation_config,
                seed=b.seed,
                description=f"head-blend champ + entry feature {fs} (iter-11 re-judge via blend).",
                hypothesis="feature-add judged via the head-blend (not masked direct use).",
                universe_slug=b.universe_slug,
                model_mode=b.model_mode,
            )
            out[fs] = t.id
        await s.commit()
    await async_engine.dispose()
    return out


def comp(rid):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score,mdd_per_symbol FROM leaderboard_runs WHERE run_id=%s", (rid,)
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    print(f"baseline 2417 (entry_lvup126_recov) head-blend = 688.6 multi-seed\n")
    ids = asyncio.run(clone_all())
    for fs in FEATS:
        r = run_template_experiment(template_id=ids[fs], seed=42)
        c = comp(r.get("run_id"))
        print(
            f"  {fs:30} seed42 comp={c[0]:.1f} mdd={c[1]:.3f}  (vs recov seed42 691.3)", flush=True
        )
    print("SCREEN_DONE")


if __name__ == "__main__":
    main()
