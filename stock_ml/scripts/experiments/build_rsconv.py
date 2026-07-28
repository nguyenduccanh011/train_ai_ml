"""BREAK-THROUGH attempt 2 — RS x conviction-timing. Turn ON entry_pullback_conv_rs_w (default 0.0)
on the double-RS frontier (3115): RS-vs-VNINDEX feeds the pullback-depth conviction, so leaders
outperforming the index get a SHALLOWER pullback (caught before they run) — entry-TIMING edge (not
sizing). Complements the RS SELECTION (cross-sectional feature ranks). Engine-only -> cache-hit.
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
BASE_TMPL = 3115  # double-RS
SEEDS = [42]
VARIANTS = [
    ("rc_rs15", {"entry_pullback_conv_rs_w": 0.15}),
    ("rc_rs25", {"entry_pullback_conv_rs_w": 0.25}),
    ("rc_rs35", {"entry_pullback_conv_rs_w": 0.35}),
    ("rc_rs25_cap12", {"entry_pullback_conv_rs_w": 0.25, "entry_pullback_conv_rs_cap": 0.12}),
]


async def make_all() -> dict:
    ids = {}
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_TMPL)
        base_slots = []
        for sl in base.component_slots:
            tc = sl.target_config
            tc = json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
            base_slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                               "rule_component_id": sl.rule_component_id,
                               "feature_set_name": sl.feature_set_name, "target_config": tc})
        base_eng = base.engine_config
        base_eng = json.loads(base_eng) if isinstance(base_eng, str) else dict(base_eng)
        for new_name, ov in VARIANTS:
            ex = await repo.get_by_name(new_name)
            if ex:
                print(f"exists {new_name} id={ex.id}"); ids[new_name] = ex.id; continue
            eng = copy.deepcopy(base_eng); eng.update(ov)
            t = await repo.create(
                name=new_name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(base_slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"double-RS + RS-conviction timing {ov} (RS-vs-VNINDEX -> shallower pullback for leaders).",
                hypothesis="RS selection (features) + RS timing (conv pullback depth) compounds.",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit()
            print(f"created {new_name} id={t.id} ov={ov}")
            ids[new_name] = t.id
    return ids


def read(run_id):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score,total_pnl,avg_pnl,trades FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    ids = asyncio.run(make_all())
    asyncio.run(async_engine.dispose())
    for name, tid in ids.items():
        try:
            r = run_template_experiment(template_id=tid, seed=SEEDS[0])
            rid = r.get("run_id"); row = read(rid)
            print(f"  {name}(t{tid}): comp={row[0]} pnl={row[1]:.1f} avg={row[2]:.4f} tr={row[3]} run_id={rid}" if row
                  else f"  {name}: NO ROW {rid}", flush=True)
        except Exception as ex:
            print(f"  {name}: ERROR {type(ex).__name__}: {str(ex)[:200]}", flush=True)
    print("BUILD_RSCONV_DONE")


if __name__ == "__main__":
    main()
