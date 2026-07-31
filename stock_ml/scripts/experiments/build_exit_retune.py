"""BREAK-THROUGH attempt — re-tune the EXIT ENGINE on the double-RS entry (3115). The exit config
(max_hold14, rgskip MA300, overext_trail0.03, snr) was co-optimized with the OLD non-RS entry; the
RS entry selects a different trade population (leaders) whose optimal hold/skip may differ. Engine-only
knobs -> predictions cache-hit, backtest re-runs (fast). NAV-score via --run-like er_.
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
VARIANTS = [  # (name, engine_config override)
    ("er_mh12", {"max_hold_bars": 12}),
    ("er_mh16", {"max_hold_bars": 16}),
    ("er_mh18", {"max_hold_bars": 18}),
    ("er_mh20", {"max_hold_bars": 20}),
    ("er_skip250", {"signal_exit_skip_if_mkt_above_ma": 250}),
    ("er_skip350", {"signal_exit_skip_if_mkt_above_ma": 350}),
    ("er_oxt04", {"overext_trail_pct": 0.04}),
    ("er_mh16_skip350", {"max_hold_bars": 16, "signal_exit_skip_if_mkt_above_ma": 350}),
    ("er_mh16_oxt04", {"max_hold_bars": 16, "overext_trail_pct": 0.04}),
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
            base_slots.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": sl.feature_set_name,
                    "target_config": tc,
                }
            )
        base_eng = base.engine_config
        base_eng = json.loads(base_eng) if isinstance(base_eng, str) else dict(base_eng)
        for new_name, ov in VARIANTS:
            ex = await repo.get_by_name(new_name)
            if ex:
                print(f"exists {new_name} id={ex.id}")
                ids[new_name] = ex.id
                continue
            eng = copy.deepcopy(base_eng)
            eng.update(ov)
            t = await repo.create(
                name=new_name,
                market=base.market,
                strategy=base.strategy,
                feature_set_id=base.feature_set_id,
                target_id=base.target_id,
                component_slots=copy.deepcopy(base_slots),
                direction=base.direction,
                signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=eng,
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"double-RS + exit re-tune {ov} (break-through: co-optimize exit with RS entry).",
                hypothesis="RS entry selects leaders -> optimal max_hold/rgskip/overext may differ from old base.",
                universe_slug=base.universe_slug,
                model_mode=base.model_mode,
            )
            await s.commit()
            print(f"created {new_name} id={t.id} ov={ov}")
            ids[new_name] = t.id
    return ids


def read(run_id):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score,total_pnl,avg_pnl,trades FROM leaderboard_runs WHERE run_id=%s",
        (run_id,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    ids = asyncio.run(make_all())
    asyncio.run(async_engine.dispose())
    for name, tid in ids.items():
        try:
            r = run_template_experiment(template_id=tid, seed=SEEDS[0])
            rid = r.get("run_id")
            row = read(rid)
            print(
                f"  {name}(t{tid}): comp={row[0]} pnl={row[1]:.1f} avg={row[2]:.4f} tr={row[3]} run_id={rid}"
                if row
                else f"  {name}: NO ROW {rid}",
                flush=True,
            )
        except Exception as ex:
            print(f"  {name}: ERROR {type(ex).__name__}: {str(ex)[:200]}", flush=True)
    print("BUILD_EXIT_RETUNE_DONE")


if __name__ == "__main__":
    main()
