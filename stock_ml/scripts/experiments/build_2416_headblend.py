"""Loop iter-8 champion: clone 2416 (vol-gate + combo entry modulator) + rpos-heavy combo weights
+ ML head-z blend 0.5 (the masked forward-peak signal, channelled through the entry modulator).
3-seed A/B = 688.6 (mdd 0.199), +11.5 over 2409 baseline / +9.5 over vol-gate. Confirm as a trained
leaderboard template. Usage: python stock_ml/scripts/build_2416_headblend.py
"""

from __future__ import annotations
import asyncio, copy, json, statistics, sys
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
BASE, NEW_NAME, SEEDS = 2416, "n2_consw20_conv04_vg_combo_hb", [42, 7, 99]
OV = {"entry_pullback_conv_combo_w": [0.30, 0.25, 0.20, 0.25], "entry_pullback_conv_head_w": 0.50}


async def make():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"exists {ex.id}")
            return ex.id
        b = await repo.get_by_id(BASE)
        slots = [
            {
                "slot_type": sl.slot_type,
                "ml_component_id": sl.ml_component_id,
                "rule_component_id": sl.rule_component_id,
                "feature_set_name": sl.feature_set_name,
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
        eng.update(OV)
        t = await repo.create(
            name=NEW_NAME,
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
            description="2416 + rpos-heavy combo weights + ML head-z blend 0.5: the masked "
            "forward-peak head prediction channelled through the conv entry modulator (price+head 50/50).",
            hypothesis="the ML head's forward-peak signal is realizable as a graded MODULATOR of the "
            "entry rule (which runners to shallow-fill), not via its own entry/exit decision (masked).",
            universe_slug=b.universe_slug,
            model_mode=b.model_mode,
        )
        await s.commit()
        print(f"created {t.id}")
        return t.id


def rd(rid):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades FROM leaderboard_runs WHERE run_id=%s",
        (rid,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    nid = asyncio.run(make())
    asyncio.run(async_engine.dispose())
    seeds = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=nid, seed=sd)
        row = rd(r.get("run_id"))
        seeds[sd] = float(row[0]) if row and row[0] is not None else None
        print(
            f"  seed {sd}: comp={seeds[sd]} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}",
            flush=True,
        )
    cs = [v for v in seeds.values() if v is not None]
    print(
        f"\n== {NEW_NAME}: MEAN={statistics.mean(cs):.1f} std={statistics.pstdev(cs):.1f} seeds={seeds}"
    )
    print(f"== vs 2416 combo MEAN=680.9 (Δ={statistics.mean(cs) - 680.9:+.1f}); vs 2409 base 677.1")
    print("BUILD_HEADBLEND_DONE")


if __name__ == "__main__":
    main()
