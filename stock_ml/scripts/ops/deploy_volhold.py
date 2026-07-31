"""Deploy the VOL-ADAPTIVE selective exit-hold champion = structure-ride 2482 + signal_exit_hold_ext_atr
(sell the per-stock ATR-stretch, not the mean) gated on the continuation head (score3 z). Multi-seed
robust +2.25 (all 4 seeds +) on score_summary. Config-only over 2482 (cached preds). Multi-seed leaderboard.
Usage: python stock_ml/scripts/deploy_volhold.py [seed ...]   default 42 7 99 555
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

import os  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = int(os.environ.get("BASE_TMPL", "2482"))
HOLD = {
    "signal_exit_hold_ext_atr": 1.3,
    "signal_exit_hold_ma": int(os.environ.get("HOLD_MA", "20")),
    "signal_exit_hold_min_score3_z": float(os.environ.get("HOLD_Z", "0.4")),
    "signal_exit_hold_profit_floor": float(os.environ.get("HOLD_PF", "0.0")),
}
if os.environ.get("CV"):
    HOLD["signal_exit_skip_if_score3_z"] = float(os.environ["CV"])
if os.environ.get("REL"):
    HOLD["signal_exit_protect_release_drop_k"] = float(os.environ["REL"])
if os.environ.get("RS_SCALE"):
    HOLD["signal_exit_hold_rs_scale"] = float(os.environ["RS_SCALE"])
SEEDS = [int(x) for x in sys.argv[1:]] or [42, 7, 99, 555]
NEW_NAME = os.environ.get("NEW_NAME", "n2_2482_volhold_s3z")


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
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
            for sl in base.component_slots
        ]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        eng.update(HOLD)
        t = await repo.create(
            name=NEW_NAME,
            market=base.market,
            strategy=base.strategy,
            feature_set_id=base.feature_set_id,
            target_id=base.target_id,
            component_slots=copy.deepcopy(slots),
            direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold,
            exit_threshold=base.exit_threshold,
            split_config=base.split_config,
            engine_config=eng,
            validation_config=base.validation_config,
            seed=base.seed,
            description="2482 structure-ride + VOL-ADAPTIVE selective exit-hold: hold an in-profit trending "
            "winner while it is < 1.3*ATR above MA20 AND the continuation head (score3 z) >= 0.4 "
            "— sell the per-stock ATR-stretch into strength, not the mean. Multi-seed +2.25 (all 4 +).",
            hypothesis="the fixed signal-exit fires at the mean (ext -0.4%) yet winners run +2.7 ATR further; "
            "holding the un-stretched, continuation-confirmed winners to their ATR-extension captures "
            "that run (PF+MDD+PnL all up) without the throughput loss the uniform hold incurred.",
            universe_slug=base.universe_slug,
            model_mode=base.model_mode,
        )
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades FROM leaderboard_runs WHERE run_id=%s",
        (run_id,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    new_id = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    comps = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=new_id, seed=sd)
        row = read_row(r.get("run_id"))
        comps[sd] = float(row[0]) if row and row[0] is not None else None
        print(
            f"  {NEW_NAME} seed={sd}: comp={comps[sd]} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}",
            flush=True,
        )
    cv = [v for v in comps.values() if v is not None]
    if cv:
        print(
            f"\n== {NEW_NAME} (tmpl {new_id}): leaderboard MEAN={statistics.mean(cv):.1f} seeds={comps}"
        )
    print("DEPLOY_VOLHOLD_DONE")


if __name__ == "__main__":
    main()
