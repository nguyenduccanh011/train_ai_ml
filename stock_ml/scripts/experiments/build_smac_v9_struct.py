"""SMAC v9 — Stage-1 position-state-aware: add STRUCTURE/state PROXY features to the v8 model.

Stage 1 of the position-state-aware breakthrough (validate the signal before the engine
rewrite). The TA forensic showed the model is blind to: DEPTH (drawdown from recent peak),
AGE (bars since high/low), SWING STRUCTURE (higher-low vs lower-low, support break — the
96%-WR held-support discriminator) and DIVERGENCE. These are batch market-feature PROXIES
for position-state. v9 = v8 target (regime ENTER + confirm06) but with feature set
`entry_struct` (entry_lvup126_recov + those features). If it beats v8 (-33.0), the true
interactive position-state engine (Stage 2) is worth building.

Usage: python stock_ml/scripts/build_smac_v9_struct.py
"""

from __future__ import annotations

import asyncio
import copy
import statistics
import sys
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
BASE_TMPL = 2459
SEEDS = [42, 7, 99]
# v8 best target (regime ENTER + entry confirmation 6%)
ENTRY_TARGET = {
    "type": "action_oracle",
    "pct": 0.10,
    "min_fwd_leg": 0.10,
    "min_leg_bars": 3,
    "entry_min_ret_120": -0.10,
    "entry_confirm_pct": 0.06,
}
NEW_NAME = "n2_smac_v9_struct"
ENTRY_FS = "entry_struct"


async def make_template() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"template exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            if sl.slot_type == "entry":
                tc = dict(ENTRY_TARGET)
                fs = ENTRY_FS
            else:
                tc = copy.deepcopy(sl.target_config)
                fs = sl.feature_set_name
            slots.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": fs,
                    "target_config": tc,
                }
            )
        t = await repo.create(
            name=NEW_NAME,
            market=base.market,
            strategy=base.strategy,
            feature_set_id=base.feature_set_id,
            target_id=base.target_id,
            component_slots=slots,
            direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=None,
            exit_threshold=None,
            split_config=copy.deepcopy(base.split_config),
            engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config,
            seed=42,
            description="SMAC v9: v8 (regime ENTER + confirm06) + entry_struct features "
            "(structure/position-state proxies + divergence) — Stage 1 of "
            "position-state-aware. Tests the TA blind-spot signals.",
            hypothesis="The model is blind to swing structure (higher-low/lower-low, support "
            "break = 96%-WR discriminator), depth/age, and divergence. Feeding these "
            "batch proxies should improve the HOLD/EXIT decision and cut the residual mdd.",
            universe_slug=base.universe_slug,
            model_mode="ml_only",
        )
        await s.commit()
        print(f"created template id={t.id} name={NEW_NAME} fs={ENTRY_FS}")
        return t.id


def read(run_id):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades,wr,avg_hold "
        "FROM leaderboard_runs WHERE run_id=%s",
        (run_id,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    tid = asyncio.run(make_template())
    asyncio.run(async_engine.dispose())
    seeds = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=tid, seed=sd)
        row = read(r.get("run_id")) if r.get("run_id") else None
        comp = float(row[0]) if row and row[0] is not None else None
        seeds[sd] = comp
        if row:
            print(
                f"  {NEW_NAME} seed={sd}: comp={comp} pnl={row[1]:.1f} pf={row[2]:.2f} "
                f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f}",
                flush=True,
            )
        else:
            print(f"  {NEW_NAME} seed={sd}: NO RESULT", flush=True)
    comps = [v for v in seeds.values() if v is not None]
    if comps:
        mean = statistics.mean(comps)
        std = statistics.pstdev(comps) if len(comps) > 1 else 0.0
        print(f"\n== {NEW_NAME}: MEAN={mean:.1f} std={std:.1f} seeds={seeds}")
        print(
            "== vs v8 conf06 -33.0 (pnl+22 pf1.58 mdd0.337 WR0.72) / v5 -46.7 ; baselines 197.4/397.8"
        )
    print("BUILD_SMAC_V9_DONE")


if __name__ == "__main__":
    main()
