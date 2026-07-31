"""SMAC v3 — attack the ENTRY weakness (pure-ML: better entry features, NO rule).

v1 forensic: the worst losers are 2022 bear-market KNIFE-CATCHES (NVL -81%, all entered
into a falling tape). The recov feature set is blind to the falling-knife / structural-
decliner profile and to the market regime. Fix at the SOURCE (don't enter the bad trade)
by giving the SINGLE action model richer ENTER context — same v1 exit policy, only the
entry slot's feature set changes (clean A/B).

Variants (each clones v1 tmpl 2459, swaps feature_set_name only, multi-seed):
  v3a entry_recov_knife      = recov + dist_63d_high + sma_200_ratio (per-stock knife axis)
  v3b entry_recov_knife_mkt  = v3a + market_trend + market_volatility_regime (tape regime)

Usage: python stock_ml/scripts/build_smac_v3_entry.py
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
BASE_TMPL = 2459  # n2_smac_action_v1
SEEDS = [42, 7, 99]
VARIANTS = {
    "n2_smac_v3a_knife": "entry_recov_knife",
    "n2_smac_v3b_knife_mkt": "entry_recov_knife_mkt",
}


async def make_template(new_name: str, feature_set: str) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(new_name)
        if ex:
            print(f"template exists: id={ex.id} ({new_name})")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            fs = feature_set if sl.slot_type == "entry" else sl.feature_set_name
            slots.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": fs,
                    "target_config": copy.deepcopy(sl.target_config),
                }
            )
        t = await repo.create(
            name=new_name,
            market=base.market,
            strategy=base.strategy,  # single_ml_action_classifier (v1 exit policy)
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
            description=f"SMAC v3 entry-fix: action model with ENTER feature set '{feature_set}' "
            "(knife/structural-decliner +/- market regime). Same v1 exit policy; "
            "addresses 2022 bear knife-catches at the source. Pure-ML, no rule.",
            hypothesis="v1's worst losers are bear-market knife entries; recov is blind to the "
            "falling-knife profile + tape regime. Richer ENTER context lets the single "
            "model avoid those buys natively (vs an exit rule that would mask the ML).",
            universe_slug=base.universe_slug,
            model_mode="ml_only",
        )
        await s.commit()
        print(f"created template id={t.id} name={new_name} entry_fs={feature_set}")
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
    for new_name, fs in VARIANTS.items():
        tid = asyncio.run(make_template(new_name, fs))
        asyncio.run(async_engine.dispose())
        seeds = {}
        for sd in SEEDS:
            r = run_template_experiment(template_id=tid, seed=sd)
            row = read(r.get("run_id")) if r.get("run_id") else None
            comp = float(row[0]) if row and row[0] is not None else None
            seeds[sd] = comp
            if row:
                print(
                    f"  {new_name} seed={sd}: comp={comp} pnl={row[1]:.1f} pf={row[2]:.2f} "
                    f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f}",
                    flush=True,
                )
            else:
                print(f"  {new_name} seed={sd}: NO RESULT (run failed?)", flush=True)
        comps = [v for v in seeds.values() if v is not None]
        if comps:
            mean = statistics.mean(comps)
            std = statistics.pstdev(comps) if len(comps) > 1 else 0.0
            print(f"== {new_name} ({fs}): MEAN={mean:.1f} std={std:.1f} seeds={seeds}\n")
    print("== vs SMAC v1 (entry_lvup126_recov) MEAN=-201.0 mdd0.518 WR0.62 ; baselines 197.4/397.8")
    print("BUILD_SMAC_V3_DONE")


if __name__ == "__main__":
    main()
