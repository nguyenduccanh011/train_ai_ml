"""Build the SINGLE-MODEL ACTION CLASSIFIER (SMAC) line and register it on the leaderboard.

ONE multiclass model decides the per-bar action {OUT, ENTER, HOLD, EXIT} from a single
feature set + single oracle target (action_oracle) — no decoupled entry/exit heads, no
recombine. This directly tests whether a single coherent model beats the decoupled 2-slot
system whose realized PnL couples the heads (the "masking" problem).

Sandbox: mechanical exit rules OFF (no pullback / overext / trailing / gates). The MODEL's
own decision drives entry & exit, so its edge is judged on its own merit — compare the
delta vs the pullback-OFF baselines (n2_abl_nopullback 197.4 / n2_nopullback_red_dcf_ma8
397.8), NOT vs the crutch-loaded champion (704).

Usage: python stock_ml/scripts/build_smac_action.py
"""
from __future__ import annotations

import asyncio
import copy
import json
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

BASE_TMPL = 2429  # champion: reuse its feature set (entry_lvup126_recov), split (gap 85),
#                   universe, costs, and the lightgbm entry component (id 82).
ENTRY_ML_COMPONENT = 82
ENTRY_FEATURE_SET = "entry_lvup126_recov"
NEW_NAME = "n2_smac_action_v1"
STRATEGY = "single_ml_action_classifier"
SEEDS = [42, 7, 99]

# Oracle action labels (zigzag swing segmentation).
ACTION_TARGET = {
    "type": "action_oracle",
    "pct": 0.10,        # 10% reversal = a swing
    "min_fwd_leg": 0.10,  # only ENTER bottoms with a >=10% up-leg ahead
    "min_leg_bars": 3,   # drop sub-3-bar noise legs
}

# Rules-OFF sandbox: clean next-close fill (NO pullback discount), the model's own signal
# exit, a loose hold backstop, and the standard costs. Every optional engine lever is
# omitted → defaults OFF, so nothing mechanical masks the model's decision.
SANDBOX_ENGINE = {
    "costs": {"tax": 0.001, "slippage": 0.0015, "commission": 0.0015},
    "hard_stop_pct": None,
    "max_hold_bars": 40,   # backstop only (model is expected to EXIT at peaks)
    "min_hold_bars": 2,
    "entry_bar_fill_type": "close_next",
    "exit_priority": ["signal"],
    "signal_exit_enabled": True,
}


async def make_template() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"template exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)

        # ONE entry slot only — no exit slot, no ensembles.
        slots = [
            {
                "slot_type": "entry",
                "ml_component_id": ENTRY_ML_COMPONENT,
                "rule_component_id": None,
                "feature_set_name": ENTRY_FEATURE_SET,
                "target_config": copy.deepcopy(ACTION_TARGET),
            }
        ]

        t = await repo.create(
            name=NEW_NAME,
            market=base.market,
            strategy=STRATEGY,
            feature_set_id=base.feature_set_id,
            target_id=base.target_id,
            component_slots=slots,
            direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=None,  # v1: argmax decision (tune P(ENTER) cutoff later)
            exit_threshold=None,   # v1: argmax decision (tune P(EXIT) cutoff later)
            split_config=copy.deepcopy(base.split_config),
            engine_config=copy.deepcopy(SANDBOX_ENGINE),
            validation_config=base.validation_config,
            seed=42,
            description="SMAC: a SINGLE multiclass model decides per-bar action "
                        "{OUT,ENTER,HOLD,EXIT} (action_oracle target, entry_lvup126_recov "
                        "features). No decoupled entry/exit heads, no recombine; rules-off "
                        "sandbox so the model's own decision drives entry & exit.",
            hypothesis="A single coherent model trained on a whole-trade oracle avoids the "
                       "decoupled 2-slot masking (each trade's realized PnL couples the two "
                       "heads, blocking independent improvement). Judge by sandbox delta vs "
                       "the pullback-OFF baselines (197.4 / 397.8), not the crutch champion.",
            universe_slug=base.universe_slug,
            model_mode="ml_only",
        )
        await s.commit()
        print(f"created template id={t.id} name={NEW_NAME} strategy={STRATEGY}")
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
    new_id = asyncio.run(make_template())
    asyncio.run(async_engine.dispose())
    seeds = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=new_id, seed=sd)
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
            print(f"  {NEW_NAME} seed={sd}: NO RESULT (run failed?)", flush=True)
    comps = [v for v in seeds.values() if v is not None]
    if comps:
        mean = statistics.mean(comps)
        std = statistics.pstdev(comps) if len(comps) > 1 else 0.0
        print(f"\n== {NEW_NAME}: MEAN={mean:.1f} std={std:.1f} seeds={seeds}")
        print("== baselines (pullback-OFF sandbox): n2_abl_nopullback=197.4 / "
              "n2_nopullback_red_dcf_ma8=397.8 ; champion(crutch)=704")
    print("BUILD_SMAC_DONE")


if __name__ == "__main__":
    main()
