"""SMAC v2 — state-machine exit policy (pure-ML fix, NO new rule).

Forensic of v1 (n2_smac_action_v1, MEAN -201): the model ENTERS well (WR 62%) but
"cuts winners early / lets losers run" — WIN avg +7.5% held 19d vs LOSS avg -12.8%
held 45d (worst -81%). Root cause is at the ML/label level: the oracle EXIT class =
upside-peak proximity, so a FAILING trade (a knife that never peaks) keeps P(EXIT)
low and is held until a late peak forms. BUT the oracle labels the down-leg as OUT, so
the model already PREDICTS OUT while a losing trade falls — v1's inference simply
ignored it (sold only on argmax==EXIT).

v2 flips ONE inference bit (engine flag smac_exit_on_out): exit the moment the model
leaves the HOLD state, i.e. sell on argmax in {OUT, EXIT}. This uses the model's own
existing prediction to cut losers — no hard_stop, no rule, no relabel. If losers still
bleed after this, the next step is a CUT class in the oracle.

Usage: python stock_ml/scripts/build_smac_v2.py
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
NEW_NAME = "n2_smac_action_v2_exitout"
SEEDS = [42, 7, 99]


async def make_template() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"template exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)

        slots = [
            {
                "slot_type": sl.slot_type,
                "ml_component_id": sl.ml_component_id,
                "rule_component_id": sl.rule_component_id,
                "feature_set_name": sl.feature_set_name,
                "target_config": copy.deepcopy(sl.target_config),
            }
            for sl in base.component_slots
        ]
        # The ONLY change vs v1 is the strategy variant (sell when the model leaves HOLD).
        # The flag rides on the strategy string, NOT engine_config (EngineConfig rejects
        # unknown keys), so the sandbox engine_config is reused verbatim.
        eng = copy.deepcopy(base.engine_config)

        t = await repo.create(
            name=NEW_NAME,
            market=base.market,
            strategy="single_ml_action_classifier_exitout",
            feature_set_id=base.feature_set_id,
            target_id=base.target_id,
            component_slots=slots,
            direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=None,
            exit_threshold=None,
            split_config=copy.deepcopy(base.split_config),
            engine_config=eng,
            validation_config=base.validation_config,
            seed=42,
            description="SMAC v2: state-machine exit — sell when the model LEAVES the HOLD "
            "state (argmax in {OUT,EXIT}), not only on EXIT. Pure-ML fix using "
            "the model's own OUT prediction to cut losers (v1 ignored it). No rule.",
            hypothesis="v1 forensic: losers held 45d to -12.8% because EXIT=upside-peak only; "
            "a failing trade never peaks. The oracle labels down-legs OUT and the "
            "model predicts OUT there, so acting on OUT cuts losers early without a "
            "stop-loss rule (which would mask the ML).",
            universe_slug=base.universe_slug,
            model_mode="ml_only",
        )
        await s.commit()
        print(f"created template id={t.id} name={NEW_NAME} (smac_exit_on_out=True)")
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
        print("== vs SMAC v1 MEAN=-201.0 (mdd 0.518, WR 0.62) ; baselines 197.4/397.8 ; champ 704")
    print("BUILD_SMAC_V2_DONE")


if __name__ == "__main__":
    main()
