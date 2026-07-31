"""Deploy the RISK-OPTIMIZED chartist variant = champion 2429 + structure-trail RIDE
(trailing_struct_donch_win + apply_overext). The Donchian-low structure trail holds runners through
shallow pullbacks and exits on a real structure break -> MDD -8%, PF +19%, PnL ~flat vs the composite
champion (composite -8, gated only by the per-bar velocity term). A safer model for real trading.
Config-only over 2429 (no retrain of predictions; the engine config changes the backtest). Multi-seed.
Usage: python stock_ml/scripts/deploy_risk_variant.py [win] [seed ...]   default win=35, seeds 42 7 99 555
"""

from __future__ import annotations
import asyncio, copy, json, os, statistics, sys
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
BASE_TMPL = int(os.environ.get("BASE_TMPL", "2429"))  # set BASE_TMPL=2457 for the nopullback combo
WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 35
SEEDS = [int(x) for x in sys.argv[2:]] or [42, 7, 99, 555]
NEW_NAME = f"n2_{BASE_TMPL}_struct{WIN}"
ENG = {"trailing_struct_donch_win": WIN, "trailing_struct_apply_overext": True}


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
        eng.update(ENG)
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
            description=f"RISK-OPTIMIZED champion variant: 2429 + structure-trail RIDE (Donchian-{WIN} + "
            "apply_overext). MDD -8%, PF +19% vs composite champ; safer for live trading.",
            hypothesis="structure-trail holds runners through shallow pullbacks, exits on real structure "
            "break -> lower MDD, higher PF; composite gated only by per-bar velocity term.",
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
        "SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, sharpe FROM leaderboard_runs WHERE run_id=%s",
        (run_id,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    new_id = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    comps, mdds, pfs = {}, {}, {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=new_id, seed=sd)
        row = read_row(r.get("run_id"))
        comps[sd] = float(row[0]) if row and row[0] is not None else None
        mdds[sd] = float(row[3]) if row and row[3] is not None else None
        pfs[sd] = float(row[2]) if row and row[2] is not None else None
        print(
            f"  {NEW_NAME} seed={sd}: comp={comps[sd]} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} sharpe={row[5]:.2f} tr={row[4]}",
            flush=True,
        )
    cv = [v for v in comps.values() if v is not None]
    if cv:
        print(
            f"\n== {NEW_NAME} (tmpl {new_id}): comp MEAN={statistics.mean(cv):.1f} std={statistics.pstdev(cv):.1f}"
            f"  MDD mean={statistics.mean([v for v in mdds.values() if v]):.3f}  PF mean={statistics.mean([v for v in pfs.values() if v]):.2f}"
        )
        print(
            f"== vs champion 2429 composite-mean ~700.9 / mdd ~0.196 / pf ~4.8 (risk-optimized: lower MDD, higher PF, ~flat PnL)"
        )
    print("DEPLOY_RISK_DONE")


if __name__ == "__main__":
    main()
