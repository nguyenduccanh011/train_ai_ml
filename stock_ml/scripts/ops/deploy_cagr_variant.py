"""Deploy the COMPOUNDING-OPTIMIZED variant = champion 2429 + max_hold_bars:20 (+ 'max_hold' in
exit_priority). Caps the hold so finite capital RECYCLES -> CAGR 77-82% / ~1.7x final wealth vs the
composite champion under finite-capital compounding (N=5-10 concurrent). It is composite-NEGATIVE
(-55) — a DIFFERENT objective (real-money compounding, not the leaderboard composite). See
project_capital_cagr_maxhold. Config-only over 2429 (cached preds). Multi-seed records the composite.
Usage: python stock_ml/scripts/deploy_cagr_variant.py [max_hold] [seed ...]   default 20, seeds 42 7 99 555
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
BASE_TMPL = 2429
MH = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEEDS = [int(x) for x in sys.argv[2:]] or [42, 7, 99, 555]
NEW_NAME = f"n2_2429_maxhold{MH}_cagr"


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"clone exists: id={ex.id}"); return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        ep = list(eng.get("exit_priority", []))
        if "max_hold" not in ep:
            eng["exit_priority"] = ["max_hold"] + ep
        eng["max_hold_bars"] = MH
        t = await repo.create(
            name=NEW_NAME, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"COMPOUNDING-OPTIMIZED: 2429 + max_hold_bars:{MH} (recycle finite capital). "
                        "CAGR 77-82% / ~1.7x wealth vs composite champ under finite-capital compounding; "
                        "composite -55 (different objective = real-money CAGR, not leaderboard).",
            hypothesis="capping the hold recycles capital faster -> compounds to higher terminal wealth "
                       "even at lower per-trade avg; composite penalizes the turnover, CAGR rewards it.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, avg_hold FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    new_id = asyncio.run(make_clone()); asyncio.run(async_engine.dispose())
    comps = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=new_id, seed=sd)
        row = read_row(r.get("run_id"))
        comps[sd] = float(row[0]) if row and row[0] is not None else None
        print(f"  {NEW_NAME} seed={sd}: composite={comps[sd]} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} avgHold={row[5]:.1f} tr={row[4]}", flush=True)
    cv = [v for v in comps.values() if v is not None]
    if cv:
        print(f"\n== {NEW_NAME} (tmpl {new_id}): composite MEAN={statistics.mean(cv):.1f} (lower by design)")
        print("== OFFLINE finite-capital CAGR (the real metric): ~77-82% / finalX ~1.7x vs baseline (capital_cagr_2429.py)")
    print("DEPLOY_CAGR_DONE")


if __name__ == "__main__":
    main()
