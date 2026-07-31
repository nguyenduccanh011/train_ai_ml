"""Improvement experiment: clone the multi-seed champion 2409 (n2_consw20_conv04 — conviction
pullback entry, MEAN 730.3) and SWAP only its exit feature set exit_vol_market -> exit_vol_dist2.
Rationale: X-ray showed 2409 exits fire 7.5 bars past peak / capture 0.22; exit_vol_market is
magnitude-only (lags tops), while exit_vol_dist2 (volume-distribution, used by 2378) is the
validated top-predictor (+2.78 robust on the prior line). Stacks 2409's entry edge with 2378's
exit edge. Train seeds 42/7/99, report multi-seed mean vs 730.3.

Usage: python stock_ml/scripts/build_2409_distexit.py
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
BASE_TMPL = 2409
NEW_NAME = "n2_consw20_conv04_distexit"
NEW_EXIT_FEAT = "exit_vol_dist2"
SEEDS = [42, 7, 99]


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            tc = sl.target_config
            tc = json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
            feat = sl.feature_set_name
            if sl.slot_type == "exit":
                feat = NEW_EXIT_FEAT  # the swap
            slots.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": feat,
                    "target_config": tc,
                }
            )
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else eng
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
            engine_config=copy.deepcopy(eng),
            validation_config=base.validation_config,
            seed=base.seed,
            description="2409 conv-pullback entry + exit_vol_dist2 (validated vol-distribution exit). "
            "Stack entry edge (730.3) with 2378's robust top-predicting exit.",
            hypothesis="exit_vol_dist2 predicts tops better than the lagging exit_vol_market -> "
            "tighter capture (X-ray: 2409 exits fire 7.5 bars late, capture 0.22).",
            universe_slug=base.universe_slug,
            model_mode=base.model_mode,
        )
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME} exit_feat={NEW_EXIT_FEAT}")
        return t.id


def read_composite(run_id: str):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades FROM leaderboard_runs "
        "WHERE run_id=%s",
        (run_id,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    new_id = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    seeds = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=new_id, seed=sd)
        row = read_composite(r.get("run_id"))
        comp = float(row[0]) if row and row[0] is not None else None
        seeds[sd] = comp
        print(
            f"  {NEW_NAME} seed={sd}: comp={comp} pnl={row[1]:.1f} pf={row[2]:.2f} "
            f"mdd={row[3]:.3f} tr={row[4]}",
            flush=True,
        )
    comps = [v for v in seeds.values() if v is not None]
    mean = statistics.mean(comps)
    std = statistics.pstdev(comps) if len(comps) > 1 else 0.0
    print(f"\n== {NEW_NAME}: MEAN={mean:.1f} std={std:.1f} seeds={seeds}")
    print(f"== vs 2409 baseline MEAN=730.3 (Δ={mean - 730.3:+.1f})")
    print("BUILD_DISTEXIT_DONE")


if __name__ == "__main__":
    main()
