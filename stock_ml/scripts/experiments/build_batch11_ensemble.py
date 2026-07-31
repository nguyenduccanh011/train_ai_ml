"""BROAD batch 11 — ENSEMBLE HEAD recomposition (never touched this campaign). The 4 union entry
heads = reversal(ens) / continuation(ens2) / mfe(ens3) / forward_penalized(ens4). Changing WHICH
targets they train on changes the union entry composition. On double-RS frontier (3115). Multi-seed
only seed-42 winners (noise discipline). NAV via --run-like eh_.
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
BASE_TMPL = 3115
SEEDS = [42]
# (name, {ensemble_key: new_target_dict}) — replace that head's target, keep its z_threshold
VARIANTS = [
    ("eh_rr4", {"entry_ensemble4": {"type": "reward_risk_regression", "horizon": 20}}),
    ("eh_swing4", {"entry_ensemble4": {"type": "swing_value_regression", "horizon": 20}}),
    (
        "eh_bottom1",
        {
            "entry_ensemble": {
                "type": "bottom_structure_entry_regression",
                "horizon": 10,
                "park_window": 20,
            }
        },
    ),
    ("eh_multih2", {"entry_ensemble2": {"type": "multi_horizon_return", "horizons": [5, 10, 20]}}),
    (
        "eh_add5dl",
        {"entry_ensemble5": {"type": "downleg_depth_regression", "max_span": 40}},
    ),  # NEW 5th head
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
            for ekey, tgt in ov.items():
                d = dict(eng.get(ekey) or {})
                d["target"] = tgt
                d.setdefault("z_threshold", 0.7)
                eng[ekey] = d
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
                description=f"double-RS + ensemble recompose {ov} (batch11 union-head composition).",
                hypothesis="different union entry-head targets change the entry composition / add fresh winners.",
                universe_slug=base.universe_slug,
                model_mode=base.model_mode,
            )
            await s.commit()
            print(f"created {new_name} id={t.id} ov={list(ov.keys())}")
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
    print("BUILD_BATCH11_DONE")


if __name__ == "__main__":
    main()
