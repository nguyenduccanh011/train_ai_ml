"""Clone 2409 + regime-adaptive conv vol-gate (entry_pullback_conv_vol_z=1.0, lb=40). Seed-42 A/B
(probe_conv_volgate) held composite (733.6) while cutting mdd 0.213->0.203 and raising pf 4.45->4.62
— a Pareto-on-risk move the free-zone scoring is blind to. Confirm it holds multi-seed (42/7/99).

Usage: python stock_ml/scripts/build_2409_volgate.py
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
BASE_TMPL, NEW_NAME, SEEDS = 2409, "n2_consw20_conv04_volgate", [42, 7, 99]
GATE = {"entry_pullback_conv_vol_z": 1.0, "entry_pullback_conv_vol_lb": 40}


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
            slots.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": sl.feature_set_name,
                    "target_config": tc,
                }
            )
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else dict(eng)
        eng = copy.deepcopy(eng)
        eng.update(GATE)  # the only change
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
            description="2409 + regime-adaptive conv vol-gate (disable conv shallow-fill when 20-bar "
            "vol z>1.0, lb40). Pareto-on-risk: holds comp, mdd 0.213->0.203, pf->4.62.",
            hypothesis="conv-pullback's MDD damage is concentrated in hi-vol bars (probe_conv_regime_sep: "
            "hi-vol conv = -8u pnl / -115u drawdown). Gating it there cuts MDD off the 0.22 cliff.",
            universe_slug=base.universe_slug,
            model_mode=base.model_mode,
        )
        await s.commit()
        print(f"created clone id={t.id} name={NEW_NAME} gate={GATE}")
        return t.id


def read(run_id):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades FROM leaderboard_runs WHERE run_id=%s",
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
        row = read(r.get("run_id"))
        comp = float(row[0]) if row and row[0] is not None else None
        seeds[sd] = comp
        print(
            f"  {NEW_NAME} seed={sd}: comp={comp} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}",
            flush=True,
        )
    comps = [v for v in seeds.values() if v is not None]
    mean = statistics.mean(comps)
    std = statistics.pstdev(comps) if len(comps) > 1 else 0.0
    print(f"\n== {NEW_NAME}: MEAN={mean:.1f} std={std:.1f} seeds={seeds}")
    print(f"== vs 2409 baseline MEAN=730.3 (Δ={mean - 730.3:+.1f}); watch mdd (target <0.213)")
    print("BUILD_VOLGATE_DONE")


if __name__ == "__main__":
    main()
