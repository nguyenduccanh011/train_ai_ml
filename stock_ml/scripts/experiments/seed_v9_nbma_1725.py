"""v9: refine the nonbull_ma_win breakthrough. Base = 1725 (no_incubate + nonbull_ma_win=40,
multiseed 416.0). v8 showed faster regime MA (40<50) helps, slower (60) hurts -> sweep finer
around 40 (30/35/45) to find the optimum, plus combine with a faster/shorter inner belowma
and re-test sl10/div08 on this newest base. Multi-seeded directly.
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402

BASE = 1725  # n2_v8_nb_ma40
TB = lambda pt, sl, h: {"type": "triple_barrier", "horizon": h, "pt": pt, "sl": sl, "direction": "long"}
GRID = [
    ("n2_v9_ma30", None, {"nonbull_ma_win": 30}),
    ("n2_v9_ma35", None, {"nonbull_ma_win": 35}),
    ("n2_v9_ma45", None, {"nonbull_ma_win": 45}),
    ("n2_v9_ma40_p2", None, {"exit_force_gate_nonbull": "belowma20p2"}),
    ("n2_v9_ma40_nb15", None, {"exit_force_gate_nonbull": "belowma15p3"}),
    ("n2_v9_ma40_div08", None, {"overext_pct": 0.08, "overext_reversal_mode": "bear_div", "overext_div_threshold": 0.30}),
    ("n2_v9_ma40_sl10", TB(0.15, 0.10, 30), None),
    ("n2_v9_ma35_p2", None, {"nonbull_ma_win": 35, "exit_force_gate_nonbull": "belowma20p2"}),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE)
        es = next(s for s in base.component_slots if s.slot_type == "entry")
        xs = next(s for s in base.component_slots if s.slot_type == "exit")

        def tc(sl):
            t = sl.target_config
            return json.loads(t) if isinstance(t, str) else copy.deepcopy(t)

        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else be
        created = []
        for name, etgt, eng_ov in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(be)
            if eng_ov:
                eng.update(eng_ov)
            slots = [
                {"slot_type": "entry", "ml_component_id": es.ml_component_id, "rule_component_id": None,
                 "feature_set_name": es.feature_set_name, "target_config": copy.deepcopy(etgt) if etgt else tc(es)},
                {"slot_type": "exit", "ml_component_id": xs.ml_component_id, "rule_component_id": None,
                 "feature_set_name": xs.feature_set_name, "target_config": tc(xs)},
            ]
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"v9 nonbull-MA refine entry={etgt} eng={eng_ov} on base {BASE}.",
                hypothesis="Find nonbull_ma_win optimum + combos -> push past 416.0 robust.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
