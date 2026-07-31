"""Directional-only exit head on t1378 — the subtractive root-cause fix.

Proof (exit_head_shape_cmp.py + feature top/bottom separation): the exit head fires at
BOTTOMS not tops because the vol-magnitude features are direction-symmetric (~0.01σ).
Drop them; keep only directional features (extension/dist-high/macd/slope/divergence).
Verify at the PREDICTION level whether exit_z now spikes at tops.
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

BASE_ID = 1378
# (tag, exit_feature_set, exit_target_override_or_None)
GRID = [
    ("xdir_rr", "exit_directional", None),
    ("xdir_fdd", "exit_directional", {"type": "forward_drawdown_regression", "horizon": 10}),
    ("xdirmkt_rr", "exit_directional_mkt", None),
    ("xdirmkt_fdd", "exit_directional_mkt", {"type": "forward_drawdown_regression", "horizon": 10}),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        base_ec = base.engine_config
        base_ec = json.loads(base_ec) if isinstance(base_ec, str) else copy.deepcopy(base_ec)

        def _tc(slot):
            tc = slot.target_config
            return json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)

        created = []
        for tag, fs, xt in GRID:
            name = f"n2_1378_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists (id={ex.id})")
                created.append((ex.id, name))
                continue
            new_slots = []
            for s in base.component_slots:
                tc = _tc(s)
                fsn = s.feature_set_name
                if s.slot_type == "exit":
                    fsn = fs
                    if xt is not None:
                        tc = copy.deepcopy(xt)
                new_slots.append(
                    {
                        "slot_type": s.slot_type,
                        "ml_component_id": s.ml_component_id,
                        "rule_component_id": s.rule_component_id,
                        "feature_set_name": fsn,
                        "target_config": tc,
                    }
                )
            tmpl = await repo.create(
                name=name,
                market=base.market,
                strategy=base.strategy,
                feature_set_id=base.feature_set_id,
                target_id=base.target_id,
                component_slots=new_slots,
                direction=base.direction,
                signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=copy.deepcopy(base_ec),
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"Directional-only exit head ({fs}) {tag} on t1378; target="
                f"{'reward_risk' if xt is None else xt['type']}. Drops symmetric vol feats.",
                hypothesis="Symmetric vol feats make the head fire at bottoms; directional-only feats "
                "should let exit_z spike at tops. Verify prediction shape; test vs t1378 405.0.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} created (id={tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
