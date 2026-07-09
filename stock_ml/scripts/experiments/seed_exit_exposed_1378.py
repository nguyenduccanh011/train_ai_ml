"""Fair EXPOSED-baseline test of exit-head feature quality.

The masked champion (overext + trail + downleg + z5.5) hides the exit head — feature
swaps look flat. To judge head QUALITY fairly, EXPOSE it: overext OFF + sell-z 3.0 so
the ML head makes the sell decisions (downleg/trailing kept as tail safety). Then swap
ONLY the exit feature set; the control is the SAME exposed config with the current vol
features. If the user's directional top-tells (down-volume bars, distribution days,
divergence) beat the symmetric-vol features HERE, that's the breakthrough direction.

Compare feature-vs-feature at the SAME exposed config — NOT vs the 405 champion.
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
EXPOSE = {"overext_ma_window": 0}   # overext OFF
SELL_Z = 3.0                         # engage the head
FEATURE_SETS = [
    ("exp_vol", "exit_vol_market"),       # control (same exposed config, current features)
    ("exp_toptells", "exit_toptells"),    # the user's directional top-tells
    ("exp_dir", "exit_directional"),
    ("exp_dist", "exit_vol_dist"),
    ("exp_phase", "exit_vol_phase"),
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
        for tag, fs in FEATURE_SETS:
            name = f"n2_1378_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists (id={ex.id})"); created.append((ex.id, name)); continue
            ec = copy.deepcopy(base_ec); ec.update(EXPOSE)
            new_slots = []
            for s in base.component_slots:
                fsn = fs if s.slot_type == "exit" else s.feature_set_name
                new_slots.append({"slot_type": s.slot_type, "ml_component_id": s.ml_component_id,
                                  "rule_component_id": s.rule_component_id, "feature_set_name": fsn,
                                  "target_config": _tc(s)})
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
                target_id=base.target_id, component_slots=new_slots, direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=SELL_Z,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=ec,
                validation_config=base.validation_config, seed=base.seed,
                description=f"EXPOSED exit head (overext off, sell-z {SELL_Z}) feature={fs} ({tag}). "
                            f"Fair feature-vs-feature test; control = exp_vol (same config, vol feats).",
                hypothesis="Masked champion hides exit-head quality. Exposed, the user's directional "
                           "top-tells (down-volume/distribution/divergence) should beat symmetric vol "
                           "feats — control is exp_vol at the SAME exposed config.",
                universe_slug=base.universe_slug)
            print(f"* {name} created (id={tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
