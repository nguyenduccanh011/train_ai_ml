"""Tier-1b on champion 1586: free the runners from the overext +12% hard cap.
Forensic: only 6.9% of overext sells were real tops; 75.6% kept running (+14.3% over 20 bars);
the strategy's whole edge is the 719 runners (>=16% MFE = +116u). Champion HARD-sells at
overext +12%/MA20 (hold 7.7 bars). Tests: (a) overext_trail_pct -> arm a give-back trail
instead of hard sell (ride continuation); (b) raise overext_pct (cap later); (c) skip overext
while the per-symbol SMA is in a strong uptrend (let trends run). Backtest vs 407.7.
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

BASE = 1586
GRID = [
    ("n2_ox_trail04", {"overext_trail_pct": 0.04}),
    ("n2_ox_trail06", {"overext_trail_pct": 0.06}),
    ("n2_ox_trail08", {"overext_trail_pct": 0.08}),
    ("n2_ox16", {"overext_pct": 0.16}),
    ("n2_ox20", {"overext_pct": 0.20}),
    ("n2_ox_slope03", {"overext_skip_ma_slope_pct": 0.03, "overext_skip_lookback": 5}),
    ("n2_ox_slope03_trail06", {"overext_skip_ma_slope_pct": 0.03, "overext_skip_lookback": 5, "overext_trail_pct": 0.06}),
    ("n2_ox16_slope03", {"overext_pct": 0.16, "overext_skip_ma_slope_pct": 0.03, "overext_skip_lookback": 5}),
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
        for name, ov in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(be); eng.update(ov)
            slots = [
                {"slot_type": "entry", "ml_component_id": es.ml_component_id, "rule_component_id": None,
                 "feature_set_name": es.feature_set_name, "target_config": tc(es)},
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
                description=f"free runners from overext cap {ov} on champ {BASE}.",
                hypothesis="Let runners run past overext +12% (trail/raise/skip) -> capture +14.3% continuation -> beat 407.7.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
