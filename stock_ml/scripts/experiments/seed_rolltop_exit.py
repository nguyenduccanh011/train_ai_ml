"""Batch 3 — capture the rolling-top giveback (104.8u, 87% of all giveback) that overext is
blind to. Those tops peak at <14% extension (overext needs 14%) and 90% make a lower-high the
next bar, but they fall to the late directionless signal head (69u giveback, 8 bars late) or a
trailing stop that never arms (peak <15%, 31u giveback, 20 bars late).

Instrument: a MODERATE-extension reversal sell — overext_pct lowered to 6-10% with
overext_reversal=True (fire only on a down-bar while extended = a rollover, not every bar)
and overext_skip_ma_slope_pct set (skip when the MA is still rising fast = strong uptrend,
let the runner run). overext runs BEFORE signal in exit_priority, so it replaces the late
signal exit at these tops with an earlier rollover sell. Sweep pct x slope-skip x MA window.
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

BASE = 1327
# (name, overext overrides) — keep everything else (incl. the +14% overext is replaced)
GRID = [
    ("n2_af04_rt08_sk04", {"overext_pct": 0.08, "overext_reversal": True,
                           "overext_skip_ma_slope_pct": 0.04, "overext_skip_lookback": 5}),
    ("n2_af04_rt06_sk04", {"overext_pct": 0.06, "overext_reversal": True,
                           "overext_skip_ma_slope_pct": 0.04, "overext_skip_lookback": 5}),
    ("n2_af04_rt08_sk025", {"overext_pct": 0.08, "overext_reversal": True,
                            "overext_skip_ma_slope_pct": 0.025, "overext_skip_lookback": 5}),
    ("n2_af04_rt10_sk04", {"overext_pct": 0.10, "overext_reversal": True,
                           "overext_skip_ma_slope_pct": 0.04, "overext_skip_lookback": 5}),
    ("n2_af04_rt08_sk04_m10", {"overext_pct": 0.08, "overext_reversal": True,
                               "overext_ma_window": 10, "overext_skip_ma_slope_pct": 0.04,
                               "overext_skip_lookback": 5}),
    ("n2_af04_rt08_noskip", {"overext_pct": 0.08, "overext_reversal": True}),
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
                description=f"rolling-top reversal exit: {ov}; base {BASE}.",
                hypothesis="Catch <14%-ext rolling tops on a reversal bar (skip strong trends) -> capture 104u giveback, beat 404.3.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
