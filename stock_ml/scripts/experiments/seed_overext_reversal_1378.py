"""Reversal-confirmed overext for the faded-winner band — A/B on champion t1378.

Pop-lock trail (seed_poplock_1378.py) tested negative: a blind price-trail clips
runners passing through the +5-9% band. The non-clipping alternative is to sell the
faded band only on a CONFIRMED reversal bar (sell AT the turn, not on a retrace).

Engine overext is a single tier: trig = (ext>=overext_pct over SMA) AND reversal_ok,
with skip-guards while the SMA / market is rising (protect runners). t1378 runs it at
pct=0.14, reversal=False -> fires immediately on a +14% stretch (the +67u, lag-0.4
mechanism), but those sells see +9.4% MORE upside after (sell a touch early), and the
+9.3%-peak faded winners never reach +14% so they fall through to the slow signal exit.

Hypothesis (double win): LOWER overext_pct to reach the faded band + require an
ema_cross reversal so (a) the faded band sells at its actual roll-over, and (b) the big
parabolas keep running until they truly cross down (capturing the +9.4% they currently
leave). skip_ma_slope protects strong per-symbol uptrends. Clone t1378, change ONLY
overext params, A/B vs FRESH t1378 (re-run = 405.0).
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402

BASE_ID = 1378

# (tag, overext overrides). reversal_mode=ema_cross sells at the turn; skip_ma_slope
# lets strong uptrends run so the lower threshold doesn't clip parabolas.
GRID = [
    # isolate: add reversal-confirm to the EXISTING +14% threshold (does waiting for the
    # turn improve the proven overext sells, which currently leave +9.4% on the table?)
    ("ox14_ema", {"overext_pct": 0.14, "overext_reversal_mode": "ema_cross"}),
    # reach into the faded band (+11%/+9%) WITH reversal-confirm + uptrend skip-guard
    ("ox11_ema_skip", {"overext_pct": 0.11, "overext_reversal_mode": "ema_cross",
                       "overext_skip_ma_slope_pct": 0.02, "overext_skip_lookback": 5}),
    ("ox09_ema_skip", {"overext_pct": 0.09, "overext_reversal_mode": "ema_cross",
                       "overext_skip_ma_slope_pct": 0.02, "overext_skip_lookback": 5}),
    # deepest reach + stronger candlestick confirm (red bar, >=2% drop) instead of ema_cross
    ("ox09_strongdown_skip", {"overext_pct": 0.09, "overext_reversal_mode": "strong_down",
                              "overext_strong_down_pct": 0.02,
                              "overext_skip_ma_slope_pct": 0.02, "overext_skip_lookback": 5}),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        if base is None:
            raise ValueError(f"base template id={BASE_ID} not found")

        def _tc(slot):
            tc = slot.target_config
            return json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)

        base_ec = base.engine_config
        base_ec = json.loads(base_ec) if isinstance(base_ec, str) else copy.deepcopy(base_ec)

        created = []
        for tag, ov in GRID:
            name = f"n2_1378_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists (id={ex.id}), skip")
                created.append((ex.id, name))
                continue
            ec = copy.deepcopy(base_ec)
            ec.update(ov)
            new_slots = [
                {"slot_type": s.slot_type, "ml_component_id": s.ml_component_id,
                 "rule_component_id": s.rule_component_id, "feature_set_name": s.feature_set_name,
                 "target_config": _tc(s)}
                for s in base.component_slots
            ]
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=new_slots, direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=ec,
                validation_config=base.validation_config, seed=base.seed,
                description=(
                    f"Reversal-confirmed overext {tag} on t1378: {ov}; everything else = t1378. "
                    f"Sell the faded band AT a confirmed turn instead of a blind trail."
                ),
                hypothesis="723 signal-exit faded winners peak +9.3% then give back ~9.6% (lag 7.4 "
                           "bars), below the +14% overext / +15% trail. Lowering overext_pct + an "
                           "ema_cross/strong_down reversal-confirm should sell the faded band at its "
                           "roll-over AND let parabolas run to the true cross (skip_ma_slope guards "
                           "strong uptrends). Test vs fresh t1378 405.0.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} created (id={tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(tid) for tid, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
