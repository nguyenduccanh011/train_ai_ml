"""Batch 3 on champion 1586: CONDITIONAL protection of the giveback cohort.
Forensic: 386 'giveback' losers peaked +8.8% MFE then round-tripped to a signal-exit loss
(-23.3u) -- they sit in the +5-12% dead zone BELOW the overext +12% cap and BELOW the +15%
trail-arm, so nothing protects them. Tier 1 (uniform early trail) clipped runners. The fix
must be CONDITIONAL: fire a low overext (~+8%) ONLY when momentum has rolled over
(bear_div / strong_down), separating the fader from a runner pausing. Also test composing
the slope-skip (runners run, PnL+) with bull-skip / early-trail on disjoint cohorts. vs 407.7.
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
    # conditional low-overext: fire at +8% ONLY on a bear-divergence / strong-down turn
    (
        "n2_gb_div08_th30",
        {"overext_pct": 0.08, "overext_reversal_mode": "bear_div", "overext_div_threshold": 0.30},
    ),
    (
        "n2_gb_div08_th50",
        {"overext_pct": 0.08, "overext_reversal_mode": "bear_div", "overext_div_threshold": 0.50},
    ),
    (
        "n2_gb_div10_th30",
        {"overext_pct": 0.10, "overext_reversal_mode": "bear_div", "overext_div_threshold": 0.30},
    ),
    (
        "n2_gb_sdn08",
        {
            "overext_pct": 0.08,
            "overext_reversal_mode": "strong_down",
            "overext_strong_down_pct": 0.02,
        },
    ),
    # conditional + arm a trail instead of hard sell (ride if it recovers)
    (
        "n2_gb_div08_trail05",
        {
            "overext_pct": 0.08,
            "overext_reversal_mode": "bear_div",
            "overext_div_threshold": 0.30,
            "overext_trail_pct": 0.05,
        },
    ),
    # composing: let per-symbol uptrend runners run (PnL+) AND protect flat-zone faders
    (
        "n2_gb_slope03_act08",
        {
            "overext_skip_ma_slope_pct": 0.03,
            "overext_skip_lookback": 5,
            "trailing_activate_pct": 0.08,
        },
    ),
    (
        "n2_gb_slope03_bull",
        {
            "overext_skip_ma_slope_pct": 0.03,
            "overext_skip_lookback": 5,
            "overext_skip_bull_enabled": True,
            "overext_bull_threshold": 1.0,
        },
    ),
    ("n2_gb_bullskip", {"overext_skip_bull_enabled": True, "overext_bull_threshold": 1.0}),
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
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(be)
            eng.update(ov)
            slots = [
                {
                    "slot_type": "entry",
                    "ml_component_id": es.ml_component_id,
                    "rule_component_id": None,
                    "feature_set_name": es.feature_set_name,
                    "target_config": tc(es),
                },
                {
                    "slot_type": "exit",
                    "ml_component_id": xs.ml_component_id,
                    "rule_component_id": None,
                    "feature_set_name": xs.feature_set_name,
                    "target_config": tc(xs),
                },
            ]
            tmpl = await repo.create(
                name=name,
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
                description=f"conditional giveback protection {ov} on champ {BASE}.",
                hypothesis="Fire low-overext only on a momentum-rollover (sep fader from runner) -> recover -23.3u giveback w/o clipping runners -> beat 407.7.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
