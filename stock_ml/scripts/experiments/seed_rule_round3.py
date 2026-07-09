"""Round 3: attack the two residual weaknesses of the rule winner r17_trailox_pb (345.0).
Forensic: 53% of big losses are ENTRY-driven (never worked) -> try a hard-stop.
The rule signal-exit (macd<0 AND below_sma20) still bleeds -32.3 -> try dropping it.
Winner config = r2_17 + pullback(0.03/25) + ATR-trail(2.0/0.15/0.08) + overext(20/0.14) + hold250.
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

BASE_ID = 67


def _winner(e):
    e["entry_pullback_pct"] = 0.03; e["entry_pullback_window"] = 25
    e["trailing_stop_pct"] = 0.08; e["trailing_activate_pct"] = 0.15; e["trailing_atr_mult"] = 2.0
    e["overext_ma_window"] = 20; e["overext_pct"] = 0.14; e["overext_reversal"] = False
    e["max_hold_bars"] = 250
    e["exit_priority"] = ["trailing_stop", "overext", "signal"]


def m_hs08(e): _winner(e); e["hard_stop_pct"] = -0.08; e["exit_priority"] = ["hard_stop", "trailing_stop", "overext", "signal"]
def m_hs06(e): _winner(e); e["hard_stop_pct"] = -0.06; e["exit_priority"] = ["hard_stop", "trailing_stop", "overext", "signal"]
def m_hs10(e): _winner(e); e["hard_stop_pct"] = -0.10; e["exit_priority"] = ["hard_stop", "trailing_stop", "overext", "signal"]
def m_nosig(e): _winner(e); e["exit_priority"] = ["trailing_stop", "overext"]
def m_hs08_nosig(e): _winner(e); e["hard_stop_pct"] = -0.08; e["exit_priority"] = ["hard_stop", "trailing_stop", "overext"]

GRID = [
    ("r17_win_hs08",      m_hs08,      "winner + hard-stop -8% (cut entry-driven losers)"),
    ("r17_win_hs06",      m_hs06,      "winner + hard-stop -6%"),
    ("r17_win_hs10",      m_hs10,      "winner + hard-stop -10%"),
    ("r17_win_nosig",     m_nosig,     "winner WITHOUT signal-exit (drop the -32.3 bleeder)"),
    ("r17_win_hs08_nosig",m_hs08_nosig,"winner + hard-stop -8% + drop signal-exit"),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        entry_slot = next(s for s in base.component_slots if s.slot_type == "entry")
        exit_slot = next(s for s in base.component_slots if s.slot_type == "exit")

        def _tc(slot):
            tc = slot.target_config
            return json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)

        slots_def = [
            {"slot_type": "entry", "ml_component_id": entry_slot.ml_component_id,
             "rule_component_id": entry_slot.rule_component_id,
             "feature_set_name": entry_slot.feature_set_name, "target_config": _tc(entry_slot)},
            {"slot_type": "exit", "ml_component_id": exit_slot.ml_component_id,
             "rule_component_id": exit_slot.rule_component_id,
             "feature_set_name": exit_slot.feature_set_name, "target_config": _tc(exit_slot)},
        ]
        base_engine = base.engine_config
        if isinstance(base_engine, str):
            base_engine = json.loads(base_engine)

        created = []
        for name, mutate, desc in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists (id={ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(base_engine); mutate(eng)
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots_def), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"Rule round3: {desc}.",
                hypothesis="Cut entry-driven losers (hard-stop) and/or drop bleeding signal-exit.",
                universe_slug=base.universe_slug, model_mode=base.model_mode,
            )
            print(f"* {name} created (id={tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
