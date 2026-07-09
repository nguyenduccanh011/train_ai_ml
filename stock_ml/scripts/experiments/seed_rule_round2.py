"""Round 2 on the rule breakthrough r17_trailox_pb (comp 345.0, base r2_17 id 67).

R1 finding: trail+overext ALONE on the hot momentum entry got WORSE (76.7) — the
ATR-trail whipsaws on extended fills. PULLBACK fill (better entry price) is the
enabling condition; pb+trail+overext -> 345.0. This round (a) ISOLATES each lever
on top of pullback, (b) optimizes pullback depth + overext threshold.
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


def _pb(e, pct=0.03, win=25):
    e["entry_pullback_pct"] = pct; e["entry_pullback_window"] = win
def _trail(e):
    e["trailing_stop_pct"] = 0.08; e["trailing_activate_pct"] = 0.15; e["trailing_atr_mult"] = 2.0
def _ox(e, pct=0.14):
    e["overext_ma_window"] = 20; e["overext_pct"] = pct; e["overext_reversal"] = False
def _hold(e, n): e["max_hold_bars"] = n

# isolation
def m_pb(e): _pb(e)                                                   # pullback only, hold 20, signal exit
def m_pb_h250(e): _pb(e); _hold(e, 250)                              # pullback only + lift cap
def m_pb_trail(e): _pb(e); _hold(e, 250); _trail(e); e["exit_priority"] = ["trailing_stop", "signal"]
def m_pb_ox(e): _pb(e); _hold(e, 250); _ox(e); e["exit_priority"] = ["overext", "signal"]
# optimize the winner (pb + trail + overext, hold 250)
def _win(e, pbpct=0.03, pbwin=25, oxpct=0.14, act=0.15):
    _pb(e, pbpct, pbwin); _hold(e, 250); _trail(e); _ox(e, oxpct)
    e["trailing_activate_pct"] = act; e["exit_priority"] = ["trailing_stop", "overext", "signal"]
def m_pb02(e): _win(e, pbpct=0.02)
def m_pb04w35(e): _win(e, pbpct=0.04, pbwin=35)
def m_ox12(e): _win(e, oxpct=0.12)
def m_ox16(e): _win(e, oxpct=0.16)
def m_act10(e): _win(e, act=0.10)

GRID = [
    ("r17_pb",        m_pb,      "pullback-fill ONLY (hold20, signal exit) — isolate pullback"),
    ("r17_pb_h250",   m_pb_h250, "pullback ONLY + hold-cap 250 — isolate lift-cap"),
    ("r17_pb_trail",  m_pb_trail,"pullback + ATR-trail (no overext)"),
    ("r17_pb_ox",     m_pb_ox,   "pullback + overext (no trail)"),
    ("r17_win_pb02",  m_pb02,    "winner, pullback 0.02 (fills more)"),
    ("r17_win_pb04w35", m_pb04w35,"winner, pullback 0.04/window35 (deeper)"),
    ("r17_win_ox12",  m_ox12,    "winner, overext_pct 0.12 (sell tops earlier)"),
    ("r17_win_ox16",  m_ox16,    "winner, overext_pct 0.16 (let run)"),
    ("r17_win_act10", m_act10,   "winner, trailing_activate 0.10 (lock earlier)"),
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
                description=f"Rule round2: {desc}.",
                hypothesis="Isolate pullback vs protective-exit; optimize winner params.",
                universe_slug=base.universe_slug, model_mode=base.model_mode,
            )
            print(f"* {name} created (id={tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
