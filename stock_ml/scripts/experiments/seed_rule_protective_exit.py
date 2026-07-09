"""Improve the top RULE models by adding the champion's mechanical PROFIT-PROTECTION
exits (trailing + overext) — which rule templates never enabled (exit_priority=["signal"]).

Forensic (rule_forensic.py): r2_17 winners give back 150 pnl-units (58% of peak),
holding 7.3 bars past the peak; mdd 7.13 is round-tripping, not blowups. The ML
champion gets +129 pnl from overext+trailing. Port that lever to the rule book.

Bases:  r2_17_multi_trigger_or (id 67, comp 207.1 PF1.93 mdd7.13)
        r2_06_breakout_di_pb_p030_w15 (id 1098, comp 197.4 PF2.83 mdd2.06, has pullback)
Proven champion exit params: trailing_atr_mult2.0 / activate0.15 / pct0.08,
                             overext ma20 pct0.14, exit_priority trail>overext>signal.
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


def _trail(eng):
    eng["trailing_stop_pct"] = 0.08
    eng["trailing_activate_pct"] = 0.15
    eng["trailing_atr_mult"] = 2.0


def _ox(eng):
    eng["overext_ma_window"] = 20
    eng["overext_pct"] = 0.14
    eng["overext_reversal"] = False


def _hold(eng, n):
    eng["max_hold_bars"] = n


def _pb(eng, pct=0.03, win=25):
    eng["entry_pullback_pct"] = pct
    eng["entry_pullback_window"] = win


# (name, base_id, mutator, desc)
def m_trail_h20(e): _trail(e); e["exit_priority"] = ["trailing_stop", "signal"]
def m_trail(e): _trail(e); _hold(e, 250); e["exit_priority"] = ["trailing_stop", "signal"]
def m_ox(e): _ox(e); _hold(e, 250); e["exit_priority"] = ["overext", "signal"]
def m_trailox(e): _trail(e); _ox(e); _hold(e, 250); e["exit_priority"] = ["trailing_stop", "overext", "signal"]
def m_trailox_h20(e): _trail(e); _ox(e); e["exit_priority"] = ["trailing_stop", "overext", "signal"]
def m_trailox_pb(e): _trail(e); _ox(e); _hold(e, 250); _pb(e); e["exit_priority"] = ["trailing_stop", "overext", "signal"]
def m06_trailox(e): _trail(e); _ox(e); _hold(e, 250); e["exit_priority"] = ["trailing_stop", "overext", "signal"]

GRID = [
    ("r17_trail_h20",   67,  m_trail_h20,  "r2_17 + ATR-trail, keep hold-cap 20"),
    ("r17_trail",       67,  m_trail,      "r2_17 + ATR-trail, hold-cap 250"),
    ("r17_ox",          67,  m_ox,         "r2_17 + overext top-sell, hold-cap 250"),
    ("r17_trailox",     67,  m_trailox,    "r2_17 + trail + overext, hold-cap 250"),
    ("r17_trailox_h20", 67,  m_trailox_h20,"r2_17 + trail + overext, keep hold-cap 20"),
    ("r17_trailox_pb",  67,  m_trailox_pb, "r2_17 + trail + overext + pullback-fill, hold-cap 250"),
    ("r06_trailox",     1098, m06_trailox, "r2_06 (breakout+DI+pullback) + trail + overext, hold-cap 250"),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        cache = {}
        created = []
        for name, base_id, mutate, desc in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists (id={ex.id})"); created.append((ex.id, name)); continue
            if base_id not in cache:
                cache[base_id] = await repo.get_by_id(base_id)
            base = cache[base_id]
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
            eng = copy.deepcopy(base_engine)
            mutate(eng)
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots_def), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"Rule protective-exit: {desc}.",
                hypothesis="Rule book has NO profit protection; winners give back 58% of peak. "
                           "Port champion trailing+overext -> mdd down + pnl up.",
                universe_slug=base.universe_slug, model_mode=base.model_mode,
            )
            print(f"* {name} created (id={tmpl.id})  base={base_id}"); created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
