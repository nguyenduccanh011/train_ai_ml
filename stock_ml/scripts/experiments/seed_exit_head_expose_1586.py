"""STRUCTURAL: expose + rebuild the EXIT head on champion 1586 (user-chosen direction).
Proof this session: decoupled uses use_raw_exit=False so the ONLY head-driven sell is
z(exit) > signal_threshold; champ signal_threshold=5.5 (~never) + reward_risk target is
SIGN-INVERTED (sells when reward_risk HIGH = good) -> head fully DORMANT, mechanical rules
(overext/downleg) do all exits (exit-feature swap was an exact no-op = proof).
Fix: (1) overext OFF (let the head replace it); (2) lower sell-z 5.5 -> 1.0/1.5/2.0 (make
the head fire); (3) swap exit target to a CORRECT-SIGN one (forward_drawdown / risk_exit:
HIGH = sell, matching 'sell on z(exit) HIGH'). Keep downleg as the tail backstop (removing
it spiked mdd 4x historically). Multi-seed the winner (single-seed noise is +-3). vs champ 406.0 (multiseed mean).
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
DD = lambda h: {"type": "forward_drawdown_regression", "horizon": h}
RX = lambda h: {"type": "risk_exit_regression", "horizon": h}
OFF = {"overext_ma_window": 0}  # expose: disable overext
OFF_NODL = {"overext_ma_window": 0, "exit_force_gate": None}  # also remove downleg backstop
# (name, signal_threshold(sell-z), exit_target, engine_override)
GRID = [
    ("n2_xh_dd_z10", 1.0, DD(10), OFF),
    ("n2_xh_dd_z15", 1.5, DD(10), OFF),
    ("n2_xh_dd_z20", 2.0, DD(10), OFF),
    ("n2_xh_rx_z10", 1.0, RX(10), OFF),
    ("n2_xh_rx_z15", 1.5, RX(10), OFF),
    ("n2_xh_rx_z20", 2.0, RX(10), OFF),
    ("n2_xh_dd_z15_nodl", 1.5, DD(10), OFF_NODL),  # head must control the tail alone
    ("n2_xh_dd_z15_keepox", 1.5, DD(10), None),  # head ON TOP of overext (additive)
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
        for name, sigthr, xtgt, eng_ov in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(be)
            if eng_ov:
                eng.update(eng_ov)
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
                    "target_config": copy.deepcopy(xtgt),
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
                signal_threshold=sigthr,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=eng,
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"EXPOSE exit head: overext_off sell-z={sigthr} target={xtgt['type']} eng={eng_ov} on {BASE}.",
                hypothesis="Correct-sign exposed exit head replaces overext -> head does real exit work -> beat champ (clear +-3 noise via multiseed).",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
