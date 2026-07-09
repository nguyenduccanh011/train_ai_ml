"""Exit-head retrain batch 2 — the REACTIVE/confirmed-leg fix. Batch 1 showed EVERY predictive
target (reward_risk, forward_drawdown, risk_exit) still SELLS WINNERS, because forward_drawdown
also lights up in-uptrend pullback tops (the winners). downleg_depth_regression lights ONLY
confirmed real down-legs (>=pct reversal that actually fell); peak_decay>0 makes the label high
at the leg TOP and fade fast -> the head fires EARLY at a real roll-over, NOT at pullback tops.
max_span=30 keeps required_gap=65 == champion gap (clean, no leakage).
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

BASE_ID = 1789
def DL(decay):
    return {"type": "downleg_depth_regression", "pct": 0.06, "max_span": 30, "min_leg_bars": 0, "peak_decay": decay}
GRID = [
    ("xdir_dl5",   "exit_directional", DL(5.0)),   # fire EARLY at confirmed-leg top
    ("xdir_dl3",   "exit_directional", DL(3.0)),   # even sharper/earlier
    ("xdir_dl0",   "exit_directional", DL(0.0)),   # loud through whole leg (control, sells late)
    ("xphase_dl5", "exit_vol_phase",   DL(5.0)),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        base_ec = base.engine_config
        base_ec = json.loads(base_ec) if isinstance(base_ec, str) else copy.deepcopy(base_ec)
        def _tc(s):
            tc = s.target_config
            return json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
        created = []
        for tag, fs, xt in GRID:
            name = f"n2_xr_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} (id={ex.id})"); created.append((ex.id, name)); continue
            slots = []
            for s in base.component_slots:
                tc = _tc(s); fsn = s.feature_set_name
                if s.slot_type == "exit":
                    fsn = fs; tc = copy.deepcopy(xt)
                slots.append({"slot_type": s.slot_type, "ml_component_id": s.ml_component_id,
                              "rule_component_id": s.rule_component_id, "feature_set_name": fsn,
                              "target_config": tc})
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
                target_id=base.target_id, component_slots=slots, direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=copy.deepcopy(base_ec),
                validation_config=base.validation_config, seed=base.seed,
                description=f"Exit retrain {tag}: exit={fs} downleg_depth decay={xt['peak_decay']} on champion 1789.",
                hypothesis="downleg_depth lights only CONFIRMED legs (not pullback tops) -> the head fires at "
                           "real roll-overs not winners. peak_decay>0 fires early at the top. Beat hard rule un-masked.",
                universe_slug=base.universe_slug)
            print(f"* {name} (id={tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
