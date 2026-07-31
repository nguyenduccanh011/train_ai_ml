"""ROOT-CAUSE fix: give the exit head LEADING phase features (exit_vol_phase) on t1378.

Diagnosis (_tmp_analysis/exit_head_shape_cmp.py): the exit head fires at BOTTOMS
(exit_z +0.19..+0.25) not TOPS (-0.13) — because the exit feature set (exit_vol_market)
is built from LAGGING realized-vol/drawdown features that spike AFTER a drop. No exit
TARGET can fix this (proven: triple_barrier-short / forward_drawdown made tops WORSE);
the FEATURES lack top-information. exit_vol_phase already exists for exactly this — it
adds momentum deceleration (macd_hist_chg), ma rollover slopes, rsi/macd divergence and
extension context so ONE head sees both the MAGNITUDE and the WHEN axes. It was only ever
tested vs the OLD model 958, never on the current champion.

A/B: clone t1378, swap ONLY the exit feature_set_name -> exit_vol_phase, across the
current exit target and two top-shaped targets. VERIFY at the PREDICTION level (does
exit_z now spike at tops?) before judging the masked backtest. Compare vs FRESH t1378.
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
EXIT_FS = "exit_vol_phase"
# (tag, exit target override or None to keep base reward_risk)
GRID = [
    ("xph_rr", None),
    ("xph_fdd", {"type": "forward_drawdown_regression", "horizon": 10}),
    (
        "xph_tb10",
        {"type": "triple_barrier", "horizon": 20, "pt": 0.10, "sl": 0.05, "direction": "short"},
    ),
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
        for tag, xt in GRID:
            name = f"n2_1378_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists (id={ex.id}), skip")
                created.append((ex.id, name))
                continue
            new_slots = []
            for s in base.component_slots:
                tc = _tc(s)
                fs = s.feature_set_name
                if s.slot_type == "exit":
                    fs = EXIT_FS
                    if xt is not None:
                        tc = copy.deepcopy(xt)
                new_slots.append(
                    {
                        "slot_type": s.slot_type,
                        "ml_component_id": s.ml_component_id,
                        "rule_component_id": s.rule_component_id,
                        "feature_set_name": fs,
                        "target_config": tc,
                    }
                )
            tmpl = await repo.create(
                name=name,
                market=base.market,
                strategy=base.strategy,
                feature_set_id=base.feature_set_id,
                target_id=base.target_id,
                component_slots=new_slots,
                direction=base.direction,
                signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=copy.deepcopy(base_ec),
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"Exit head on LEADING phase features (exit_vol_phase) {tag} on t1378; "
                f"target={'base reward_risk' if xt is None else xt['type']}. All else = t1378.",
                hypothesis="Exit head fires at BOTTOMS not tops because exit_vol_market features lag. "
                "exit_vol_phase adds decel/divergence/extension (leading) so the head can fire "
                "AT tops. Verify exit_z spikes at tops; test vs fresh t1378 405.0.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} created (id={tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(tid) for tid, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
