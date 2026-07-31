"""SMAC v15 — NEW ANGLE: cross-sectional RELATIVE STRENGTH entry (buy leaders' dips).

New niche beyond single-symbol price structure: a TA buys dips in stocks showing relative
strength vs the universe (leaders recover; laggards keep failing). Adds momentum_rank (20d-
return rank), cs_rank_trend (leadership rotation = 10-bar change of that rank), and
price_strength_rank to the mfl30 best (action_oracle: regime + confirm06 + min_fwd_leg 0.30,
peak exit, sandbox). Feature set entry_recov_rs.

Result: MEAN +197.1 (std 6.5, pnl +58, pf 3.49, mdd 0.265, WR 0.73, 410 tr, hold 172) vs
mfl30 +190.5 = +6.6, robust. Matches the pullback-OFF baselines (197/398) — a CLEAN single
model. n2_smac_v15_mfl30_rs is the current SMAC best.

Usage: python stock_ml/scripts/build_smac_v15_rs.py
"""

from __future__ import annotations

import asyncio
import copy
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2459
SEEDS = [42, 7, 99]
NAME = "n2_smac_v15_mfl30_rs"
FS = "entry_recov_rs"
TARGET = {
    "type": "action_oracle",
    "pct": 0.10,
    "min_fwd_leg": 0.30,
    "min_leg_bars": 3,
    "entry_min_ret_120": -0.10,
    "entry_confirm_pct": 0.06,
}


async def make_template() -> int:
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NAME)
        if ex:
            print(f"exists {ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            if sl.slot_type == "entry":
                tc, fs = dict(TARGET), FS
            else:
                tc, fs = copy.deepcopy(sl.target_config), sl.feature_set_name
            slots.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": fs,
                    "target_config": tc,
                }
            )
        t = await repo.create(
            name=NAME,
            market=base.market,
            strategy=base.strategy,
            feature_set_id=base.feature_set_id,
            target_id=base.target_id,
            component_slots=slots,
            direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=None,
            exit_threshold=None,
            split_config=copy.deepcopy(base.split_config),
            engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config,
            seed=42,
            description="SMAC v15: mfl30 + cross-sectional relative-strength entry (buy leaders' dips)",
            hypothesis="RS/leadership distinguishes leader-bottoms (recover) from laggard-bottoms.",
            universe_slug=base.universe_slug,
            model_mode="ml_only",
        )
        await s.commit()
        print(f"created {t.id} {NAME} fs={FS}")
        return t.id


def rd(rid):
    c = psycopg2.connect(**PG)
    cur = c.cursor()
    cur.execute(
        "SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades,wr,avg_hold "
        "FROM leaderboard_runs WHERE run_id=%s",
        (rid,),
    )
    r = cur.fetchone()
    c.close()
    return r


def main():
    tid = asyncio.run(make_template())
    asyncio.run(async_engine.dispose())
    seeds = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=tid, seed=sd)
        row = rd(r.get("run_id")) if r.get("run_id") else None
        comp = float(row[0]) if row and row[0] is not None else None
        seeds[sd] = comp
        if row:
            print(
                f"  {NAME} seed={sd}: comp={comp} pnl={row[1]:.1f} pf={row[2]:.2f} "
                f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f}",
                flush=True,
            )
    cs = [v for v in seeds.values() if v is not None]
    if cs:
        print(
            f"== {NAME}: MEAN={statistics.mean(cs):.1f} "
            f"std={statistics.pstdev(cs) if len(cs) > 1 else 0:.1f} seeds={seeds}"
        )
    print("BUILD_SMAC_V15_DONE")


if __name__ == "__main__":
    main()
