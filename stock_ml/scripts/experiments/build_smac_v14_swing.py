"""SMAC v14 — swing-size frontier (min_fwd_leg sweep), the entry-selectivity breakthrough.

The entry label is the only productive axis; requiring a bigger hindsight up-leg (min_fwd_leg)
= the model learns higher-quality big-mover bottoms. Monotonic: mfl15 +41 → mfl20 +99 →
mfl25 +158 → mfl30 +190 (pf 1.87→3.36, mdd 0.334→0.26). All on v8 base (regime ENTER + confirm06,
peak exit, sandbox). NEW BEST = n2_smac_v14_mfl30 (+190.5, pf 3.36, mdd 0.26, robust std 4.8).

CAVEAT: high min_fwd_leg thins trades (880→420 @ mfl30); beyond mfl30 = overfitting / too-few-
trades risk. The label is hindsight-SELECTIVE — holdout-validate before pushing further.

Usage: python stock_ml/scripts/build_smac_v14_swing.py
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
V8 = {"type": "action_oracle", "pct": 0.10, "min_fwd_leg": 0.10, "min_leg_bars": 3,
      "entry_min_ret_120": -0.10, "entry_confirm_pct": 0.06}
VARIANTS = {
    "n2_smac_v14_mfl20_conf08": {"min_fwd_leg": 0.20, "entry_confirm_pct": 0.08},
    "n2_smac_v14_mfl25": {"min_fwd_leg": 0.25},
    "n2_smac_v14_mfl30": {"min_fwd_leg": 0.30},
}


async def make_template(name: str, ov: dict) -> int:
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"exists {ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            tc = (dict(V8) | ov) if sl.slot_type == "entry" else copy.deepcopy(sl.target_config)
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id,
                          "feature_set_name": sl.feature_set_name, "target_config": tc})
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=None, exit_threshold=None,
            split_config=copy.deepcopy(base.split_config),
            engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42,
            description=f"SMAC v14 swing-size {ov} (entry-selectivity breakthrough; peak exit, sandbox)",
            hypothesis="Bigger required up-leg = higher-quality big-mover bottoms the model can learn.",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit()
        print(f"created {t.id} {name} {ov}")
        return t.id


def rd(rid):
    c = psycopg2.connect(**PG); cur = c.cursor()
    cur.execute("SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades,wr,avg_hold "
                "FROM leaderboard_runs WHERE run_id=%s", (rid,))
    r = cur.fetchone(); c.close(); return r


def main():
    for name, ov in VARIANTS.items():
        tid = asyncio.run(make_template(name, ov))
        asyncio.run(async_engine.dispose())
        seeds = {}
        for sd in SEEDS:
            r = run_template_experiment(template_id=tid, seed=sd)
            row = rd(r.get("run_id")) if r.get("run_id") else None
            comp = float(row[0]) if row and row[0] is not None else None
            seeds[sd] = comp
            if row:
                print(f"  {name} seed={sd}: comp={comp} pnl={row[1]:.1f} pf={row[2]:.2f} "
                      f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f}", flush=True)
        cs = [v for v in seeds.values() if v is not None]
        if cs:
            print(f"== {name} {ov}: MEAN={statistics.mean(cs):.1f} "
                  f"std={statistics.pstdev(cs) if len(cs) > 1 else 0:.1f} seeds={seeds}\n")
    print("BUILD_SMAC_V14_DONE")


if __name__ == "__main__":
    main()
