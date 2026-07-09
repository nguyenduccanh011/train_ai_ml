"""SMAC v7 — synthesis: MAX-AMPLITUDE target + REGIME constraint (the two best ideas).

v5 (regime-gated zigzag) = best so far (-46.7, pnl +21.5, pf 1.53): the win was making the
ENTER target PREDICTABLE (only healthy-regime bottoms). v6 (unconstrained max-amplitude DP)
failed (-211..-233) because it enters every hindsight low incl. unpredictable counter-trend
knives. v7 combines them: the perfect-foresight optimal long-only DP (joint entry+exit max
amplitude) BUT constrained to only OPEN a position where ret_120d > threshold — keeps v6's
two-way amplitude optimization, drops the unpredictable downtrend entries.

Sweep (fee, entry_min_ret_120). Clones v1 (tmpl 2459); only the entry target changes; multi-seed.

Usage: python stock_ml/scripts/build_smac_v7_ampreg.py
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
SWEEP = {  # name -> (fee, entry_min_ret_120)
    "n2_smac_v7_amp06_reg10": (0.06, -0.10),
    "n2_smac_v7_amp10_reg10": (0.10, -0.10),
}


async def make_template(new_name: str, fee: float, gate: float) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(new_name)
        if ex:
            print(f"template exists: id={ex.id} ({new_name})")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            if sl.slot_type == "entry":
                tc = {"type": "amplitude_oracle", "fee": fee, "entry_min_ret_120": gate}
            else:
                tc = copy.deepcopy(sl.target_config)
            slots.append({
                "slot_type": sl.slot_type,
                "ml_component_id": sl.ml_component_id,
                "rule_component_id": sl.rule_component_id,
                "feature_set_name": sl.feature_set_name,
                "target_config": tc,
            })
        t = await repo.create(
            name=new_name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=None, exit_threshold=None,
            split_config=copy.deepcopy(base.split_config),
            engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42,
            description=f"SMAC v7: max-amplitude DP target (fee={fee}) + regime gate "
                        f"(entry_min_ret_120={gate}) — joint entry+exit amplitude on predictable "
                        "(healthy-regime) swings only. Synthesis of v5+v6.",
            hypothesis="v6's amplitude target is theoretically optimal but enters unpredictable "
                       "downtrend knives; v5's regime gate fixes predictability. Combining: optimal "
                       "two-way amplitude restricted to learnable healthy-regime entries.",
            universe_slug=base.universe_slug, model_mode="ml_only",
        )
        await s.commit()
        print(f"created template id={t.id} name={new_name} fee={fee} gate={gate}")
        return t.id


def read(run_id):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades,wr,avg_hold "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    for new_name, (fee, gate) in SWEEP.items():
        tid = asyncio.run(make_template(new_name, fee, gate))
        asyncio.run(async_engine.dispose())
        seeds = {}
        for sd in SEEDS:
            r = run_template_experiment(template_id=tid, seed=sd)
            row = read(r.get("run_id")) if r.get("run_id") else None
            comp = float(row[0]) if row and row[0] is not None else None
            seeds[sd] = comp
            if row:
                print(f"  {new_name} seed={sd}: comp={comp} pnl={row[1]:.1f} pf={row[2]:.2f} "
                      f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f}", flush=True)
            else:
                print(f"  {new_name} seed={sd}: NO RESULT", flush=True)
        comps = [v for v in seeds.values() if v is not None]
        if comps:
            mean = statistics.mean(comps); std = statistics.pstdev(comps) if len(comps) > 1 else 0.0
            print(f"== {new_name} (fee={fee},gate={gate}): MEAN={mean:.1f} std={std:.1f} seeds={seeds}\n")
    print("== vs v5_reg10 -46.7 (pnl+21.5 pf1.53 mdd0.341) / v6 amp -211..-233 ; baselines 197.4/397.8")
    print("BUILD_SMAC_V7_DONE")


if __name__ == "__main__":
    main()
