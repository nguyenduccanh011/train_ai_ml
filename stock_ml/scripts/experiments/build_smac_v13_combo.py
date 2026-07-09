"""SMAC v13 — COMBINE the winning entry knobs (tổ hợp): min_fwd_leg x confirmation, + mfl20.

v12: min_fwd_leg 0.15 -> +41.0 (best, robust); more confirmation also helped. v13 stacks
them and pushes the swing-size knob:
  mfl15_conf08 / mfl15_conf10 = bigger swings + more reversal confirmation
  mfl20_conf06                = only >=20% swings (cleanest), v8 confirmation

All on v8 base (regime ENTER, peak exit). Single position.

Usage: python stock_ml/scripts/build_smac_v13_combo.py
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
    "n2_smac_v13_mfl15_conf08": {"min_fwd_leg": 0.15, "entry_confirm_pct": 0.08},
    "n2_smac_v13_mfl15_conf10": {"min_fwd_leg": 0.15, "entry_confirm_pct": 0.10},
    "n2_smac_v13_mfl20":        {"min_fwd_leg": 0.20},
}


async def make_template(new_name: str, override: dict) -> int:
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
                tc = dict(V8); tc.update(override)
            else:
                tc = copy.deepcopy(sl.target_config)
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id,
                          "feature_set_name": sl.feature_set_name, "target_config": tc})
        t = await repo.create(
            name=new_name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=None, exit_threshold=None,
            split_config=copy.deepcopy(base.split_config),
            engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42,
            description=f"SMAC v13 entry combo on v8: {override}",
            hypothesis="Stack the winning entry knobs (swing-size + confirmation) for cleaner entries.",
            universe_slug=base.universe_slug, model_mode="ml_only",
        )
        await s.commit()
        print(f"created template id={t.id} name={new_name} override={override}")
        return t.id


def read(run_id):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades,wr,avg_hold "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    for new_name, ov in VARIANTS.items():
        tid = asyncio.run(make_template(new_name, ov))
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
            print(f"== {new_name} {ov}: MEAN={mean:.1f} std={std:.1f} seeds={seeds}\n")
    print("== vs v12 mfl15 +41.0 (pf1.87 mdd0.334) / v8 -33 ; baselines 197.4/397.8 ; champ 704")
    print("BUILD_SMAC_V13_DONE")


if __name__ == "__main__":
    main()
