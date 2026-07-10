# -*- coding: utf-8 -*-
"""FAMILY_CHAMPIONS step 7: sweep lever re cho vo dich rule-only fc_rule2 (t2005 clone).

Counterfactual fc04 (rule2): clip12 +22.1 / no-bear2022 +22.8 / ca hai +38.5 (tran ly tuong).
Lever config-only chua thu trong nhanh n3 (5 template):
  fc_r2_hs12  : hard_stop_pct -0.12 (exit_priority them hard_stop dau) — clip12 proxy
  fc_r2_mgf30 : entry_market_abs_floor -0.03  (goc -0.04) — siet washout skip
  fc_r2_mgf25 : entry_market_abs_floor -0.025
Deterministic -> seed 42.
Luu y an tu lien quan: DEEP_STOP (ho gb) chet vi stop cat lenh sap rally — o day
ho rule KHAC model tin hieu, thu 1 lan cho co so; neu am thi dong luon.
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = "fc_rule2"
VARIANTS = [
    ("fc_r2_hs12", {"hard_stop_pct": -0.12,
                    "exit_priority": ["hard_stop", "trailing_stop", "overext", "signal"]}),
    ("fc_r2_mgf30", {"entry_market_abs_floor": -0.03}),
    ("fc_r2_mgf25", {"entry_market_abs_floor": -0.025}),
]


async def make_clones():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    tids = {}
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_name(BASE)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        for name, patch in VARIANTS:
            ex = await repo.get_by_name(name)
            if ex:
                tids[name] = ex.id
                print(f"exists: {name} id={ex.id}", flush=True)
                continue
            eng = base.engine_config
            eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
            eng = copy.deepcopy(eng); eng.update(patch)
            t = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"FAMILY_CHAMPIONS rule-only n3 lever sweep: {BASE} + {patch}",
                hypothesis="config-only lever from fc04 counterfactual (clip12 +22 / bear-gate +23 ideal)",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            tids[name] = t.id
            print(f"created: {name} id={t.id}", flush=True)
        await s.commit()
    await async_engine.dispose()
    return tids


def main():
    tids = asyncio.run(make_clones())
    con = psycopg2.connect(**PG); cur = con.cursor()
    for name, _ in VARIANTS:
        r = run_template_experiment(template_id=tids[name], seed=42)
        cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, avg_hold "
                    "FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
        row = cur.fetchone()
        print(f"FC7_RESULT {name} s42 comp={row[0]:.1f} d_base={row[0]-568.3:+.1f} pnl={row[1]:.1f} "
              f"pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f} "
              f"run_id={r.get('run_id')}", flush=True)
    con.close()
    print("FC07_DONE")


if __name__ == "__main__":
    main()
