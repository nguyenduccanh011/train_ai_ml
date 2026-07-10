# -*- coding: utf-8 -*-
"""FAMILY_CHAMPIONS step 5: sweep lever re cho ho RULE-ONLY (seed 42).

Lever chua tung thu trong 79 template rule-only: cac gate regime rule-based o engine
(entry_market_gate washout-skip, entry_market_chop ER-gate). Counterfactual fc04:
bo entry bear-2022 = +46.5 (tran ly tuong). Gate that se thap hon.

Variants (clone tu fc_ruleonly t2830, doi engine_config):
  fc_ro_mg    : entry_market_gate_enabled (cumret w5 thr -0.03)
  fc_ro_chop  : entry_market_chop_enabled (w20 thr 0.30)
  fc_ro_mgchop: ca hai
Rule-only deterministic theo seed -> chi can s42.
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
BASE = "fc_ruleonly"
VARIANTS = [
    ("fc_ro_mg", {"entry_market_gate_enabled": True}),
    ("fc_ro_chop", {"entry_market_chop_enabled": True}),
    ("fc_ro_mgchop", {"entry_market_gate_enabled": True, "entry_market_chop_enabled": True}),
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
                description=f"FAMILY_CHAMPIONS rule-only lever sweep: {BASE} + {patch}",
                hypothesis="rule-based regime entry gates (never tried in rule-only family); "
                           "counterfactual ceiling no-bear2022 = +46.5",
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
        print(f"FC5_RESULT {name} s42 comp={row[0]:.1f} d_base={row[0]-435.1:+.1f} pnl={row[1]:.1f} "
              f"pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f} "
              f"run_id={r.get('run_id')}", flush=True)
    con.close()
    print("FC05_DONE")


if __name__ == "__main__":
    main()
