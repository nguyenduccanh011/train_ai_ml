# -*- coding: utf-8 -*-
"""UNION_TWO_SOURCES step 2: sweep seed-42 cac bien the union tren clone gb_x08 (t2783).

Co che union (config-only, khong code moi): kenh rule cua t2005 = buy MOI bar qua
entry_gate upleg_abovema20 (score hang + entry_raw_threshold 0). gb_x08 dung CUNG
entry_gate + CUNG market gate; main buy cua no = zE > entry_threshold (-1.9).
=> UNION DU o tang signal == ha entry_threshold xuong -99: buy = moi bar qua gate
   (= buy mask rule) UNION 4 ensemble head. Chung nguyen stack exit/pullback gb.
Bien the:
  un_full : entry_threshold -99 (union du)
  un_w50  : union du + entry_pullback_window 50 (cua so fill cua rule; bat dip bar 41-50)
Dry-run un01: additive net first-order -3.1u (displacement 3.42u > additive 0.32u) —
day la run xac nhan re truoc khi dong an.
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
BASE_ID = 2783  # gb_x08
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\combo"
# (name, entry_threshold override, engine patch)
VARIANTS = [
    ("un_full", -99.0, {}),
    ("un_w50", -99.0, {"entry_pullback_window": 50}),
]


async def make_clones():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    tids = {}
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_ID)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        for name, ethr, patch in VARIANTS:
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
                entry_threshold=ethr, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"UNION_TWO_SOURCES: gb_x08 + kenh rule price-gate cua t2005 hop nhat "
                            f"config-only (entry_threshold {ethr} -> buy moi bar qua upleg gate = "
                            f"buy mask rule; chung stack exit/pullback gb). patch={patch}",
                hypothesis="union 2 nguon entry (ML gb + rule price-gate fc_rule2, overlap trades 39%) "
                           "lap slot trong 2024-26; rui ro so 1 = occupancy displacement (an bs_/bch_)",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            tids[name] = t.id
            print(f"created: {name} id={t.id}", flush=True)
        await s.commit()
    await async_engine.dispose()
    return tids


def main():
    tids = asyncio.run(make_clones())
    con = psycopg2.connect(**PG)
    import pandas as pd
    for name, _, _ in VARIANTS:
        cur = con.cursor()
        cur.execute("SELECT run_id FROM leaderboard_runs WHERE run_id LIKE %s", (f"template/{name}-%",))
        done = cur.fetchone()
        if done:
            run_id = done[0]
        else:
            r = run_template_experiment(template_id=tids[name], seed=42)
            run_id = r.get("run_id")
        cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, avg_hold "
                    "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
        row = cur.fetchone()
        print(f"UN2_RESULT {name} s42 comp={row[0]:.1f} d_gb={row[0]-735.0:+.1f} pnl={row[1]:.1f} "
              f"pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f} "
              f"run_id={run_id}", flush=True)
        # export trades for occupancy autopsy
        tr = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                         "holding_days, pnl_pct, exit_reason, entry_signal_date from run_trades "
                         "where run_id=%s order by symbol, entry_date", con, params=(run_id,))
        tr.to_csv(rf"{OUT}\{name}_s42_trades.csv", index=False)
        print(f"saved {len(tr)} trades -> {name}_s42_trades.csv", flush=True)
    con.close()
    print("UN02_DONE")


if __name__ == "__main__":
    main()
