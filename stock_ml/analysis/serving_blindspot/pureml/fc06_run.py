# -*- coding: utf-8 -*-
"""FAMILY_CHAMPIONS step 6: vo dich rule-only THAT SU = t2005 n3_dtstop06 (568.3).

fc01 bucket nham (slot giu ml_component_id nhung rule_only_no_ml=true -> ML bypass).
Clone -> fc_rule2, chay 5 seed + dump trades s42.
"""
from __future__ import annotations
import asyncio, copy, json, statistics, sys
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
OUT = Path(__file__).parent
BASE_TMPL = 2005
NAME = "fc_rule2"
SEEDS = [42, 7, 99, 555, 123]
GBX08 = {42: 735.0, 7: 736.7, 99: 728.1, 555: 735.7, 123: 733.3}


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NAME)
        if ex:
            print(f"clone exists: id={ex.id}", flush=True)
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        t = await repo.create(
            name=NAME, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description="FAMILY_CHAMPIONS clone of t2005 n3_dtstop06 (TRUE rule-only champ, ML bypassed; no engine change)",
            hypothesis="best rule-only (rule_only_no_ml, champion engine stack) template; "
                       "multi-seed + trades for family autopsy vs gb_x08",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={NAME}", flush=True)
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, avg_hold "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    tid = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    comps = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=tid, seed=sd)
        rid = r.get("run_id")
        row = read_row(rid)
        comps[sd] = float(row[0]) if row and row[0] is not None else None
        print(f"FC6_RESULT seed={sd} comp={comps[sd]} dGBX08={comps[sd]-GBX08[sd]:+.1f} "
              f"pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} "
              f"hold={row[6]:.1f} run_id={rid}", flush=True)
        if sd == 42:
            import pandas as pd
            con = psycopg2.connect(**PG)
            t = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                            "holding_days, pnl_pct, exit_reason, entry_signal_date from run_trades "
                            "where run_id=%s", con, params=(rid,))
            con.close()
            t.to_csv(OUT / "fc_rule2_s42_trades.csv", index=False)
            print(f"dumped {len(t)} trades -> fc_rule2_s42_trades.csv", flush=True)
    cv = [v for v in comps.values() if v is not None]
    print(f"\n== {NAME} MEAN={statistics.mean(cv):.2f} seeds={comps}")
    print(f"== gb_x08 MEAN={statistics.mean(GBX08.values()):.2f} d={statistics.mean(cv)-statistics.mean(GBX08.values()):+.2f}")
    print("FC06_DONE")


if __name__ == "__main__":
    main()
