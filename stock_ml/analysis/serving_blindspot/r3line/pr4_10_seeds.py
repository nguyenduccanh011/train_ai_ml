# -*- coding: utf-8 -*-
"""pr4_10: CONG TO T1d — seed-luck check. Clone pr4_mh16 (= t2429 + mh16, y het
r3_mh16/t2907) va chay seeds 7/99/555; dump trades + NAV 2 che do x full/f22/f23.
KHONG dung canonical."""
from __future__ import annotations
import asyncio
import copy
import json
import os
import sys
from pathlib import Path

os.environ["STOCK_DATA_DIR"] = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "r2line" / "na_audit"))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_ID = 2907  # r3_mh16
NAME = "pr4_mh16"
SEEDS = [7, 99, 555]


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NAME)
        if ex:
            print(f"clone exists: {NAME} id={ex.id}", flush=True)
            await async_engine.dispose()
            return ex.id
        base = await repo.get_by_id(BASE_ID)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config)
                                    if isinstance(sl.target_config, str)
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
            description="PR4 cong to: clone r3_mh16 (t2429+mh16), seed-luck 7/99/555",
            hypothesis="T1d: loai seed-luck cho dai dien plateau mh16",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created: {NAME} id={t.id} base={BASE_ID}", flush=True)
        tid = t.id
    await async_engine.dispose()
    return tid


def main():
    tid = asyncio.run(make_clone())
    for seed in SEEDS:
        csv_path = HERE / f"pr4_mh16_s{seed}_trades.csv"
        if not csv_path.exists():
            r = run_template_experiment(template_id=tid, seed=seed)
            con = psycopg2.connect(**PG)
            cur = con.cursor()
            cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, "
                        "avg_hold FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
            row = cur.fetchone()
            print(f"PR4RUN pr4_mh16 s{seed} comp={row[0]:.1f} pnl={row[1]:.1f} pf={row[2]:.2f} "
                  f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f}", flush=True)
            tdf = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                              "holding_days, pnl_pct, exit_reason from run_trades "
                              "where run_id=%s", con, params=(r.get("run_id"),))
            con.close()
            tdf.to_csv(csv_path, index=False)
            print(f"dumped {len(tdf)} -> {csv_path.name}", flush=True)
        for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22"), ("2023-01-01", "f23")):
            sim = NavSim2(str(csv_path), date_lo=lo)
            a = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)
            n = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None, n=20)
            print(f"PR4NAV pr4_mh16 s{seed} {tag}: adv x{a['mean']:.2f}±{a['sd']:.2f} "
                  f"DDw {a['dd_worst']*100:.1f}% | noadv x{n['mean']:.2f}±{n['sd']:.2f} "
                  f"DDw {n['dd_worst']*100:.1f}%", flush=True)
    print("PR4_10_DONE", flush=True)


if __name__ == "__main__":
    main()
