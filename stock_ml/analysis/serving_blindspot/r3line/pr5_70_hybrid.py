# -*- coding: utf-8 -*-
"""pr5_70: T6 — hybrid thu: t2936 stack (gb - T keys) + head csrank dynamics
(entry_ensemble3 cua t2531) + max_hold 25. 1 run seed 42, cham nh_nav2 2 che do.
PHAI chay cwd=repo root (xsec/breadth path tuong doi). KHONG commit."""
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
T_KEYS = ["trailing_struct_apply_overext", "trailing_struct_donch_win"]
DYN_E3 = {"target": {"type": "continuation_entry_regression", "horizon": 10,
                     "penalty": 0.5, "trend_window": 50},
          "features": "entry_dyn_pure", "norm": "csrank", "z_threshold": 0.88}
NAME = "pr5_dynclean_mh25"


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NAME)
        if ex:
            print(f"clone exists: {NAME} id={ex.id}", flush=True)
            await async_engine.dispose()
            return ex.id
        base = await repo.get_by_id(2783)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config)
                                    if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        pri = list(eng.get("exit_priority") or ["trailing_stop", "overext", "signal"])
        if "max_hold" not in pri:
            pri = ["max_hold"] + pri
        eng["exit_priority"] = pri
        eng["max_hold_bars"] = 25
        eng["entry_ensemble3"] = DYN_E3
        for k in T_KEYS:
            eng.pop(k, None)
        t = await repo.create(
            name=NAME, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description="pr5 T6: t2936 stack (gb-T) + head csrank dyn (e3 t2531) + mh25",
            hypothesis="pr5_70: recent-tilt cua dyncsr88 co cong duoc len nen bo-T khong",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created: {NAME} id={t.id}", flush=True)
        tid = t.id
    await async_engine.dispose()
    return tid


def main():
    tid = asyncio.run(make_clone())
    csv_path = HERE / f"{NAME}_s42_trades.csv"
    if not csv_path.exists():
        r = run_template_experiment(template_id=tid, seed=42)
        con = psycopg2.connect(**PG)
        cur = con.cursor()
        cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, avg_hold "
                    "FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
        row = cur.fetchone()
        print(f"PR5RUN {NAME} s42 comp={row[0]:.1f} pnl={row[1]:.1f} pf={row[2]:.2f} "
              f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f}", flush=True)
        tdf = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                          "holding_days, pnl_pct, exit_reason from run_trades where run_id=%s",
                          con, params=(r.get("run_id"),))
        con.close()
        tdf.to_csv(csv_path, index=False)
        print(f"dumped {len(tdf)} -> {csv_path.name}", flush=True)
    for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22"),
                    ("2023-01-01", "f23"), ("2025-01-01", "f25")):
        sim = NavSim2(str(csv_path), date_lo=lo)
        a = shuffle_stats(sim, K=25, advance_fee=0.0008, n=20)
        n = shuffle_stats(sim, K=25, advance_fee=None, n=20)
        print(f"PR5NAV {NAME} {tag}: adv x{a['mean']:.2f}±{a['sd']:.2f} "
              f"DDw {a['dd_worst']*100:.1f}% | noadv x{n['mean']:.2f}±{n['sd']:.2f} "
              f"DDw {n['dd_worst']*100:.1f}%", flush=True)
    print("PR5_70_DONE", flush=True)


if __name__ == "__main__":
    main()
