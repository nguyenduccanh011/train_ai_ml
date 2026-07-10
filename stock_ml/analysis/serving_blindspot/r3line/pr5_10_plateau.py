# -*- coding: utf-8 -*-
"""pr5_10: CONG TO t2936 (ab_noT) — T1 plateau check: bo-T co song o mh12/mh20/mh25
khong hay chi dinh tai 16? Clone t2783 - T_KEYS + max_hold {12,20,25}, seed 42,
cham nh_nav2 CA HAI che do x full/f22/f23/f25. KHONG dung canonical, KHONG commit.
Usage: python pr5_10_plateau.py mh12 mh20 mh25  (cwd = repo root)
"""
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
VARIANTS = {"mh12": 12, "mh20": 20, "mh25": 25}


async def make_clone(variant: str) -> int:
    mh = VARIANTS[variant]
    name = f"pr5_noT_{variant}"
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: {name} id={ex.id}", flush=True)
            await async_engine.dispose()
            return ex.id
        base = await repo.get_by_id(2783)
        slots = []
        for sl in base.component_slots:
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id,
                          "feature_set_name": sl.feature_set_name,
                          "target_config": (json.loads(sl.target_config)
                                            if isinstance(sl.target_config, str)
                                            else copy.deepcopy(sl.target_config))})
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        pri = list(eng.get("exit_priority") or ["trailing_stop", "overext", "signal"])
        if "max_hold" not in pri:
            pri = ["max_hold"] + pri
        eng["exit_priority"] = pri
        eng["max_hold_bars"] = mh
        for k in T_KEYS:
            eng.pop(k, None)
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"pr5 cong to t2936: clone t2783 - trailing_struct + max_hold {mh}",
            hypothesis="pr5_10: plateau bo-T — dinh co chi tai mh16?",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created: {name} id={t.id} mh={mh}", flush=True)
        tid = t.id
    await async_engine.dispose()
    return tid


def run_variant(variant: str):
    tid = asyncio.run(make_clone(variant))
    csv_path = HERE / f"pr5_noT_{variant}_s42_trades.csv"
    if not csv_path.exists():
        r = run_template_experiment(template_id=tid, seed=42)
        con = psycopg2.connect(**PG)
        cur = con.cursor()
        cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, avg_hold "
                    "FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
        row = cur.fetchone()
        print(f"PR5RUN pr5_noT_{variant} s42 comp={row[0]:.1f} pnl={row[1]:.1f} pf={row[2]:.2f} "
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
        a = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)
        n = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None, n=20)
        print(f"PR5NAV pr5_noT_{variant} {tag}: adv x{a['mean']:.2f}±{a['sd']:.2f} "
              f"DDw {a['dd_worst']*100:.1f}% | noadv x{n['mean']:.2f}±{n['sd']:.2f} "
              f"DDw {n['dd_worst']*100:.1f}%", flush=True)


def main():
    for v in sys.argv[1:]:
        if v not in VARIANTS:
            print(f"UNKNOWN {v}")
            continue
        print(f"\n===== VARIANT {v} =====", flush=True)
        try:
            run_variant(v)
        except Exception as e:  # noqa: BLE001
            print(f"PR5FAIL {v}: {e!r}", flush=True)
    print("PR5_10_DONE", flush=True)


if __name__ == "__main__":
    main()
