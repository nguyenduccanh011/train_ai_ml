# -*- coding: utf-8 -*-
"""HY manh A (run): clone 2783 (KHONG dung canonical) + entry_ensemble5 =
continuation_entry_regression h6/pen1.0/tw70, features entry_lvup126_lean (head t1058),
z_threshold {0.7, 0.9} theo pattern ensemble hien co. Seed 42.

Baseline gb_x08 s42 = 735.0 / pnl 128.5 / 1378 tr.
"""
from __future__ import annotations
import asyncio, copy, csv, json, logging, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # _load_market_breadth (entry_ensemble3) dung duong dan tuong doi market_data/

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2783
HERE = Path(__file__).resolve().parent

CELLS = [("hy_ens5_z07", 0.7), ("hy_ens5_z09", 0.9)]

ENS5 = {
    "target": {"type": "continuation_entry_regression", "horizon": 6,
               "penalty": 1.0, "trend_window": 70},
    "features": "entry_lvup126_lean",
}


async def make_clone(name: str, zthr: float) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: id={ex.id} name={name}", flush=True)
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        ens = copy.deepcopy(ENS5)
        ens["z_threshold"] = zthr
        eng["entry_ensemble5"] = ens
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"hy ens5: gb_x08 + entry_ensemble5 continuation h6 lean (t1058 head), z {zthr}",
            hypothesis="manh A HYBRID_VERDICT: head continuation h6 lean them buy-bars vung doi "
                       "2024-26 cua gb_x08 (offline: add z0.7 = 87/111/469 bar 2024/25/26H1)",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} z={zthr}", flush=True)
        return t.id


def read_metrics(run_id: str, name: str, seed: int):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, avg_hold "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    row = cur.fetchone()
    cur.execute("SELECT symbol, entry_date, entry_price, exit_date, exit_price, holding_days, "
                "pnl_pct, exit_reason FROM run_trades WHERE run_id=%s ORDER BY entry_date, symbol",
                (run_id,))
    trades = cur.fetchall()
    cols = ["symbol", "entry_date", "entry_price", "exit_date", "exit_price",
            "holding_days", "pnl_pct", "exit_reason"]
    out = HERE / f"hy_{name}_s{seed}_trades.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(trades)
    nyr = {}
    pyr = {}
    for t in trades:
        y = t[1].year
        nyr[y] = nyr.get(y, 0) + 1
        pyr[y] = pyr.get(y, 0.0) + float(t[6])
    con.close()
    return row, nyr, {k: round(v, 2) for k, v in pyr.items()}, len(trades)


async def make_all() -> dict:
    ids = {}
    try:
        for name, z in CELLS:
            ids[name] = await make_clone(name, z)
    finally:
        await async_engine.dispose()
    return ids


def main():
    ids = asyncio.run(make_all())
    for name, z in CELLS:
        r = run_template_experiment(template_id=ids[name], seed=42)
        rid = r.get("run_id")
        row, nyr, pyr, ntr = read_metrics(rid, name, 42)
        comp = float(row[0]) if row and row[0] is not None else None
        print(f"HY_CELL {name} z={z} comp={comp} pnl={row[1]:.1f} pf={row[2]:.2f} "
              f"mdd={row[3]:.4f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f} run_id={rid}",
              flush=True)
        print(f"  n theo nam: {sorted(nyr.items())}", flush=True)
        print(f"  pnl theo nam: {sorted(pyr.items())}", flush=True)
    print("HY_ENS5_DONE (baseline gb_x08 s42: comp 735.0 pnl 128.5 tr 1378)")


if __name__ == "__main__":
    main()
