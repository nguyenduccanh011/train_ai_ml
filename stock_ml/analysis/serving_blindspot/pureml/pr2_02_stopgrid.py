# -*- coding: utf-8 -*-
"""pr2 (cong to): bien quanh hard_stop -0.10 — chay -0.08/-0.09/-0.11 (zX 2.5, seed 42).

Muc dich buoc toi: neu -0.10 la dinh nhon (curve khong phang) -> overfit 1 tham so.
Clone tu t1058 nhu pm2_01_grid.py, prefix pr2_, KHONG dong canonical.
"""
from __future__ import annotations
import asyncio, copy, csv, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 1058
HERE = Path(__file__).resolve().parent

CELLS = [
    ("pr2_hs08_zx25", -0.08, 2.5),
    ("pr2_hs09_zx25", -0.09, 2.5),
    ("pr2_hs11_zx25", -0.11, 2.5),
]


async def make_clone(name: str, hs: float, zx: float) -> int:
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
        eng["hard_stop_pct"] = hs
        eng["exit_priority"] = ["hard_stop", "signal"]
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=zx,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"pr2 prosecution stop-margin probe: hard_stop {hs}, zX {zx}",
            hypothesis="kiem tra -0.10 co phai dinh nhon (overfit) hay plateau",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} hs={hs}", flush=True)
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
    out = HERE / f"{name}_s{seed}_trades.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(trades)
    worst = min((float(t[6]) for t in trades), default=float("nan"))
    con.close()
    return row, worst, len(trades)


async def make_all() -> dict:
    ids = {}
    try:
        for name, hs, zx in CELLS:
            ids[name] = await make_clone(name, hs, zx)
    finally:
        await async_engine.dispose()
    return ids


def main():
    ids = asyncio.run(make_all())
    print("== reference: hs10 775.1 / hs12 751.9 / hs15 719.4 (zx25 s42) ==", flush=True)
    for name, hs, zx in CELLS:
        r = run_template_experiment(template_id=ids[name], seed=42)
        rid = r.get("run_id")
        row, worst, ntr = read_metrics(rid, name, 42)
        print(f"PR2_CELL {name} hs={hs} comp={float(row[0]):.1f} pnl={float(row[1]):.1f} "
              f"pf={float(row[2]):.2f} mdd={float(row[3]):.3f} tr={row[4]} wr={float(row[5]):.3f} "
              f"hold={float(row[6]):.1f} worst={worst:.3f} run_id={rid}", flush=True)
    print("PR2_STOPGRID_DONE")


if __name__ == "__main__":
    main()
