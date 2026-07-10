# -*- coding: utf-8 -*-
"""pm2 vong 1: luoi tail-cut config-only tren t1058 (pure-ML champ), seed 42.

Truc hard_stop {-0.10,-0.12,-0.15} x truc zX signal_threshold {2.0,2.5,3.0} = 9 o.
hard_stop_pct chi tac dung khi "hard_stop" in exit_priority (dau AM) — trap LINE_A #5.
zX = signal_threshold (decoupled: SELL z(exit) > signal_threshold, experiment.py:2555).

Moi o: clone moi tu t1058 (KHONG dong den 1058/2808 canonical), chay seed 42,
doc leaderboard row + dem lech entry 2023-2025 tu run_trades, dump trades CSV.

Usage: python pm2_01_grid.py            # 9 o luoi
       python pm2_01_grid.py extra      # 2 o chong-nghen max_hold (sau khi biet o tot nhat)
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

# (name, hard_stop_pct, signal_threshold zX, max_hold_bars or None)
GRID = []
for hs_tag, hs in [("hs10", -0.10), ("hs12", -0.12), ("hs15", -0.15)]:
    for zx_tag, zx in [("zx20", 2.0), ("zx25", 2.5), ("zx30", 3.0)]:
        GRID.append((f"pm2_{hs_tag}_{zx_tag}", hs, zx, None))

# vong 1b chong-nghen so: max_hold tren o tot nhat (hs10_zx25 = 775.1 s42)
EXTRA = [
    ("pm2_hs10_zx25_mh120", -0.10, 2.5, 120),
    ("pm2_hs10_zx25_mh200", -0.10, 2.5, 200),
]


async def make_clone(name: str, hs: float, zx: float, max_hold: int | None) -> int:
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
        prio = ["hard_stop", "signal"]
        if max_hold is not None:
            eng["max_hold_bars"] = max_hold
            prio = ["hard_stop", "signal", "max_hold"]
        eng["exit_priority"] = prio
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=zx,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"pm2 tail-cut grid on t1058: hard_stop {hs}, zX {zx}, max_hold {max_hold}",
            hypothesis="t1058 pure-ML thieu tail-cut (counterfactual clip -12% = +140 comp); "
                       "thr2.5 x hstop12/15 la o trong chua ai chay (t1053 chi la thr2.0+hstop15)",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} hs={hs} zx={zx} mh={max_hold}", flush=True)
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
    out = HERE / f"pm2_{name}_s{seed}_trades.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(trades)
    n2325 = sum(1 for t in trades if 2023 <= t[1].year <= 2025)
    nyr = {}
    for t in trades:
        nyr[t[1].year] = nyr.get(t[1].year, 0) + 1
    worst = min((float(t[6]) for t in trades), default=float("nan"))
    con.close()
    return row, n2325, nyr, worst, len(trades)


async def make_all(cells) -> dict:
    """Tao het clone trong MOT event loop (tranh loi asyncpg pool + loop cu tren Windows)."""
    ids = {}
    try:
        for name, hs, zx, mh in cells:
            ids[name] = await make_clone(name, hs, zx, mh)
    finally:
        await async_engine.dispose()
    return ids


def main():
    cells = EXTRA if (len(sys.argv) > 1 and sys.argv[1] == "extra") else GRID
    ids = asyncio.run(make_all(cells))
    results = []
    for name, hs, zx, mh in cells:
        r = run_template_experiment(template_id=ids[name], seed=42)
        rid = r.get("run_id")
        row, n2325, nyr, worst, ntr = read_metrics(rid, name, 42)
        comp = float(row[0]) if row and row[0] is not None else None
        print(f"PM2_CELL {name} hs={hs} zx={zx} mh={mh} comp={comp} "
              f"pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} "
              f"hold={row[6]:.1f} n2325={n2325} worst={worst:.3f} nyr={sorted(nyr.items())} "
              f"run_id={rid}", flush=True)
        results.append((name, hs, zx, mh, comp, float(row[1]), float(row[2]), float(row[3]),
                        int(row[4]), n2325, worst))
    print("\n== PM2 GRID SUMMARY (seed 42; pm_sxrr25 baseline 644.8; champ 729.6; gb_x08 735.0) ==")
    for r in sorted(results, key=lambda x: -(x[4] or 0)):
        print(f"  {r[0]:16s} hs={r[1]} zx={r[2]} mh={r[3]} comp={r[4]:.1f} pnl={r[5]:.1f} "
              f"pf={r[6]:.2f} mdd={r[7]:.3f} tr={r[8]} n2325={r[9]} worst={r[10]:.3f}")
    print("PM2_GRID_DONE")


if __name__ == "__main__":
    main()
