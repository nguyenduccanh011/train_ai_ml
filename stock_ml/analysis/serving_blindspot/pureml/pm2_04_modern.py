# -*- coding: utf-8 -*-
"""pm2 vong 3: 2 bien the hien-dai-hoa NHE tren o tot nhat (pm2_hs10_zx25), van thuan-ML exit.

(a) pm2_hs10_zx25_recovfs: doi feature set dau ENTRY entry_lvup126_lean -> entry_lvup126_recov
    (bo feature hien dai cua champ 2646/gb_x08; config-only o slot, head tu retrain).
(b) pm2_hs10_zx25_snr: chong 3 key exit_snr_extend cua gb_x08 (rule MEM defer-signal cho winner
    trong regime SNR cao — khong force sell, khong chan trade):
    threshold 0.8 / min_gain 0.27 / defer_min_giveback 0.08 (window 20 = default).

Seed 42 truoc. Usage: python pm2_04_modern.py [a|b|ab]
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
BASE = "pm2_hs10_zx25"  # 775.1 s42
HERE = Path(__file__).resolve().parent

SNR = {"exit_snr_extend_threshold": 0.8, "exit_snr_extend_window": 20,
       "exit_snr_min_gain": 0.27, "exit_snr_defer_min_giveback": 0.08}


async def make(variant: str) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        name = f"{BASE}_recovfs" if variant == "a" else f"{BASE}_snr"
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: id={ex.id} name={name}", flush=True)
            return ex.id
        base = await repo.get_by_name(BASE)
        slots = []
        for sl in base.component_slots:
            fs = sl.feature_set_name
            if variant == "a" and sl.slot_type == "entry":
                fs = "entry_lvup126_recov"
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id, "feature_set_name": fs,
                          "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                            else copy.deepcopy(sl.target_config))})
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        if variant == "b":
            eng.update(SNR)
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=("pm2 modernize (a): entry feature set -> entry_lvup126_recov" if variant == "a"
                         else "pm2 modernize (b): + exit_snr_extend keys of gb_x08 (soft defer)"),
            hypothesis="pure-ML exit + tail-cut hs10/zx25 da 775 s42; thu bo feature hien dai / "
                       "runner-extension mem da chung minh tren gb_x08",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name}", flush=True)
        return t.id


async def make_all(variants):
    ids = {}
    try:
        for v in variants:
            ids[v] = await make(v)
    finally:
        await async_engine.dispose()
    return ids


def read_and_dump(run_id: str, name: str, seed: int):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, avg_hold "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    row = cur.fetchone()
    cur.execute("SELECT symbol, entry_date, entry_price, exit_date, exit_price, holding_days, "
                "pnl_pct, exit_reason FROM run_trades WHERE run_id=%s ORDER BY entry_date, symbol", (run_id,))
    rows = cur.fetchall(); con.close()
    out = HERE / f"pm2_{name}_s{seed}_trades.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "entry_date", "entry_price", "exit_date", "exit_price",
                    "holding_days", "pnl_pct", "exit_reason"])
        w.writerows(rows)
    n2325 = sum(1 for t in rows if 2023 <= t[1].year <= 2025)
    return row, n2325


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "ab"
    variants = list(arg)
    ids = asyncio.run(make_all(variants))
    for v in variants:
        name = f"{BASE}_recovfs" if v == "a" else f"{BASE}_snr"
        r = run_template_experiment(template_id=ids[v], seed=42)
        rid = r.get("run_id")
        row, n2325 = read_and_dump(rid, name.replace("pm2_", ""), 42)
        print(f"PM2_MOD {name} comp={float(row[0]):.1f} pnl={row[1]:.1f} pf={row[2]:.2f} "
              f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f} n2325={n2325} "
              f"(base {BASE} 775.1 s42) run_id={rid}", flush=True)
    print("PM2_MODERN_DONE")


if __name__ == "__main__":
    main()
